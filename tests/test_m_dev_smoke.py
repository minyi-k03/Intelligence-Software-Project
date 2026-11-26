"""PR A-5b 통합 스모크: M을 실제 코퍼스·개발 세트 60문항 전체에 돌려 로그가 규격대로 나오는지 본다.

모델만 가짜다(CPU 전용, Ollama 불필요). 나머지는 모두 진짜다: 동결 해시 검증, BM25 색인(실제
`corpus.jsonl`), 문장 창 압축, LangGraph, 호출 예산·잘림 처리(B-1 콜백), 러너와 로그 기록.
실모델 동작(정확도·잘림률)은 이 테스트가 보증하지 않는다 → `notebooks/verify_a5b.ipynb`.

가짜 모델은 문항 유형을 모른다. 질문 글자만 보고 라우터·근거 판정 응답을 정해, 한 번에 답하기 ·
재검색 · 기권 · 수학 경로가 모두 나오게 한다.
"""

from __future__ import annotations

import dataclasses
import json
import zlib
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("langgraph")

from src.common.types import PipelineResult  # noqa: E402
from src.config.settings import REPO_ROOT, JudgeConfig, load_settings  # noqa: E402
from src.eval.runner import run_round  # noqa: E402
from src.meba.graph import build_m  # noqa: E402
from src.meba.router import ROUTER_SYSTEM_PROMPT  # noqa: E402

FROZEN = REPO_ROOT / "data" / "frozen"
README = REPO_ROOT / "data" / "README.md"
ENV = {"git_commit": "abc1234", "git_dirty": False, "gpu": None, "ollama_version": "test"}
TYPE_NAMES = ("hotpotqa", "squad2", "gsm8k")

# INTERFACES §7.2에서 로그 한 줄에 있어야 하는 메타·채점·측정 필드.
LOG_META_FIELDS = {
    "qid", "type", "subtype", "answer_kind", "pipeline", "bundle", "round", "order_idx", "session_idx",
    "gold", "em", "f1", "numeric_correct", "correct", "timeout", "latency_s", "max_input_tokens",
    "evidence_kept", "is_warmup",
}


_ANSWER, _RETRIEVE, _ABSTAIN = "ANSWER", "RETRIEVE", "ABSTAIN"


class _FakeSampler:
    """GPU가 없는 환경용 가짜 VRAM 샘플러."""

    peak_mib = None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _Model:
    """ChatOllama 대역. 진짜 B-1 콜백을 구동하고, 모델에 간 모든 입력을 기록한다."""

    def __init__(self, callback: Any, kwargs: dict[str, Any], seen: list[str]) -> None:
        self.callback = callback
        self.kwargs = kwargs
        self.seen = seen

    def _reply(self, model_input: Any) -> str:
        """입력 글자만 보고 응답을 정한다 (문항 유형은 모른다)."""
        first = model_input[0]
        system = first["content"] if isinstance(first, dict) else first[1]
        human = model_input[-1]
        human_text = human["content"] if isinstance(human, dict) else human[1]
        if system == ROUTER_SYSTEM_PROMPT:
            digits = sum(ch.isdigit() for ch in human_text)
            return "ROUTE: math\nFinal answer: 42" if digits >= 4 else "ROUTE: search"
        question = human_text.split("\n\nEvidence:")[0].removeprefix("Question: ")
        schema = self.kwargs.get("format_schema")  # json 형식이면 스키마가 있고, tag 형식이면 없다
        if schema is None:
            can_retrieve = "RETRIEVE" in system.split("DECISION: <one of ")[1].split(">")[0]
        else:
            can_retrieve = "RETRIEVE" in schema["properties"]["decision"]["enum"]
        outcome = (_ANSWER, _RETRIEVE, _ABSTAIN)[zlib.crc32(question.encode()) % 3]
        if outcome == _RETRIEVE and can_retrieve:
            decision, missing, answer = _RETRIEVE, f"{max(question.split(), key=len)} history", ""
        elif outcome == _ABSTAIN:
            decision, missing, answer = _ABSTAIN, "", ""
        else:
            decision, missing, answer = _ANSWER, "", "some answer"
        if schema is None:
            return (f"The evidence was weighed.\nDECISION: {decision}\n"
                    f"MISSING: {missing or 'NONE'}\nANSWER: {answer or 'NONE'}")
        return json.dumps({"reasoning": "ok", "decision": decision, "missing": missing, "answer": answer})

    def invoke(self, model_input: Any, **kwargs: Any) -> SimpleNamespace:
        self.seen.append(json.dumps(model_input, ensure_ascii=False, default=str))
        text = self._reply(model_input)
        cap = self.kwargs["num_predict"]
        self.callback.on_chat_model_start({}, [[]], invocation_params={"options": {"num_predict": cap}})
        message = SimpleNamespace(
            content=text,
            usage_metadata={"input_tokens": 50, "output_tokens": 8},
            response_metadata={"done_reason": "stop", "prompt_eval_count": 50, "eval_count": 8},
        )
        self.callback.on_llm_end(
            SimpleNamespace(
                generations=[[SimpleNamespace(message=message, text="", generation_info=None)]], llm_output=None
            )
        )
        return SimpleNamespace(content=text, tool_calls=[])


def _run_dev_round(
    runs_dir: Path, seen: list[str], *, judge_format: str = "json", limit: int | None = None
) -> list[dict[str, Any]]:
    """M을 개발 세트에 한 라운드 돌리고 `m.jsonl`의 줄을 읽어 돌려준다.

    Args:
        runs_dir: 결과 폴더 (테스트 임시 폴더)
        seen: 모델에 간 입력을 모으는 리스트
        judge_format: 설정 `judge.format` 값 (`json` | `tag`)
        limit: 앞에서부터 이 수만큼만 돈다 (중단된 세션 흉내)

    Returns:
        로그 줄 리스트
    """
    settings = dataclasses.replace(load_settings("configs/default.yaml"), judge=JudgeConfig(format=judge_format))

    def factory(**kwargs: Any) -> _Model:
        return _Model(kwargs["callbacks"][0], kwargs, seen)

    meta = run_round(
        bundle="dev", round_name="r1", item_set="dev", settings=settings,
        pipelines={"m": build_m(settings, llm_factory=factory)},
        frozen_dir=FROZEN, readme_path=README, runs_dir=runs_dir,
        environment=ENV, sampler_factory=_FakeSampler, drain_s=2, limit=limit,
    )
    assert meta["settings"]["judge"] == {"format": judge_format}
    path = runs_dir / "dev" / "r1" / "m.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module", params=["json", "tag"])
def smoke(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, list[dict[str, Any]], list[str]]:
    """M을 개발 세트 전체에 한 번 돌린 결과. 판정 형식(json·tag)마다 한 번씩 (모듈 안에서 공유)."""
    tmp = tmp_path_factory.mktemp(f"runs-{request.param}")
    seen: list[str] = []
    return tmp, _run_dev_round(tmp, seen, judge_format=request.param), seen


def test_m_finishes_the_whole_dev_set_without_errors(smoke) -> None:
    """개발 세트 60문항이 모두 로그에 남고, 시간 초과·예외·예산 초과가 없다."""
    _, rows, _ = smoke

    assert [r["order_idx"] for r in rows] == list(range(60))
    assert {r["qid"] for r in rows} == {
        json.loads(line)["qid"] for line in (FROZEN / "dev.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert all(r["error"] is None and r["timeout"] is False for r in rows)
    assert all(r["pipeline"] == "m" and r["bundle"] == "dev" and r["round"] == "r1" for r in rows)
    assert [r["is_warmup"] for r in rows] == [True] + [False] * 59


def test_every_log_line_has_the_interface_fields(smoke) -> None:
    """로그 한 줄에 INTERFACES §7.2의 메타·채점 필드와 `PipelineResult`의 모든 필드가 있다."""
    _, rows, _ = smoke
    expected = LOG_META_FIELDS | {f.name for f in fields(PipelineResult)}

    assert all(expected <= set(r) for r in rows)
    call_fields = {"step", "cap", "input_tokens", "output_tokens", "truncated", "is_retry", "parse_ok", "latency_s"}
    assert all(call_fields <= set(call) for r in rows for call in r["calls"])


def test_log_lines_are_consistent_with_the_m_design(smoke) -> None:
    """호출 수는 예산 4 이하, 경로·판정·검색·압축 기록이 서로 맞고, 세 가지 판정과 수학 경로가 모두 나온다."""
    _, rows, _ = smoke

    for r in rows:
        assert r["route"] in {"math", "search"}
        assert r["n_calls"] <= 4 and r["budget_exhausted"] is False
        assert r["n_calls"] == sum(1 for c in r["calls"] if not c["is_retry"])
        assert len(r["retrieved_doc_ids"]) == len(r["kept_windows"])
        if r["route"] == "math":
            assert r["decisions"] == [] and r["retrieved_doc_ids"] == []
        else:
            assert r["decisions"] and r["decisions"][-1] != "RETRIEVE"
            # 같은 검색이 막히거나 후속 검색어가 없으면 검색 없이 마지막 판정만 받으므로 판정 ≥ 검색.
            assert 1 <= len(r["retrieved_doc_ids"]) <= len(r["decisions"])
            assert r["decisions"].count("RETRIEVE") <= 2  # budget.max_requery
        assert all(c["step"] in {"router", "judge", "finalize"} for c in r["calls"])
        assert all(c["parse_ok"] is True for c in r["calls"] if c["step"] == "judge")

    assert {"math", "search"} == {r["route"] for r in rows}
    assert {"ANSWER", "RETRIEVE", "ABSTAIN"} <= {d for r in rows for d in r["decisions"]}
    assert any(r["abstained"] and r["answer"] == "UNANSWERABLE" for r in rows)
    assert any(r["evidence_kept"] is not None for r in rows)


def test_the_model_never_sees_the_question_type_or_qid(smoke) -> None:
    """모델 입력에는 질문과 근거만 간다 (유형·출처 이름, qid 없음)."""
    _, rows, seen = smoke
    joined = "\n".join(seen).lower()

    assert len(seen) == sum(len(r["calls"]) for r in rows)
    assert not any(name in joined for name in TYPE_NAMES)
    assert not any(r["qid"].lower() in joined for r in rows[:5])


def test_rerunning_the_same_round_resumes_without_new_rows(smoke) -> None:
    """끝난 라운드를 다시 돌려도 로그가 그대로다 (이어 돌기)."""
    tmp, rows, _ = smoke
    meta = json.loads((tmp / "dev" / "r1" / "RUN_META.json").read_text(encoding="utf-8"))
    judge_format = meta["settings"]["judge"]["format"]  # 같은 조건이어야 이어 돌 수 있다

    assert _run_dev_round(tmp, [], judge_format=judge_format) == rows


def test_an_interrupted_round_continues_where_it_stopped(tmp_path: Path) -> None:
    """앞 10문항만 돈 뒤 같은 라운드를 이어 돌리면, 이미 쓴 줄은 그대로 두고 나머지 50문항만 더한다."""
    first_session = _run_dev_round(tmp_path, [], limit=10)
    seen: list[str] = []

    rows = _run_dev_round(tmp_path, seen)

    assert [r["order_idx"] for r in rows] == list(range(60))
    assert rows[:10] == first_session
    assert {r["session_idx"] for r in rows[:10]} == {0} and {r["session_idx"] for r in rows[10:]} == {1}
    assert len(seen) == sum(len(r["calls"]) for r in rows[10:])  # 모델은 남은 50문항에만 불렸다
