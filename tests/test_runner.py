"""러너 단위 테스트 (PR A-3). 가짜 파이프라인만 쓰므로 모델·GPU 없이 돈다.

seam: src/eval/runner.py의 `round_order`, `run_item`, `run_round`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from src.common.types import CallRecord, PipelineResult
from src.config.settings import RunConfig, Settings, load_settings
from src.eval.logger import FrozenDataError
from src.eval.runner import StuckWorkerError, round_order, run_item, run_round

# --- 가짜 문항·파이프라인 ---------------------------------------------------

HOTPOT = {
    "qid": "hotpotqa-0001",
    "type": "hotpotqa",
    "subtype": "bridge",
    "question": "Who wrote it?",
    "answers": ["Tony Benn"],
    "answer_kind": "text",
    "gold_doc_ids": ["hp::Alpha", "hp::Beta"],
    "supporting_facts": [["Alpha", 0], ["Beta", 2]],
    "source_id": "x1",
}
UNANSWERABLE_ITEM = {
    "qid": "squad2-0001",
    "type": "squad2",
    "subtype": "unanswerable",
    "question": "When?",
    "answers": ["UNANSWERABLE"],
    "answer_kind": "unanswerable",
    "gold_doc_ids": ["sq::Doc::0"],
    "supporting_facts": [],
    "source_id": "x2",
}
GSM_ITEM = {
    "qid": "gsm8k-0001",
    "type": "gsm8k",
    "subtype": None,
    "question": "How many?",
    "answers": ["18"],
    "answer_kind": "numeric",
    "gold_doc_ids": [],
    "supporting_facts": [],
    "source_id": "gsm8k-test-3",
}


def _result(answer: str, **overrides: Any) -> PipelineResult:
    """기본값이 채워진 PipelineResult."""
    fields: dict[str, Any] = dict(
        answer=answer,
        abstained=answer == "UNANSWERABLE",
        budget_exhausted=False,
        route=None,
        decisions=[],
        n_calls=1,
        n_retries=0,
    )
    fields.update(overrides)
    return PipelineResult(**fields)


def _call(input_tokens: int, output_tokens: int = 5) -> CallRecord:
    return CallRecord(
        step="p0",
        cap=512,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        truncated=False,
        is_retry=False,
        parse_ok=None,
        latency_s=0.1,
    )


def _fixed(result: PipelineResult) -> Callable[[str, str], PipelineResult]:
    """항상 같은 결과를 돌려주는 가짜 파이프라인."""
    return lambda qid, question: result


def _run(item: dict, pipeline: Callable[[str, str], PipelineResult], **overrides: Any) -> dict:
    """기본 인자로 run_item을 부른다."""
    args: dict[str, Any] = dict(
        pipeline_name="p0",
        bundle="main",
        round_name="r1",
        order_idx=3,
        timeout_s=5,
        window=1,
        is_warmup=False,
        drain_s=2,
    )
    args.update(overrides)
    return run_item(item, pipeline, **args)


# --- round_order ----------------------------------------------------------

_PIPELINES = ["p0", "p0_raw", "p1", "m"]


def test_round_order_rotates_each_round() -> None:
    """라운드마다 시작 파이프라인이 바뀐다 (한 파이프라인이 늘 먼저·나중에 돌지 않게)."""
    assert round_order(_PIPELINES, "r1") == ["p0", "p0_raw", "p1", "m"]
    assert round_order(_PIPELINES, "r2") == ["p0_raw", "p1", "m", "p0"]
    assert round_order(_PIPELINES, "r3") == ["p1", "m", "p0", "p0_raw"]


def test_round_order_m_has_different_position_in_each_round() -> None:
    """본 비교 3라운드에서 M이 같은 순번에 놓이지 않는다."""
    positions = {round_order(_PIPELINES, r).index("m") for r in ("r1", "r2", "r3")}
    assert len(positions) == 3


def test_round_order_keeps_all_pipelines_exactly_once() -> None:
    """순서만 바뀌고 파이프라인은 하나도 빠지거나 겹치지 않는다."""
    for r in ("r1", "r2", "r3", "r4"):
        assert sorted(round_order(_PIPELINES, r)) == sorted(_PIPELINES)


def test_round_order_rejects_unknown_round() -> None:
    """`r<번호>` 형식이 아니면 거부한다."""
    with pytest.raises(ValueError):
        round_order(_PIPELINES, "round1")


# --- run_item: 로그 한 줄 -----------------------------------------------------


def test_run_item_row_has_meta_result_and_score() -> None:
    """로그 한 줄 = 문항 메타 + PipelineResult + 채점 (INTERFACES §7.2)."""
    result = _result("Tony Benn", n_calls=2, calls=[_call(300), _call(900)], decisions=["ANSWER"])
    row = _run(HOTPOT, _fixed(result), order_idx=7, is_warmup=False)
    assert row["qid"] == "hotpotqa-0001"
    assert row["type"] == "hotpotqa" and row["subtype"] == "bridge" and row["answer_kind"] == "text"
    assert row["pipeline"] == "p0" and row["bundle"] == "main" and row["round"] == "r1"
    assert row["order_idx"] == 7
    assert row["gold"] == ["Tony Benn"]
    assert row["answer"] == "Tony Benn" and row["n_calls"] == 2 and row["decisions"] == ["ANSWER"]
    assert row["em"] == 1.0 and row["f1"] == 1.0 and row["correct"] is True
    assert row["numeric_correct"] is None
    assert row["timeout"] is False and row["error"] is None
    assert row["is_warmup"] is False
    assert row["max_input_tokens"] == 900
    assert [c["input_tokens"] for c in row["calls"]] == [300, 900]


def test_run_item_row_is_json_serializable() -> None:
    """한 줄이 그대로 JSON으로 써진다 (CallRecord도 dict로 풀려 있다)."""
    row = _run(HOTPOT, _fixed(_result("Tony Benn", calls=[_call(10)])))
    assert json.loads(json.dumps(row, ensure_ascii=False)) == row


def test_run_item_passes_only_qid_and_question_to_pipeline() -> None:
    """파이프라인에는 qid와 question만 넘긴다 (type·answers 등은 숨긴다)."""
    seen: list[tuple] = []

    def spy(*args: Any, **kwargs: Any) -> PipelineResult:
        seen.append((args, kwargs))
        return _result("x")

    _run(HOTPOT, spy)
    assert seen == [(("hotpotqa-0001", "Who wrote it?"), {})]


def test_run_item_scores_numeric_item() -> None:
    """수치 문항은 수치 비교로 채점하고 EM·F1은 null이다."""
    row = _run(GSM_ITEM, _fixed(_result("Final answer: 18")))
    assert row["numeric_correct"] is True and row["correct"] is True
    assert row["em"] is None and row["f1"] is None


def test_run_item_scores_unanswerable_item() -> None:
    """답할 수 없음 문항은 기권해야 정답이고, EM·F1은 같은 값이다."""
    abstain = _run(UNANSWERABLE_ITEM, _fixed(_result("UNANSWERABLE")))
    wrong = _run(UNANSWERABLE_ITEM, _fixed(_result("1998")))
    assert abstain["correct"] is True and abstain["em"] == abstain["f1"] == 1.0
    assert wrong["correct"] is False and wrong["em"] == wrong["f1"] == 0.0


def test_run_item_warmup_flag_is_passed_through() -> None:
    """워밍업 표시는 호출자가 정한 값 그대로 기록한다."""
    assert _run(HOTPOT, _fixed(_result("a")), is_warmup=True)["is_warmup"] is True


def test_run_item_latency_is_measured_by_runner() -> None:
    """시간은 문항 입력 ~ 최종 답을 러너가 직접 잰다."""

    def slow(qid: str, question: str) -> PipelineResult:
        time.sleep(0.05)
        return _result("a", calls=[_call(10)])

    assert _run(HOTPOT, slow)["latency_s"] >= 0.05


def test_run_item_without_calls_has_zero_max_input_tokens() -> None:
    """호출 기록이 없으면 최대 입력 토큰은 0이다."""
    assert _run(HOTPOT, _fixed(_result("a")))["max_input_tokens"] == 0


# --- run_item: 예외·타임아웃 ---------------------------------------------------


def test_run_item_exception_becomes_wrong_row_with_error() -> None:
    """파이프라인이 예외를 던져도 러너는 멈추지 않고 오답 한 줄을 남긴다."""

    def broken(qid: str, question: str) -> PipelineResult:
        raise ValueError("boom")

    row = _run(HOTPOT, broken)
    assert row["error"] is not None and "ValueError" in row["error"] and "boom" in row["error"]
    assert row["answer"] == "" and row["correct"] is False
    assert row["timeout"] is False


def test_run_item_timeout_marks_wrong_even_if_answer_would_be_right() -> None:
    """제한 시간을 넘기면 늦게 나올 답과 상관없이 오답이다."""
    done = threading.Event()

    def slow_but_right(qid: str, question: str) -> PipelineResult:
        try:
            for _ in range(300):
                time.sleep(0.01)
        finally:
            done.set()
        return _result("Tony Benn")

    start = time.perf_counter()
    row = _run(HOTPOT, slow_but_right, timeout_s=0.1)
    assert time.perf_counter() - start < 2.0
    assert row["timeout"] is True and row["correct"] is False
    assert row["error"] == "timeout"
    assert row["latency_s"] >= 0.1


def test_run_item_timeout_stops_the_worker_before_next_item() -> None:
    """시간 초과한 작업은 다음 문항이 시작되기 전에 멈춘다 (다음 문항의 시간·VRAM 오염 방지)."""
    finished = threading.Event()

    def endless(qid: str, question: str) -> PipelineResult:
        try:
            while True:
                time.sleep(0.01)
        finally:
            finished.set()

    _run(HOTPOT, endless, timeout_s=0.1, drain_s=5)
    assert finished.is_set()


def test_run_item_stops_the_round_when_timed_out_worker_cannot_be_cancelled() -> None:
    """모델 호출에서 막혀 취소가 닿지 않는 작업이 남으면, 다음 문항을 오염시키지 않도록 멈춘다."""
    release = threading.Event()

    def blocked(qid: str, question: str) -> PipelineResult:
        release.wait()  # 파이썬 코드를 실행하지 않는 대기라 비동기 취소 신호가 닿지 않는다
        return _result("late")

    try:
        with pytest.raises(StuckWorkerError):
            _run(HOTPOT, blocked, timeout_s=0.1, drain_s=0.2)
    finally:
        release.set()


def test_run_item_normal_completion_is_not_a_timeout() -> None:
    """제한 시간 안에 끝난 작업은 취소되지 않고 정상 채점된다."""
    row = _run(HOTPOT, _fixed(_result("Tony Benn")), timeout_s=5)
    assert row["timeout"] is False and row["correct"] is True


# --- run_item: evidence_kept ----------------------------------------------


def _kept(*windows: str) -> PipelineResult:
    """Alpha·Beta 두 문단을 검색해 주어진 문장 창을 남긴 결과."""
    return _result(
        "Tony Benn",
        retrieved_doc_ids=[["hp::Alpha", "hp::Beta", "hp::Noise"]],
        kept_windows=[list(windows)],
    )


def test_evidence_kept_true_when_all_supporting_sentences_are_covered() -> None:
    """근거 문장(Alpha#0, Beta#2)이 남긴 창(앞뒤 1문장)에 모두 들어 있으면 True."""
    row = _run(HOTPOT, _fixed(_kept("hp::Alpha#0", "hp::Beta#1")), window=1)
    assert row["evidence_kept"] is True


def test_evidence_kept_false_when_a_supporting_sentence_is_dropped() -> None:
    """근거 문장 하나라도 창 밖이면 False."""
    row = _run(HOTPOT, _fixed(_kept("hp::Alpha#0", "hp::Beta#5")), window=1)
    assert row["evidence_kept"] is False


def test_evidence_kept_uses_window_radius() -> None:
    """창 반지름이 0이면 가운데 문장만 인정한다."""
    row = _run(HOTPOT, _fixed(_kept("hp::Alpha#0", "hp::Beta#1")), window=0)
    assert row["evidence_kept"] is False


def test_evidence_kept_ignores_gold_paragraph_that_was_never_retrieved() -> None:
    """검색이 못 찾은 문단의 근거 문장은 압축 탓이 아니므로 세지 않는다 (C-2 측정과 같은 기준)."""
    result = _result(
        "x",
        retrieved_doc_ids=[["hp::Alpha"]],
        kept_windows=[["hp::Alpha#0"]],
    )
    assert _run(HOTPOT, _fixed(result), window=1)["evidence_kept"] is True


def test_evidence_kept_is_none_without_compression() -> None:
    """압축을 쓰지 않았으면(kept_windows 비어 있음) null이다."""
    assert _run(HOTPOT, _fixed(_result("x", retrieved_doc_ids=[["hp::Alpha"]])))["evidence_kept"] is None


def test_evidence_kept_is_none_for_non_hotpot_items() -> None:
    """HotpotQA가 아니면 null이다."""
    result = _result("18", kept_windows=[["hp::Alpha#0"]], retrieved_doc_ids=[["hp::Alpha"]])
    assert _run(GSM_ITEM, _fixed(result))["evidence_kept"] is None


def test_evidence_kept_merges_rounds_and_duplicate_title_doc_ids() -> None:
    """재검색 회차의 창도 합쳐 보고, 같은 제목의 `::2` 문단도 gold_doc_ids로 찾는다."""
    item = {**HOTPOT, "gold_doc_ids": ["hp::Alpha::2", "hp::Beta"]}
    result = _result(
        "x",
        retrieved_doc_ids=[["hp::Alpha::2"], ["hp::Beta"]],
        kept_windows=[["hp::Alpha::2#0"], ["hp::Beta#2"]],
    )
    assert _run(item, _fixed(result), window=1)["evidence_kept"] is True


# --- run_round --------------------------------------------------------------

_ENV = {
    "git_commit": "abc1234",
    "git_dirty": False,
    "gpu": {"name": "Tesla T4", "driver": "550.54"},
    "ollama_version": "0.5.1",
}
_EVAL_ITEMS = [HOTPOT, UNANSWERABLE_ITEM, GSM_ITEM]
_DEV_ITEMS = [{**HOTPOT, "qid": "hotpotqa-0040"}, {**GSM_ITEM, "qid": "gsm8k-0040"}]
_NL = "\n"


class _FakeSampler:
    """start/stop 호출을 기록하고 정해 둔 최대치를 돌려주는 가짜 VRAM 샘플러."""

    events: list[str] = []

    def __init__(self) -> None:
        self.peak_mib: Optional[int] = None

    def start(self) -> None:
        _FakeSampler.events.append("start")
        self.peak_mib = 7000 + len(_FakeSampler.events)

    def stop(self) -> None:
        _FakeSampler.events.append("stop")


@pytest.fixture(autouse=True)
def _reset_fake_sampler() -> None:
    _FakeSampler.events = []


def _write_jsonl(path: Path, rows: list[dict]) -> bytes:
    data = "".join(json.dumps(r, ensure_ascii=False) + _NL for r in rows).encode("utf-8")
    path.write_bytes(data)
    return data


class _Env:
    """임시 동결 폴더(+README 해시 표)와 결과 폴더."""

    def __init__(self, tmp_path: Path) -> None:
        self.frozen = tmp_path / "frozen"
        self.frozen.mkdir()
        files = {
            "eval.jsonl": _write_jsonl(self.frozen / "eval.jsonl", _EVAL_ITEMS),
            "dev.jsonl": _write_jsonl(self.frozen / "dev.jsonl", _DEV_ITEMS),
            "corpus.jsonl": _write_jsonl(self.frozen / "corpus.jsonl", [{"doc_id": "hp::Alpha"}]),
        }
        header = ["| 파일 | 수 | SHA-256 |", "|---|---|---|"]
        rows = [f"| `frozen/{n}` | 1 | `{hashlib.sha256(d).hexdigest()}` |" for n, d in files.items()]
        self.readme = tmp_path / "README.md"
        self.readme.write_text(_NL.join(header + rows) + _NL, encoding="utf-8")
        self.runs = tmp_path / "runs"

    def round_dir(self, bundle: str = "main", round_name: str = "r1") -> Path:
        return self.runs / bundle / round_name

    def rows(self, pipeline: str, bundle: str = "main", round_name: str = "r1") -> list[dict]:
        path = self.round_dir(bundle, round_name) / f"{pipeline}.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def meta(self, bundle: str = "main", round_name: str = "r1") -> dict:
        path = self.round_dir(bundle, round_name) / "RUN_META.json"
        return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def env(tmp_path: Path) -> _Env:
    return _Env(tmp_path)


@pytest.fixture
def settings() -> Settings:
    return load_settings()


class _Spy:
    """파이프라인 호출 순서를 (파이프라인 이름, qid)로 모으는 가짜 파이프라인 묶음."""

    def __init__(self, names: list[str], fail_on: Optional[tuple[str, str]] = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self._fail_on = fail_on
        self.pipelines = {n: self._make(n) for n in names}

    def _make(self, name: str) -> Callable[[str, str], PipelineResult]:
        def pipeline(qid: str, question: str) -> PipelineResult:
            self.calls.append((name, qid))
            if self._fail_on == (name, qid):
                raise ValueError("boom")
            return _result("Tony Benn", calls=[_call(100)])

        return pipeline


def _round(env: _Env, settings: Settings, spy: _Spy, **overrides: Any) -> dict:
    args: dict[str, Any] = dict(
        bundle="main",
        round_name="r1",
        item_set="eval",
        settings=settings,
        pipelines=spy.pipelines,
        frozen_dir=env.frozen,
        readme_path=env.readme,
        runs_dir=env.runs,
        environment=_ENV,
        sampler_factory=_FakeSampler,
        drain_s=2,
    )
    args.update(overrides)
    return run_round(**args)


def test_run_round_writes_one_log_per_pipeline_with_every_item(env: _Env, settings: Settings) -> None:
    """파이프라인마다 `<bundle>/<round>/<pipeline>.jsonl`에 평가 세트 전 문항을 순서대로 쓴다."""
    _round(env, settings, _Spy(["p0", "m"]))
    for name in ("p0", "m"):
        rows = env.rows(name)
        assert [r["qid"] for r in rows] == [i["qid"] for i in _EVAL_ITEMS]
        assert [r["order_idx"] for r in rows] == [0, 1, 2]
        assert {r["pipeline"] for r in rows} == {name}
        assert {r["bundle"] for r in rows} == {"main"} and {r["round"] for r in rows} == {"r1"}


def test_run_round_marks_only_first_item_of_each_pipeline_as_warmup(env: _Env, settings: Settings) -> None:
    """워밍업은 파이프라인마다 첫 문항 하나다 (모든 파이프라인이 같은 조건)."""
    _round(env, settings, _Spy(["p0", "m"]))
    for name in ("p0", "m"):
        assert [r["is_warmup"] for r in env.rows(name)] == [True, False, False]


def test_run_round_runs_pipelines_in_rotated_order(env: _Env, settings: Settings) -> None:
    """라운드 이름에 따라 파이프라인 실행 순서가 바뀌고, RUN_META에도 같은 순서가 남는다."""
    spy = _Spy(["p0", "p0_raw", "p1", "m"])
    meta = _round(env, settings, spy, round_name="r2")
    executed = list(dict.fromkeys(name for name, _ in spy.calls))
    assert executed == ["p0_raw", "p1", "m", "p0"]
    assert meta["pipeline_order"] == executed


def test_run_round_gives_every_pipeline_the_same_items_in_the_same_order(env: _Env, settings: Settings) -> None:
    """파이프라인 사이에 문항과 순서가 같다 (비교 공정성)."""
    spy = _Spy(["p0", "p1"])
    _round(env, settings, spy)
    per_pipeline = {n: [q for p, q in spy.calls if p == n] for n in ("p0", "p1")}
    assert per_pipeline["p0"] == per_pipeline["p1"] == [i["qid"] for i in _EVAL_ITEMS]


def test_run_round_records_run_meta_with_per_pipeline_vram(env: _Env, settings: Settings) -> None:
    """RUN_META에 설정 스냅샷·해시·환경·종료 시각·파이프라인별 최대 VRAM이 들어간다."""
    _round(env, settings, _Spy(["p0", "m"]))
    meta = env.meta()
    assert meta["settings"] == settings.to_dict()
    assert meta["model_tag"] == settings.model.tag
    assert set(meta["data_hashes"]) == {"eval.jsonl", "dev.jsonl", "corpus.jsonl"}
    assert meta["git_commit"] == "abc1234" and meta["gpu"]["name"] == "Tesla T4"
    assert meta["finished_at"] is not None
    assert set(meta["max_vram_mib_by_pipeline"]) == {"p0", "m"}
    assert meta["max_vram_mib"] == max(meta["max_vram_mib_by_pipeline"].values())


def test_run_round_samples_vram_around_each_pipeline_separately(env: _Env, settings: Settings) -> None:
    """파이프라인마다 샘플러를 따로 시작·종료해 파이프라인별 최대치를 얻는다."""
    _round(env, settings, _Spy(["p0", "m"]))
    assert _FakeSampler.events == ["start", "stop", "start", "stop"]


def test_run_round_resumes_after_partial_log(env: _Env, settings: Settings) -> None:
    """이미 기록된 문항은 다시 돌리지 않고 남은 문항만 이어서 쓴다."""
    _round(env, settings, _Spy(["p0"]))
    log = env.round_dir() / "p0.jsonl"
    first_line = log.read_bytes().split(_NL.encode())[0]
    log.write_bytes(first_line + _NL.encode())  # 첫 문항만 남기고 끊긴 상태

    spy = _Spy(["p0"])
    _round(env, settings, spy)

    assert [q for _, q in spy.calls] == ["squad2-0001", "gsm8k-0001"]
    rows = env.rows("p0")
    assert [r["qid"] for r in rows] == [i["qid"] for i in _EVAL_ITEMS]
    assert [r["is_warmup"] for r in rows] == [True, False, False]
    assert len(env.meta()["sessions"]) == 2


def test_run_round_skips_pipeline_that_is_already_complete(env: _Env, settings: Settings) -> None:
    """끝난 파이프라인은 호출도, VRAM 샘플링도 하지 않는다."""
    _round(env, settings, _Spy(["p0", "m"]))
    _FakeSampler.events = []
    spy = _Spy(["p0", "m"])
    _round(env, settings, spy)
    assert spy.calls == []
    assert _FakeSampler.events == []


def test_run_round_keeps_vram_and_unfinished_state_when_interrupted(env: _Env, settings: Settings) -> None:
    """도중에 끊겨도 그때까지의 VRAM은 RUN_META에 남고, 종료 시각은 비워 둔다. 이어 돌면 끝까지 간다."""
    seen: list[str] = []

    def interrupt_after_two(row: dict) -> None:
        seen.append(row["qid"])
        if len(seen) == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _round(env, settings, _Spy(["p0", "m"]), on_row=interrupt_after_two)
    interrupted = env.meta()
    assert interrupted["finished_at"] is None
    assert interrupted["max_vram_mib_by_pipeline"]["p0"] is not None
    assert _FakeSampler.events[-1] == "stop"  # 샘플러도 정리됐다

    _round(env, settings, _Spy(["p0", "m"]))
    assert [r["qid"] for r in env.rows("p0")] == [i["qid"] for i in _EVAL_ITEMS]
    assert [r["qid"] for r in env.rows("m")] == [i["qid"] for i in _EVAL_ITEMS]
    assert env.meta()["finished_at"] is not None


def test_run_round_refuses_resume_with_changed_settings(env: _Env, settings: Settings) -> None:
    """설정을 바꿔 이어 돌리면 한 라운드에 조건이 섞이므로 거부한다."""
    from src.eval.logger import RunMetaMismatch

    _round(env, settings, _Spy(["p0"]))
    changed = dataclasses.replace(settings, run=RunConfig(timeout_s=settings.run.timeout_s + 1))
    with pytest.raises(RunMetaMismatch):
        _round(env, changed, _Spy(["p0"]))


def test_run_round_stops_before_anything_runs_if_frozen_data_changed(env: _Env, settings: Settings) -> None:
    """동결 파일이 바뀌었으면 파이프라인도 로그도 만들지 않고 멈춘다."""
    (env.frozen / "eval.jsonl").write_bytes(b"tampered")
    spy = _Spy(["p0"])
    with pytest.raises(FrozenDataError):
        _round(env, settings, spy)
    assert spy.calls == []
    assert not env.runs.exists()


def test_run_round_continues_after_pipeline_exception(env: _Env, settings: Settings) -> None:
    """한 문항에서 예외가 나도 다음 문항을 계속 돌고, 그 문항은 오답 한 줄로 남는다."""
    spy = _Spy(["p0"], fail_on=("p0", "squad2-0001"))
    _round(env, settings, spy)
    rows = env.rows("p0")
    assert [r["qid"] for r in rows] == [i["qid"] for i in _EVAL_ITEMS]
    assert rows[1]["error"] is not None and rows[1]["correct"] is False
    assert rows[2]["error"] is None


def test_run_round_enforces_timeout_from_settings(env: _Env, settings: Settings) -> None:
    """제한 시간은 설정(`run.timeout_s`)에서 읽는다."""
    short = dataclasses.replace(settings, run=RunConfig(timeout_s=1))

    def slow(qid: str, question: str) -> PipelineResult:
        for _ in range(500):
            time.sleep(0.01)
        return _result("Tony Benn")

    spy = _Spy([])
    spy.pipelines = {"p0": slow}
    _round(env, short, spy, limit=1)
    row = env.rows("p0")[0]
    assert row["timeout"] is True and row["correct"] is False


def test_run_round_limit_runs_only_first_items(env: _Env, settings: Settings) -> None:
    """`limit`은 앞에서부터 그 수만큼만 돈다 (검증 모드)."""
    spy = _Spy(["p0"])
    _round(env, settings, spy, bundle="_scratch", limit=2)
    assert [q for _, q in spy.calls] == ["hotpotqa-0001", "squad2-0001"]


def test_run_round_dev_bundle_uses_dev_set(env: _Env, settings: Settings) -> None:
    """개발 세트 묶음은 dev.jsonl을 읽는다."""
    spy = _Spy(["p0"])
    _round(env, settings, spy, bundle="dev", item_set="dev")
    assert [q for _, q in spy.calls] == ["hotpotqa-0040", "gsm8k-0040"]


@pytest.mark.parametrize(
    ("bundle", "item_set"),
    [("main", "dev"), ("ablation", "dev"), ("dev", "eval"), ("nonsense", "eval"), ("main", "other")],
)
def test_run_round_rejects_wrong_bundle_and_item_set_pair(
    env: _Env, settings: Settings, bundle: str, item_set: str
) -> None:
    """정식 묶음(main·ablation)은 평가 세트, dev는 개발 세트만 쓴다. 섞으면 정식 폴더가 오염된다."""
    with pytest.raises(ValueError):
        _round(env, settings, _Spy(["p0"]), bundle=bundle, item_set=item_set)
