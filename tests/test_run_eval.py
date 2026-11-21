"""측정 러너 CLI(`scripts/run_eval.py`) 테스트 (PR A-3). 모델·GPU 없이 가짜 파이프라인으로 돈다.

seam: `scripts.run_eval.main(argv, registry, frozen_dir, readme_path)`.
정상 경로는 실제 `data/frozen/`을 **읽기만** 하고, 변조 경로는 임시 폴더에 만든 가짜 동결 파일을 쓴다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from scripts.run_eval import main
from src.common.types import PipelineResult
from src.config.settings import Settings


def _answer(qid: str, question: str) -> PipelineResult:
    """항상 같은 답을 내는 가짜 파이프라인."""
    return PipelineResult(
        answer="UNANSWERABLE",
        abstained=True,
        budget_exhausted=False,
        route=None,
        decisions=[],
        n_calls=1,
        n_retries=0,
    )


class _Factories:
    """파이프라인 생성 함수 등록부. 어떤 이름이 언제 만들어졌는지 기록한다."""

    def __init__(self, *names: str) -> None:
        self.built: list[str] = []
        self.registry: dict[str, Callable[[Settings], Any]] = {n: self._factory(n) for n in names}

    def _factory(self, name: str) -> Callable[[Settings], Any]:
        def build(settings: Settings) -> Callable[[str, str], PipelineResult]:
            self.built.append(name)
            return _answer

        return build


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_verify_mode_writes_first_n_items_to_scratch(tmp_path: Path) -> None:
    """`--verify N`은 앞 N문항만 돌려 `_scratch/<round>/`에 규격대로 남긴다 (정식 묶음을 줘도 _scratch)."""
    factories = _Factories("p0", "m")
    main(
        ["--verify", "2", "--bundle", "main", "--pipelines", "p0,m", "--runs-dir", str(tmp_path)],
        registry=factories.registry,
    )
    round_dir = tmp_path / "_scratch" / "r1"
    assert not (tmp_path / "main").exists()
    for name in ("p0", "m"):
        rows = _rows(round_dir / f"{name}.jsonl")
        assert len(rows) == 2
        assert [r["is_warmup"] for r in rows] == [True, False]
        assert {r["bundle"] for r in rows} == {"_scratch"} and {r["session_idx"] for r in rows} == {0}
    meta = json.loads((round_dir / "RUN_META.json").read_text(encoding="utf-8"))
    assert meta["item_set"] == "dev" and meta["pipeline_order"] == ["p0", "m"]
    assert meta["finished_at"] is not None


def _tampered_frozen(tmp_path: Path) -> tuple[Path, Path]:
    """README 표와 해시가 맞지 않는 가짜 동결 폴더를 만든다."""
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    rows = []
    for name in ("eval.jsonl", "dev.jsonl", "corpus.jsonl"):
        (frozen / name).write_bytes(b"{}\n")
        rows.append(f"| `frozen/{name}` | 1 | `{hashlib.sha256(b'original').hexdigest()}` |")
    readme = tmp_path / "README.md"
    readme.write_text("\n".join(["| 파일 | 수 | SHA-256 |", "|---|---|---|", *rows]) + "\n", encoding="utf-8")
    return frozen, readme


def test_frozen_hash_mismatch_stops_before_any_pipeline_is_built(tmp_path: Path) -> None:
    """동결 데이터가 바뀌었으면 파이프라인(검색 인덱스 등)을 만들기 전에 멈춘다."""
    frozen, readme = _tampered_frozen(tmp_path)
    factories = _Factories("p0")
    with pytest.raises(SystemExit, match="해시"):
        main(
            ["--verify", "1", "--pipelines", "p0", "--runs-dir", str(tmp_path / "runs")],
            registry=factories.registry,
            frozen_dir=frozen,
            readme_path=readme,
        )
    assert factories.built == []
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("pipelines", [",", " , "])
def test_empty_pipeline_list_is_rejected(tmp_path: Path, pipelines: str) -> None:
    """`--pipelines`가 비어 있으면 인자 오류로 멈춘다."""
    factories = _Factories("p0")
    with pytest.raises(SystemExit):
        main(["--verify", "1", "--pipelines", pipelines, "--runs-dir", str(tmp_path)], registry=factories.registry)
    assert factories.built == []


_SEEN_SETTINGS: list[Any] = []


def run_fake(qid: str, question: str, *, settings: Settings | None = None) -> PipelineResult:
    """`call:` 등록용 가짜 실행 함수. 받은 설정을 기록한다."""
    _SEEN_SETTINGS.append(settings)
    return _answer(qid, question)


def test_call_style_entry_gets_settings_bound(tmp_path: Path) -> None:
    """`call:모듈:함수` 항목은 `fn(qid, question, *, settings=...)`에 설정을 묶어 (qid, question)으로 돈다."""
    _SEEN_SETTINGS.clear()
    main(
        ["--verify", "2", "--pipelines", "p0", "--runs-dir", str(tmp_path)],
        registry={"p0": f"call:{__name__}:run_fake"},
    )
    rows = _rows(tmp_path / "_scratch" / "r1" / "p0.jsonl")
    assert len(rows) == 2 and len(_SEEN_SETTINGS) == 2
    assert all(isinstance(s, Settings) for s in _SEEN_SETTINGS)


def test_call_style_entry_with_missing_function_is_rejected(tmp_path: Path) -> None:
    """`call:` 항목도 함수를 못 찾으면 어떤 항목인지 밝히며 멈춘다."""
    entry = f"call:{__name__}:nope"
    with pytest.raises(SystemExit, match="nope"):
        main(["--verify", "1", "--pipelines", "p0", "--runs-dir", str(tmp_path)], registry={"p0": entry})


def test_unknown_pipeline_is_rejected(tmp_path: Path) -> None:
    """등록되지 않은 이름은 등록된 이름 목록과 함께 멈춘다."""
    with pytest.raises(SystemExit, match="p0"):
        main(["--verify", "1", "--pipelines", "nope", "--runs-dir", str(tmp_path)], registry=_Factories("p0").registry)
