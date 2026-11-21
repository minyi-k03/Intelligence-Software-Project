"""로거·VRAM·러너 단위 테스트 (PR A-3). 모델·GPU·nvidia-smi 없이 돈다.

seam: src/eval/vram.py, src/eval/logger.py, src/eval/runner.py의 공개 함수·클래스.
실제 Ollama·GPU 값은 여기서 다루지 않는다 (수동 검증 모드 `--verify`로 확인).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from src.eval import vram
from src.eval.logger import (
    FrozenDataError,
    RunLog,
    RunMeta,
    RunMetaMismatch,
    collect_environment,
    verify_frozen_hashes,
)
from src.eval.vram import VramSampler, query_gpu_memory_mib


# --- query_gpu_memory_mib --------------------------------------------------


def _fake_nvidia_smi(monkeypatch: pytest.MonkeyPatch, stdout: str | None) -> None:
    """`nvidia-smi` 출력을 정해 둔다. stdout이 None이면 명령이 없는 환경이다."""

    def fake_run(cmd, **kwargs):
        if stdout is None:
            raise FileNotFoundError(cmd[0])
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(vram.subprocess, "run", fake_run)


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("5120\n", 5120),
        ("5120\n7300\n\n", 7300),  # GPU 두 장이면 큰 값, 빈 줄은 무시
        ("", None),  # 출력 없음
        ("[N/A]\n", None),  # 값을 못 읽음
        (None, None),  # nvidia-smi 없음
    ],
    ids=["one-gpu", "two-gpus", "empty", "not-available", "missing"],
)
def test_query_gpu_memory_reads_nvidia_smi_output(
    monkeypatch: pytest.MonkeyPatch, stdout: str | None, expected: int | None
) -> None:
    """nvidia-smi `memory.used`(MiB) 출력을 숫자로 읽고, 못 읽으면 None이다."""
    _fake_nvidia_smi(monkeypatch, stdout)
    assert query_gpu_memory_mib() == expected


# --- VramSampler ----------------------------------------------------------


class _FakeGpu:
    """질의할 때마다 미리 정한 값을 차례로 돌려주는 가짜 nvidia-smi."""

    def __init__(self, values: list) -> None:
        self._values = list(values)
        self._last = values[-1]
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            self.calls += 1
            value = self._values.pop(0) if self._values else self._last
        if isinstance(value, Exception):
            raise value
        return value


def _wait_until(predicate, timeout_s: float = 5.0) -> bool:
    """조건이 참이 될 때까지 기다린다 (스레드 표본을 결정적으로 기다리는 용도)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_sampler_reports_peak_of_samples() -> None:
    """측정 구간에서 본 값 중 최댓값을 최대 VRAM으로 돌려준다."""
    gpu = _FakeGpu([1000, 5000, 3000, 2000])
    sampler = VramSampler(query=gpu, interval_s=0.01)
    sampler.start()
    assert _wait_until(lambda: gpu.calls >= 4)
    sampler.stop()
    assert sampler.peak_mib == 5000


def test_sampler_short_interval_still_has_start_and_stop_samples() -> None:
    """구간이 폴링 간격보다 짧아도 시작·종료 시점 값은 남는다."""
    gpu = _FakeGpu([1200, 7000])
    sampler = VramSampler(query=gpu, interval_s=60.0)
    sampler.start()
    sampler.stop()
    assert sampler.peak_mib == 7000
    assert gpu.calls == 2


def test_sampler_without_nvidia_smi_has_no_peak() -> None:
    """질의가 계속 None이면(nvidia-smi 없음) 최대치는 None이고 예외가 없다."""
    sampler = VramSampler(query=_FakeGpu([None]), interval_s=0.01)
    sampler.start()
    sampler.stop()
    assert sampler.peak_mib is None


def test_sampler_ignores_failed_samples() -> None:
    """질의가 예외를 던져도 그 표본만 버리고 이전 최댓값을 지킨다."""
    gpu = _FakeGpu([4000, RuntimeError("boom"), RuntimeError("boom")])
    sampler = VramSampler(query=gpu, interval_s=0.01)
    sampler.start()
    assert _wait_until(lambda: gpu.calls >= 3)
    sampler.stop()
    assert sampler.peak_mib == 4000


def test_sampler_stop_ends_polling_thread() -> None:
    """stop() 뒤에는 더 읽지 않는다."""
    gpu = _FakeGpu([1000])
    sampler = VramSampler(query=gpu, interval_s=0.01)
    sampler.start()
    sampler.stop()
    calls_at_stop = gpu.calls
    time.sleep(0.05)
    assert gpu.calls == calls_at_stop


# --- verify_frozen_hashes -------------------------------------------------


def _make_frozen(tmp_path: Path, contents: dict[str, bytes]) -> tuple[Path, Path]:
    """가짜 frozen 폴더와, 그 해시 표가 든 README를 만든다.

    Args:
        tmp_path: 작업 폴더.
        contents: 파일 이름 -> 바이트. README 표에는 이 바이트의 SHA-256을 적는다.

    Returns:
        (frozen 폴더, README 경로)
    """
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    for name, data in contents.items():
        (frozen / name).write_bytes(data)
    return frozen, _write_readme(tmp_path / "README.md", contents)


def _write_readme(path: Path, contents: dict[str, bytes]) -> Path:
    """`data/README.md`와 같은 모양의 해시 표를 쓴다."""
    rows = [f"| `frozen/{n}` | 3 | `{hashlib.sha256(d).hexdigest()}` |" for n, d in contents.items()]
    header = ["# data", "", "| 파일 | 수 | SHA-256 |", "|---|---|---|"]
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    return path


_FROZEN_FILES = {"eval.jsonl": b"e\n", "dev.jsonl": b"d\n", "corpus.jsonl": b"c\n"}


def test_verify_frozen_hashes_returns_hashes_when_all_match(tmp_path: Path) -> None:
    """README 표의 해시와 파일 바이트 해시가 모두 같으면 해시를 돌려준다."""
    frozen, readme = _make_frozen(tmp_path, _FROZEN_FILES)
    result = verify_frozen_hashes(frozen, readme)
    assert result == {n: hashlib.sha256(d).hexdigest() for n, d in _FROZEN_FILES.items()}


def test_verify_frozen_hashes_stops_on_changed_file(tmp_path: Path) -> None:
    """파일이 한 바이트라도 바뀌면 멈춘다 (평가 세트 사후 변경 방지)."""
    frozen, readme = _make_frozen(tmp_path, _FROZEN_FILES)
    (frozen / "eval.jsonl").write_bytes(b"e changed\n")
    with pytest.raises(FrozenDataError, match="eval.jsonl"):
        verify_frozen_hashes(frozen, readme)


def test_verify_frozen_hashes_stops_on_missing_file(tmp_path: Path) -> None:
    """파일이 없으면 멈춘다."""
    frozen, readme = _make_frozen(tmp_path, _FROZEN_FILES)
    (frozen / "dev.jsonl").unlink()
    with pytest.raises(FrozenDataError, match="dev.jsonl"):
        verify_frozen_hashes(frozen, readme)


def test_verify_frozen_hashes_stops_when_readme_lacks_a_hash(tmp_path: Path) -> None:
    """기준 해시가 README에 없으면 비교할 수 없으므로 멈춘다."""
    frozen, _ = _make_frozen(tmp_path, _FROZEN_FILES)
    two_only = {k: v for k, v in _FROZEN_FILES.items() if k != "corpus.jsonl"}
    readme = _write_readme(tmp_path / "README_two.md", two_only)
    with pytest.raises(FrozenDataError, match="corpus.jsonl"):
        verify_frozen_hashes(frozen, readme)


# --- RunLog ---------------------------------------------------------------


def _lines(path: Path) -> list[dict]:
    """로그 파일의 모든 줄을 JSON으로 읽는다 (줄 하나라도 깨져 있으면 예외)."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_runlog_writes_each_row_immediately(tmp_path: Path) -> None:
    """append가 끝난 시점에 이미 파일에 한 줄이 들어 있다 (세션이 끊겨도 남는다)."""
    path = tmp_path / "results" / "runs" / "dev" / "r1" / "p0.jsonl"
    log = RunLog(path)
    log.append({"qid": "q1", "answer": "한글"})
    assert _lines(path) == [{"qid": "q1", "answer": "한글"}]
    log.append({"qid": "q2"})
    assert [r["qid"] for r in _lines(path)] == ["q1", "q2"]


def test_runlog_resume_reports_finished_qids(tmp_path: Path) -> None:
    """같은 파일을 다시 열면 이미 기록한 qid를 알려 주고, 이어서 쓴다."""
    path = tmp_path / "p1.jsonl"
    first = RunLog(path)
    first.append({"qid": "q1"})
    first.append({"qid": "q2"})

    resumed = RunLog(path)
    assert resumed.done_qids == {"q1", "q2"}
    resumed.append({"qid": "q3"})
    assert [r["qid"] for r in _lines(path)] == ["q1", "q2", "q3"]


def test_runlog_new_file_has_no_finished_qids(tmp_path: Path) -> None:
    """파일이 없으면 처음부터 돈다."""
    assert RunLog(tmp_path / "new.jsonl").done_qids == set()


def test_runlog_drops_half_written_last_line_on_resume(tmp_path: Path) -> None:
    """쓰다 끊겨 깨진 마지막 줄은 버리고, 그 문항은 다시 돌린다."""
    path = tmp_path / "m.jsonl"
    path.write_bytes(b'{"qid": "q1"}\n{"qid": "q2"}\n{"qid": "q3", "answ')
    log = RunLog(path)
    assert log.done_qids == {"q1", "q2"}
    log.append({"qid": "q3"})
    assert [r["qid"] for r in _lines(path)] == ["q1", "q2", "q3"]


def test_runlog_empty_file_starts_from_scratch(tmp_path: Path) -> None:
    """빈 파일(만들기만 하고 끊김)은 기록이 없는 것으로 본다."""
    path = tmp_path / "m.jsonl"
    path.write_bytes(b"")
    log = RunLog(path)
    assert log.done_qids == set()
    log.append({"qid": "q1"})
    assert [r["qid"] for r in _lines(path)] == ["q1"]


def test_runlog_drops_file_holding_only_a_half_written_line(tmp_path: Path) -> None:
    """첫 줄을 쓰다 끊겨 개행이 하나도 없으면 그 조각을 지우고 처음부터 쓴다."""
    path = tmp_path / "m.jsonl"
    path.write_bytes('{"qid": "q1", "answer": "한'.encode("utf-8")[:-1])  # 한글 바이트 중간에서 끊김
    log = RunLog(path)
    assert log.done_qids == set()
    log.append({"qid": "q1"})
    assert path.read_bytes() == b'{"qid": "q1"}\n'


def test_runlog_accepts_crlf_lines_written_elsewhere(tmp_path: Path) -> None:
    """다른 도구가 CRLF로 저장한 로그도 이어 돌 수 있다."""
    path = tmp_path / "m.jsonl"
    path.write_bytes(b'{"qid": "q1"}\r\n{"qid": "q2"}\r\n')
    assert RunLog(path).done_qids == {"q1", "q2"}


def test_runlog_refuses_corrupt_line_in_the_middle(tmp_path: Path) -> None:
    """마지막이 아닌 줄이 깨져 있으면 이어 쓰지 않고 멈춘다 (로그가 손상된 것)."""
    path = tmp_path / "m.jsonl"
    path.write_bytes(b'{"qid": "q1"}\nnot json\n{"qid": "q3"}\n')
    with pytest.raises(ValueError, match="2"):
        RunLog(path)


def test_runlog_writes_utf8_with_lf_newlines(tmp_path: Path) -> None:
    """데이터 파일과 같게 UTF-8·LF로 쓴다 (Windows에서도 CRLF가 섞이지 않는다)."""
    path = tmp_path / "u.jsonl"
    RunLog(path).append({"qid": "q1", "answer": "한글"})
    raw = path.read_bytes()
    assert raw == '{"qid": "q1", "answer": "한글"}\n'.encode("utf-8")


# --- RunMeta --------------------------------------------------------------

_ENV = {
    "git_commit": "abc1234",
    "git_dirty": False,
    "gpu": {"name": "Tesla T4", "driver": "550.54"},
    "ollama_version": "0.5.1",
}
_SETTINGS = {"model": {"tag": "qwen2.5:7b-instruct-q4_K_M", "temperature": 0}, "run": {"timeout_s": 120}}
_HASHES = {"eval.jsonl": "a" * 64, "dev.jsonl": "b" * 64, "corpus.jsonl": "c" * 64}
_ORDER = ["p0", "p0_raw", "p1", "m"]


def _begin(meta: RunMeta, **overrides) -> None:
    """기본 인자로 세션을 시작한다 (테스트마다 바꿀 것만 overrides로 받는다)."""
    args = dict(
        bundle="main",
        round_name="r1",
        item_set="eval",
        settings=_SETTINGS,
        data_hashes=_HASHES,
        order=_ORDER,
        environment=_ENV,
        now="2026-11-28T09:00:00+00:00",
    )
    args.update(overrides)
    meta.begin(**args)


def _read_meta(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_runmeta_begin_records_everything_required_by_interfaces(tmp_path: Path) -> None:
    """INTERFACES §7.3의 항목: 커밋·해시·설정 스냅샷·GPU·Ollama·모델 태그·실행 순서·시작 시각."""
    path = tmp_path / "RUN_META.json"
    _begin(RunMeta(path))
    data = _read_meta(path)
    assert data["git_commit"] == "abc1234"
    assert data["data_hashes"] == _HASHES
    assert data["settings"] == _SETTINGS
    assert data["gpu"] == {"name": "Tesla T4", "driver": "550.54"}
    assert data["ollama_version"] == "0.5.1"
    assert data["model_tag"] == "qwen2.5:7b-instruct-q4_K_M"
    assert data["pipeline_order"] == _ORDER
    assert data["bundle"] == "main" and data["round"] == "r1" and data["item_set"] == "eval"
    assert data["started_at"] == "2026-11-28T09:00:00+00:00"
    assert data["finished_at"] is None


def test_runmeta_finish_records_end_time_and_peak_vram(tmp_path: Path) -> None:
    """종료 시각과 최대 VRAM(전체·파이프라인별)을 남긴다. 전체는 파이프라인별 최댓값이다."""
    path = tmp_path / "RUN_META.json"
    meta = RunMeta(path)
    _begin(meta)
    meta.finish(vram_by_pipeline={"p0": 5100, "p1": 7300, "m": None}, now="2026-11-28T10:00:00+00:00")
    data = _read_meta(path)
    assert data["finished_at"] == "2026-11-28T10:00:00+00:00"
    assert data["max_vram_mib_by_pipeline"] == {"p0": 5100, "p1": 7300, "m": None}
    assert data["max_vram_mib"] == 7300


def test_runmeta_peak_vram_is_none_when_no_gpu_sample(tmp_path: Path) -> None:
    """GPU 표본이 하나도 없으면(nvidia-smi 없음) 최대 VRAM은 None이다."""
    path = tmp_path / "RUN_META.json"
    meta = RunMeta(path)
    _begin(meta)
    meta.finish(vram_by_pipeline={"p0": None}, now="2026-11-28T10:00:00+00:00")
    assert _read_meta(path)["max_vram_mib"] is None


def test_runmeta_resume_keeps_first_start_and_merges_peak_vram(tmp_path: Path) -> None:
    """끊긴 라운드를 이어 돌면 세션을 덧붙이고, 시작은 처음 것, 최대 VRAM은 세션들의 최댓값이다."""
    path = tmp_path / "RUN_META.json"
    first = RunMeta(path)
    _begin(first)
    first.finish(vram_by_pipeline={"p0": 5000, "p1": 6000}, now="2026-11-28T09:30:00+00:00")

    resumed = RunMeta(path)
    _begin(resumed, environment={**_ENV, "git_commit": "def5678"}, now="2026-11-28T11:00:00+00:00")
    resumed.finish(vram_by_pipeline={"p1": 6500, "m": 8000}, now="2026-11-28T12:00:00+00:00")

    data = _read_meta(path)
    assert data["started_at"] == "2026-11-28T09:00:00+00:00"
    assert data["finished_at"] == "2026-11-28T12:00:00+00:00"
    assert data["max_vram_mib_by_pipeline"] == {"p0": 5000, "p1": 6500, "m": 8000}
    assert data["max_vram_mib"] == 8000
    assert [s["git_commit"] for s in data["sessions"]] == ["abc1234", "def5678"]


def test_runmeta_resume_after_crash_keeps_unfinished_session(tmp_path: Path) -> None:
    """finish 없이 끊긴 세션도 기록에 남고, 이어 돈 뒤 종료 시각은 마지막 세션 것이다."""
    path = tmp_path / "RUN_META.json"
    _begin(RunMeta(path))  # 여기서 끊김 (finish 없음)
    resumed = RunMeta(path)
    _begin(resumed, now="2026-11-28T11:00:00+00:00")
    assert _read_meta(path)["finished_at"] is None
    resumed.finish(vram_by_pipeline={"p0": 100}, now="2026-11-28T12:00:00+00:00")
    data = _read_meta(path)
    assert len(data["sessions"]) == 2
    assert data["sessions"][0]["finished_at"] is None
    assert data["finished_at"] == "2026-11-28T12:00:00+00:00"


@pytest.mark.parametrize(
    "override",
    [
        {"settings": {**_SETTINGS, "run": {"timeout_s": 60}}},
        {"data_hashes": {**_HASHES, "eval.jsonl": "d" * 64}},
        {"order": ["p1", "m", "p0", "p0_raw"]},
        {"round_name": "r2"},
        {"item_set": "dev"},
        {"bundle": "ablation"},
    ],
    ids=["settings", "data-hash", "order", "round", "item-set", "bundle"],
)
def test_runmeta_resume_refuses_when_conditions_changed(tmp_path: Path, override: dict) -> None:
    """설정·데이터·실행 순서가 달라진 채로 이어 돌면 한 라운드에 조건이 섞이므로 거부한다."""
    path = tmp_path / "RUN_META.json"
    _begin(RunMeta(path))
    with pytest.raises(RunMetaMismatch):
        _begin(RunMeta(path), **override)


def test_collect_environment_parses_tool_output() -> None:
    """git·nvidia-smi·ollama 출력을 읽어 환경 정보로 만든다."""
    outputs = {
        ("git", "rev-parse", "HEAD"): "0123456789abcdef0123456789abcdef01234567",
        ("git", "status", "--porcelain"): " M src/eval/runner.py",
        ("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"): "Tesla T4, 550.54.15",
        ("ollama", "--version"): "ollama version is 0.5.1",
    }

    def fake_run(cmd: tuple[str, ...]):
        return outputs.get(cmd)

    env = collect_environment(run_cmd=fake_run)
    assert env["git_commit"] == "0123456789abcdef0123456789abcdef01234567"
    assert env["git_dirty"] is True
    assert env["gpu"] == {"name": "Tesla T4", "driver": "550.54.15"}
    assert env["ollama_version"] == "0.5.1"


@pytest.mark.skipif(shutil.which("git") is None, reason="git이 없는 환경")
def test_collect_environment_reads_repo_commit_from_any_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """레포 밖(예: Colab의 /content)에서 실행해도 이 레포의 커밋을 기록한다."""
    monkeypatch.chdir(tmp_path)
    env = collect_environment()
    assert env["git_commit"] is not None and re.fullmatch(r"[0-9a-f]{40}", env["git_commit"])


def test_collect_environment_missing_tools_give_none() -> None:
    """도구가 없거나 실패해도 예외 없이 None으로 남긴다 (로컬 CPU 환경 등)."""
    env = collect_environment(run_cmd=lambda cmd: None)
    assert env == {"git_commit": None, "git_dirty": None, "gpu": {"name": None, "driver": None}, "ollama_version": None}


def test_runmeta_record_vram_keeps_peak_without_finishing(tmp_path: Path) -> None:
    """중단된 세션의 VRAM도 보존한다: 최대치만 합치고 종료 시각은 비워 둔다."""
    path = tmp_path / "RUN_META.json"
    meta = RunMeta(path)
    _begin(meta)
    meta.record_vram({"p0": 5200, "p1": None})
    data = _read_meta(path)
    assert data["max_vram_mib_by_pipeline"] == {"p0": 5200, "p1": None}
    assert data["max_vram_mib"] == 5200
    assert data["finished_at"] is None
    assert data["sessions"][-1]["finished_at"] is None
