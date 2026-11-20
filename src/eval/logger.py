"""측정 로거: 동결 데이터 해시 검증, 문항별 즉시 기록·이어 돌기, `RUN_META.json`.

Reference: docs/INTERFACES.md §1.3, §7, docs/EVALUATION.md §4.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

# 측정 시작 전에 해시를 확인하는 동결 파일 (INTERFACES §1.3).
FROZEN_FILES = ("eval.jsonl", "dev.jsonl", "corpus.jsonl")

# `data/README.md`의 동결 기록 표 한 줄: | `frozen/eval.jsonl` | 120 | `<sha256>` |
_README_HASH_ROW = re.compile(r"\|\s*`frozen/([\w.]+)`\s*\|[^|]*\|\s*`([0-9a-f]{64})`\s*\|")

_NEWLINE = b"\n"


class FrozenDataError(RuntimeError):
    """동결 데이터가 기록된 해시와 다르거나, 비교할 수 없을 때."""


def sha256_file(path: Path) -> str:
    """파일 바이트의 SHA-256(16진수)을 돌려준다.

    Args:
        path: 파일 경로.

    Returns:
        소문자 16진수 해시.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_hashes(frozen_dir: Path, readme_path: Path) -> dict[str, str]:
    """동결 파일 세 개의 해시를 `data/README.md` 표와 비교한다. 다르면 멈춘다.

    Args:
        frozen_dir: `eval.jsonl`, `dev.jsonl`, `corpus.jsonl`이 있는 폴더.
        readme_path: 동결 기록 표가 있는 `data/README.md`.

    Returns:
        파일 이름 -> 다시 계산한 SHA-256.

    Raises:
        FrozenDataError: 파일이 없거나, README에 기준 해시가 없거나, 해시가 다를 때.
    """
    expected = dict(_README_HASH_ROW.findall(readme_path.read_text(encoding="utf-8")))
    actual: dict[str, str] = {}
    for name in FROZEN_FILES:
        if name not in expected:
            raise FrozenDataError(f"{readme_path}에 {name}의 기준 해시가 없다")
        path = frozen_dir / name
        if not path.is_file():
            raise FrozenDataError(f"동결 파일이 없다: {path}")
        actual[name] = sha256_file(path)
        if actual[name] != expected[name]:
            raise FrozenDataError(
                f"{name} 해시가 다르다 (기록 {expected[name]}, 실제 {actual[name]}). "
                "동결 데이터가 바뀌었으므로 측정을 멈춘다."
            )
    return actual


class RunLog:
    """파이프라인 한 개의 문항별 `.jsonl` 로그. 한 줄씩 즉시 쓰고, 끊긴 세션을 이어 돌게 한다.

    파일을 열 때 기존 줄을 읽어 `done_qids`를 만든다. 개행으로 끝나지 않은 **마지막 조각**은
    쓰다 끊긴 것으로 보고 버린다(파일에서도 잘라내며, 그 문항은 다시 돈다).
    개행으로 끝난 줄이 JSON이 아니면 로그 손상이므로 멈춘다.

    Attributes:
        path: 로그 파일 경로.
        done_qids: 이미 기록된 문항의 `qid` 집합.
    """

    def __init__(self, path: Path) -> None:
        """
        Args:
            path: `.jsonl` 경로. 폴더가 없으면 만든다.

        Raises:
            ValueError: 개행으로 끝난 줄이 JSON이 아니거나 `qid`가 없을 때.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done_qids: set[str] = set()
        if self.path.exists():
            self._recover()

    def _recover(self) -> None:
        """기존 파일을 읽어 `done_qids`를 채우고, 불완전한 마지막 조각을 잘라낸다."""
        data = self.path.read_bytes()
        complete, _, _ = data.rpartition(_NEWLINE)  # 마지막 개행 앞까지가 온전한 줄들
        has_complete_line = _NEWLINE in data
        good_bytes = len(complete) + 1 if has_complete_line else 0
        if has_complete_line:
            for number, line in enumerate(complete.split(_NEWLINE), start=1):
                try:
                    self.done_qids.add(json.loads(line)["qid"])
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{self.path} {number}번째 줄이 깨져 있다: {exc}") from exc
        if good_bytes != len(data):
            with self.path.open("r+b") as f:
                f.truncate(good_bytes)

    def append(self, row: dict[str, Any]) -> None:
        """한 줄을 쓰고 디스크까지 내린다.

        Args:
            row: `qid`를 포함한 문항 로그 한 줄.
        """
        line = json.dumps(row, ensure_ascii=False).encode("utf-8") + _NEWLINE
        with self.path.open("ab") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        self.done_qids.add(row["qid"])


# --- RUN_META.json --------------------------------------------------------

CommandRunner = Callable[[tuple[str, ...]], Optional[str]]


def _run_cmd(cmd: tuple[str, ...]) -> Optional[str]:
    """명령을 실행해 표준 출력(양끝 공백 제거)을 돌려준다. 실패하면 None.

    Args:
        cmd: 실행할 명령과 인자.

    Returns:
        표준 출력 문자열. 명령이 없거나 0이 아닌 코드로 끝나면 None.
    """
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def collect_environment(run_cmd: CommandRunner = _run_cmd) -> dict[str, Any]:
    """측정 환경(git 커밋, GPU, Ollama 버전)을 모은다. 알 수 없는 값은 None이다.

    Ollama 버전은 CLI(`ollama --version`)로 읽는다. 모델 호출 경로(LangChain)와 별개다.

    Args:
        run_cmd: 명령 실행 함수. 테스트에서 가짜로 바꾼다.

    Returns:
        `git_commit`, `git_dirty`, `gpu`(`name`, `driver`), `ollama_version`.
    """
    commit = run_cmd(("git", "rev-parse", "HEAD"))
    status = run_cmd(("git", "status", "--porcelain"))
    gpu_line = run_cmd(("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"))
    name: Optional[str] = None
    driver: Optional[str] = None
    if gpu_line:
        # GPU가 여러 장이면 첫 줄만 기록한다 (측정 환경은 GPU 한 장).
        name, _, driver = (part.strip() for part in gpu_line.splitlines()[0].partition(","))
    version = run_cmd(("ollama", "--version"))
    match = re.search(r"\d+(?:\.\d+)+\S*", version) if version else None
    return {
        "git_commit": commit or None,
        "git_dirty": None if status is None else bool(status),
        "gpu": {"name": name, "driver": driver},
        "ollama_version": match.group(0) if match else None,
    }


class RunMetaMismatch(RuntimeError):
    """이어 돌려는 라운드의 조건이 처음 시작할 때와 다를 때."""


def _now() -> str:
    """UTC 현재 시각(ISO 8601)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _peak(values: list[Optional[int]]) -> Optional[int]:
    """None을 건너뛴 최댓값. 값이 없으면 None."""
    present = [v for v in values if v is not None]
    return max(present) if present else None


class RunMeta:
    """한 라운드의 `RUN_META.json`. 세션마다 환경을 덧붙이고, 이어 돌 때 조건이 같은지 확인한다.

    같은 라운드 안에서는 설정·데이터 해시·실행 순서·묶음·라운드·문항 세트가 같아야 한다.
    GPU·드라이버·git 커밋은 세션마다 다를 수 있으므로 `sessions`에 각각 남긴다.
    """

    # 이어 돌 때 처음과 같아야 하는 항목.
    _FIXED = ("bundle", "round", "item_set", "settings", "data_hashes", "pipeline_order")

    def __init__(self, path: Path) -> None:
        """
        Args:
            path: `RUN_META.json` 경로.
        """
        self.path = Path(path)
        self.data: dict[str, Any] = {}

    def begin(
        self,
        *,
        bundle: str,
        round_name: str,
        item_set: str,
        settings: dict[str, Any],
        data_hashes: dict[str, str],
        order: list[str],
        environment: dict[str, Any],
        now: Optional[str] = None,
    ) -> None:
        """세션을 시작한다. 파일이 이미 있으면 이어 돌기로 보고 조건을 비교한다.

        Args:
            bundle: `main` | `ablation` | `dev` | `_scratch`.
            round_name: `r1` | `r2` | `r3`.
            item_set: `eval` | `dev`.
            settings: 설정 전체 스냅샷 (`Settings.to_dict()`).
            data_hashes: 동결 파일 이름 -> SHA-256.
            order: 이 라운드의 파이프라인 실행 순서.
            environment: `collect_environment()` 결과.
            now: 시작 시각(테스트용). 없으면 현재 시각.

        Raises:
            RunMetaMismatch: 이어 돌기인데 설정·데이터·순서 등이 처음과 다를 때.
        """
        started = now or _now()
        fresh = {
            "bundle": bundle,
            "round": round_name,
            "item_set": item_set,
            "model_tag": settings["model"]["tag"],
            "settings": settings,
            "data_hashes": data_hashes,
            "pipeline_order": list(order),
        }
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            for key in self._FIXED:
                if self.data.get(key) != fresh[key]:
                    raise RunMetaMismatch(
                        f"{self.path}: 이어 돌려는 라운드의 `{key}`가 처음과 다르다. "
                        "한 라운드에 조건이 섞이므로 새 라운드로 돌려야 한다."
                    )
        else:
            self.data = {
                **fresh,
                "started_at": started,
                "finished_at": None,
                "max_vram_mib": None,
                "max_vram_mib_by_pipeline": {},
                "sessions": [],
            }
        self.data["sessions"].append({"started_at": started, "finished_at": None, **environment})
        self.data["finished_at"] = None
        self._refresh_environment_view(environment)
        self._write()

    def _refresh_environment_view(self, environment: dict[str, Any]) -> None:
        """최상위의 git·GPU·Ollama 항목을 가장 최근 세션 값으로 맞춘다."""
        for key in ("git_commit", "git_dirty", "gpu", "ollama_version"):
            self.data[key] = environment[key]

    def finish(self, *, vram_by_pipeline: dict[str, Optional[int]], now: Optional[str] = None) -> None:
        """세션을 끝낸다. 종료 시각과 최대 VRAM(이전 세션과 합쳐 최댓값)을 쓴다.

        Args:
            vram_by_pipeline: 파이프라인 이름 -> 이번 세션에서 본 최대 VRAM(MiB), 없으면 None.
            now: 종료 시각(테스트용). 없으면 현재 시각.
        """
        ended = now or _now()
        self.data["sessions"][-1]["finished_at"] = ended
        self.data["finished_at"] = ended
        self.record_vram(vram_by_pipeline)

    def record_vram(self, vram_by_pipeline: dict[str, Optional[int]]) -> None:
        """파이프라인별 최대 VRAM을 기존 값과 합쳐(최댓값) 쓴다. 종료 시각은 건드리지 않는다.

        세션이 중간에 끊겨도 그때까지 본 VRAM을 남기려고 `finish`와 따로 둔다.

        Args:
            vram_by_pipeline: 파이프라인 이름 -> 최대 VRAM(MiB), 표본이 없으면 None.
        """
        merged: dict[str, Optional[int]] = dict(self.data["max_vram_mib_by_pipeline"])
        for name, peak in vram_by_pipeline.items():
            merged[name] = _peak([merged.get(name), peak])
        self.data["max_vram_mib_by_pipeline"] = merged
        self.data["max_vram_mib"] = _peak(list(merged.values()))
        self._write()

    def _write(self) -> None:
        """임시 파일에 쓴 뒤 바꿔치기해서, 쓰는 도중 끊겨도 파일이 깨지지 않게 한다."""
        tmp = self.path.with_name(self.path.name + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        os.replace(tmp, self.path)
