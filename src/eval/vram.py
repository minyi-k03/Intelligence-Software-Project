"""VRAM 샘플러. `nvidia-smi`를 0.5초 간격으로 읽어 측정 구간의 최대치를 남긴다.

Reference: docs/EVALUATION.md §4, docs/INTERFACES.md §7.3.
"""

from __future__ import annotations

import subprocess
import threading
from typing import Callable, Optional

# EVALUATION §4: "nvidia-smi 0.5초 간격 최대치". 설계안 고정값이라 설정 키가 아니다.
SAMPLE_INTERVAL_S = 0.5

MemoryQuery = Callable[[], Optional[int]]


def query_gpu_memory_mib() -> Optional[int]:
    """`nvidia-smi`로 지금 사용 중인 GPU 메모리(MiB)를 읽는다.

    GPU가 여러 장이면 가장 큰 값을 쓴다 (측정 환경은 GPU 한 장이다).

    Returns:
        사용 중인 메모리(MiB). `nvidia-smi`가 없거나 실패하면 None.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        return max(int(line.strip()) for line in out.splitlines() if line.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class VramSampler:
    """측정 구간 동안 GPU 메모리를 주기적으로 읽어 최대치를 기록한다.

    `start()`와 `stop()`은 각각 한 번씩 동기로 읽으므로 구간이 짧아도 값이 하나는 남는다.
    질의가 None을 돌려주거나 예외를 던지면 그 표본은 건너뛴다.

    Attributes:
        peak_mib: 지금까지 본 최댓값(MiB). 표본이 하나도 없으면 None.
    """

    def __init__(
        self,
        query: MemoryQuery = query_gpu_memory_mib,
        interval_s: float = SAMPLE_INTERVAL_S,
    ) -> None:
        """
        Args:
            query: GPU 메모리(MiB)를 돌려주는 함수. 테스트에서 가짜로 바꾼다.
            interval_s: 표본 간격(초).
        """
        self._query = query
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.peak_mib: Optional[int] = None

    def _sample(self) -> None:
        """한 번 읽어 최대치를 갱신한다 (실패한 표본은 버린다)."""
        try:
            value = self._query()
        except Exception:
            return
        if value is None:
            return
        with self._lock:
            if self.peak_mib is None or value > self.peak_mib:
                self.peak_mib = value

    def _loop(self) -> None:
        """`stop()`이 올 때까지 간격마다 한 번씩 읽는다 (폴링 스레드 본체)."""
        while not self._stop.wait(self._interval_s):
            self._sample()

    def start(self) -> None:
        """폴링을 시작한다. 시작 시점의 값을 한 번 읽는다."""
        self._stop.clear()
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="vram-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """폴링을 멈추고 마지막으로 한 번 더 읽는다."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._sample()
