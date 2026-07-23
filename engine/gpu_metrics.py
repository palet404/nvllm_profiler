"""
engine/gpu_metrics.py
nvidia-smi를 subprocess로 호출해 실시간 GPU 지표(VRAM, Utilization)를
수집하는 프로파일링 데몬.

Continuous Batching이 실제로 GPU SM Occupancy를 얼마나 끌어올리는지는
더미 엔진의 시뮬레이션 값만으로는 증명력이 부족하다. 그래서 이 모듈은
백그라운드 스레드에서 실제 하드웨어 카운터(nvidia-smi)를 주기적으로
폴링해, 대시보드가 "실측 GPU 지표"를 함께 보여줄 수 있게 한다.
"""
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

_NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=utilization.gpu,memory.used,memory.total",
    "--format=csv,noheader,nounits",
]


@dataclass
class GPUSnapshot:
    timestamp: float
    gpu_util_pct: float
    mem_used_mb: float
    mem_total_mb: float

    @property
    def mem_util_pct(self) -> float:
        return 100.0 * self.mem_used_mb / self.mem_total_mb if self.mem_total_mb else 0.0


def read_gpu_snapshot() -> Optional[GPUSnapshot]:
    """
    nvidia-smi를 1회 호출해 GPU 사용률/VRAM을 파싱한다.
    CSV 한 줄 예시: "23, 4521, 24564"  (util%, used MB, total MB)
    GPU가 없거나 드라이버 문제로 호출이 실패하면 None을 반환해 호출부가
    graceful하게(대시보드가 죽지 않고) 처리할 수 있게 한다.
    """
    try:
        result = subprocess.run(
            _NVIDIA_SMI_QUERY,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None

    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        return None

    try:
        util, used, total = (float(p) for p in parts)
    except ValueError:
        return None

    return GPUSnapshot(
        timestamp=time.time(),
        gpu_util_pct=util,
        mem_used_mb=used,
        mem_total_mb=total,
    )


class GPUProfilerDaemon:
    """
    백그라운드 스레드에서 일정 주기로 nvidia-smi를 폴링해 히스토리를 쌓는
    '프로파일링 데몬'. 호출 스레드를 막지 않도록 별도 스레드로 동작하며,
    최근 GPU_HISTORY_MAXLEN개의 스냅샷만 보관한다(deque).
    """

    def __init__(self, interval_s: float = 0.5, maxlen: int = 600):
        self._interval_s = interval_s
        self._history: deque[GPUSnapshot] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            snapshot = read_gpu_snapshot()
            if snapshot is not None:
                with self._lock:
                    self._history.append(snapshot)
            self._stop_event.wait(self._interval_s)

    def history(self) -> list[GPUSnapshot]:
        with self._lock:
            return list(self._history)

    def latest(self) -> Optional[GPUSnapshot]:
        with self._lock:
            return self._history[-1] if self._history else None
