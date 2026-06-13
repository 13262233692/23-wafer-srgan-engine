import gc
import logging
import threading
import time
from typing import Optional, Callable, List, Dict
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    timestamp: float
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    fragmentation_ratio: float


@dataclass
class OOMWarning:
    timestamp: float
    severity: str
    usage_ratio: float
    message: str


class GPUMemoryMonitor:
    def __init__(
        self,
        device: torch.device,
        check_interval: float = 2.0,
        warning_threshold: float = 0.75,
        critical_threshold: float = 0.90,
        oom_threshold: float = 0.95,
        auto_reclaim: bool = True,
        history_size: int = 100,
        callback: Optional[Callable[[OOMWarning], None]] = None,
    ):
        self.device = device
        self.check_interval = check_interval
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.oom_threshold = oom_threshold
        self.auto_reclaim = auto_reclaim
        self.history_size = history_size
        self.callback = callback

        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._history: List[MemorySnapshot] = []
        self._warnings: List[OOMWarning] = []
        self._oom_count = 0
        self._reclaim_count = 0

        self._reclaim_callbacks: List[Callable[[], None]] = []

        self._cuda_available = torch.cuda.is_available() and device.type == "cuda"

    def add_reclaim_callback(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._reclaim_callbacks.append(callback)

    def start(self) -> None:
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            logger.warning("Memory monitor already running")
            return

        if not self._cuda_available:
            logger.info("CUDA not available, running memory monitor in CPU-only mode")

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name=f"GPUMemoryMonitor-{self.device}",
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info(f"GPU Memory Monitor started on {self.device}")

    def stop(self) -> None:
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        logger.info(f"GPU Memory Monitor stopped on {self.device}")

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                snapshot = self._take_snapshot()
                self._record_snapshot(snapshot)
                self._check_and_respond(snapshot)
            except Exception as e:
                logger.debug(f"Memory monitor error: {e}")

            self._stop_event.wait(self.check_interval)

    def _take_snapshot(self) -> Optional[MemorySnapshot]:
        if not self._cuda_available:
            return MemorySnapshot(
                timestamp=time.time(),
                allocated_mb=0.0,
                reserved_mb=0.0,
                peak_allocated_mb=0.0,
                peak_reserved_mb=0.0,
                fragmentation_ratio=0.0,
            )

        try:
            with torch.cuda.device(self.device):
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                peak_allocated = torch.cuda.max_memory_allocated()
                peak_reserved = torch.cuda.max_memory_reserved()

                fragmentation = 0.0
                if reserved > 0:
                    fragmentation = 1.0 - (allocated / reserved)

                return MemorySnapshot(
                    timestamp=time.time(),
                    allocated_mb=allocated / 1e6,
                    reserved_mb=reserved / 1e6,
                    peak_allocated_mb=peak_allocated / 1e6,
                    peak_reserved_mb=peak_reserved / 1e6,
                    fragmentation_ratio=fragmentation,
                )
        except Exception as e:
            logger.debug(f"Failed to take memory snapshot: {e}")
            return None

    def _record_snapshot(self, snapshot: Optional[MemorySnapshot]) -> None:
        if snapshot is None:
            return

        with self._lock:
            self._history.append(snapshot)
            if len(self._history) > self.history_size:
                self._history.pop(0)

    def _check_and_respond(self, snapshot: Optional[MemorySnapshot]) -> None:
        if snapshot is None:
            return

        if not self._cuda_available:
            return

        try:
            total_mem = torch.cuda.get_device_properties(self.device).total_memory
            usage_ratio = snapshot.reserved_mb * 1e6 / total_mem

            warning = None
            if usage_ratio >= self.oom_threshold:
                warning = OOMWarning(
                    timestamp=time.time(),
                    severity="OOM_IMMINENT",
                    usage_ratio=usage_ratio,
                    message=f"OOM imminent: {usage_ratio:.2%} memory used",
                )
                self._handle_oom_imminent(snapshot)
            elif usage_ratio >= self.critical_threshold:
                warning = OOMWarning(
                    timestamp=time.time(),
                    severity="CRITICAL",
                    usage_ratio=usage_ratio,
                    message=f"Critical memory usage: {usage_ratio:.2%}",
                )
                self._handle_critical(snapshot)
            elif usage_ratio >= self.warning_threshold:
                warning = OOMWarning(
                    timestamp=time.time(),
                    severity="WARNING",
                    usage_ratio=usage_ratio,
                    message=f"High memory usage: {usage_ratio:.2%}",
                )
                self._handle_warning(snapshot)

            if warning is not None:
                with self._lock:
                    self._warnings.append(warning)
                    if len(self._warnings) > self.history_size:
                        self._warnings.pop(0)

                if self.callback is not None:
                    try:
                        self.callback(warning)
                    except Exception as e:
                        logger.debug(f"Memory monitor callback error: {e}")

        except Exception as e:
            logger.debug(f"Memory check failed: {e}")

    def _handle_warning(self, snapshot: MemorySnapshot) -> None:
        logger.warning(
            f"[Memory Monitor] WARNING: {snapshot.usage_ratio:.2%} used | "
            f"Allocated: {snapshot.allocated_mb:.1f}MB | Reserved: {snapshot.reserved_mb:.1f}MB | "
            f"Fragmentation: {snapshot.fragmentation_ratio:.2%}"
        )

        if self.auto_reclaim and snapshot.fragmentation_ratio > 0.3:
            logger.info("Memory fragmentation detected, triggering reclaim")
            self._trigger_reclaim()

    def _handle_critical(self, snapshot: MemorySnapshot) -> None:
        logger.error(
            f"[Memory Monitor] CRITICAL: {snapshot.usage_ratio:.2%} used | "
            f"Allocated: {snapshot.allocated_mb:.1f}MB | Reserved: {snapshot.reserved_mb:.1f}MB"
        )
        self._trigger_reclaim()

    def _handle_oom_imminent(self, snapshot: MemorySnapshot) -> None:
        logger.critical(
            f"[Memory Monitor] OOM IMMINENT: {snapshot.usage_ratio:.2%} used | "
            f"Allocated: {snapshot.allocated_mb:.1f}MB | Reserved: {snapshot.reserved_mb:.1f}MB"
        )
        self._oom_count += 1
        self._trigger_emergency_reclaim()

    def _trigger_reclaim(self) -> None:
        with self._lock:
            callbacks = list(self._reclaim_callbacks)

        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logger.debug(f"Reclaim callback error: {e}")

        gc.collect()
        if self._cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)

        self._reclaim_count += 1
        logger.info("Memory reclaim completed")

    def _trigger_emergency_reclaim(self) -> None:
        logger.warning("=== EMERGENCY MEMORY RECLAIM TRIGGERED BY MONITOR ===")

        with self._lock:
            callbacks = list(self._reclaim_callbacks)

        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logger.debug(f"Emergency reclaim callback error: {e}")

        gc.collect()

        if self._cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)
            torch.cuda.ipc_collect()

        self._reclaim_count += 1
        logger.warning("Emergency memory reclaim completed")

    def get_stats(self) -> Dict:
        with self._lock:
            latest = self._history[-1] if self._history else None

            stats = {
                "running": self._monitor_thread is not None and self._monitor_thread.is_alive(),
                "total_snapshots": len(self._history),
                "total_warnings": len(self._warnings),
                "oom_count": self._oom_count,
                "reclaim_count": self._reclaim_count,
                "latest_snapshot": None,
                "trend": None,
            }

            if latest is not None:
                stats["latest_snapshot"] = {
                    "allocated_mb": latest.allocated_mb,
                    "reserved_mb": latest.reserved_mb,
                    "peak_allocated_mb": latest.peak_allocated_mb,
                    "peak_reserved_mb": latest.peak_reserved_mb,
                    "fragmentation_ratio": latest.fragmentation_ratio,
                }

                if len(self._history) >= 10:
                    recent = self._history[-10:]
                    avg_alloc = sum(s.allocated_mb for s in recent) / len(recent)
                    trend = "stable"
                    if latest.allocated_mb > avg_alloc * 1.1:
                        trend = "increasing"
                    elif latest.allocated_mb < avg_alloc * 0.9:
                        trend = "decreasing"
                    stats["trend"] = trend

            return stats

    def get_usage_ratio(self) -> float:
        if not self._cuda_available:
            return 0.0

        try:
            total_mem = torch.cuda.get_device_properties(self.device).total_memory
            reserved = torch.cuda.memory_reserved(self.device)
            return reserved / total_mem
        except Exception:
            return 0.0

    def reset_peak_stats(self) -> None:
        if self._cuda_available:
            torch.cuda.reset_peak_memory_stats(self.device)
            logger.info("Peak memory stats reset")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def close(self) -> None:
        self.stop()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class MemoryGuard:
    def __init__(
        self,
        device: torch.device,
        monitor: Optional[GPUMemoryMonitor] = None,
        cleanup_on_exit: bool = True,
        description: str = "",
    ):
        self.device = device
        self.monitor = monitor
        self.cleanup_on_exit = cleanup_on_exit
        self.description = description
        self._start_allocated = 0
        self._start_reserved = 0
        self._cuda_available = torch.cuda.is_available() and device.type == "cuda"

    def __enter__(self):
        if self._cuda_available:
            torch.cuda.synchronize(self.device)
            self._start_allocated = torch.cuda.memory_allocated(self.device)
            self._start_reserved = torch.cuda.memory_reserved(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._cuda_available or not self.cleanup_on_exit:
            return False

        try:
            torch.cuda.synchronize(self.device)

            end_allocated = torch.cuda.memory_allocated(self.device)
            end_reserved = torch.cuda.memory_reserved(self.device)

            leaked = end_allocated - self._start_allocated
            if leaked > 10 * 1e6:
                logger.warning(
                    f"[MemoryGuard {self.description}] Potential memory leak: "
                    f"{leaked / 1e6:.1f}MB not released"
                )

            if self.monitor is not None and self.monitor.get_usage_ratio() > 0.8:
                logger.info(f"[MemoryGuard {self.description}] Triggering cleanup")
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(self.device)

        except Exception as e:
            logger.debug(f"MemoryGuard exit error: {e}")

        return False
