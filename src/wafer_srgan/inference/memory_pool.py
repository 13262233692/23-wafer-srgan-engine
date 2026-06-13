import gc
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Tuple
from collections import deque, OrderedDict

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class BufferBlock:
    index: int
    tensor: Optional[torch.Tensor]
    shape: Optional[Tuple[int, ...]]
    dtype: torch.dtype
    device: torch.device
    acquired: bool = False
    last_used: float = 0.0


class GPUMemoryRingBuffer:
    def __init__(
        self,
        device: torch.device,
        max_buffers: int = 64,
        buffer_dtype: torch.dtype = torch.float32,
        enable_logging: bool = False,
        auto_reclaim_threshold: float = 0.85,
        preallocate: bool = False,
    ):
        self.device = device
        self.max_buffers = max_buffers
        self.buffer_dtype = buffer_dtype
        self.enable_logging = enable_logging
        self.auto_reclaim_threshold = auto_reclaim_threshold

        self._lock = threading.Lock()
        self._buffers: OrderedDict[int, BufferBlock] = OrderedDict()
        self._shape_cache: Dict[Tuple[int, ...], deque[int]] = {}
        self._next_index = 0
        self._peak_allocated = 0
        self._peak_reserved = 0

        self._cuda_available = torch.cuda.is_available() and device.type == "cuda"

        if preallocate:
            self._preallocate_default_buffers()

    def _preallocate_default_buffers(self):
        default_shapes = [
            (1, 3, 512, 512),
            (1, 3, 1024, 1024),
            (1, 3, 2048, 2048),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
        ]
        for shape in default_shapes:
            for _ in range(2):
                self._allocate_buffer(shape, self.buffer_dtype)

        if self._cuda_available:
            torch.cuda.synchronize(self.device)
        logger.info(f"Preallocated {len(self._buffers)} default buffers on {self.device}")

    def _allocate_buffer(self, shape: Tuple[int, ...], dtype: torch.dtype) -> int:
        index = self._next_index
        self._next_index += 1

        try:
            tensor = torch.empty(shape, dtype=dtype, device=self.device, requires_grad=False)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM when allocating buffer {shape}, triggering emergency reclaim")
            self.emergency_reclaim()
            tensor = torch.empty(shape, dtype=dtype, device=self.device, requires_grad=False)

        block = BufferBlock(
            index=index,
            tensor=tensor,
            shape=shape,
            dtype=dtype,
            device=self.device,
            last_used=torch.cuda.Event(enable_timing=True) if self._cuda_available else 0.0,
        )

        self._buffers[index] = block
        shape_key = (shape, dtype)
        if shape_key not in self._shape_cache:
            self._shape_cache[shape_key] = deque()
        self._shape_cache[shape_key].append(index)

        if self.enable_logging:
            logger.debug(f"Allocated buffer {index}: shape={shape}, dtype={dtype}")

        return index

    def _find_compatible_buffer(self, shape: Tuple[int, ...], dtype: torch.dtype) -> Optional[int]:
        shape_key = (shape, dtype)
        if shape_key in self._shape_cache and self._shape_cache[shape_key]:
            available = [idx for idx in self._shape_cache[shape_key] if not self._buffers[idx].acquired]
            if available:
                return available[0]

        for idx, block in self._buffers.items():
            if not block.acquired and block.shape == shape and block.dtype == dtype:
                if shape_key not in self._shape_cache:
                    self._shape_cache[shape_key] = deque()
                self._shape_cache[shape_key].append(idx)
                return idx

        return None

    def acquire(self, shape: Tuple[int, ...], dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        dtype = dtype or self.buffer_dtype

        with self._lock:
            if self.auto_reclaim_threshold > 0:
                self._check_and_reclaim()

            buf_idx = self._find_compatible_buffer(shape, dtype)

            if buf_idx is None:
                if len(self._buffers) >= self.max_buffers:
                    logger.warning(f"Buffer pool full ({self.max_buffers}), evicting LRU")
                    self._evict_lru()
                buf_idx = self._allocate_buffer(shape, dtype)

            block = self._buffers[buf_idx]
            block.acquired = True
            block.last_used = torch.cuda.Event(enable_timing=True) if self._cuda_available else 0.0

            if self._cuda_available and isinstance(block.last_used, torch.cuda.Event):
                block.last_used.record()

            block.tensor.zero_()

            if self.enable_logging:
                self._log_stats(f"acquired buffer {buf_idx}: shape={shape}")

            return block.tensor

    def release(self, tensor: torch.Tensor) -> None:
        if tensor is None or tensor.device != self.device:
            return

        with self._lock:
            for idx, block in self._buffers.items():
                if block.tensor is tensor:
                    block.acquired = False
                    if self._cuda_available:
                        block.last_used = torch.cuda.Event(enable_timing=True)
                        block.last_used.record()
                    else:
                        import time
                        block.last_used = time.time()
                    tensor = None
                    if self.enable_logging:
                        self._log_stats(f"released buffer {idx}")
                    return

            logger.warning("Tried to release tensor not owned by this pool")
            del tensor

    def _deallocate_buffer(self, buf_idx: int) -> None:
        block = self._buffers.pop(buf_idx, None)
        if block is None:
            return

        for shape_key, queue in self._shape_cache.items():
            if buf_idx in queue:
                queue.remove(buf_idx)

        if block.tensor is not None:
            block.tensor.data = torch.empty(0, device=self.device)
            del block.tensor

        block.shape = None
        block.tensor = None

        if self.enable_logging:
            logger.debug(f"Deallocated buffer {buf_idx}")

    def _evict_lru(self) -> Optional[int]:
        oldest_idx = None
        oldest_time = float("inf")

        for idx, block in self._buffers.items():
            if block.acquired:
                continue
            try:
                if self._cuda_available and isinstance(block.last_used, torch.cuda.Event):
                    if not block.last_used.query():
                        continue
                    elapsed = block.last_used.elapsed_time()
                else:
                    import time
                    elapsed = time.time() - block.last_used

                if elapsed < oldest_time:
                    oldest_time = elapsed
                    oldest_idx = idx
            except Exception:
                continue

        if oldest_idx is not None:
            self._deallocate_buffer(oldest_idx)
            return oldest_idx
        return None

    def _check_and_reclaim(self) -> None:
        if not self._cuda_available:
            return

        try:
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)

            self._peak_allocated = max(self._peak_allocated, allocated)
            self._peak_reserved = max(self._peak_reserved, reserved)

            total_mem = torch.cuda.get_device_properties(self.device).total_memory
            if reserved / total_mem > self.auto_reclaim_threshold:
                logger.warning(
                    f"Memory usage {reserved / total_mem:.2%} > threshold {self.auto_reclaim_threshold}, "
                    f"triggering reclaim"
                )
                self._reclaim_unused_buffers()
        except Exception as e:
            logger.debug(f"Memory check failed: {e}")

    def _reclaim_unused_buffers(self) -> None:
        to_dealloc = []
        for idx, block in self._buffers.items():
            if not block.acquired:
                to_dealloc.append(idx)

        for idx in to_dealloc:
            self._deallocate_buffer(idx)

        gc.collect()
        if self._cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)

        if self.enable_logging:
            self._log_stats("after reclaim")

    def emergency_reclaim(self) -> None:
        logger.warning("=== EMERGENCY MEMORY RECLAIM TRIGGERED ===")
        with self._lock:
            to_dealloc = list(self._buffers.keys())
            for idx in to_dealloc:
                block = self._buffers[idx]
                if block.tensor is not None:
                    block.tensor.data = torch.empty(0, device=self.device)
                    del block.tensor
                    block.tensor = None
                    block.shape = None

            self._buffers.clear()
            self._shape_cache.clear()

        gc.collect()
        if self._cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)
            torch.cuda.ipc_collect()

        logger.warning("Emergency reclaim complete")

    def get_stats(self) -> dict:
        with self._lock:
            stats = {
                "total_buffers": len(self._buffers),
                "acquired_buffers": sum(1 for b in self._buffers.values() if b.acquired),
                "available_buffers": sum(1 for b in self._buffers.values() if not b.acquired),
                "max_buffers": self.max_buffers,
                "peak_allocated_bytes": self._peak_allocated,
                "peak_reserved_bytes": self._peak_reserved,
            }
            if self._cuda_available:
                try:
                    stats["current_allocated_bytes"] = torch.cuda.memory_allocated(self.device)
                    stats["current_reserved_bytes"] = torch.cuda.memory_reserved(self.device)
                    stats["total_memory"] = torch.cuda.get_device_properties(self.device).total_memory
                except Exception:
                    pass
            return stats

    def _log_stats(self, prefix: str = "") -> None:
        stats = self.get_stats()
        logger.debug(
            f"[RingBuffer {self.device}] {prefix} | "
            f"buffers: {stats['total_buffers']}/{stats['max_buffers']} "
            f"(acquired: {stats['acquired_buffers']}, available: {stats['available_buffers']})"
        )

    def close(self) -> None:
        with self._lock:
            for idx in list(self._buffers.keys()):
                self._deallocate_buffer(idx)
            self._buffers.clear()
            self._shape_cache.clear()

        gc.collect()
        if self._cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)

        logger.info(f"RingBuffer on {self.device} closed")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class TensorPoolManager:
    _instances: Dict[str, "TensorPoolManager"] = {}
    _instance_lock = threading.Lock()

    def __init__(self):
        self._pools: Dict[str, GPUMemoryRingBuffer] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "TensorPoolManager":
        with cls._instance_lock:
            if "global" not in cls._instances:
                cls._instances["global"] = TensorPoolManager()
            return cls._instances["global"]

    def get_pool(
        self,
        device: torch.device,
        max_buffers: int = 64,
        buffer_dtype: torch.dtype = torch.float32,
        auto_reclaim_threshold: float = 0.85,
        preallocate: bool = False,
    ) -> GPUMemoryRingBuffer:
        device_key = str(device)
        with self._lock:
            if device_key not in self._pools:
                self._pools[device_key] = GPUMemoryRingBuffer(
                    device=device,
                    max_buffers=max_buffers,
                    buffer_dtype=buffer_dtype,
                    auto_reclaim_threshold=auto_reclaim_threshold,
                    preallocate=preallocate,
                )
            return self._pools[device_key]

    def clear_all(self) -> None:
        with self._lock:
            for pool in self._pools.values():
                pool.close()
            self._pools.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_all_stats(self) -> Dict[str, dict]:
        stats = {}
        with self._lock:
            for device_key, pool in self._pools.items():
                stats[device_key] = pool.get_stats()
        return stats


def pooled_tensor(
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pool = TensorPoolManager.get_instance().get_pool(device=device)
    return pool.acquire(shape, dtype)


def release_pooled_tensor(tensor: torch.Tensor) -> None:
    if tensor is None or tensor.device.type == "cpu":
        return
    try:
        pool = TensorPoolManager.get_instance().get_pool(device=tensor.device)
        pool.release(tensor)
    except Exception as e:
        logger.debug(f"Failed to release pooled tensor: {e}")
