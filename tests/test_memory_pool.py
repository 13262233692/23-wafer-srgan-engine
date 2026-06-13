import pytest
import gc
import time

import numpy as np

from wafer_srgan.inference.memory_pool import (
    GPUMemoryRingBuffer,
    TensorPoolManager,
    pooled_tensor,
    release_pooled_tensor,
)


class TestGPUMemoryRingBuffer:
    def test_create_pool_cpu(self):
        import torch
        device = torch.device("cpu")
        pool = GPUMemoryRingBuffer(device=device, max_buffers=8, preallocate=False)
        assert pool.device == device
        assert pool.max_buffers == 8

    def test_acquire_release(self):
        import torch
        device = torch.device("cpu")
        pool = GPUMemoryRingBuffer(device=device, max_buffers=8, preallocate=False)

        t = pool.acquire((1, 3, 64, 64), torch.float32)
        assert t.shape == (1, 3, 64, 64)
        assert t.dtype == torch.float32

        stats = pool.get_stats()
        assert stats["acquired_buffers"] == 1
        assert stats["total_buffers"] == 1

        pool.release(t)
        stats = pool.get_stats()
        assert stats["acquired_buffers"] == 0

    def test_reuse_buffer(self):
        import torch
        device = torch.device("cpu")
        pool = GPUMemoryRingBuffer(device=device, max_buffers=8, preallocate=False)

        t1 = pool.acquire((1, 3, 64, 64), torch.float32)
        pool.release(t1)

        t2 = pool.acquire((1, 3, 64, 64), torch.float32)
        stats = pool.get_stats()
        assert stats["total_buffers"] == 1

        pool.release(t2)

    def test_different_shapes(self):
        import torch
        device = torch.device("cpu")
        pool = GPUMemoryRingBuffer(device=device, max_buffers=8, preallocate=False)

        t1 = pool.acquire((1, 3, 64, 64), torch.float32)
        t2 = pool.acquire((1, 3, 128, 128), torch.float32)

        stats = pool.get_stats()
        assert stats["total_buffers"] == 2
        assert stats["acquired_buffers"] == 2

        pool.release(t1)
        pool.release(t2)

    def test_eviction(self):
        import torch
        device = torch.device("cpu")
        pool = GPUMemoryRingBuffer(device=device, max_buffers=2, preallocate=False)

        t1 = pool.acquire((1, 3, 64, 64), torch.float32)
        t2 = pool.acquire((1, 3, 128, 128), torch.float32)
        pool.release(t1)
        pool.release(t2)

        time.sleep(0.1)

        t3 = pool.acquire((1, 3, 256, 256), torch.float32)
        pool.release(t3)

        stats = pool.get_stats()
        assert stats["total_buffers"] <= 2

    def test_pooled_tensor_helper(self):
        import torch
        TensorPoolManager.get_instance().clear_all()

        t = pooled_tensor((1, 3, 64, 64), torch.float32, torch.device("cpu"))
        assert t.shape == (1, 3, 64, 64)

        release_pooled_tensor(t)

        stats = TensorPoolManager.get_instance().get_all_stats()
        assert "cpu" in stats

    def test_emergency_reclaim(self):
        import torch
        device = torch.device("cpu")
        pool = GPUMemoryRingBuffer(device=device, max_buffers=8, preallocate=False)

        t1 = pool.acquire((1, 3, 64, 64), torch.float32)
        t2 = pool.acquire((1, 3, 128, 128), torch.float32)

        stats_before = pool.get_stats()
        assert stats_before["total_buffers"] == 2

        pool.emergency_reclaim()

        stats_after = pool.get_stats()
        assert stats_after["total_buffers"] == 0

    def test_singleton_manager(self):
        mgr1 = TensorPoolManager.get_instance()
        mgr2 = TensorPoolManager.get_instance()
        assert mgr1 is mgr2

    def test_close(self):
        import torch
        device = torch.device("cpu")
        pool = GPUMemoryRingBuffer(device=device, max_buffers=8, preallocate=False)

        t = pool.acquire((1, 3, 64, 64), torch.float32)
        pool.release(t)

        pool.close()
        stats = pool.get_stats()
        assert stats["total_buffers"] == 0
