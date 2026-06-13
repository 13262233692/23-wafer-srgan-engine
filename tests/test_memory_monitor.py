import pytest
import time
import threading

import numpy as np

from wafer_srgan.inference.memory_monitor import GPUMemoryMonitor, MemoryGuard, OOMWarning


class TestGPUMemoryMonitor:
    def test_create_monitor(self):
        import torch
        device = torch.device("cpu")
        monitor = GPUMemoryMonitor(
            device=device,
            check_interval=0.1,
            auto_reclaim=False,
        )
        assert monitor.device == device

    def test_monitor_lifecycle(self):
        import torch
        device = torch.device("cpu")
        monitor = GPUMemoryMonitor(
            device=device,
            check_interval=0.05,
            auto_reclaim=False,
        )
        monitor.start()
        time.sleep(0.2)

        stats = monitor.get_stats()
        assert "total_snapshots" in stats
        assert stats["total_snapshots"] > 0

        monitor.stop()
        stats = monitor.get_stats()
        assert stats["running"] is False

    def test_oom_warning_dataclass(self):
        warning = OOMWarning(
            timestamp=time.time(),
            severity="WARNING",
            usage_ratio=0.8,
            message="Test warning",
        )
        assert warning.severity == "WARNING"
        assert warning.usage_ratio == 0.8

    def test_callback(self):
        import torch
        device = torch.device("cpu")

        received = []
        def callback(warning):
            received.append(warning)

        monitor = GPUMemoryMonitor(
            device=device,
            check_interval=0.05,
            warning_threshold=0.0,
            auto_reclaim=False,
            callback=callback,
        )
        monitor.start()
        time.sleep(0.15)
        monitor.stop()

    def test_context_manager(self):
        import torch
        device = torch.device("cpu")
        monitor = GPUMemoryMonitor(
            device=device,
            check_interval=0.05,
            auto_reclaim=False,
        )

        with monitor:
            time.sleep(0.1)
            stats = monitor.get_stats()
            assert stats["running"] is True

        stats = monitor.get_stats()
        assert stats["running"] is False

    def test_memory_guard(self):
        import torch
        device = torch.device("cpu")

        with MemoryGuard(device=device, description="test_guard") as guard:
            a = torch.ones(1000, 1000)
            b = torch.ones(1000, 1000)
            c = a + b
            del c

        assert guard is not None

    def test_memory_guard_with_monitor(self):
        import torch
        device = torch.device("cpu")

        monitor = GPUMemoryMonitor(
            device=device,
            check_interval=0.05,
            auto_reclaim=False,
        )

        with MemoryGuard(device=device, monitor=monitor, description="test"):
            pass


class TestGPUGaussianBlender:
    def test_gpu_blender_cpu_fallback(self):
        import torch
        from wafer_srgan.inference.gpu_stitcher import GPUGaussianBlender

        blender = GPUGaussianBlender(
            sigma=4.0,
            feather_width=8,
            use_gpu=False,
            device=torch.device("cpu"),
            enable_memory_pool=False,
            chunk_size=4,
        )
        assert blender.device.type == "cpu"

    def test_gpu_blender_stitch_small(self):
        import torch
        from wafer_srgan.inference.gpu_stitcher import GPUGaussianBlender
        from wafer_srgan.inference.sliding_window import TilePacket

        blender = GPUGaussianBlender(
            sigma=2.0,
            feather_width=4,
            use_gpu=False,
            device=torch.device("cpu"),
            enable_memory_pool=False,
            chunk_size=4,
        )

        tiles = []
        for row in range(2):
            for col in range(2):
                tile = TilePacket(
                    tile_id=f"tile_{row}_{col}",
                    image=np.ones((64, 64, 3), dtype=np.uint8) * 128,
                    row=row,
                    col=col,
                    tile_h=64,
                    tile_w=64,
                    overlap_top=8 if row > 0 else 0,
                    overlap_left=8 if col > 0 else 0,
                    overlap_bottom=8 if row < 1 else 0,
                    overlap_right=8 if col < 1 else 0,
                    layer_index=0,
                    metadata={"src_y": row * 56, "src_x": col * 56, "src_h": 120, "src_w": 120},
                )
                tiles.append(tile)

        result = blender.stitch(tiles, target_h=120, target_w=120, scale_factor=1)
        assert result.image.shape == (120, 120, 3)
        assert result.image.dtype == np.uint8

    def test_close(self):
        import torch
        from wafer_srgan.inference.gpu_stitcher import GPUGaussianBlender

        blender = GPUGaussianBlender(
            sigma=2.0,
            feather_width=4,
            use_gpu=False,
            device=torch.device("cpu"),
            enable_memory_pool=False,
        )
        blender.close()
