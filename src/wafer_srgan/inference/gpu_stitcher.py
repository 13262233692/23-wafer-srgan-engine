import gc
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from wafer_srgan.inference.sliding_window import TilePacket
from wafer_srgan.inference.memory_pool import TensorPoolManager, pooled_tensor, release_pooled_tensor

logger = logging.getLogger(__name__)


@dataclass
class StitchResult:
    image: np.ndarray
    weight_map: np.ndarray
    device: str = "cpu"


@dataclass
class _StitchBuffer:
    canvas: Optional[torch.Tensor]
    weight_canvas: Optional[torch.Tensor]
    weight_map_cache: dict = field(default_factory=dict)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    current_h: int = 0
    current_w: int = 0
    channels: int = 3


class GPUGaussianBlender:
    def __init__(
        self,
        sigma: float = 16.0,
        feather_width: int = 32,
        use_gpu: bool = True,
        device: Optional[torch.device] = None,
        enable_memory_pool: bool = True,
        memory_pool_max_buffers: int = 32,
        memory_pool_reclaim_threshold: float = 0.85,
        stream_tiles: bool = True,
        chunk_size: int = 16,
    ):
        self.sigma = sigma
        self.feather_width = feather_width
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = device or (torch.device("cuda" if self.use_gpu else "cpu"))
        self.enable_memory_pool = enable_memory_pool
        self.stream_tiles = stream_tiles
        self.chunk_size = chunk_size

        self._buffer: Optional[_StitchBuffer] = None
        self._memory_pool = None

        if self.use_gpu:
            self._init_memory_pool(memory_pool_max_buffers, memory_pool_reclaim_threshold)

        self._precompute_weight_kernels()

    def _init_memory_pool(self, max_buffers: int, reclaim_threshold: float):
        try:
            import torch
            self._memory_pool = TensorPoolManager.get_instance().get_pool(
                device=self.device,
                max_buffers=max_buffers,
                buffer_dtype=torch.float32,
                auto_reclaim_threshold=reclaim_threshold,
                preallocate=False,
            )
            logger.info(f"GPUGaussianBlender memory pool initialized on {self.device}")
        except Exception as e:
            logger.warning(f"Failed to initialize blender memory pool: {e}")
            self._memory_pool = None

    def _precompute_weight_kernels(self):
        self._weight_cache = {}
        for size in [128, 256, 512, 1024, 2048, 4096]:
            for overlap in [0, 16, 32, 64, 128, 256]:
                self._weight_cache[(size, overlap)] = None

    def _create_weight_map_gpu(
        self,
        tile_h: int,
        tile_w: int,
        overlap_top: int,
        overlap_left: int,
        overlap_bottom: int,
        overlap_right: int,
    ) -> torch.Tensor:
        cache_key = (tile_h, tile_w, overlap_top, overlap_left, overlap_bottom, overlap_right)

        if cache_key in self._buffer.weight_map_cache:
            return self._buffer.weight_map_cache[cache_key]

        weight = torch.ones((tile_h, tile_w), dtype=torch.float32, device=self.device)

        if overlap_top > 0:
            ramp = torch.linspace(0, 1, overlap_top, device=self.device, dtype=torch.float32)
            weight[:overlap_top, :] *= ramp.view(-1, 1)

        if overlap_bottom > 0:
            ramp = torch.linspace(1, 0, overlap_bottom, device=self.device, dtype=torch.float32)
            weight[-overlap_bottom:, :] *= ramp.view(-1, 1)

        if overlap_left > 0:
            ramp = torch.linspace(0, 1, overlap_left, device=self.device, dtype=torch.float32)
            weight[:, :overlap_left] *= ramp.view(1, -1)

        if overlap_right > 0:
            ramp = torch.linspace(1, 0, overlap_right, device=self.device, dtype=torch.float32)
            weight[:, -overlap_right:] *= ramp.view(1, -1)

        if self.sigma > 0:
            weight_4d = weight.unsqueeze(0).unsqueeze(0)
            kernel_size = int(2 * math.ceil(3 * self.sigma) + 1)
            sigma = self.sigma
            channels = 1
            kernel = self._generate_gaussian_kernel(kernel_size, sigma, channels)
            kernel = kernel.to(self.device).to(torch.float32)

            pad = kernel_size // 2
            weight_padded = F.pad(weight_4d, (pad, pad, pad, pad), mode="reflect")
            weight_smoothed = F.conv2d(weight_padded, kernel, groups=channels, padding=0)
            weight = weight_smoothed.squeeze(0).squeeze(0)

        weight = torch.clamp(weight, min=1e-8)

        if len(self._buffer.weight_map_cache) < 128:
            self._buffer.weight_map_cache[cache_key] = weight

        return weight

    @staticmethod
    def _generate_gaussian_kernel(kernel_size: int, sigma: float, channels: int) -> torch.Tensor:
        x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        gaussian = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel_1d = gaussian / gaussian.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        return kernel

    def _acquire_canvas(self, h: int, w: int, channels: int) -> None:
        if self._buffer is not None and self._buffer.current_h >= h and self._buffer.current_w >= w and self._buffer.channels == channels:
            self._buffer.canvas.zero_()
            self._buffer.weight_canvas.zero_()
            return

        self._release_buffer()

        if self.enable_memory_pool and self._memory_pool is not None:
            canvas_shape = (h, w, channels) if channels > 1 else (h, w)
            canvas = pooled_tensor(canvas_shape, torch.float32, device=self.device)
            weight_canvas = pooled_tensor((h, w), torch.float32, device=self.device)
        else:
            canvas_shape = (h, w, channels) if channels > 1 else (h, w)
            canvas = torch.zeros(canvas_shape, dtype=torch.float32, device=self.device)
            weight_canvas = torch.zeros((h, w), dtype=torch.float32, device=self.device)

        self._buffer = _StitchBuffer(
            canvas=canvas,
            weight_canvas=weight_canvas,
            device=self.device,
            current_h=h,
            current_w=w,
            channels=channels,
        )

    def _release_buffer(self) -> None:
        if self._buffer is None:
            return

        if self.enable_memory_pool and self._memory_pool is not None:
            if self._buffer.canvas is not None:
                release_pooled_tensor(self._buffer.canvas)
            if self._buffer.weight_canvas is not None:
                release_pooled_tensor(self._buffer.weight_canvas)

        self._buffer.canvas = None
        self._buffer.weight_canvas = None
        self._buffer.weight_map_cache.clear()
        self._buffer = None

    def _preprocess_tile(self, tile_image: np.ndarray) -> torch.Tensor:
        if tile_image.dtype != np.float32:
            tile_image = tile_image.astype(np.float32)

        if tile_image.max() > 1.0 and tile_image.max() <= 255.0:
            tile_image = tile_image / 255.0

        if self.enable_memory_pool and self._memory_pool is not None:
            tensor = pooled_tensor(tile_image.shape, torch.float32, device=self.device)
            tensor.copy_(torch.from_numpy(tile_image))
        else:
            tensor = torch.from_numpy(tile_image).to(self.device)

        return tensor

    def stitch(
        self,
        tiles: List[TilePacket],
        target_h: int,
        target_w: int,
        scale_factor: int = 4,
    ) -> StitchResult:
        if not tiles:
            raise ValueError("No tiles to stitch")

        sr_h = target_h * scale_factor
        sr_w = target_w * scale_factor

        channels = tiles[0].image.shape[-1] if tiles[0].image.ndim == 3 else 1

        self._acquire_canvas(sr_h, sr_w, channels)
        canvas = self._buffer.canvas
        weight_canvas = self._buffer.weight_canvas

        tile_tensors = []
        weight_tensors = []

        try:
            for chunk_start in range(0, len(tiles), self.chunk_size):
                chunk = tiles[chunk_start : chunk_start + self.chunk_size]
                chunk_tensors = []
                chunk_weights = []
                chunk_meta = []

                for tile in chunk:
                    tile_h_sr = tile.image.shape[0] * scale_factor if tile.image.shape[0] * scale_factor <= sr_h else tile.image.shape[0]
                    tile_w_sr = tile.image.shape[1] * scale_factor if tile.image.shape[1] * scale_factor <= sr_w else tile.image.shape[1]

                    if tile.image.shape[0] * scale_factor != tile.image.shape[0]:
                        actual_h = tile.image.shape[0]
                        actual_w = tile.image.shape[1]
                    else:
                        actual_h = tile_h_sr // scale_factor
                        actual_w = tile_w_sr // scale_factor

                    tile_tensor = self._preprocess_tile(tile.image)
                    chunk_tensors.append(tile_tensor)

                    weight = self._create_weight_map_gpu(
                        tile.image.shape[0],
                        tile.image.shape[1],
                        tile.overlap_top * scale_factor // scale_factor,
                        tile.overlap_left * scale_factor // scale_factor,
                        tile.overlap_bottom * scale_factor // scale_factor,
                        tile.overlap_right * scale_factor // scale_factor,
                    )
                    chunk_weights.append(weight)

                    src_y = tile.metadata.get("src_y", 0)
                    src_x = tile.metadata.get("src_x", 0)
                    dst_y = src_y * scale_factor
                    dst_x = src_x * scale_factor

                    chunk_meta.append((
                        dst_y,
                        dst_x,
                        min(tile.image.shape[0], sr_h - dst_y),
                        min(tile.image.shape[1], sr_w - dst_x),
                    ))

                self._process_chunk(
                    canvas,
                    weight_canvas,
                    chunk_tensors,
                    chunk_weights,
                    chunk_meta,
                    channels,
                )

                for t in chunk_tensors:
                    if self.enable_memory_pool and self._memory_pool is not None:
                        release_pooled_tensor(t)

                chunk_tensors.clear()
                chunk_weights.clear()

            if self._memory_pool is not None:
                self._memory_pool._check_and_reclaim()

            result = self._finalize_stitch(canvas, weight_canvas, channels)

            return StitchResult(image=result, weight_map=weight_canvas.cpu().numpy(), device=str(self.device))

        except torch.cuda.OutOfMemoryError as oom:
            logger.error(f"CUDA OOM during stitching: {oom}")
            self._emergency_cleanup()
            return self._fallback_cpu_stitch(tiles, target_h, target_w, scale_factor)

        finally:
            for t in tile_tensors:
                if self.enable_memory_pool and self._memory_pool is not None and t is not None:
                    release_pooled_tensor(t)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

    def _process_chunk(
        self,
        canvas: torch.Tensor,
        weight_canvas: torch.Tensor,
        tile_tensors: List[torch.Tensor],
        weight_tensors: List[torch.Tensor],
        chunk_meta: List[Tuple[int, int, int, int]],
        channels: int,
    ) -> None:
        for tile_tensor, weight_tensor, (dst_y, dst_x, actual_h, actual_w) in zip(tile_tensors, weight_tensors, chunk_meta):
            y_end = dst_y + actual_h
            x_end = dst_x + actual_w

            tile_view = tile_tensor[:actual_h, :actual_w]
            weight_view = weight_tensor[:actual_h, :actual_w]

            if channels == 1:
                canvas[dst_y:y_end, dst_x:x_end] += tile_view * weight_view
            else:
                for c in range(channels):
                    canvas[dst_y:y_end, dst_x:x_end, c] += tile_view[:actual_h, :actual_w, c] * weight_view

            weight_canvas[dst_y:y_end, dst_x:x_end] += weight_view

        del tile_tensors, weight_tensors

    def _finalize_stitch(
        self,
        canvas: torch.Tensor,
        weight_canvas: torch.Tensor,
        channels: int,
    ) -> np.ndarray:
        mask = weight_canvas > 0

        if channels == 1:
            result_tensor = canvas.clone()
            result_tensor[mask] /= weight_canvas[mask]
        else:
            result_tensor = canvas.clone()
            for c in range(channels):
                result_tensor[:, :, c][mask] /= weight_canvas[mask]

        result_tensor = torch.clamp(result_tensor * 255.0, 0.0, 255.0)
        result = result_tensor.to(torch.uint8).cpu().numpy()

        del result_tensor

        if self.enable_memory_pool and self._memory_pool is not None:
            release_pooled_tensor(canvas)
            release_pooled_tensor(weight_canvas)

        self._buffer.canvas = None
        self._buffer.weight_canvas = None

        return result

    def _fallback_cpu_stitch(
        self,
        tiles: List[TilePacket],
        target_h: int,
        target_w: int,
        scale_factor: int = 4,
    ) -> StitchResult:
        logger.warning("Falling back to CPU stitching")

        from wafer_srgan.inference.stitcher import GaussianBlender as CPUGaussianBlender
        blender = CPUGaussianBlender(sigma=self.sigma, feather_width=self.feather_width)
        result = blender.stitch(tiles, target_h, target_w, scale_factor)
        return StitchResult(image=result.image, weight_map=result.weight_map, device="cpu")

    def _emergency_cleanup(self) -> None:
        logger.warning("=== EMERGENCY STITCHER MEMORY CLEANUP ===")

        gc.collect()

        try:
            if self._memory_pool is not None:
                self._memory_pool.emergency_reclaim()
        except Exception:
            pass

        self._release_buffer()

        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)
            torch.cuda.ipc_collect()

        logger.warning("Stitcher emergency cleanup complete")

    def get_memory_stats(self) -> dict:
        stats = {}

        if self._memory_pool is not None:
            stats["memory_pool"] = self._memory_pool.get_stats()

        if self._buffer is not None:
            stats["buffer"] = {
                "canvas_shape": list(self._buffer.canvas.shape) if self._buffer.canvas is not None else None,
                "weight_canvas_shape": list(self._buffer.weight_canvas.shape) if self._buffer.weight_canvas is not None else None,
                "weight_map_cache_size": len(self._buffer.weight_map_cache),
            }

        if torch.cuda.is_available() and self.device.type == "cuda":
            stats["cuda"] = {
                "allocated_mb": torch.cuda.memory_allocated(self.device) / 1e6,
                "reserved_mb": torch.cuda.memory_reserved(self.device) / 1e6,
            }

        return stats

    def close(self) -> None:
        self._release_buffer()
        self._weight_cache.clear()

        if self._memory_pool is not None:
            try:
                self._memory_pool.close()
            except Exception:
                pass
            self._memory_pool = None

        gc.collect()
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info("GPUGaussianBlender closed")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
