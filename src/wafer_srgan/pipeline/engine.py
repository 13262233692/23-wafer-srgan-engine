import gc
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from wafer_srgan.config import load_config
from wafer_srgan.inference.stream_reader import RawImageStreamReader
from wafer_srgan.inference.sliding_window import SlidingWindowEngine
from wafer_srgan.inference.triton_client import TritonSRClient
from wafer_srgan.inference.stitcher import GaussianBlender
from wafer_srgan.inference.gpu_stitcher import GPUGaussianBlender
from wafer_srgan.inference.edge_postprocess import EdgePostProcessor
from wafer_srgan.export.ome_tiff import OMETiffWriter
from wafer_srgan.inference.memory_monitor import GPUMemoryMonitor, MemoryGuard, OOMWarning
from wafer_srgan.inference.memory_pool import TensorPoolManager

logger = logging.getLogger(__name__)


class SRGANPipeline:
    def __init__(self, cfg=None, checkpoint_path: Optional[str] = None):
        if cfg is None:
            cfg = load_config()

        self.cfg = cfg
        self.scale_factor = cfg.pipeline.scale_factor

        self.device = cfg.pipeline.device
        import torch
        self._torch_device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        self.stream_reader = RawImageStreamReader(chunk_size=cfg.server.stream_chunk_size)
        self.sliding_window = SlidingWindowEngine(
            tile_size=cfg.sliding_window.tile_size,
            overlap=cfg.sliding_window.overlap,
            scale_factor=cfg.pipeline.scale_factor,
            max_tiles_per_batch=cfg.sliding_window.max_tiles_per_batch,
        )

        triton_cfg = cfg.triton
        self.triton_client = TritonSRClient(
            model_name=triton_cfg.model_name,
            model_version=triton_cfg.model_version,
            grpc_url=triton_cfg.grpc_url,
            http_url=triton_cfg.http_url,
            protocol=triton_cfg.protocol,
            timeout=triton_cfg.timeout,
            max_concurrent_requests=triton_cfg.max_concurrent_requests,
            enable_memory_pool=triton_cfg.enable_memory_pool,
            memory_pool_max_buffers=triton_cfg.memory_pool_max_buffers,
            memory_pool_reclaim_threshold=triton_cfg.memory_pool_reclaim_threshold,
            auto_cleanup_interval=triton_cfg.auto_cleanup_interval,
        )
        if checkpoint_path:
            self.triton_client.load_local_weights(checkpoint_path)

        mem_cfg = cfg.memory_management
        stitch_cfg = cfg.stitching
        self._use_gpu_stitching = mem_cfg.use_gpu_stitching and stitch_cfg.use_gpu and self._torch_device.type == "cuda"

        if self._use_gpu_stitching:
            logger.info("Initializing GPU-accelerated Gaussian blender with memory pool")
            self.blender = GPUGaussianBlender(
                sigma=stitch_cfg.gaussian_sigma,
                feather_width=stitch_cfg.feather_width,
                use_gpu=True,
                device=self._torch_device,
                enable_memory_pool=mem_cfg.enable_memory_pool,
                memory_pool_max_buffers=stitch_cfg.memory_pool_max_buffers,
                memory_pool_reclaim_threshold=mem_cfg.memory_pool_reclaim_threshold,
                stream_tiles=True,
                chunk_size=stitch_cfg.chunk_size,
            )
        else:
            logger.info("Initializing CPU Gaussian blender")
            self.blender = GaussianBlender(
                sigma=stitch_cfg.gaussian_sigma,
                feather_width=stitch_cfg.feather_width,
            )

        self.ome_writer = OMETiffWriter(
            pixel_type=cfg.ome_tiff.pixel_type,
            channel_names=cfg.ome_tiff.channel_names,
            physical_size_x=cfg.ome_tiff.physical_size_x,
            physical_size_y=cfg.ome_tiff.physical_size_y,
            compression=cfg.ome_tiff.compression,
            tile_size=cfg.ome_tiff.tile_size,
        )
        self.edge_processor = EdgePostProcessor(
            canny_threshold1=cfg.edge_postprocess.canny_threshold1,
            canny_threshold2=cfg.edge_postprocess.canny_threshold2,
            sobel_ksize=cfg.edge_postprocess.sobel_ksize,
            morph_kernel_size=cfg.edge_postprocess.morph_kernel_size,
            morph_iterations=cfg.edge_postprocess.morph_iterations,
        )

        self._memory_monitor: Optional[GPUMemoryMonitor] = None
        if mem_cfg.enable_gpu_monitor and self._torch_device.type == "cuda":
            self._init_memory_monitor(mem_cfg)

        self._pipeline_counter = 0

    def _init_memory_monitor(self, mem_cfg):
        try:
            def on_warning(warning: OOMWarning):
                if warning.severity == "OOM_IMMINENT":
                    logger.critical(f"OOM Imminent detected! Usage: {warning.usage_ratio:.2%}")
                    self._emergency_memory_reclaim()
                elif warning.severity == "CRITICAL" and mem_cfg.auto_reclaim_on_warning:
                    logger.error(f"Critical memory usage: {warning.usage_ratio:.2%}, triggering reclaim")
                    self.triton_client.manual_cleanup()

            self._memory_monitor = GPUMemoryMonitor(
                device=self._torch_device,
                check_interval=mem_cfg.monitor_check_interval,
                warning_threshold=mem_cfg.monitor_warning_threshold,
                critical_threshold=mem_cfg.monitor_critical_threshold,
                oom_threshold=mem_cfg.monitor_oom_threshold,
                auto_reclaim=mem_cfg.auto_reclaim_on_warning,
                callback=on_warning,
            )

            self._memory_monitor.add_reclaim_callback(self._on_memory_reclaim)
            self._memory_monitor.start()

            logger.info(f"GPU Memory Monitor started on {self._torch_device}")
        except Exception as e:
            logger.warning(f"Failed to initialize memory monitor: {e}")
            self._memory_monitor = None

    def _on_memory_reclaim(self) -> None:
        logger.debug("Memory reclaim callback triggered")
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available() and self._torch_device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize(self._torch_device)
        except Exception:
            pass

    def _emergency_memory_reclaim(self) -> None:
        logger.warning("=== EMERGENCY PIPELINE MEMORY RECLAIM ===")

        try:
            TensorPoolManager.get_instance().clear_all()
        except Exception as e:
            logger.debug(f"Tensor pool clear error: {e}")

        try:
            self.triton_client._emergency_cleanup()
        except Exception as e:
            logger.debug(f"Triton cleanup error: {e}")

        try:
            if isinstance(self.blender, GPUGaussianBlender):
                self.blender._emergency_cleanup()
        except Exception as e:
            logger.debug(f"Blender cleanup error: {e}")

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available() and self._torch_device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize(self._torch_device)
                torch.cuda.ipc_collect()
        except Exception as e:
            logger.debug(f"CUDA cleanup error: {e}")

        logger.warning("Pipeline emergency memory reclaim complete")

    def _get_memory_stats(self) -> dict:
        stats = {}

        if self._memory_monitor is not None:
            stats["monitor"] = self._memory_monitor.get_stats()

        try:
            stats["triton_client"] = self.triton_client.get_memory_stats()
        except Exception:
            pass

        if isinstance(self.blender, GPUGaussianBlender):
            try:
                stats["blender"] = self.blender.get_memory_stats()
            except Exception:
                pass

        try:
            stats["tensor_pools"] = TensorPoolManager.get_instance().get_all_stats()
        except Exception:
            pass

        return stats

    def run(
        self,
        input_path: str,
        output_path: str,
        edge_detection: bool = False,
        edge_method: str = "canny",
    ) -> dict:
        self._pipeline_counter += 1
        start = time.time()
        logger.info(f"Pipeline #{self._pipeline_counter} started: {input_path} -> {output_path}")

        with MemoryGuard(
            device=self._torch_device,
            monitor=self._memory_monitor,
            cleanup_on_exit=True,
            description=f"pipeline_run_{self._pipeline_counter}",
        ):

            logger.info("Step 1/5: Reading and parsing raw image layers...")
            images = self.stream_reader.stream_from_file(input_path)
            if not images:
                raise ValueError(f"Failed to parse image from {input_path}")
            logger.info(f"  Parsed {len(images)} layer(s)")

            logger.info("Step 2/5: Sliding-window tiling...")
            all_tiles = self.sliding_window.tile_image_batch(images)
            logger.info(f"  Generated {len(all_tiles)} tiles")

            logger.info("Step 3/5: Super-resolution inference via Triton...")
            sr_tiles = []
            total_batches = 0

            with MemoryGuard(
                device=self._torch_device,
                monitor=self._memory_monitor,
                cleanup_on_exit=True,
                description="inference_phase",
            ):
                for batch in self.sliding_window.batch_tiles(all_tiles):
                    batch_images = [t.image for t in batch]
                    sr_batch = self.triton_client.infer_batch(batch_images)

                    for tile, sr_img in zip(batch, sr_batch):
                        if sr_img.ndim == 4:
                            sr_img = sr_img[0]
                        if sr_img.ndim == 3 and sr_img.shape[0] in (1, 3):
                            sr_img = sr_img.transpose(1, 2, 0)
                        sr_img = np.clip(sr_img * 255.0, 0, 255).astype(np.uint8)
                        tile.image = sr_img
                        tile.tile_h = sr_img.shape[0]
                        tile.tile_w = sr_img.shape[1]
                        sr_tiles.append(tile)

                    total_batches += 1

                    del batch_images, sr_batch

                logger.info(f"  Inference done: {total_batches} batches, {len(sr_tiles)} tiles")

            logger.info("Step 4/5: Gaussian-weighted stitching...")
            layer_tiles = {}
            for t in sr_tiles:
                layer_tiles.setdefault(t.layer_index, []).append(t)

            sr_images = []

            with MemoryGuard(
                device=self._torch_device,
                monitor=self._memory_monitor,
                cleanup_on_exit=True,
                description="stitching_phase",
            ):
                for layer_idx in sorted(layer_tiles.keys()):
                    tiles = layer_tiles[layer_idx]
                    src_h = tiles[0].metadata["src_h"]
                    src_w = tiles[0].metadata["src_w"]

                    logger.info(f"  Stitching layer {layer_idx}: {len(tiles)} tiles")
                    result = self.blender.stitch(tiles, src_h, src_w, self.scale_factor)
                    sr_images.append(result.image)

                    del tiles
                    if self._memory_monitor is not None and self._memory_monitor.get_usage_ratio() > 0.7:
                        self.triton_client.manual_cleanup()

            logger.info(f"  Stitched {len(sr_images)} layer(s)")

            if edge_detection:
                logger.info("Step 4.5/5: Edge post-processing...")
                edge_images = []
                for sr in sr_images:
                    edges = self.edge_processor.detect_defect_edges(sr, method=edge_method)
                    edge_images.append(edges)
                logger.info(f"  Detected edges on {len(edge_images)} layer(s)")

            logger.info("Step 5/5: Writing OME-TIFF...")
            output_path = Path(output_path)
            self.ome_writer.write(sr_images, output_path)

        elapsed = time.time() - start

        memory_stats = self._get_memory_stats()

        result = {
            "input": str(input_path),
            "output": str(output_path),
            "layers": len(sr_images),
            "scale_factor": self.scale_factor,
            "elapsed_s": round(elapsed, 2),
            "output_size_mb": round(output_path.stat().st_size / 1e6, 2),
            "pipeline_run": self._pipeline_counter,
        }

        if memory_stats:
            result["memory_stats"] = memory_stats

        logger.info(f"Pipeline complete in {elapsed:.2f}s")
        return result

    def manual_memory_cleanup(self) -> None:
        logger.info("Manual memory cleanup triggered")
        self.triton_client.manual_cleanup()

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available() and self._torch_device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize(self._torch_device)
        except Exception:
            pass

    def close(self) -> None:
        logger.info("Closing SRGANPipeline...")

        if self._memory_monitor is not None:
            try:
                self._memory_monitor.stop()
            except Exception:
                pass
            self._memory_monitor = None

        try:
            self.triton_client.close()
        except Exception:
            pass

        if isinstance(self.blender, GPUGaussianBlender):
            try:
                self.blender.close()
            except Exception:
                pass

        try:
            TensorPoolManager.get_instance().clear_all()
        except Exception:
            pass

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("SRGANPipeline closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
