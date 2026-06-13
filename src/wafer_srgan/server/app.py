import gc
import io
import logging
import time
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify, Response

from wafer_srgan.config import load_config
from wafer_srgan.inference.stream_reader import RawImageStreamReader
from wafer_srgan.inference.sliding_window import SlidingWindowEngine
from wafer_srgan.inference.triton_client import TritonSRClient
from wafer_srgan.inference.stitcher import GaussianBlender
from wafer_srgan.inference.gpu_stitcher import GPUGaussianBlender
from wafer_srgan.inference.edge_postprocess import EdgePostProcessor
from wafer_srgan.inference.memory_monitor import GPUMemoryMonitor, MemoryGuard, OOMWarning
from wafer_srgan.inference.memory_pool import TensorPoolManager
from wafer_srgan.export.ome_tiff import OMETiffWriter

logger = logging.getLogger(__name__)


class InferenceServer:
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = load_config()

        self.cfg = cfg
        self.scale_factor = cfg.pipeline.scale_factor

        import torch
        self._torch_device = torch.device(cfg.pipeline.device if torch.cuda.is_available() else "cpu")

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

        mem_cfg = cfg.memory_management
        stitch_cfg = cfg.stitching
        self._use_gpu_stitching = mem_cfg.use_gpu_stitching and stitch_cfg.use_gpu and self._torch_device.type == "cuda"

        if self._use_gpu_stitching:
            logger.info("Server: Initializing GPU-accelerated Gaussian blender")
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
            logger.info("Server: Initializing CPU Gaussian blender")
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

        self._memory_monitor = None
        if mem_cfg.enable_gpu_monitor and self._torch_device.type == "cuda":
            self._init_memory_monitor(mem_cfg)

        self._request_counter = 0

        self.app = Flask(__name__)
        self.app.config["MAX_CONTENT_LENGTH"] = cfg.server.max_upload_size_mb * 1024 * 1024
        self._register_routes()

    def _init_memory_monitor(self, mem_cfg):
        try:
            def on_warning(warning: OOMWarning):
                if warning.severity == "OOM_IMMINENT":
                    logger.critical(f"[Server] OOM Imminent! Usage: {warning.usage_ratio:.2%}")
                    self._emergency_memory_reclaim()
                elif warning.severity == "CRITICAL" and mem_cfg.auto_reclaim_on_warning:
                    logger.error(f"[Server] Critical memory: {warning.usage_ratio:.2%}")
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
            self._memory_monitor.start()
            logger.info(f"[Server] GPU Memory Monitor started on {self._torch_device}")
        except Exception as e:
            logger.warning(f"[Server] Failed to initialize memory monitor: {e}")
            self._memory_monitor = None

    def _emergency_memory_reclaim(self) -> None:
        logger.warning("[Server] === EMERGENCY MEMORY RECLAIM ===")
        try:
            TensorPoolManager.get_instance().clear_all()
        except Exception:
            pass
        try:
            self.triton_client._emergency_cleanup()
        except Exception:
            pass
        try:
            if isinstance(self.blender, GPUGaussianBlender):
                self.blender._emergency_cleanup()
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available() and self._torch_device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize(self._torch_device)
                torch.cuda.ipc_collect()
        except Exception:
            pass
        logger.warning("[Server] Emergency memory reclaim complete")

    def _register_routes(self):
        self.app.add_url_rule("/health", "health", self._health, methods=["GET"])
        self.app.add_url_rule("/memory/stats", "memory_stats", self._memory_stats, methods=["GET"])
        self.app.add_url_rule("/memory/cleanup", "memory_cleanup", self._memory_cleanup, methods=["POST"])
        self.app.add_url_rule("/infer", "infer", self._infer, methods=["POST"])
        self.app.add_url_rule("/infer/stream", "infer_stream", self._infer_stream, methods=["POST"])

    def _health(self):
        triton_alive = self.triton_client.is_server_live()
        mem_stats = {}
        if self._memory_monitor is not None:
            mem_stats = self._memory_monitor.get_stats()
        return jsonify({
            "status": "ok" if triton_alive else "degraded",
            "triton": triton_alive,
            "memory_monitor_running": self._memory_monitor is not None,
            "memory": mem_stats,
        })

    def _memory_stats(self):
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
        return jsonify(stats)

    def _memory_cleanup(self):
        logger.info("[Server] Manual memory cleanup requested")
        self.triton_client.manual_cleanup()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return jsonify({"status": "ok", "message": "Memory cleanup completed"})

    def _infer(self):
        self._request_counter += 1
        start = time.time()
        job_id = str(uuid.uuid4())[:8]

        with MemoryGuard(
            device=self._torch_device,
            monitor=self._memory_monitor,
            cleanup_on_exit=True,
            description=f"request_{job_id}",
        ):
            if "file" not in request.files:
                return jsonify({"error": "No file uploaded"}), 400

            file = request.files["file"]
            image_format = request.form.get("format", "png")
            run_edge = request.form.get("edge", "false").lower() == "true"
            output_format = request.form.get("output_format", "ome_tiff")

            images = self.stream_reader.parse_stream(io.BytesIO(file.read()), image_format)
            if not images:
                return jsonify({"error": "Failed to parse image"}), 400

            logger.info(f"[{job_id}] Received {len(images)} layer(s)")

            all_sr_images = []
            for layer_idx, img in enumerate(images):
                sr_img = self._process_single_layer(img, job_id, layer_idx)
                all_sr_images.append(sr_img)

            result = {
                "job_id": job_id,
                "layers": len(all_sr_images),
                "elapsed_s": round(time.time() - start, 2),
                "request_num": self._request_counter,
            }

            if output_format == "ome_tiff":
                output_path = Path("output") / f"{job_id}" / f"sr_result.ome.tif"
                self.ome_writer.write(all_sr_images, output_path)
                result["output_path"] = str(output_path)
            else:
                for i, sr in enumerate(all_sr_images):
                    out_path = Path("output") / f"{job_id}" / f"layer_{i}.png"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    from PIL import Image as PILImage
                    PILImage.fromarray(sr).save(str(out_path))
                    result[f"layer_{i}_path"] = str(out_path)

            if run_edge:
                edge_images = []
                for sr in all_sr_images:
                    edges = self.edge_processor.detect_defect_edges(sr, method="canny")
                    edge_images.append(edges)
                result["edge_layers"] = len(edge_images)

            if self._memory_monitor is not None:
                result["memory_stats"] = self._memory_monitor.get_stats()

        return jsonify(result)

    def _infer_stream(self):
        def generate():
            self._request_counter += 1
            job_id = str(uuid.uuid4())[:8]
            yield f'data: {{"job_id": "{job_id}", "status": "started", "request_num": {self._request_counter}}}\n\n'

            try:
                data = request.get_data()
                images = self.stream_reader.parse_stream(io.BytesIO(data), "tiff")
                yield f'data: {{"job_id": "{job_id}", "status": "parsed", "layers": {len(images)}}}\n\n'

                with MemoryGuard(
                    device=self._torch_device,
                    monitor=self._memory_monitor,
                    cleanup_on_exit=True,
                    description=f"stream_{job_id}",
                ):
                    for layer_idx, img in enumerate(images):
                        sr = self._process_single_layer(img, job_id, layer_idx)
                        mem_usage = self._memory_monitor.get_usage_ratio() if self._memory_monitor else 0
                        yield f'data: {{"job_id": "{job_id}", "status": "layer_done", "layer": {layer_idx}, "memory_usage": {mem_usage:.3f}}}\n\n'

                yield f'data: {{"job_id": "{job_id}", "status": "completed"}}\n\n'

            except Exception as e:
                logger.error(f"[{job_id}] Stream error: {e}")
                yield f'data: {{"job_id": "{job_id}", "status": "error", "error": "{str(e)}"}}\n\n'

        return Response(generate(), mimetype="text/event-stream")

    def _process_single_layer(self, image, job_id: str, layer_idx: int):
        tiles = self.sliding_window.tile_image(image, layer_index=layer_idx)
        logger.info(f"[{job_id}] Layer {layer_idx}: {len(tiles)} tiles")

        sr_tiles = []

        with MemoryGuard(
            device=self._torch_device,
            monitor=self._memory_monitor,
            cleanup_on_exit=True,
            description=f"infer_layer_{layer_idx}",
        ):
            for batch in self.sliding_window.batch_tiles(tiles):
                batch_images = [t.image for t in batch]
                sr_batch = self.triton_client.infer_batch(batch_images)
                for tile, sr_img in zip(batch, sr_batch):
                    if sr_img.ndim == 4:
                        sr_img = sr_img[0]
                    if sr_img.ndim == 3 and sr_img.shape[0] in (1, 3):
                        sr_img = sr_img.transpose(1, 2, 0)
                    sr_img = np.clip(sr_img * 255.0, 0, 255).astype(np.uint8)
                    sr_tile = tile
                    sr_tile.image = sr_img
                    sr_tile.tile_h = sr_img.shape[0]
                    sr_tile.tile_w = sr_img.shape[1]
                    sr_tiles.append(sr_tile)

                del batch_images, sr_batch

        with MemoryGuard(
            device=self._torch_device,
            monitor=self._memory_monitor,
            cleanup_on_exit=True,
            description=f"stitch_layer_{layer_idx}",
        ):
            h, w = image.shape[:2]
            stitch_result = self.blender.stitch(sr_tiles, h, w, self.cfg.pipeline.scale_factor)
            return stitch_result.image

    def run(self, **kwargs):
        host = kwargs.get("host", self.cfg.server.host)
        port = kwargs.get("port", self.cfg.server.port)
        logger.info(f"Starting inference server on {host}:{port}")
        self.app.run(host=host, port=port, threaded=True)

    def close(self):
        logger.info("Closing InferenceServer...")
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
        logger.info("InferenceServer closed")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def create_app(cfg=None):
    server = InferenceServer(cfg)
    return server.app
