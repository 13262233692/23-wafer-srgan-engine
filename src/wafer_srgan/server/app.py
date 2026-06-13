import io
import logging
import time
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, Response

from wafer_srgan.config import load_config
from wafer_srgan.inference.stream_reader import RawImageStreamReader
from wafer_srgan.inference.sliding_window import SlidingWindowEngine
from wafer_srgan.inference.triton_client import TritonSRClient
from wafer_srgan.inference.stitcher import GaussianBlender
from wafer_srgan.inference.edge_postprocess import EdgePostProcessor
from wafer_srgan.export.ome_tiff import OMETiffWriter

logger = logging.getLogger(__name__)


class InferenceServer:
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = load_config()

        self.cfg = cfg
        self.stream_reader = RawImageStreamReader(chunk_size=cfg.server.stream_chunk_size)
        self.sliding_window = SlidingWindowEngine(
            tile_size=cfg.sliding_window.tile_size,
            overlap=cfg.sliding_window.overlap,
            scale_factor=cfg.pipeline.scale_factor,
            max_tiles_per_batch=cfg.sliding_window.max_tiles_per_batch,
        )
        self.triton_client = TritonSRClient(
            model_name=cfg.triton.model_name,
            model_version=cfg.triton.model_version,
            grpc_url=cfg.triton.grpc_url,
            http_url=cfg.triton.http_url,
            protocol=cfg.triton.protocol,
            timeout=cfg.triton.timeout,
            max_concurrent_requests=cfg.triton.max_concurrent_requests,
        )
        self.blender = GaussianBlender(
            sigma=cfg.stitching.gaussian_sigma,
            feather_width=cfg.stitching.feather_width,
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

        self.app = Flask(__name__)
        self.app.config["MAX_CONTENT_LENGTH"] = cfg.server.max_upload_size_mb * 1024 * 1024
        self._register_routes()

    def _register_routes(self):
        self.app.add_url_rule("/health", "health", self._health, methods=["GET"])
        self.app.add_url_rule("/infer", "infer", self._infer, methods=["POST"])
        self.app.add_url_rule("/infer/stream", "infer_stream", self._infer_stream, methods=["POST"])

    def _health(self):
        triton_alive = self.triton_client.is_server_live()
        return jsonify({
            "status": "ok" if triton_alive else "degraded",
            "triton": triton_alive,
        })

    def _infer(self):
        start = time.time()
        job_id = str(uuid.uuid4())[:8]

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

        result = {"job_id": job_id, "layers": len(all_sr_images), "elapsed_s": round(time.time() - start, 2)}

        if output_format == "ome_tiff":
            output_path = Path("output") / f"{job_id}" / f"sr_result.ome.tif"
            self.ome_writer.write(all_sr_images, output_path)
            result["output_path"] = str(output_path)
        else:
            import numpy as np
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

        return jsonify(result)

    def _infer_stream(self):
        def generate():
            job_id = str(uuid.uuid4())[:8]
            yield f'data: {{"job_id": "{job_id}", "status": "started"}}\n\n'

            data = request.get_data()
            images = self.stream_reader.parse_stream(io.BytesIO(data), "tiff")
            yield f'data: {{"job_id": "{job_id}", "status": "parsed", "layers": {len(images)}}}\n\n'

            for layer_idx, img in enumerate(images):
                sr = self._process_single_layer(img, job_id, layer_idx)
                yield f'data: {{"job_id": "{job_id}", "status": "layer_done", "layer": {layer_idx}}}\n\n'

            yield f'data: {{"job_id": "{job_id}", "status": "completed"}}\n\n'

        return Response(generate(), mimetype="text/event-stream")

    def _process_single_layer(self, image, job_id: str, layer_idx: int):
        tiles = self.sliding_window.tile_image(image, layer_index=layer_idx)
        logger.info(f"[{job_id}] Layer {layer_idx}: {len(tiles)} tiles")

        sr_tiles = []
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

        h, w = image.shape[:2]
        stitch_result = self.blender.stitch(sr_tiles, h, w, self.cfg.pipeline.scale_factor)
        return stitch_result.image

    def run(self, **kwargs):
        host = kwargs.get("host", self.cfg.server.host)
        port = kwargs.get("port", self.cfg.server.port)
        self.app.run(host=host, port=port, threaded=True)


import numpy as np


def create_app(cfg=None):
    server = InferenceServer(cfg)
    return server.app
