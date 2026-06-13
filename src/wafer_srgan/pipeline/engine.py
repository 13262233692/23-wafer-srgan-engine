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
from wafer_srgan.inference.edge_postprocess import EdgePostProcessor
from wafer_srgan.export.ome_tiff import OMETiffWriter

logger = logging.getLogger(__name__)


class SRGANPipeline:
    def __init__(self, cfg=None, checkpoint_path: Optional[str] = None):
        if cfg is None:
            cfg = load_config()

        self.cfg = cfg
        self.scale_factor = cfg.pipeline.scale_factor

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
        if checkpoint_path:
            self.triton_client.load_local_weights(checkpoint_path)

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

    def run(
        self,
        input_path: str,
        output_path: str,
        edge_detection: bool = False,
        edge_method: str = "canny",
    ) -> dict:
        start = time.time()
        logger.info(f"Pipeline started: {input_path} -> {output_path}")

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
        logger.info(f"  Inference done: {total_batches} batches, {len(sr_tiles)} tiles")

        logger.info("Step 4/5: Gaussian-weighted stitching...")
        layer_tiles = {}
        for t in sr_tiles:
            layer_tiles.setdefault(t.layer_index, []).append(t)

        sr_images = []
        for layer_idx in sorted(layer_tiles.keys()):
            tiles = layer_tiles[layer_idx]
            src_h = tiles[0].metadata["src_h"]
            src_w = tiles[0].metadata["src_w"]
            result = self.blender.stitch(tiles, src_h, src_w, self.scale_factor)
            sr_images.append(result.image)
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
        logger.info(f"Pipeline complete in {elapsed:.2f}s")

        return {
            "input": str(input_path),
            "output": str(output_path),
            "layers": len(sr_images),
            "scale_factor": self.scale_factor,
            "elapsed_s": round(elapsed, 2),
            "output_size_mb": round(output_path.stat().st_size / 1e6, 2),
        }
