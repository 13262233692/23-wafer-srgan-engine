import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from wafer_srgan.inference.sliding_window import TilePacket

logger = logging.getLogger(__name__)


@dataclass
class StitchResult:
    image: np.ndarray
    weight_map: np.ndarray


class GaussianBlender:
    def __init__(self, sigma: float = 16.0, feather_width: int = 32):
        self.sigma = sigma
        self.feather_width = feather_width

    def _create_weight_map(self, tile_h: int, tile_w: int, overlap_top: int, overlap_left: int,
                           overlap_bottom: int, overlap_right: int) -> np.ndarray:
        weight = np.ones((tile_h, tile_w), dtype=np.float64)

        if overlap_top > 0:
            ramp = np.linspace(0, 1, overlap_top)
            weight[:overlap_top, :] *= ramp[:, np.newaxis]
        if overlap_bottom > 0:
            ramp = np.linspace(1, 0, overlap_bottom)
            weight[-overlap_bottom:, :] *= ramp[:, np.newaxis]
        if overlap_left > 0:
            ramp = np.linspace(0, 1, overlap_left)
            weight[:, :overlap_left] *= ramp[np.newaxis, :]
        if overlap_right > 0:
            ramp = np.linspace(1, 0, overlap_right)
            weight[:, -overlap_right:] *= ramp[np.newaxis, :]

        weight = gaussian_filter(weight, sigma=self.sigma)
        weight = np.maximum(weight, 1e-8)

        return weight

    def stitch(self, tiles: list[TilePacket], target_h: int, target_w: int, scale_factor: int = 4) -> StitchResult:
        sr_h = target_h * scale_factor
        sr_w = target_w * scale_factor

        channels = tiles[0].image.shape[-1] if tiles[0].image.ndim == 3 else 1
        if channels == 1:
            canvas = np.zeros((sr_h, sr_w), dtype=np.float64)
        else:
            canvas = np.zeros((sr_h, sr_w, channels), dtype=np.float64)

        weight_canvas = np.zeros((sr_h, sr_w), dtype=np.float64)

        for tile in tiles:
            sr_tile = tile.image
            if sr_tile.dtype != np.float64:
                sr_tile = sr_tile.astype(np.float64)
            if sr_tile.max() > 1.0 and sr_tile.max() <= 255.0:
                sr_tile = sr_tile / 255.0

            src_y = tile.metadata.get("src_y", 0)
            src_x = tile.metadata.get("src_x", 0)

            dst_y = src_y * scale_factor
            dst_x = src_x * scale_factor

            tile_h_sr = sr_tile.shape[0]
            tile_w_sr = sr_tile.shape[1]

            weight = self._create_weight_map(
                tile_h_sr, tile_w_sr,
                tile.overlap_top * scale_factor,
                tile.overlap_left * scale_factor,
                tile.overlap_bottom * scale_factor,
                tile.overlap_right * scale_factor,
            )

            y_end = min(dst_y + tile_h_sr, sr_h)
            x_end = min(dst_x + tile_w_sr, sr_w)
            actual_h = y_end - dst_y
            actual_w = x_end - dst_x

            if channels == 1:
                canvas[dst_y:y_end, dst_x:x_end] += sr_tile[:actual_h, :actual_w] * weight[:actual_h, :actual_w]
            else:
                for c in range(channels):
                    canvas[dst_y:y_end, dst_x:x_end, c] += sr_tile[:actual_h, :actual_w, c] * weight[:actual_h, :actual_w]

            weight_canvas[dst_y:y_end, dst_x:x_end] += weight[:actual_h, :actual_w]

        mask = weight_canvas > 0
        if channels == 1:
            canvas[mask] /= weight_canvas[mask]
        else:
            for c in range(channels):
                canvas[:, :, c][mask] /= weight_canvas[mask]

        result = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
        return StitchResult(image=result, weight_map=weight_canvas)
