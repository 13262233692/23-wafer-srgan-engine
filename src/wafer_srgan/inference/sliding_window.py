import logging
import uuid
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TilePacket:
    tile_id: str
    image: np.ndarray
    row: int
    col: int
    tile_h: int
    tile_w: int
    overlap_top: int
    overlap_left: int
    overlap_bottom: int
    overlap_right: int
    layer_index: int = 0
    metadata: dict = field(default_factory=dict)


class SlidingWindowEngine:
    def __init__(self, tile_size: int = 512, overlap: int = 64, scale_factor: int = 4, max_tiles_per_batch: int = 8):
        self.tile_size = tile_size
        self.overlap = overlap
        self.scale_factor = scale_factor
        self.max_tiles_per_batch = max_tiles_per_batch

    def tile_image(self, image: np.ndarray, layer_index: int = 0) -> list[TilePacket]:
        h, w = image.shape[:2]
        tiles = []

        step = self.tile_size - self.overlap

        rows = []
        y = 0
        while y < h:
            tile_h = min(self.tile_size, h - y)
            overlap_top = self.overlap if y > 0 else 0
            overlap_bottom = self.overlap if y + tile_h < h else 0
            rows.append((y, tile_h, overlap_top, overlap_bottom))
            y += step
            if y + self.overlap >= h and y < h:
                break

        cols = []
        x = 0
        while x < w:
            tile_w = min(self.tile_size, w - x)
            overlap_left = self.overlap if x > 0 else 0
            overlap_right = self.overlap if x + tile_w < w else 0
            cols.append((x, tile_w, overlap_left, overlap_right))
            x += step
            if x + self.overlap >= w and x < w:
                break

        for row_idx, (y, tile_h, o_top, o_bot) in enumerate(rows):
            for col_idx, (x, tile_w, o_left, o_right) in enumerate(cols):
                patch = image[y : y + tile_h, x : x + tile_w]
                tile = TilePacket(
                    tile_id=str(uuid.uuid4()),
                    image=patch,
                    row=row_idx,
                    col=col_idx,
                    tile_h=tile_h,
                    tile_w=tile_w,
                    overlap_top=o_top,
                    overlap_left=o_left,
                    overlap_bottom=o_bot,
                    overlap_right=o_right,
                    layer_index=layer_index,
                    metadata={
                        "src_y": y,
                        "src_x": x,
                        "src_h": h,
                        "src_w": w,
                    },
                )
                tiles.append(tile)

        logger.info(f"Tiled image ({h}x{w}) into {len(tiles)} tiles (grid: {len(rows)}x{len(cols)})")
        return tiles

    def batch_tiles(self, tiles: list[TilePacket]) -> Iterator[list[TilePacket]]:
        for i in range(0, len(tiles), self.max_tiles_per_batch):
            yield tiles[i : i + self.max_tiles_per_batch]

    def tile_image_batch(self, images: list[np.ndarray]) -> list[TilePacket]:
        all_tiles = []
        for layer_idx, img in enumerate(images):
            tiles = self.tile_image(img, layer_index=layer_idx)
            all_tiles.extend(tiles)
        return all_tiles
