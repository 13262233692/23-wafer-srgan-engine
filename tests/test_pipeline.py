import pytest
import numpy as np

from wafer_srgan.inference.sliding_window import SlidingWindowEngine, TilePacket
from wafer_srgan.inference.stitcher import GaussianBlender


class TestSlidingWindow:
    def test_exact_tile(self):
        engine = SlidingWindowEngine(tile_size=64, overlap=8, scale_factor=4)
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        tiles = engine.tile_image(image)
        assert len(tiles) >= 1
        assert all(t.tile_h <= 64 for t in tiles)
        assert all(t.tile_w <= 64 for t in tiles)

    def test_large_image(self):
        engine = SlidingWindowEngine(tile_size=128, overlap=16, scale_factor=4)
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        tiles = engine.tile_image(image)
        assert len(tiles) > 1
        for t in tiles:
            assert t.image.shape[0] <= 128
            assert t.image.shape[1] <= 128

    def test_batch_tiles(self):
        engine = SlidingWindowEngine(tile_size=64, overlap=8, scale_factor=4, max_tiles_per_batch=2)
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        tiles = engine.tile_image(image)
        batches = list(engine.batch_tiles(tiles))
        assert len(batches) >= 1
        for batch in batches:
            assert len(batch) <= 2


class TestStitcher:
    def test_single_tile_stitch(self):
        engine = SlidingWindowEngine(tile_size=64, overlap=0, scale_factor=4)
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        tiles = engine.tile_image(image)

        for t in tiles:
            h4 = t.image.shape[0] * 4
            w4 = t.image.shape[1] * 4
            t.image = np.random.randint(0, 255, (h4, w4, 3), dtype=np.uint8)
            t.tile_h = h4
            t.tile_w = w4

        blender = GaussianBlender(sigma=4.0)
        result = blender.stitch(tiles, 64, 64, scale_factor=4)
        assert result.image.shape[0] == 256
        assert result.image.shape[1] == 256
