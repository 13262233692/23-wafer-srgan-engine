import pytest
import numpy as np
from wafer_srgan.config import load_config


class TestConfig:
    def test_load_default(self):
        cfg = load_config()
        assert cfg.pipeline.scale_factor == 4
        assert cfg.model.generator.upscale_factor == 4
        assert cfg.sliding_window.tile_size == 512
        assert cfg.ome_tiff.pixel_type == "uint16"

    def test_override(self):
        cfg = load_config(overrides=["pipeline.scale_factor=2", "training.batch_size=8"])
        assert cfg.pipeline.scale_factor == 2
        assert cfg.training.batch_size == 8
