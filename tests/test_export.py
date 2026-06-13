import pytest
import numpy as np
import tempfile
from pathlib import Path

from wafer_srgan.export.ome_tiff import OMETiffWriter


class TestOMETiffWriter:
    def test_write_single_layer(self):
        writer = OMETiffWriter(pixel_type="uint16", compression="none")
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test.ome.tif"
            writer.write([image], out_path)
            assert out_path.exists()
            assert out_path.stat().st_size > 0

    def test_write_multi_layer(self):
        writer = OMETiffWriter(pixel_type="uint8", compression="none")
        images = [
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "multi.ome.tif"
            writer.write(images, out_path)
            assert out_path.exists()
