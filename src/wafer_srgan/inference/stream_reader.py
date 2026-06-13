import io
import logging
from pathlib import Path
from typing import Generator as GenType

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class RawImageStreamReader:
    def __init__(self, chunk_size: int = 65536):
        self.chunk_size = chunk_size

    def parse_stream(self, stream: io.BytesIO, image_format: str = "png") -> list[np.ndarray]:
        images = []
        buf = bytearray()
        while True:
            chunk = stream.read(self.chunk_size)
            if not chunk:
                break
            buf.extend(chunk)

        if not buf:
            return images

        data = bytes(buf)

        if image_format in ("tiff", "tif"):
            images = self._parse_tiff_stack(data)
        else:
            images = self._parse_single(data, image_format)

        logger.info(f"Parsed {len(images)} image(s) from stream")
        return images

    def _parse_single(self, data: bytes, fmt: str) -> list[np.ndarray]:
        try:
            img = Image.open(io.BytesIO(data))
            return [np.array(img.convert("RGB"))]
        except Exception as e:
            logger.error(f"Failed to parse image: {e}")
            return []

    def _parse_tiff_stack(self, data: bytes) -> list[np.ndarray]:
        try:
            import tifffile
            tif = tifffile.TiffFile(io.BytesIO(data))
            images = []
            for page in tif.pages:
                img_array = page.asarray()
                if img_array.ndim == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.ndim == 3 and img_array.shape[-1] > 3:
                    img_array = img_array[:, :, :3]
                images.append(img_array)
            return images
        except Exception as e:
            logger.error(f"Failed to parse TIFF stack: {e}")
            return []

    def stream_from_file(self, filepath: str | Path) -> list[np.ndarray]:
        filepath = Path(filepath)
        suffix = filepath.suffix.lstrip(".").lower()
        if suffix in ("tif", "tiff"):
            fmt = "tiff"
        elif suffix == "png":
            fmt = "png"
        else:
            fmt = "png"

        with open(filepath, "rb") as f:
            return self.parse_stream(io.BytesIO(f.read()), fmt)
