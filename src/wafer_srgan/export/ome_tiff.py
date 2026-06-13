import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class OMETiffWriter:
    def __init__(
        self,
        pixel_type: str = "uint16",
        channel_names: list[str] | None = None,
        physical_size_x: float = 0.1,
        physical_size_y: float = 0.1,
        physical_size_x_unit: str = "um",
        physical_size_y_unit: str = "um",
        compression: str = "zstd",
        tile_size: int = 512,
    ):
        self.pixel_type = pixel_type
        self.channel_names = channel_names or ["R", "G", "B"]
        self.physical_size_x = physical_size_x
        self.physical_size_y = physical_size_y
        self.physical_size_x_unit = physical_size_x_unit
        self.physical_size_y_unit = physical_size_y_unit
        self.compression = compression
        self.tile_size = tile_size

    def _numpy_to_ome_dtype(self, data: np.ndarray) -> np.ndarray:
        if self.pixel_type == "uint8":
            return data.astype(np.uint8)
        elif self.pixel_type == "uint16":
            if data.dtype == np.uint8:
                return (data.astype(np.float32) * 257).astype(np.uint16)
            return data.astype(np.uint16)
        elif self.pixel_type == "float32":
            return data.astype(np.float32)
        else:
            return data.astype(np.uint16)

    def _build_ome_xml(self, shape: tuple, num_layers: int = 1) -> str:
        size_y, size_x = shape[:2]
        size_c = len(self.channel_names)
        size_z = num_layers

        ome_ns = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
        ET.register_namespace("", ome_ns)

        root = ET.Element("OME", xmlns=ome_ns)

        for layer_idx in range(num_layers):
            image_el = ET.SubElement(root, "Image", ID=f"Image:{layer_idx}", Name=f"Layer_{layer_idx}")

            creation_date = ET.SubElement(image_el, "CreationDate")
            creation_date.text = "2026-06-13T00:00:00"

            pixels = ET.SubElement(
                image_el, "Pixels",
                ID=f"Pixels:{layer_idx}",
                DimensionOrder="XYCZT",
                Type=self.pixel_type,
                SizeX=str(size_x),
                SizeY=str(size_y),
                SizeC=str(size_c),
                SizeZ=str(size_z),
                SizeT="1",
                PhysicalSizeX=str(self.physical_size_x),
                PhysicalSizeY=str(self.physical_size_y),
                PhysicalSizeXUnit=self.physical_size_x_unit,
                PhysicalSizeYUnit=self.physical_size_y_unit,
            )

            for c_idx, ch_name in enumerate(self.channel_names):
                channel = ET.SubElement(pixels, "Channel", ID=f"Channel:{layer_idx}:{c_idx}", Name=ch_name)
                if c_idx == 0:
                    channel.set("SamplesPerPixel", str(size_c))

            for z in range(size_z):
                for c in range(size_c):
                    tiff_data = ET.SubElement(
                        pixels, "TiffData",
                        FirstC=str(c),
                        FirstZ=str(z),
                        FirstT="0",
                        PlaneCount="1",
                    )

        xml_str = ET.tostring(root, encoding="unicode", xml_declaration=False)
        return f'<?xml version="1.0" encoding="UTF-8"?>{xml_str}'

    def write(
        self,
        images: list[np.ndarray],
        output_path: str | Path,
        metadata: Optional[dict] = None,
    ) -> Path:
        import tifffile

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not images:
            raise ValueError("No images to write")

        processed = []
        for img in images:
            data = self._numpy_to_ome_dtype(img)
            if data.ndim == 2:
                data = data[np.newaxis, :, :]
            elif data.ndim == 3 and data.shape[-1] == len(self.channel_names):
                data = data.transpose(2, 0, 1)
            processed.append(data)

        ome_xml = self._build_ome_xml(processed[0].shape[1:], num_layers=len(images))

        compression_map = {
            "zstd": "zstd",
            "lzw": "lzw",
            "deflate": "deflate",
            "none": None,
        }
        comp = compression_map.get(self.compression, None)

        with tifffile.TiffWriter(str(output_path), ome=True, bigtiff=True) as tw:
            for layer_idx, data in enumerate(processed):
                tw.write(
                    data,
                    photometric="rgb" if data.shape[0] == 3 else "minisblack",
                    compression=comp,
                    tile=(self.tile_size, self.tile_size) if self.tile_size else None,
                    metadata=None if layer_idx > 0 else {"Creator": "wafer-srgan-engine"},
                )

        desc = tw._fh if hasattr(tw, '_fh') else None
        logger.info(f"OME-TIFF written: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return output_path

    def write_single_layer(
        self,
        image: np.ndarray,
        output_path: str | Path,
    ) -> Path:
        return self.write([image], output_path)
