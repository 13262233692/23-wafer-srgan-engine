import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.maxpool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetLite(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2, base_filters: int = 16):
        super().__init__()
        f = base_filters
        self.input_conv = DoubleConv(in_channels, f)
        self.down1 = DownBlock(f, f * 2)
        self.down2 = DownBlock(f * 2, f * 4)
        self.down3 = DownBlock(f * 4, f * 8)
        self.bottleneck = DoubleConv(f * 8, f * 16)
        self.up3 = UpBlock(f * 16, f * 4)
        self.up2 = UpBlock(f * 8, f * 2)
        self.up1 = UpBlock(f * 4, f)
        self.out_conv = nn.Conv2d(f, num_classes, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x = self.up3(x5, x4)
        x = self.up2(x, x3)
        x = self.up1(x, x2)
        logits = self.out_conv(x)
        return logits


class DefectSegmentor:
    def __init__(
        self,
        num_classes: int = 2,
        base_filters: int = 16,
        device: Optional[torch.device] = None,
        threshold: float = 0.5,
        min_defect_area: int = 16,
        use_gpu_if_available: bool = True,
    ):
        self.num_classes = num_classes
        self.threshold = threshold
        self.min_defect_area = min_defect_area

        if device is None:
            self.device = torch.device(
                "cuda" if use_gpu_if_available and torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.model = UNetLite(in_channels=3, num_classes=num_classes, base_filters=base_filters)
        self.model.to(self.device)
        self.model.eval()

        self._total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"DefectSegmentor initialized: UNet-Lite "
            f"({self._total_params:,} params) on {self.device}"
        )

    def load_weights(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info(f"Loaded defect segmentor weights from {checkpoint_path}")

    @torch.no_grad()
    def segment(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            input_tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        elif image.ndim == 3:
            input_tensor = torch.from_numpy(
                image.transpose(2, 0, 1).astype(np.float32)
            ).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if input_tensor.max() > 1.0:
            input_tensor = input_tensor / 255.0

        input_tensor = input_tensor.to(self.device)

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            input_tensor = F.pad(input_tensor, [0, pad_w, 0, pad_h], mode="reflect")

        logits = self.model(input_tensor)
        logits = logits[:, :, :h, :w]

        if self.num_classes == 1:
            probs = torch.sigmoid(logits[:, 0])
            mask = (probs > self.threshold).cpu().numpy().astype(np.uint8)
        else:
            probs = F.softmax(logits, dim=1)
            mask = torch.argmax(probs, dim=1).cpu().numpy().astype(np.uint8)

        return mask[0]

    @torch.no_grad()
    def segment_tiled(
        self,
        image: np.ndarray,
        tile_size: int = 1024,
        overlap: int = 64,
    ) -> np.ndarray:
        h, w = image.shape[:2]
        if h <= tile_size and w <= tile_size:
            return self.segment(image)

        full_mask = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        step = tile_size - overlap
        for y in range(0, h, step):
            for x in range(0, w, step):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                tile = image[y:y_end, x:x_end]

                th, tw = tile.shape[:2]
                weight = np.ones((th, tw), dtype=np.float32)
                if overlap > 0:
                    if y > 0:
                        ramp = np.linspace(0, 1, min(overlap, th)).reshape(-1, 1)
                        weight[:min(overlap, th), :] *= ramp
                    if y_end < h:
                        ramp = np.linspace(1, 0, min(overlap, th)).reshape(-1, 1)
                        weight[-min(overlap, th):, :] *= ramp
                    if x > 0:
                        ramp = np.linspace(0, 1, min(overlap, tw)).reshape(1, -1)
                        weight[:, :min(overlap, tw)] *= ramp
                    if x_end < w:
                        ramp = np.linspace(1, 0, min(overlap, tw)).reshape(1, -1)
                        weight[:, -min(overlap, tw):] *= ramp

                tile_mask = self.segment(tile).astype(np.float32)

                full_mask[y:y_end, x:x_end] += tile_mask * weight
                weight_map[y:y_end, x:x_end] += weight

        mask_valid = weight_map > 0
        result = np.zeros((h, w), dtype=np.uint8)
        result[mask_valid] = (full_mask[mask_valid] / weight_map[mask_valid] > self.threshold).astype(np.uint8)
        return result

    @staticmethod
    def cleanup_small_regions(mask: np.ndarray, min_area: Optional[int] = None) -> np.ndarray:
        import cv2
        min_area = min_area if min_area is not None else 16
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        cleaned = np.zeros_like(mask)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == label] = 1
        return cleaned

    def get_line_width_mask(self, mask: np.ndarray, kernel_size: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        import cv2
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        line_width = dist * 2.0
        skeleton = cv2.ximgproc.thinning(mask) if hasattr(cv2, 'ximgproc') else None
        return line_width, skeleton


__all__ = ["UNetLite", "DefectSegmentor"]
