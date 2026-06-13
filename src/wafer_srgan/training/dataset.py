import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class WaferPatchDataset(Dataset):
    def __init__(
        self,
        hr_dir: str,
        lr_dir: Optional[str] = None,
        patch_size: int = 96,
        scale_factor: int = 4,
        augment: bool = True,
    ):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir) if lr_dir else None
        self.patch_size = patch_size
        self.lr_patch_size = patch_size // scale_factor
        self.scale_factor = scale_factor
        self.augment = augment

        self.hr_paths = sorted(self.hr_dir.glob("*.png"))
        if not self.hr_paths:
            self.hr_paths = sorted(self.hr_dir.glob("*.tif"))
        if not self.hr_paths:
            self.hr_paths = sorted(self.hr_dir.glob("*.tiff"))

        if self.lr_dir:
            self.lr_paths = sorted(self.lr_dir.glob("*.png"))
            if not self.lr_paths:
                self.lr_paths = sorted(self.lr_dir.glob("*.tif"))

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        hr_img = self._load_image(self.hr_paths[idx])

        if self.lr_dir and idx < len(self.lr_paths):
            lr_img = self._load_image(self.lr_paths[idx])
        else:
            h, w = hr_img.shape[:2]
            lr_h, lr_w = h // self.scale_factor, w // self.scale_factor
            lr_pil = Image.fromarray(hr_img)
            lr_pil = lr_pil.resize((lr_w, lr_h), Image.BICUBIC)
            lr_img = np.array(lr_pil)

        hr_patch, lr_patch = self._crop_patch(hr_img, lr_img)

        if self.augment:
            hr_patch, lr_patch = self._augment(hr_patch, lr_patch)

        hr_tensor = self._to_tensor(hr_patch)
        lr_tensor = self._to_tensor(lr_patch)

        return {"lr": lr_tensor, "hr": hr_tensor}

    def _load_image(self, path: Path) -> np.ndarray:
        img = Image.open(str(path)).convert("RGB")
        return np.array(img)

    def _crop_patch(self, hr_img: np.ndarray, lr_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = hr_img.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            hr_pil = Image.fromarray(hr_img)
            hr_pil = hr_pil.resize((max(w, self.patch_size), max(h, self.patch_size)), Image.BICUBIC)
            hr_img = np.array(hr_pil)
            h, w = hr_img.shape[:2]

        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        hr_patch = hr_img[top : top + self.patch_size, left : left + self.patch_size]

        lr_top = top // self.scale_factor
        lr_left = left // self.scale_factor
        lr_patch = lr_img[lr_top : lr_top + self.lr_patch_size, lr_left : lr_left + self.lr_patch_size]

        return hr_patch, lr_patch

    def _augment(self, hr: np.ndarray, lr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            hr = hr[:, ::-1, :].copy()
            lr = lr[:, ::-1, :].copy()
        if random.random() < 0.5:
            hr = hr[::-1, :, :].copy()
            lr = lr[::-1, :, :].copy()
        if random.random() < 0.5:
            hr = np.rot90(hr, k=1, axes=(0, 1)).copy()
            lr = np.rot90(lr, k=1, axes=(0, 1)).copy()
        return hr, lr

    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)


def build_dataloaders(cfg) -> tuple[DataLoader, DataLoader | None]:
    train_ds = WaferPatchDataset(
        hr_dir=cfg.dataset.train.hr_dir,
        lr_dir=cfg.dataset.train.lr_dir,
        patch_size=cfg.dataset.train.patch_size,
        scale_factor=cfg.pipeline.scale_factor,
        augment=cfg.dataset.train.augment,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        drop_last=True,
    )

    val_loader = None
    if cfg.dataset.val.hr_dir:
        val_ds = WaferPatchDataset(
            hr_dir=cfg.dataset.val.hr_dir,
            lr_dir=cfg.dataset.val.lr_dir,
            patch_size=cfg.dataset.val.patch_size,
            scale_factor=cfg.pipeline.scale_factor,
            augment=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    return train_loader, val_loader
