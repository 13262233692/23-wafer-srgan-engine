import time
import logging
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, MultiStepLR

from wafer_srgan.models.builder import build_generator, build_discriminator, build_vgg_extractor, save_checkpoint
from wafer_srgan.models.losses import SRGANLoss
from wafer_srgan.training.dataset import build_dataloaders

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.pipeline.device if torch.cuda.is_available() else "cpu")
        self.epoch = 0
        self.best_psnr = 0.0

        self.generator = build_generator(cfg).to(self.device)
        self.discriminator = build_discriminator(cfg).to(self.device)
        self.vgg_extractor = build_vgg_extractor(cfg).to(self.device).eval()

        self.g_optimizer = self._build_optimizer(self.generator, cfg.training.optimizer.generator)
        self.d_optimizer = self._build_optimizer(self.discriminator, cfg.training.optimizer.discriminator)

        self.g_scheduler = self._build_scheduler(self.g_optimizer, cfg.training.scheduler)
        self.d_scheduler = self._build_scheduler(self.d_optimizer, cfg.training.scheduler)

        self.criterion = SRGANLoss(
            generator=self.generator,
            discriminator=self.discriminator,
            vgg_extractor=self.vgg_extractor,
            pixel_weight=cfg.training.loss.pixel_weight,
            content_weight=cfg.training.loss.content_weight,
            adversarial_weight=cfg.training.loss.adversarial_weight,
            tv_weight=cfg.training.loss.tv_weight,
        )

        self.train_loader, self.val_loader = build_dataloaders(cfg)

        self.ckpt_dir = Path("checkpoints")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _build_optimizer(self, model, opt_cfg):
        lr = float(opt_cfg.lr)
        betas = tuple(float(b) for b in opt_cfg.betas)
        weight_decay = float(opt_cfg.weight_decay)
        return Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    def _build_scheduler(self, optimizer, sched_cfg):
        stype = sched_cfg.type
        if stype == "step":
            return StepLR(optimizer, step_size=sched_cfg.step_size, gamma=float(sched_cfg.gamma))
        elif stype == "cosine":
            return CosineAnnealingLR(optimizer, T_max=self.cfg.training.num_epochs)
        elif stype == "multistep":
            return MultiStepLR(optimizer, milestones=[200, 400], gamma=float(sched_cfg.gamma))
        else:
            return StepLR(optimizer, step_size=sched_cfg.step_size, gamma=float(sched_cfg.gamma))

    def train(self):
        logger.info(f"Starting training on device: {self.device}")
        logger.info(f"Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        logger.info(f"Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")

        for epoch in range(self.epoch, self.cfg.training.num_epochs):
            self.epoch = epoch
            self.generator.train()
            self.discriminator.train()

            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(self.train_loader):
                lr = batch["lr"].to(self.device)
                hr = batch["hr"].to(self.device)

                sr = self.generator(lr)

                d_losses = self.criterion.compute_discriminator_loss(sr, hr)
                self.d_optimizer.zero_grad()
                d_losses["d_total"].backward()
                self.d_optimizer.step()

                sr = self.generator(lr)
                g_losses = self.criterion.compute_generator_loss(sr, hr)
                self.g_optimizer.zero_grad()
                g_losses["g_total"].backward()
                self.g_optimizer.step()

                epoch_g_loss += g_losses["g_total"].item()
                epoch_d_loss += d_losses["d_total"].item()
                n_batches += 1

                if (batch_idx + 1) % 50 == 0:
                    logger.info(
                        f"[Epoch {epoch}] Batch {batch_idx + 1}/{len(self.train_loader)} | "
                        f"G_loss: {g_losses['g_total'].item():.4f} D_loss: {d_losses['d_total'].item():.4f}"
                    )

            self.g_scheduler.step()
            self.d_scheduler.step()

            avg_g = epoch_g_loss / max(n_batches, 1)
            avg_d = epoch_d_loss / max(n_batches, 1)
            logger.info(f"Epoch {epoch} done. Avg G_loss: {avg_g:.4f} Avg D_loss: {avg_d:.4f}")

            if (epoch + 1) % self.cfg.training.checkpoint.save_interval == 0:
                save_checkpoint(
                    self.generator, self.discriminator,
                    self.g_optimizer, self.d_optimizer,
                    epoch, self.ckpt_dir / f"ckpt_epoch_{epoch:05d}.pt",
                )
                logger.info(f"Checkpoint saved: ckpt_epoch_{epoch:05d}.pt")

            if self.val_loader and (epoch + 1) % 10 == 0:
                psnr = self._validate()
                logger.info(f"Validation PSNR: {psnr:.2f} dB")
                if psnr > self.best_psnr:
                    self.best_psnr = psnr
                    save_checkpoint(
                        self.generator, self.discriminator,
                        self.g_optimizer, self.d_optimizer,
                        epoch, self.ckpt_dir / "best_model.pt",
                    )
                    logger.info(f"New best model saved with PSNR: {psnr:.2f} dB")

        save_checkpoint(
            self.generator, self.discriminator,
            self.g_optimizer, self.d_optimizer,
            self.epoch, self.ckpt_dir / "last_model.pt",
        )
        logger.info("Training complete.")

    @torch.no_grad()
    def _validate(self) -> float:
        self.generator.eval()
        total_psnr = 0.0
        count = 0
        for batch in self.val_loader:
            lr = batch["lr"].to(self.device)
            hr = batch["hr"].to(self.device)
            sr = self.generator(lr)
            sr = sr.clamp(0.0, 1.0)
            mse = ((sr - hr) ** 2).mean()
            if mse > 0:
                psnr = 10.0 * torch.log10(1.0 / mse)
            else:
                psnr = torch.tensor(100.0)
            total_psnr += psnr.item()
            count += 1
        self.generator.train()
        return total_psnr / max(count, 1)

    def resume(self, ckpt_path: str):
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.g_optimizer.load_state_dict(ckpt["g_optimizer"])
        self.d_optimizer.load_state_dict(ckpt["d_optimizer"])
        self.epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {self.epoch}")
