import torch
import torch.nn as nn
from wafer_srgan.models.networks import Generator, Discriminator, VGGFeatureExtractor


class SRGANLoss(nn.Module):
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        vgg_extractor: VGGFeatureExtractor,
        pixel_weight: float = 1.0,
        content_weight: float = 1.0,
        adversarial_weight: float = 0.005,
        tv_weight: float = 0.0,
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.vgg_extractor = vgg_extractor
        self.pixel_weight = pixel_weight
        self.content_weight = content_weight
        self.adversarial_weight = adversarial_weight
        self.tv_weight = tv_weight

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def compute_generator_loss(self, sr: torch.Tensor, hr: torch.Tensor) -> dict[str, torch.Tensor]:
        pixel_loss = self.l1_loss(sr, hr)

        vgg_sr = self.vgg_extractor(sr)
        with torch.no_grad():
            vgg_hr = self.vgg_extractor(hr)
        content_loss = self.l1_loss(vgg_sr, vgg_hr)

        disc_sr = self.discriminator(sr)
        real_label = torch.ones_like(disc_sr)
        adversarial_loss = self.bce_loss(disc_sr, real_label)

        tv_loss = torch.tensor(0.0, device=sr.device)
        if self.tv_weight > 0:
            diff_h = sr[:, :, 1:, :] - sr[:, :, :-1, :]
            diff_w = sr[:, :, :, 1:] - sr[:, :, :, :-1]
            tv_loss = diff_h.pow(2).mean() + diff_w.pow(2).mean()

        total_loss = (
            self.pixel_weight * pixel_loss
            + self.content_weight * content_loss
            + self.adversarial_weight * adversarial_loss
            + self.tv_weight * tv_loss
        )

        return {
            "g_total": total_loss,
            "g_pixel": pixel_loss.detach(),
            "g_content": content_loss.detach(),
            "g_adversarial": adversarial_loss.detach(),
            "g_tv": tv_loss.detach(),
        }

    def compute_discriminator_loss(self, sr: torch.Tensor, hr: torch.Tensor) -> dict[str, torch.Tensor]:
        disc_hr = self.discriminator(hr)
        disc_sr = self.discriminator(sr.detach())

        real_label = torch.ones_like(disc_hr)
        fake_label = torch.zeros_like(disc_sr)

        loss_real = self.bce_loss(disc_hr, real_label)
        loss_fake = self.bce_loss(disc_sr, fake_label)

        total_loss = (loss_real + loss_fake) / 2.0

        return {
            "d_total": total_loss,
            "d_real": loss_real.detach(),
            "d_fake": loss_fake.detach(),
        }
