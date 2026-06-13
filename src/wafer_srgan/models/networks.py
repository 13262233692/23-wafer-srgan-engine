import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_features: int = 64, growth_channels: int = 32, num_layers: int = 5, residual_scaling: float = 0.2):
        super().__init__()
        self.residual_scaling = residual_scaling
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = num_features + i * growth_channels
            self.layers.append(nn.Conv2d(in_ch, growth_channels, kernel_size=3, stride=1, padding=1))
        self.conv_out = nn.Conv2d(num_features + num_layers * growth_channels, num_features, kernel_size=1, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            out = self.lrelu(layer(torch.cat(features, dim=1)))
            features.append(out)
        out = self.conv_out(torch.cat(features, dim=1))
        return x + out * self.residual_scaling


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, num_features: int = 64, growth_channels: int = 32, num_dense_layers: int = 5, num_blocks: int = 3, residual_scaling: float = 0.2):
        super().__init__()
        self.residual_scaling = residual_scaling
        self.blocks = nn.ModuleList([
            ResidualDenseBlock(num_features, growth_channels, num_dense_layers, residual_scaling)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = out + block(out) * self.residual_scaling
        return out


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_residual_blocks: int = 16,
        upscale_factor: int = 4,
        growth_channels: int = 32,
        residual_scaling: float = 0.2,
    ):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.conv_head = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1)

        self.body = nn.Sequential(
            *[ResidualInResidualDenseBlock(num_features, growth_channels, residual_scaling=residual_scaling)
              for _ in range(num_residual_blocks)]
        )
        self.conv_body = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)

        upsample_layers = []
        num_upsamples = int(math.log2(upscale_factor))
        for _ in range(num_upsamples):
            upsample_layers.append(nn.Conv2d(num_features, num_features * 4, kernel_size=3, stride=1, padding=1))
            upsample_layers.append(nn.PixelShuffle(2))
            upsample_layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.upsample = nn.Sequential(*upsample_layers)

        self.conv_tail = nn.Conv2d(num_features, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat1 = self.conv_head(x)
        body_out = self.body(feat1)
        feat2 = self.conv_body(body_out)
        out = self.upsample(feat1 + feat2)
        out = self.conv_tail(out)
        return out


import math

__all__ = ["Generator", "Discriminator", "VGGFeatureExtractor"]


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, num_features: int = 64, num_layers: int = 8):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        for i in range(1, num_layers):
            in_ch = min(num_features * (2 ** ((i - 1) // 2)), 512)
            out_ch = min(num_features * (2 ** (i // 2)), 512)
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1 + (i % 2), padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 128, 128)
            feat = self.features(dummy)
            flat_size = feat.flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat.flatten(1)
        return self.classifier(feat)


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_index: int = 36, use_input_norm: bool = True, range_norm: bool = False):
        super().__init__()
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features.children())[: layer_index + 1])

        for param in self.features.parameters():
            param.requires_grad = False

        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
        else:
            self.mean = None
            self.std = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.range_norm:
            x = (x + 1.0) / 2.0
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        return self.features(x)
