import pytest
import torch
import numpy as np

from wafer_srgan.models.networks import Generator, Discriminator, VGGFeatureExtractor


class TestGenerator:
    def test_output_shape(self):
        model = Generator(in_channels=3, out_channels=3, num_features=32, num_residual_blocks=4, upscale_factor=4)
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 3, 128, 128)

    def test_batch_inference(self):
        model = Generator(in_channels=3, out_channels=3, num_features=32, num_residual_blocks=2, upscale_factor=4)
        x = torch.randn(2, 3, 48, 48)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 3, 192, 192)

    def test_single_channel(self):
        model = Generator(in_channels=1, out_channels=1, num_features=32, num_residual_blocks=2, upscale_factor=4)
        x = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1, 128, 128)


class TestDiscriminator:
    def test_output_shape(self):
        model = Discriminator(in_channels=3, num_features=32, num_layers=6)
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1)


class TestVGGFeatureExtractor:
    def test_feature_extraction(self):
        model = VGGFeatureExtractor(layer_index=10, use_input_norm=True)
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.ndim == 4
        assert y.shape[0] == 1
