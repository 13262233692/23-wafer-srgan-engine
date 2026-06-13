import torch
from pathlib import Path
from wafer_srgan.models.networks import Generator, Discriminator, VGGFeatureExtractor


def build_generator(cfg) -> Generator:
    return Generator(
        in_channels=cfg.model.generator.in_channels,
        out_channels=cfg.model.generator.out_channels,
        num_features=cfg.model.generator.num_features,
        num_residual_blocks=cfg.model.generator.num_residual_blocks,
        upscale_factor=cfg.model.generator.upscale_factor,
        residual_scaling=cfg.model.generator.residual_scaling,
    )


def build_discriminator(cfg) -> Discriminator:
    return Discriminator(
        in_channels=cfg.model.discriminator.in_channels,
        num_features=cfg.model.discriminator.num_features,
        num_layers=cfg.model.discriminator.num_layers,
    )


def build_vgg_extractor(cfg) -> VGGFeatureExtractor:
    return VGGFeatureExtractor(
        layer_index=cfg.model.vgg.layer_index,
        use_input_norm=cfg.model.vgg.use_input_norm,
        range_norm=cfg.model.vgg.range_norm,
    )


def save_checkpoint(generator: Generator, discriminator: Discriminator, g_opt, d_opt, epoch: int, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "g_optimizer": g_opt.state_dict(),
            "d_optimizer": d_opt.state_dict(),
        },
        str(path),
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict:
    return torch.load(str(path), map_location=device, weights_only=False)


def load_generator_for_inference(path: str | Path, cfg, device: str = "cpu") -> Generator:
    generator = build_generator(cfg)
    ckpt = load_checkpoint(path, device)
    generator.load_state_dict(ckpt["generator"])
    generator.to(device)
    generator.eval()
    return generator


def export_generator_onnx(generator: Generator, output_path: str | Path, input_size: tuple[int, int] = (128, 128), device: str = "cpu"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, input_size[0], input_size[1], device=device)
    generator.to(device)
    generator.eval()
    torch.onnx.export(
        generator,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
        opset_version=17,
    )
