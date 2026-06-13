from wafer_srgan.models.builder import export_generator_onnx, build_generator
from wafer_srgan.config import load_config
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Export SRGAN generator to ONNX for Triton")
    parser.add_argument("--checkpoint", type=str, required=True, help="Generator checkpoint path")
    parser.add_argument("--output", type=str, default="triton_model_repo/srgan_generator/1/model.onnx")
    parser.add_argument("--input-size", type=int, nargs=2, default=[128, 128])
    args = parser.parse_args()

    cfg = load_config()
    generator = build_generator(cfg)

    import torch
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "generator" in ckpt:
        generator.load_state_dict(ckpt["generator"])
    else:
        generator.load_state_dict(ckpt)
    generator.eval()

    output_path = Path(args.output)
    export_generator_onnx(generator, output_path, input_size=tuple(args.input_size))
    print(f"ONNX model exported to {output_path}")


if __name__ == "__main__":
    main()
