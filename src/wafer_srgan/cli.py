import argparse
import logging
import sys

from wafer_srgan.config import load_config


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def train_entry():
    parser = argparse.ArgumentParser(description="Train SRGAN model for wafer super-resolution")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--lr-dir", type=str, default=None, help="LR training images directory")
    parser.add_argument("--hr-dir", type=str, default=None, help="HR training images directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)

    overrides = []
    if args.lr_dir:
        overrides.append(f"dataset.train.lr_dir={args.lr_dir}")
    if args.hr_dir:
        overrides.append(f"dataset.train.hr_dir={args.hr_dir}")
    if args.epochs:
        overrides.append(f"training.num_epochs={args.epochs}")
    if args.batch_size:
        overrides.append(f"training.batch_size={args.batch_size}")

    cfg = load_config(overrides=overrides if overrides else None, config_path=args.config)

    from wafer_srgan.training.trainer import Trainer
    trainer = Trainer(cfg)

    if args.resume:
        trainer.resume(args.resume)

    trainer.train()


def infer_entry():
    parser = argparse.ArgumentParser(description="Run SRGAN inference pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, required=True, help="Output OME-TIFF path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Generator checkpoint path")
    parser.add_argument("--edge", action="store_true", help="Run edge detection post-processing")
    parser.add_argument("--edge-method", type=str, default="canny", choices=["canny", "sobel"])
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    cfg = load_config(config_path=args.config)

    from wafer_srgan.pipeline.engine import SRGANPipeline
    pipeline = SRGANPipeline(cfg=cfg, checkpoint_path=args.checkpoint)
    result = pipeline.run(
        input_path=args.input,
        output_path=args.output,
        edge_detection=args.edge,
        edge_method=args.edge_method,
    )
    print(result)


def pipeline_entry():
    parser = argparse.ArgumentParser(description="Run end-to-end wafer SRGAN pipeline server")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--host", type=str, default=None, help="Server host")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    cfg = load_config(config_path=args.config)

    from wafer_srgan.server.app import InferenceServer
    server = InferenceServer(cfg)

    kwargs = {}
    if args.host:
        kwargs["host"] = args.host
    if args.port:
        kwargs["port"] = args.port

    server.run(**kwargs)
