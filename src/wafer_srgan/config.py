from omegaconf import OmegaConf, DictConfig
import yaml
import os
from pathlib import Path

_DEFAULT_CFG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"


def load_config(overrides: list[str] | None = None, config_path: str | None = None) -> DictConfig:
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            base = OmegaConf.create(yaml.safe_load(f))
    else:
        with open(_DEFAULT_CFG_PATH, "r") as f:
            base = OmegaConf.create(yaml.safe_load(f))

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        base = OmegaConf.merge(base, override_cfg)

    OmegaConf.set_struct(base, False)
    return base
