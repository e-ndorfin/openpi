"""Validate a YAM checkpoint's deployable layout without loading model weights."""

from __future__ import annotations

import argparse
import json
import pathlib

from openpi.training import config as _config


def validate_checkpoint(config_name: str, checkpoint_dir: str | pathlib.Path) -> dict:
    checkpoint = pathlib.Path(checkpoint_dir).expanduser().resolve()
    cfg = _config.get_config(config_name)
    metadata = cfg.policy_metadata or {}
    if metadata.get("contract") != "yam_bimanual_v1":
        raise ValueError(f"Config {config_name!r} does not declare contract yam_bimanual_v1")
    if int(metadata.get("action_dim", -1)) != 14:
        raise ValueError(f"Config {config_name!r} must declare action_dim=14")
    if int(metadata.get("model_action_horizon", -1)) != int(cfg.model.action_horizon):
        raise ValueError("policy_metadata model_action_horizon does not match the model config")

    params = checkpoint / "params"
    if not params.is_dir() or not any(params.iterdir()):
        raise FileNotFoundError(f"Missing or empty checkpoint params directory: {params}")

    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    if not data_cfg.asset_id:
        raise ValueError(f"Config {config_name!r} has no normalization asset_id")
    norm_stats = checkpoint / "assets" / data_cfg.asset_id / "norm_stats.json"
    if not norm_stats.is_file():
        raise FileNotFoundError(f"Missing checkpoint normalization statistics: {norm_stats}")
    with norm_stats.open() as f:
        parsed = json.load(f)
    if not parsed:
        raise ValueError(f"Normalization statistics are empty: {norm_stats}")
    return {
        "config_name": config_name,
        "checkpoint_dir": str(checkpoint),
        "params_dir": str(params),
        "norm_stats": str(norm_stats),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(validate_checkpoint(args.config_name, args.checkpoint_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
