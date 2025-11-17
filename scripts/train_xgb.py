#!/usr/bin/env python3

"""
Simple runner script to train XGBoost per-endpoint models using the existing
training orchestration and YAML configuration.

Usage:
  python scripts/train_xgb.py \
      --data-root /path/to/splits \
      --config configs/xgb.yaml \
      --out-dir runs/xgb_001

This is a thin wrapper over `admet.train.xgb_train.train_xgb_models`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from admet.data.load import load_dataset
from admet.train.xgb_train import train_xgb_models


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost per-endpoint models")
    p.add_argument("--data-root", type=Path, required=True, help="Dir with train.csv/val.csv/test.csv")
    p.add_argument("--config", type=Path, required=True, help="YAML config with models.xgboost")
    p.add_argument("--out-dir", type=Path, default=Path("xgb_artifacts"), help="Output artifacts dir")
    p.add_argument("--seed", type=int, default=None, help="Global random seed")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.exists():
        print(f"Config file not found: {args.config}")
        return 2

    cfg = yaml.safe_load(args.config.read_text()) or {}
    if not cfg:
        print(f"Failed to load config from: {args.config}")
        return 3

    endpoints = cfg.get("data", {}).get("endpoints")
    xgb_cfg = cfg.get("models", {}).get("xgboost", {})
    model_params = xgb_cfg.get("model_params")
    early_stopping_rounds = xgb_cfg.get("early_stopping_rounds", 50)

    sw_cfg = cfg.get("training", {}).get("sample_weights", {})
    sw_enabled = sw_cfg.get("enabled", False)
    sw_mapping = sw_cfg.get("weights") if sw_enabled else None

    dataset = load_dataset(args.data_root, endpoints=endpoints)
    metrics = train_xgb_models(
        dataset,
        model_params=model_params,
        early_stopping_rounds=early_stopping_rounds,
        sample_weight_mapping=sw_mapping,
        output_dir=args.out_dir,
        seed=args.seed,
    )

    # Print a compact summary to stdout
    print(json.dumps({"val_macro": metrics["val"]["macro"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
