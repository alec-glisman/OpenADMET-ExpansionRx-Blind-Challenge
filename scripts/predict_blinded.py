#!/usr/bin/env python

"""
Generate predictions on a blinded test set using the final trained model.

Supports:
  - xgboost
  - lightgbm
  - chemprop

Assumptions:
  - Final models were trained and saved by scripts/run_final_train.py
  - Config files (base.yaml, xgboost.yaml/lightgbm.yaml/chemprop.yaml, final_train.yaml)
    are present under config/
  - For classical models, blinded CSV has the same feature columns as train_val/local_eval,
    minus the target columns.
  - For Chemprop, blinded CSV has a 'smiles' column.
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import joblib
import torch

from chemprop.features.featurization import featurize_smiles

from quality_curriculum_ml.chemprop.model import ChempropLightning


def load_configs(model_type: str):
    """Load base, model-specific, and final configs (same as in run_final_train)."""
    cfg_base = OmegaConf.load("config/base.yaml")
    cfg_final = OmegaConf.load("config/final_train.yaml")

    if model_type == "xgboost":
        cfg_model = OmegaConf.load("config/xgboost.yaml")
    elif model_type == "lightgbm":
        cfg_model = OmegaConf.load("config/lightgbm.yaml")
    elif model_type == "chemprop":
        cfg_model = OmegaConf.load("config/chemprop.yaml")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    cfg = OmegaConf.merge(cfg_base, cfg_model, cfg_final)
    return cfg


# ---------------------------------------------------------------------------
# Classical (XGBoost / LightGBM) prediction
# ---------------------------------------------------------------------------


def infer_classical(
    cfg,
    model_type: str,
    blinded_df: pd.DataFrame,
    model_path: str,
) -> pd.DataFrame:
    """Run inference using a final classical model on blinded_df."""

    # Load the trained model
    model = joblib.load(model_path)

    target_cols = list(cfg.data.target_cols)
    ignore_cols = set(target_cols) | {"quality_bucket", "sample_weight"}
    # Common case: classical features are all non-target, non-quality columns.
    # You may want to customize this if you use special feature columns.
    feature_cols = [c for c in blinded_df.columns if c not in ignore_cols]

    X_blind = blinded_df[feature_cols].values.astype(np.float32)
    y_pred = model.predict(X_blind)  # shape [N, n_tasks]

    # Build prediction DataFrame
    pred_df = pd.DataFrame(
        y_pred,
        columns=[f"{t}_pred" for t in target_cols],
        index=blinded_df.index,
    )

    # Optionally keep original identifiers / smiles / etc.
    out_df = blinded_df.copy()
    for col in pred_df.columns:
        out_df[col] = pred_df[col]

    return out_df


# ---------------------------------------------------------------------------
# Chemprop prediction
# ---------------------------------------------------------------------------


def build_chemprop_model_for_inference(cfg, state_dict_path: str):
    """Rebuild ChempropLightning with the same hyperparameters and load weights."""
    target_cols = list(cfg.data.target_cols)
    task_weights = list(cfg.training.task_weights)

    model = ChempropLightning(
        model_params=dict(cfg.model.params),
        lr=float(cfg.chemprop.lr),
        weight_decay=float(cfg.chemprop.weight_decay),
        target_dim=len(target_cols),
        task_weights=task_weights,
    )

    state_dict = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def infer_chemprop(
    cfg,
    blinded_df: pd.DataFrame,
    model_path: str,
) -> pd.DataFrame:
    """Run inference with a final ChempropLightning model on blinded_df."""
    target_cols = list(cfg.data.target_cols)

    if "smiles" not in blinded_df.columns:
        raise ValueError("Blinded CSV must contain a 'smiles' column for Chemprop.")

    smiles_list = blinded_df["smiles"].tolist()
    feats_list = featurize_smiles(smiles_list)
    X = torch.tensor(np.array(feats_list), dtype=torch.float32)

    model = build_chemprop_model_for_inference(cfg, model_path)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    preds = []
    batch_size = 256
    device = model.device

    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i : i + batch_size].to(device)
            yb = model(xb)
            preds.append(yb.cpu().numpy())

    y_pred = np.vstack(preds)  # [N, n_tasks]

    pred_df = pd.DataFrame(
        y_pred,
        columns=[f"{t}_pred" for t in target_cols],
        index=blinded_df.index,
    )

    out_df = blinded_df.copy()
    for col in pred_df.columns:
        out_df[col] = pred_df[col]

    return out_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions on a blinded test set using the final trained model."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["xgboost", "lightgbm", "chemprop"],
        required=True,
        help="Which model family to use.",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to blinded input CSV (must contain necessary feature columns or 'smiles').",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Where to write the submission CSV with predictions.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory where final models are saved (default: artifacts).",
    )
    args = parser.parse_args()

    cfg = load_configs(model_type=args.model_type)

    blinded_df = pd.read_csv(args.input_path)

    if args.model_type in ["xgboost", "lightgbm"]:
        model_file = f"{args.model_type}_final_model.pkl"
    else:
        model_file = "chemprop_final_model.ckpt"

    model_path = os.path.join(args.artifacts_dir, model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. " "Run scripts/run_final_train.py first to create the final model."
        )

    if args.model_type in ["xgboost", "lightgbm"]:
        out_df = infer_classical(cfg, args.model_type, blinded_df, model_path)
    else:
        out_df = infer_chemprop(cfg, blinded_df, model_path)

    out_df.to_csv(args.output_path, index=False)
    print(f"Saved predictions to {args.output_path}")


if __name__ == "__main__":
    main()
