#!/usr/bin/env python

import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import mlflow
from mlflow import log_metric, log_params, set_tags

from quality_curriculum_ml.classical.data import (
    augment_quality as augment_quality_classical,
    get_xyw,
)
from quality_curriculum_ml.classical.models import (
    build_model,
    fit_model,
    predict_model,
)
from quality_curriculum_ml.classical.metrics import (
    build_metric_dict as build_metric_dict_classical,
)

from quality_curriculum_ml.chemprop.data import (
    augment_quality as augment_quality_chem,
    MolDataset,
    collate_mol_batch,
)
from quality_curriculum_ml.chemprop.model import ChempropLightning
from quality_curriculum_ml.chemprop.curriculum import CurriculumState, CurriculumCallback

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split


def load_configs(model_type: str):
    """Load base, model-specific, and final training configs and merge."""
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

    # Merge in order: base <- model <- final
    cfg = OmegaConf.merge(cfg_base, cfg_model, cfg_final)
    return cfg


# ------------------------ CLASSICAL FINAL TRAINING ------------------------ #


def final_train_classical(cfg):
    """
    Final training for XGBoost / LightGBM:

    Train on:
      - train_val
      - plus local_eval (if include_local_eval_in_final_train is True)

    Evaluate on:
      - local_eval (metrics logged as 'local_*' to MLflow)
    """
    df_train_val = pd.read_csv(cfg.final.train_val_path)
    df_local = pd.read_csv(cfg.final.local_eval_path)

    # augment with quality buckets and weights
    quality_weights = {
        "high": float(cfg.quality_weights.high),
        "medium": float(cfg.quality_weights.medium),
        "low": float(cfg.quality_weights.low),
    }
    df_train_val = augment_quality_classical(
        df_train_val,
        quality_col=cfg.data.quality_col,
        quality_weights=quality_weights,
    )
    df_local = augment_quality_classical(
        df_local,
        quality_col=cfg.data.quality_col,
        quality_weights=quality_weights,
    )

    target_cols = list(cfg.data.target_cols)
    task_weights = list(cfg.training.task_weights)

    # Decide what goes into the final training set
    if cfg.final.include_local_eval_in_final_train:
        df_final_train = pd.concat([df_train_val, df_local], axis=0).reset_index(drop=True)
    else:
        df_final_train = df_train_val

    X_train, y_train, w_train = get_xyw(df_final_train, target_cols)
    X_local, y_local, _ = get_xyw(df_local, target_cols)

    # Build and fit final model
    model_type = cfg.model.type  # xgboost or lightgbm
    params = dict(cfg.model.params)
    model = build_model(model_type=model_type, params=params)

    if cfg.final.use_sample_weight:
        model = fit_model(model, X_train, y_train, sample_weight=w_train)
    else:
        model = fit_model(model, X_train, y_train, sample_weight=None)

    # Evaluate on local_eval
    y_pred_local = predict_model(model, X_local)
    local_metrics = build_metric_dict_classical(
        y_true=y_local,
        y_pred=y_pred_local,
        task_weights=task_weights,
        prefix="local",
    )
    return model, local_metrics


# ------------------------ CHEMPROP FINAL TRAINING ------------------------ #


def build_chemprop_dataloaders(df_final_train, df_local, target_cols, cfg):
    """
    Create train/val loaders from df_final_train and a loader from df_local
    for post-training evaluation.
    """
    val_fraction = float(cfg.final.val_fraction)
    n_total = len(df_final_train)
    n_val = max(1, int(val_fraction * n_total))
    n_train = n_total - n_val

    ds_full = MolDataset(df_final_train, target_cols)
    ds_train, ds_val = random_split(
        ds_full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.final.random_state),
    )
    ds_local = MolDataset(df_local, target_cols)

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.chemprop.batch_size,
        shuffle=True,
        collate_fn=collate_mol_batch,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=cfg.chemprop.batch_size,
        shuffle=False,
        collate_fn=collate_mol_batch,
    )
    local_loader = DataLoader(
        ds_local,
        batch_size=cfg.chemprop.batch_size,
        shuffle=False,
        collate_fn=collate_mol_batch,
    )

    return train_loader, val_loader, local_loader


def evaluate_chemprop_on_loader(trainer, model, loader):
    """
    Run evaluation on a loader and return metrics dict.
    We reuse the validation loop (validation_step/validation_epoch_end).
    """
    results = trainer.validate(model, dataloaders=loader, verbose=False)
    return results[0] if results else {}


def final_train_chemprop(cfg):
    """
    Final training for ChempropLightning:

    Train on:
      - train_val
      - plus local_eval (if include_local_eval_in_final_train is True)

    Evaluate on:
      - local_eval (metrics logged as 'local_*' to MLflow)
    """
    df_train_val = pd.read_csv(cfg.final.train_val_path)
    df_local = pd.read_csv(cfg.final.local_eval_path)

    quality_weights = {
        "high": float(cfg.quality_weights.high),
        "medium": float(cfg.quality_weights.medium),
        "low": float(cfg.quality_weights.low),
    }
    df_train_val = augment_quality_chem(
        df_train_val,
        quality_col=cfg.data.quality_col,
        quality_weights=quality_weights,
    )
    df_local = augment_quality_chem(
        df_local,
        quality_col=cfg.data.quality_col,
        quality_weights=quality_weights,
    )

    target_cols = list(cfg.data.target_cols)
    task_weights = list(cfg.training.task_weights)

    # Decide what goes into final training set
    if cfg.final.include_local_eval_in_final_train:
        df_final_train = pd.concat([df_train_val, df_local], axis=0).reset_index(drop=True)
    else:
        df_final_train = df_train_val

    train_loader, val_loader, local_loader = build_chemprop_dataloaders(df_final_train, df_local, target_cols, cfg)

    curr_state = CurriculumState(patience=cfg.chemprop.curr_patience)
    model = ChempropLightning(
        model_params=dict(cfg.model.params),
        lr=float(cfg.chemprop.lr),
        weight_decay=float(cfg.chemprop.weight_decay),
        target_dim=len(target_cols),
        task_weights=task_weights,
    )

    cb_curr = CurriculumCallback(curr_state)
    es = pl.callbacks.EarlyStopping(
        monitor="val_combined",
        mode="min",
        patience=6,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.final.max_epochs,
        callbacks=[cb_curr, es],
        enable_checkpointing=False,
        enable_model_summary=False,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    local_metrics = evaluate_chemprop_on_loader(trainer, model, local_loader)
    local_metrics_renamed = {}
    for k, v in local_metrics.items():
        if k.startswith("val_"):
            local_metrics_renamed["local_" + k[4:]] = float(v)
        else:
            local_metrics_renamed[f"local_{k}"] = float(v)
    return model, local_metrics_renamed


# ------------------------ MAIN ENTRYPOINT ------------------------ #


def main():
    parser = argparse.ArgumentParser(
        description="Final training on train_val (+ optional local_eval) and evaluation on local_eval."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["xgboost", "lightgbm", "chemprop"],
        help="Override model type from config.final.model_type",
    )
    args = parser.parse_args()

    tmp_cfg_final = OmegaConf.load("config/final_train.yaml")
    model_type = args.model_type or tmp_cfg_final.final.model_type

    cfg = load_configs(model_type=model_type)

    mlflow.set_experiment(cfg.mlflow.experiment_name)
    run_name = f"{cfg.final.mlflow_run_name_prefix}_{model_type}"

    with mlflow.start_run(run_name=run_name):
        flat = OmegaConf.to_container(cfg, resolve=True)

        def flatten_dict(d, parent_key="", sep="."):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_params = flatten_dict(flat)
        log_params(flat_params)

        tags = cfg.final.get("tags", {})
        if tags:
            set_tags(tags)

        if model_type in ["xgboost", "lightgbm"]:
            model, local_metrics = final_train_classical(cfg)
        elif model_type == "chemprop":
            model, local_metrics = final_train_chemprop(cfg)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        for k, v in local_metrics.items():
            log_metric(k, float(v))

        import os
        import joblib

        artifacts_dir = "artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)

        if model_type in ["xgboost", "lightgbm"]:
            model_path = os.path.join(artifacts_dir, f"{model_type}_final_model.pkl")
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)
        else:
            model_path = os.path.join(artifacts_dir, "chemprop_final_model.ckpt")
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(model_path)

        print("Final local_eval metrics (not blinded-test metrics):")
        for k, v in local_metrics.items():
            print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
