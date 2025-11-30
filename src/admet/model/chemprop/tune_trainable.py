from typing import Dict, List

import numpy as np
import mlflow
from mlflow import log_metric, log_params
from ray import tune
import torch
import pytorch_lightning as pl

from .data import augment_quality, make_folds, MolDataset, collate_mol_batch
from .model import ChempropLightning
from .curriculum import CurriculumState, CurriculumCallback
from torch.utils.data import DataLoader


def tune_trainable_chemprop(
    config: Dict,
    df,
    target_cols: List[str],
    task_weights: List[float],
    n_splits: int,
    experiment_name: str,
):
    """Ray Tune trainable for Chemprop + Lightning with CV + MLflow nested runs."""

    df = augment_quality(
        df,
        quality_col=config.get("quality_col", "quality"),
        quality_weights=config["quality_weights"],
    )
    folds = make_folds(df, n_splits=n_splits, random_state=config.get("random_state", 42))

    mlflow.set_experiment(experiment_name)
    trial_id = tune.get_trial_id()

    with mlflow.start_run(run_name=f"chemprop_trial_{trial_id}") as parent_run:
        # log hyperparams
        flat_params = {f"model__{k}": v for k, v in config["model"].items()}
        flat_params["lr"] = config["lr"]
        flat_params["weight_decay"] = config["weight_decay"]
        flat_params["batch_size"] = config["batch_size"]
        flat_params["curr_patience"] = config["curr_patience"]
        log_params(flat_params)
        log_params({"quality_weights": config["quality_weights"]})

        fold_scores = []

        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            with mlflow.start_run(
                run_name=f"chemprop_trial_{trial_id}_fold_{fold_idx}",
                nested=True,
            ):
                train_df = df.iloc[tr_idx]
                val_df = df.iloc[val_idx]

                train_ds = MolDataset(train_df, target_cols)
                val_ds = MolDataset(val_df, target_cols)

                train_loader = DataLoader(
                    train_ds,
                    batch_size=config["batch_size"],
                    shuffle=True,
                    collate_fn=collate_mol_batch,
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=config["batch_size"],
                    shuffle=False,
                    collate_fn=collate_mol_batch,
                )

                curr_state = CurriculumState(patience=config["curr_patience"])
                model = ChempropLightning(
                    model_params=config["model"],
                    lr=config["lr"],
                    weight_decay=config["weight_decay"],
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
                    max_epochs=config["max_epochs"],
                    callbacks=[cb_curr, es],
                    enable_checkpointing=False,
                    enable_model_summary=False,
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices=1,
                    logger=False,
                )

                trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
                metrics = trainer.callback_metrics
                fold_loss = float(metrics["val_combined"].item())
                fold_scores.append(fold_loss)

                log_metric("fold_val_combined", fold_loss)
                log_metric("fold_val_high_loss", float(metrics["val_high_loss"].item()))
                log_metric("fold_val_medium_loss", float(metrics["val_medium_loss"].item()))
                log_metric("fold_val_low_loss", float(metrics["val_low_loss"].item()))

        mean_loss = float(np.mean(fold_scores))
        log_metric("cv_val_combined_mean", mean_loss)
        tune.report(val_combined=mean_loss)
