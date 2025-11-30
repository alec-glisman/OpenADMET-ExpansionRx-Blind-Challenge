from typing import Dict, List

import numpy as np
import mlflow
from mlflow import log_metric, log_params
from ray import tune

from .data import augment_quality, make_folds, get_xyw
from .models import build_model, fit_model, predict_model
from .metrics import build_metric_dict


def tune_trainable(
    config: Dict,
    df,
    target_cols: List[str],
    task_weights: List[float],
    model_type: str,
    n_splits: int,
    experiment_name: str,
):
    """Ray Tune trainable with MLflow nested runs + cross-fold CV for classical models."""

    df = augment_quality(
        df,
        quality_col=config.get("quality_col", "quality"),
        quality_weights=config["quality_weights"],
    )

    folds = make_folds(
        df,
        n_splits=n_splits,
        random_state=config.get("random_state", 42),
        shuffle=True,
    )

    mlflow.set_experiment(experiment_name)
    trial_id = tune.get_trial_id()

    with mlflow.start_run(run_name=f"{model_type}_trial_{trial_id}") as parent_run:
        # log hyperparameters once for the trial
        flat_params = {f"model__{k}": v for k, v in config["model"].items()}
        log_params(flat_params)
        log_params({"quality_weights": config["quality_weights"]})

        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            with mlflow.start_run(
                run_name=f"{model_type}_trial_{trial_id}_fold_{fold_idx}",
                nested=True,
            ):
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]

                X_train, y_train, w_train = get_xyw(train_df, target_cols)
                X_val, y_val, _ = get_xyw(val_df, target_cols)

                model = build_model(model_type=model_type, params=config["model"])
                model = fit_model(model, X_train, y_train, sample_weight=w_train)

                y_pred = predict_model(model, X_val)
                metrics = build_metric_dict(y_val, y_pred, task_weights, prefix="val")

                for k, v in metrics.items():
                    log_metric(k, v)

                fold_score = metrics["val_rmse_weighted"]
                fold_scores.append(fold_score)
                log_metric("fold_val_rmse_weighted", fold_score)

        mean_score = float(np.mean(fold_scores))
        log_metric("cv_val_rmse_weighted_mean", mean_score)

        tune.report(val_rmse_weighted=mean_score)
