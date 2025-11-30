import torch
import pandas as pd
from omegaconf import OmegaConf

from ray import tune
from ray.air import RunConfig
from ray.tune.schedulers import ASHAScheduler

from admet.model.classical.data import load_data
from admet.model.classical.tune_trainable import tune_trainable


def main():
    cfg_base = OmegaConf.load("config/base.yaml")
    cfg_model = OmegaConf.load("config/lightgbm.yaml")
    cfg = OmegaConf.merge(cfg_base, cfg_model)

    df = load_data(cfg.data.path)
    target_cols = list(cfg.data.target_cols)
    task_weights = list(cfg.training.task_weights)

    quality_weights = {
        "high": float(cfg.quality_weights.high),
        "medium": float(cfg.quality_weights.medium),
        "low": float(cfg.quality_weights.low),
    }

    search_space = {
        "model": {
            "n_estimators": tune.randint(200, 600),
            "num_leaves": tune.randint(31, 127),
            "learning_rate": tune.loguniform(1e-3, 1e-1),
            "subsample": tune.uniform(0.5, 1.0),
            "colsample_bytree": tune.uniform(0.5, 1.0),
            "reg_lambda": tune.loguniform(1e-2, 10.0),
            "reg_alpha": tune.loguniform(1e-4, 1.0),
            "objective": cfg.model.params.objective,
        },
        "quality_col": cfg.data.quality_col,
        "quality_weights": quality_weights,
        "random_state": cfg.cv.random_state,
    }

    scheduler = ASHAScheduler(
        metric="val_rmse_weighted",
        mode="min",
        max_t=cfg.training.get("max_epochs", 1),
        grace_period=1,
        reduction_factor=2,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                tune_trainable,
                df=df,
                target_cols=target_cols,
                task_weights=task_weights,
                model_type="lightgbm",
                n_splits=cfg.cv.n_splits,
                experiment_name=cfg.mlflow.experiment_name,
            ),
            resources={"cpu": 4, "gpu": 0},
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="val_rmse_weighted",
            mode="min",
            num_samples=cfg.ray.num_samples,
            scheduler=scheduler,
        ),
        run_config=RunConfig(
            name="lightgbm_quality_curriculum",
            local_dir=cfg.ray.local_dir,
        ),
    )

    results = tuner.fit()
    best = results.get_best_result(metric="val_rmse_weighted", mode="min")
    print("Best config:", best.config)
    print("Best val_rmse_weighted:", best.metrics["val_rmse_weighted"])


if __name__ == "__main__":
    main()
