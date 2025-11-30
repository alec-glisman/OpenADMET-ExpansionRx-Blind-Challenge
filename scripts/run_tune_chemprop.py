import torch
from omegaconf import OmegaConf

from ray import tune
from ray.air import RunConfig
from ray.tune.schedulers import ASHAScheduler

from quality_curriculum_ml.classical.data import load_data
from quality_curriculum_ml.chemprop.tune_trainable import tune_trainable_chemprop


def main():
    cfg_base = OmegaConf.load("config/base.yaml")
    cfg_model = OmegaConf.load("config/chemprop.yaml")
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
            "atom_fdim": cfg.model.params.atom_fdim,
            "bond_fdim": cfg.model.params.bond_fdim,
            "depth": tune.randint(3, 7),
            "hidden_size": tune.choice([300, 600, 900]),
            "dropout": tune.uniform(0.0, 0.4),
        },
        "lr": tune.loguniform(1e-4, 3e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "batch_size": tune.choice([32, 64, 128]),
        "curr_patience": tune.randint(2, 6),
        "max_epochs": cfg.chemprop.max_epochs,
        "quality_col": cfg.data.quality_col,
        "quality_weights": quality_weights,
        "random_state": cfg.cv.random_state,
    }

    scheduler = ASHAScheduler(
        metric="val_combined",
        mode="min",
        max_t=cfg.chemprop.max_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                tune_trainable_chemprop,
                df=df,
                target_cols=target_cols,
                task_weights=task_weights,
                n_splits=cfg.cv.n_splits,
                experiment_name=cfg.mlflow.experiment_name,
            ),
            resources={"cpu": 4, "gpu": 1 if torch.cuda.is_available() else 0},
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="val_combined",
            mode="min",
            num_samples=cfg.ray.num_samples,
            scheduler=scheduler,
        ),
        run_config=RunConfig(
            name="chemprop_quality_curriculum",
            local_dir=cfg.ray.local_dir,
        ),
    )

    results = tuner.fit()
    best = results.get_best_result(metric="val_combined", mode="min")
    print("Best config:", best.config)
    print("Best val_combined:", best.metrics["val_combined"])


if __name__ == "__main__":
    main()
