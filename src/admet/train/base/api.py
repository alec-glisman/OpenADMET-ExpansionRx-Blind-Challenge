"""Convenience wrapper functions for training APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Type, Tuple

from admet.data.load import LoadedDataset
from admet.model.base import ModelProtocol
from admet.evaluate.metrics import AllMetrics
from .model_trainer import RunSummary

from .model_trainer import BaseModelTrainer
from .ray_trainer import BaseEnsembleTrainer


def train_model(
    dataset: LoadedDataset,
    trainer_cls: Type[BaseModelTrainer],
    model_cls: Type[ModelProtocol],
    model_params: Optional[Dict[str, object]] = None,
    early_stopping_rounds: Optional[int] = None,
    sample_weight_mapping: Optional[Dict[str, float]] = None,
    output_dir: Optional[Path] = None,
    seed: Optional[int] = None,
) -> Tuple[AllMetrics, Optional[RunSummary]]:
    trainer = trainer_cls(model_cls=model_cls, model_params=model_params, seed=seed)
    return trainer.fit(
        dataset,
        sample_weight_mapping=sample_weight_mapping,
        early_stopping_rounds=early_stopping_rounds,
        output_dir=output_dir,
    )


def train_ensemble(
    root: Path,
    *,
    ensemble_trainer_cls: Type[BaseEnsembleTrainer],
    trainer_cls: Type[BaseModelTrainer],
    model_cls: Type[ModelProtocol],
    model_params: Optional[Dict[str, object]] = None,
    early_stopping_rounds: Optional[int] = None,
    sample_weight_mapping: Optional[Dict[str, float]] = None,
    output_root: Optional[Path] = None,
    seed: Optional[int] = None,
    n_fingerprint_bits: Optional[int] = None,
    num_cpus: Optional[int] = None,
    ray_address: Optional[str] = None,
    dry_run: bool = False,
    max_duration_seconds: Optional[float] = None,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment_name: Optional[str] = None,
    mlflow_parent_run_id: Optional[str] = None,
    mlflow_params: Optional[Dict[str, object]] = None,
    mlflow_cli_params: Optional[Dict[str, object]] = None,
    worker_thread_limit: Optional[int] = None,
) -> Dict[str, Dict[str, object]]:
    ray_trainer = ensemble_trainer_cls(
        trainer_cls=trainer_cls,
        trainer_kwargs={"model_cls": model_cls, "model_params": model_params, "seed": seed},
    )
    return ray_trainer.fit_ensemble(
        root,
        output_root=output_root,
        early_stopping_rounds=early_stopping_rounds,
        sample_weight_mapping=sample_weight_mapping,
        num_cpus=num_cpus,
        ray_address=ray_address,
        dry_run=dry_run,
        max_duration_seconds=max_duration_seconds,
        n_fingerprint_bits=n_fingerprint_bits,
        seed=seed,
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment_name,
        parent_run_id=mlflow_parent_run_id,
        base_params=mlflow_params,
        cli_params=mlflow_cli_params,
        worker_thread_limit=worker_thread_limit,
    )


__all__ = ["train_model", "train_ensemble"]
