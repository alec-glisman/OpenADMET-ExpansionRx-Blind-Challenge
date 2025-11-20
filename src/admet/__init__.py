"""
ADMET Prediction Package
========================

This package provides tools for training and evaluating machine learning models
for ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) property prediction.

.. module:: admet

"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

# Mapping of attribute name -> (module_path, attribute_name)
_LAZY_MAP: Dict[str, tuple[str, str]] = {
    # chem helpers (implemented under admet.data.chem)
    "canonicalize_smiles": ("admet.data.chem", "canonicalize_smiles"),
    "parallel_canonicalize_smiles": ("admet.data.chem", "parallel_canonicalize_smiles"),
    "compute_molecular_properties": ("admet.data.chem", "compute_molecular_properties"),
    # dataset/constants (under admet.data.constants)
    "COLS_WITH_UNITS": ("admet.data.constants", "COLS_WITH_UNITS"),
    "TRANSFORMATIONS": ("admet.data.constants", "TRANSFORMATIONS"),
    # data loading
    "LoadedDataset": ("admet.data.load", "LoadedDataset"),
    "load_dataset": ("admet.data.load", "load_dataset"),
    "load_blinded_dataset": ("admet.data.load", "load_blinded_dataset"),
    # plotting utilities (under admet.visualize.plots)
    "calc_stats": ("admet.visualize.plots", "calc_stats"),
    "plot_numeric_distributions": ("admet.visualize.plots", "plot_numeric_distributions"),
    "plot_correlation_matrix": ("admet.visualize.plots", "plot_correlation_matrix"),
    "plot_property_distributions": ("admet.visualize.plots", "plot_property_distributions"),
    # training
    "BaseModelTrainer": ("admet.train.base_trainer", "BaseModelTrainer"),
    "XGBoostTrainer": ("admet.train.xgb_train", "XGBoostTrainer"),
    "train_xgb_models": ("admet.train.xgb_train", "train_xgb_models"),
    "train_xgb_models_ray": ("admet.train.xgb_train", "train_xgb_models_ray"),
    # models
    "BaseModel": ("admet.model.base", "BaseModel"),
    "XGBoostMultiEndpoint": ("admet.model.xgb_wrapper", "XGBoostMultiEndpoint"),
    # evaluate
    "compute_metrics_log_and_linear": ("admet.evaluate.metrics", "compute_metrics_log_and_linear"),
    "AllMetrics": ("admet.evaluate.metrics", "AllMetrics"),
}


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    """Lazily import attributes defined in ``_LAZY_MAP``.

    On first access the target submodule is imported and the attribute is
    cached in this module's globals so subsequent access is fast and does not
    re-import.
    """
    if name in _LAZY_MAP:
        module_name, attr_name = _LAZY_MAP[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'admet' has no attribute '{name}'")


def __dir__() -> List[str]:
    """Return module attributes plus lazily exportable names."""
    return sorted(list(globals().keys()) + list(_LAZY_MAP.keys()))


__all__ = list(_LAZY_MAP.keys())
