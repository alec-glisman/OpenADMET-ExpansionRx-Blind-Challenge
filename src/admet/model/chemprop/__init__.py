from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__all__ = [
    # Modules
    "config",
    "curriculum",
    "ensemble",
    "ffn",
    "hpo",
    "hpo_config",
    "hpo_search_space",
    "hpo_trainable",
    "model",
    # Config classes
    "ChempropConfig",
    "DataConfig",
    "ModelConfig",
    "OptimizationConfig",
    "MlflowConfig",
    "EnsembleConfig",
    "EnsembleDataConfig",
    # HPO config classes
    "HPOConfig",
    "SearchSpaceConfig",
    "ParameterSpace",
    "ASHAConfig",
    "ResourceConfig",
    "TransferLearningConfig",
    # Model classes
    "ChempropModel",
    "ChempropHyperparams",
    # Ensemble classes
    "ChempropEnsemble",
    # HPO classes
    "ChempropHPO",
    "RayTuneReportCallback",
    # HPO functions
    "build_search_space",
    "get_default_search_space",
    "train_chemprop_trial",
]

if TYPE_CHECKING:
    from . import config, curriculum, ensemble, ffn, hpo, hpo_config, hpo_search_space, hpo_trainable, model
    from .config import (
        ChempropConfig,
        DataConfig,
        EnsembleConfig,
        EnsembleDataConfig,
        MlflowConfig,
        ModelConfig,
        OptimizationConfig,
    )
    from .ensemble import ChempropEnsemble
    from .hpo import ChempropHPO
    from .hpo_config import (
        ASHAConfig,
        HPOConfig,
        ParameterSpace,
        ResourceConfig,
        SearchSpaceConfig,
        TransferLearningConfig,
    )
    from .hpo_search_space import build_search_space, get_default_search_space
    from .hpo_trainable import RayTuneReportCallback, train_chemprop_trial
    from .model import ChempropHyperparams, ChempropModel


_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    # Modules
    "config": ("admet.model.chemprop.config", None),
    "curriculum": ("admet.model.chemprop.curriculum", None),
    "ensemble": ("admet.model.chemprop.ensemble", None),
    "ffn": ("admet.model.chemprop.ffn", None),
    "hpo": ("admet.model.chemprop.hpo", None),
    "hpo_config": ("admet.model.chemprop.hpo_config", None),
    "hpo_search_space": ("admet.model.chemprop.hpo_search_space", None),
    "hpo_trainable": ("admet.model.chemprop.hpo_trainable", None),
    "model": ("admet.model.chemprop.model", None),
    # Config classes
    "ChempropConfig": ("admet.model.chemprop.config", "ChempropConfig"),
    "DataConfig": ("admet.model.chemprop.config", "DataConfig"),
    "ModelConfig": ("admet.model.chemprop.config", "ModelConfig"),
    "OptimizationConfig": ("admet.model.chemprop.config", "OptimizationConfig"),
    "MlflowConfig": ("admet.model.chemprop.config", "MlflowConfig"),
    "EnsembleConfig": ("admet.model.chemprop.config", "EnsembleConfig"),
    "EnsembleDataConfig": ("admet.model.chemprop.config", "EnsembleDataConfig"),
    # HPO config classes
    "HPOConfig": ("admet.model.chemprop.hpo_config", "HPOConfig"),
    "SearchSpaceConfig": ("admet.model.chemprop.hpo_config", "SearchSpaceConfig"),
    "ParameterSpace": ("admet.model.chemprop.hpo_config", "ParameterSpace"),
    "ASHAConfig": ("admet.model.chemprop.hpo_config", "ASHAConfig"),
    "ResourceConfig": ("admet.model.chemprop.hpo_config", "ResourceConfig"),
    "TransferLearningConfig": ("admet.model.chemprop.hpo_config", "TransferLearningConfig"),
    # Model classes
    "ChempropModel": ("admet.model.chemprop.model", "ChempropModel"),
    "ChempropHyperparams": ("admet.model.chemprop.model", "ChempropHyperparams"),
    # Ensemble classes
    "ChempropEnsemble": ("admet.model.chemprop.ensemble", "ChempropEnsemble"),
    # HPO classes/functions
    "ChempropHPO": ("admet.model.chemprop.hpo", "ChempropHPO"),
    "RayTuneReportCallback": ("admet.model.chemprop.hpo_trainable", "RayTuneReportCallback"),
    "build_search_space": ("admet.model.chemprop.hpo_search_space", "build_search_space"),
    "get_default_search_space": ("admet.model.chemprop.hpo_search_space", "get_default_search_space"),
    "train_chemprop_trial": ("admet.model.chemprop.hpo_trainable", "train_chemprop_trial"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_path, attr_name = _LAZY_IMPORTS[name]
    module = importlib.import_module(module_path)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value
