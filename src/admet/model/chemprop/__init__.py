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
    "inter_task_affinity",
    "model",
    "task_affinity",
    # Config classes
    "ChempropConfig",
    "DataConfig",
    "ModelConfig",
    "OptimizationConfig",
    "MlflowConfig",
    "EnsembleConfig",
    "EnsembleDataConfig",
    "InterTaskAffinityConfig",
    # HPO config classes
    "HPOConfig",
    "SearchSpaceConfig",
    "ParameterSpace",
    "ASHAConfig",
    "ResourceConfig",
    "TransferLearningConfig",
    # Model classes
    "ChempropModel",
    "ChempropModelAdapter",
    "ChempropHyperparams",
    # Ensemble classes
    "ModelEnsemble",
    "ChempropEnsemble",  # Backward compatibility alias
    # HPO classes
    "ChempropHPO",
    "RayTuneReportCallback",
    # HPO functions
    "build_search_space",
    "get_default_search_space",
    "train_chemprop_trial",
    # Task affinity classes
    "TaskAffinityConfig",
    "TaskAffinityComputer",
    "TaskGrouper",
    "compute_task_affinity",
    # Inter-task affinity classes (paper-accurate)
    "InterTaskAffinityCallback",
    "InterTaskAffinityComputer",
]

if TYPE_CHECKING:
    from . import (
        config,
        curriculum,
        ensemble,
        ffn,
        hpo,
        hpo_config,
        hpo_search_space,
        hpo_trainable,
        inter_task_affinity,
        model,
        task_affinity,
    )
    from .adapter import ChempropModelAdapter
    from .config import (
        ChempropConfig,
        DataConfig,
        EnsembleConfig,
        EnsembleDataConfig,
        InterTaskAffinityConfig,
        MlflowConfig,
        ModelConfig,
        OptimizationConfig,
    )
    from .ensemble import ModelEnsemble, ModelEnsemble as ChempropEnsemble  # Backward compatibility
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
    from .inter_task_affinity import InterTaskAffinityCallback, InterTaskAffinityComputer
    from .model import ChempropHyperparams, ChempropModel
    from .task_affinity import TaskAffinityComputer, TaskAffinityConfig, TaskGrouper, compute_task_affinity


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
    "inter_task_affinity": ("admet.model.chemprop.inter_task_affinity", None),
    "model": ("admet.model.chemprop.model", None),
    "task_affinity": ("admet.model.chemprop.task_affinity", None),
    # Config classes
    "ChempropConfig": ("admet.model.chemprop.config", "ChempropConfig"),
    "DataConfig": ("admet.model.chemprop.config", "DataConfig"),
    "ModelConfig": ("admet.model.chemprop.config", "ModelConfig"),
    "OptimizationConfig": ("admet.model.chemprop.config", "OptimizationConfig"),
    "MlflowConfig": ("admet.model.chemprop.config", "MlflowConfig"),
    "EnsembleConfig": ("admet.model.chemprop.config", "EnsembleConfig"),
    "EnsembleDataConfig": ("admet.model.chemprop.config", "EnsembleDataConfig"),
    "InterTaskAffinityConfig": ("admet.model.chemprop.config", "InterTaskAffinityConfig"),
    # HPO config classes
    "HPOConfig": ("admet.model.chemprop.hpo_config", "HPOConfig"),
    "SearchSpaceConfig": ("admet.model.chemprop.hpo_config", "SearchSpaceConfig"),
    "ParameterSpace": ("admet.model.chemprop.hpo_config", "ParameterSpace"),
    "ASHAConfig": ("admet.model.chemprop.hpo_config", "ASHAConfig"),
    "ResourceConfig": ("admet.model.chemprop.hpo_config", "ResourceConfig"),
    "TransferLearningConfig": ("admet.model.chemprop.hpo_config", "TransferLearningConfig"),
    # Model classes
    "ChempropModel": ("admet.model.chemprop.model", "ChempropModel"),
    "ChempropModelAdapter": ("admet.model.chemprop.adapter", "ChempropModelAdapter"),
    "ChempropHyperparams": ("admet.model.chemprop.model", "ChempropHyperparams"),
    # Ensemble classes
    "ModelEnsemble": ("admet.model.chemprop.ensemble", "ModelEnsemble"),
    "ChempropEnsemble": ("admet.model.chemprop.ensemble", "ModelEnsemble"),  # Backward compatibility
    # HPO classes/functions
    "ChempropHPO": ("admet.model.chemprop.hpo", "ChempropHPO"),
    "RayTuneReportCallback": ("admet.model.chemprop.hpo_trainable", "RayTuneReportCallback"),
    "build_search_space": ("admet.model.chemprop.hpo_search_space", "build_search_space"),
    "get_default_search_space": ("admet.model.chemprop.hpo_search_space", "get_default_search_space"),
    "train_chemprop_trial": ("admet.model.chemprop.hpo_trainable", "train_chemprop_trial"),
    # Task affinity classes (legacy pre-training approach)
    "TaskAffinityConfig": ("admet.model.chemprop.task_affinity", "TaskAffinityConfig"),
    "TaskAffinityComputer": ("admet.model.chemprop.task_affinity", "TaskAffinityComputer"),
    "TaskGrouper": ("admet.model.chemprop.task_affinity", "TaskGrouper"),
    "compute_task_affinity": ("admet.model.chemprop.task_affinity", "compute_task_affinity"),
    # Inter-task affinity classes (paper-accurate during-training approach)
    "InterTaskAffinityCallback": ("admet.model.chemprop.inter_task_affinity", "InterTaskAffinityCallback"),
    "InterTaskAffinityComputer": ("admet.model.chemprop.inter_task_affinity", "InterTaskAffinityComputer"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_path, attr_name = _LAZY_IMPORTS[name]
    module = importlib.import_module(module_path)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value
