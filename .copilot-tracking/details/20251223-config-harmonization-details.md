<!-- markdownlint-disable-file -->

# Task Details: Configuration Harmonization for Multi-Model Support

## Research Reference

**Source Research**: #file:../research/config-harmonization-plan.md

---

# PART 1: Phases 1-3

---

## Phase 1: Unified Base Configs

### Task 1.1: Create UnifiedDataConfig dataclass

Create a universal data configuration that consolidates all data-related settings.

- **Files**:
  - `src/admet/model/config.py` - Add UnifiedDataConfig class after BaseDataConfig

- **Implementation**:

```python
@dataclass
class UnifiedDataConfig:
    """Universal data configuration for all model types.

    Consolidates data paths, columns, and ensemble settings into a single
    config that works across ChemProp, classical models, and Chemeleon.
    """
    # Core paths
    data_dir: str = MISSING
    train_file: str | None = None
    validation_file: str | None = None
    test_file: str | None = None
    blind_file: str | None = None
    output_dir: str | None = None

    # Column configuration
    smiles_col: str = "SMILES"
    target_cols: list[str] = field(default_factory=list)
    target_weights: list[float] = field(default_factory=list)
    quality_col: str | None = None  # For curriculum learning

    # Ensemble settings
    splits: list[int] | None = None
    folds: list[int] | None = None
```

- **Success**:
  - Class passes OmegaConf structured validation
  - Can be instantiated with YAML using `OmegaConf.structured(UnifiedDataConfig)`
  - Includes all fields from both BaseDataConfig and EnsembleDataConfig

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 95-125) - Tier 1 Core Configs

- **Dependencies**:
  - None (first task)

---

### Task 1.2: Create UnifiedMlflowConfig dataclass

Create unified MLflow tracking configuration that works for all models.

- **Files**:
  - `src/admet/model/config.py` - Add UnifiedMlflowConfig after UnifiedDataConfig

- **Implementation**:

```python
@dataclass
class UnifiedMlflowConfig:
    """Universal MLflow tracking configuration.

    Consolidates tracking settings with consistent field names.
    """
    enabled: bool = True
    tracking_uri: str | None = None
    experiment_name: str = "admet"
    run_name: str | None = None
    run_id: str | None = None
    parent_run_id: str | None = None
    nested: bool = False
    log_model: bool = True
    log_plots: bool = True
    log_metrics: bool = True
```

- **Success**:
  - Config works with existing MLflow mixin
  - Field names consistent (uses `enabled` not `tracking`)
  - Can be converted to/from ChemProp's MlflowConfig

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 95-125) - Tier 1 Core Configs

- **Dependencies**:
  - Task 1.1 complete

---

### Task 1.3: Create UnifiedOptimizationConfig dataclass

Create optimization config that works for both neural and tree-based models.

- **Files**:
  - `src/admet/model/config.py` - Add UnifiedOptimizationConfig

- **Implementation**:

```python
@dataclass
class UnifiedOptimizationConfig:
    """Universal optimization configuration.

    Contains superset of parameters for all model types. Neural models
    use learning rate scheduling; tree-based models ignore those fields.
    """
    # Common to all models
    seed: int = 42
    progress_bar: bool = False

    # Neural network training
    max_epochs: int = 150
    patience: int = 15
    batch_size: int = 32
    num_workers: int = 0

    # Learning rate (neural models)
    learning_rate: float = 0.001
    init_lr: float | None = None
    max_lr: float | None = None
    final_lr: float | None = None
    warmup_epochs: int = 5

    # Scheduler
    scheduler: str = "onecycle"  # "onecycle", "cosine", "constant", "step"

    # Loss function
    criterion: str = "MAE"

    # Regularization
    weight_decay: float = 0.0
    gradient_clip: float | None = None
```

- **Success**:
  - Contains all fields from ChemProp's OptimizationConfig
  - Tree-based models can ignore neural-specific fields
  - Default values match current ChemProp defaults

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 95-125) - Tier 1 Core Configs
  - [src/admet/model/chemprop/config.py](src/admet/model/chemprop/config.py) (Lines 110-155) - Current OptimizationConfig

- **Dependencies**:
  - Task 1.2 complete

---

### Task 1.4: Add config conversion helpers to ChempropConfig

Add methods to convert between legacy ChemProp configs and unified configs.

- **Files**:
  - `src/admet/model/chemprop/config.py` - Add conversion methods to ChempropConfig

- **Implementation**:

```python
# Add to ChempropConfig class

@classmethod
def from_unified(cls, unified: "UnifiedModelConfig") -> "ChempropConfig":
    """Convert unified config to ChempropConfig.

    Parameters
    ----------
    unified : UnifiedModelConfig
        Unified configuration object.

    Returns
    -------
    ChempropConfig
        Legacy ChempropConfig for backward compatibility.
    """
    return cls(
        data=DataConfig(
            data_dir=unified.data.data_dir,
            test_file=unified.data.test_file,
            blind_file=unified.data.blind_file,
            smiles_col=unified.data.smiles_col,
            target_cols=list(unified.data.target_cols),
            target_weights=list(unified.data.target_weights),
            output_dir=unified.data.output_dir,
        ),
        model=ModelConfig(
            depth=unified.chemprop.depth,
            message_hidden_dim=unified.chemprop.message_hidden_dim,
            # ... map all fields
        ),
        optimization=OptimizationConfig(
            criterion=unified.optimization.criterion,
            max_epochs=unified.optimization.max_epochs,
            # ... map all fields
        ),
        mlflow=MlflowConfig(
            tracking=unified.mlflow.enabled,
            # ... map all fields
        ),
        joint_sampling=JointSamplingConfig(
            enabled=unified.sampling.enabled,
            # ... map sampling and curriculum
        ),
    )

def to_unified(self) -> "UnifiedModelConfig":
    """Convert this ChempropConfig to unified format.

    Returns
    -------
    UnifiedModelConfig
        Unified configuration object.
    """
    from admet.model.config import UnifiedModelConfig
    # ... reverse mapping
```

- **Success**:
  - Roundtrip conversion: `config == ChempropConfig.from_unified(config.to_unified())`
  - All fields preserved in conversion
  - No data loss

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 280-310) - Migration Strategy

- **Dependencies**:
  - Tasks 1.1-1.3 complete

---

### Task 1.5: Create unit tests for base config conversions

Add comprehensive tests for config conversion functions.

- **Files**:
  - `tests/test_config_unified.py` - New test file

- **Implementation**:

```python
"""Tests for unified configuration system."""
import pytest
from omegaconf import OmegaConf

from admet.model.config import (
    UnifiedDataConfig,
    UnifiedMlflowConfig,
    UnifiedOptimizationConfig,
    UnifiedModelConfig,
)
from admet.model.chemprop.config import ChempropConfig


class TestUnifiedDataConfig:
    def test_default_values(self):
        config = UnifiedDataConfig()
        assert config.smiles_col == "SMILES"
        assert config.target_cols == []

    def test_omegaconf_structured(self):
        config = OmegaConf.structured(UnifiedDataConfig)
        assert OmegaConf.is_missing(config, "data_dir")

    def test_yaml_roundtrip(self):
        config = UnifiedDataConfig(
            data_dir="test/data",
            target_cols=["LogD", "KSOL"],
        )
        yaml_str = OmegaConf.to_yaml(OmegaConf.structured(config))
        loaded = OmegaConf.create(yaml_str)
        assert loaded.data_dir == "test/data"


class TestChempropConversion:
    def test_roundtrip_conversion(self, sample_chemprop_config):
        unified = sample_chemprop_config.to_unified()
        restored = ChempropConfig.from_unified(unified)
        # Compare key fields
        assert restored.data.smiles_col == sample_chemprop_config.data.smiles_col
        assert restored.model.depth == sample_chemprop_config.model.depth
```

- **Success**:
  - All tests pass
  - Coverage >95% for config conversion code
  - Edge cases tested (missing fields, None values)

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 320-340) - Testing Strategy

- **Dependencies**:
  - Task 1.4 complete

---

## Phase 2: Model-Agnostic Training Strategies

### Task 2.1: Create training module structure

Set up the new training module for model-agnostic training strategies.

- **Files**:
  - `src/admet/training/__init__.py` - New module init
  - `src/admet/training/protocols.py` - Protocol definitions
  - `src/admet/training/sampling.py` - Sampling utilities
  - `src/admet/training/curriculum.py` - Curriculum scheduler
  - `src/admet/training/task_affinity.py` - Task affinity computation

- **Implementation**:

```python
# src/admet/training/__init__.py
"""Model-agnostic training strategies.

This module provides training utilities that work across all model types:
- Sampling: Task-aware and curriculum-aware sample weighting
- Curriculum: Quality-based curriculum learning scheduler
- Task Affinity: Gradient-based task grouping
"""

from admet.training.protocols import CurriculumAwareModel, SamplingAwareModel
from admet.training.sampling import compute_sample_weights, SamplingConfig
from admet.training.curriculum import CurriculumScheduler, CurriculumConfig
from admet.training.task_affinity import TaskAffinityComputer, TaskAffinityConfig

__all__ = [
    "CurriculumAwareModel",
    "SamplingAwareModel",
    "compute_sample_weights",
    "SamplingConfig",
    "CurriculumScheduler",
    "CurriculumConfig",
    "TaskAffinityComputer",
    "TaskAffinityConfig",
]
```

- **Success**:
  - Module structure created
  - All imports work correctly
  - No circular import issues

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 155-190) - Phase 2 design

- **Dependencies**:
  - Phase 1 complete

---

### Task 2.2: Extract SamplingConfig and implement model-agnostic sampler

Create model-agnostic sampling configuration and weight computation.

- **Files**:
  - `src/admet/training/sampling.py` - New file

- **Implementation**:

```python
"""Model-agnostic sampling strategies."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class SamplingConfig:
    """Configuration for training data sampling.

    Works with any model that supports sample weights or weighted sampling.
    """
    enabled: bool = False

    # Task-aware oversampling
    task_oversampling_alpha: float = 0.5  # 0=uniform, 1=full inverse

    # General sampling
    num_samples: int | None = None
    seed: int = 42
    increment_seed_per_epoch: bool = True
    log_to_mlflow: bool = True


def compute_task_weights(
    targets: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Compute per-sample weights based on task label availability.

    Samples with labels for rare tasks get higher weights.

    Parameters
    ----------
    targets : np.ndarray
        Target values of shape (n_samples, n_tasks). NaN indicates missing.
    alpha : float
        Power law exponent. 0=uniform, 1=full inverse proportional.

    Returns
    -------
    np.ndarray
        Per-sample weights of shape (n_samples,).
    """
    # Count non-NaN values per task
    task_counts = np.sum(~np.isnan(targets), axis=0)
    task_counts = np.maximum(task_counts, 1)  # Avoid division by zero

    # Inverse frequency weights
    task_weights = (1.0 / task_counts) ** alpha
    task_weights = task_weights / task_weights.sum()  # Normalize

    # For each sample, use weight of rarest task it has
    sample_weights = np.zeros(len(targets))
    for i, row in enumerate(targets):
        valid_tasks = ~np.isnan(row)
        if valid_tasks.any():
            sample_weights[i] = task_weights[valid_tasks].max()
        else:
            sample_weights[i] = 1.0 / len(targets)

    # Normalize to sum to n_samples
    sample_weights = sample_weights / sample_weights.sum() * len(targets)
    return sample_weights


def compute_quality_weights(
    quality_labels: np.ndarray,
    quality_order: list[str],
    phase_proportions: list[float],
    quality_counts: dict[str, int],
    count_normalize: bool = True,
) -> np.ndarray:
    """Compute per-sample weights based on quality labels.

    Parameters
    ----------
    quality_labels : np.ndarray
        Quality label for each sample (e.g., "high", "medium", "low").
    quality_order : list[str]
        Ordered list of quality levels from highest to lowest.
    phase_proportions : list[float]
        Target proportions for each quality level in current phase.
    quality_counts : dict[str, int]
        Count of samples per quality level.
    count_normalize : bool
        If True, adjust for dataset size imbalance.

    Returns
    -------
    np.ndarray
        Per-sample weights.
    """
    n_samples = len(quality_labels)
    weights = np.ones(n_samples)

    for i, quality in enumerate(quality_order):
        mask = quality_labels == quality
        target_prop = phase_proportions[i]

        if count_normalize:
            # Adjust for imbalanced dataset
            actual_prop = quality_counts.get(quality, 1) / n_samples
            if actual_prop > 0:
                weights[mask] = target_prop / actual_prop
        else:
            weights[mask] = target_prop

    # Normalize
    weights = weights / weights.sum() * n_samples
    return weights


def compute_combined_weights(
    task_weights: np.ndarray,
    quality_weights: np.ndarray,
) -> np.ndarray:
    """Combine task and quality weights multiplicatively.

    Parameters
    ----------
    task_weights : np.ndarray
        Weights from task oversampling.
    quality_weights : np.ndarray
        Weights from curriculum learning.

    Returns
    -------
    np.ndarray
        Combined per-sample weights.
    """
    combined = task_weights * quality_weights
    return combined / combined.sum() * len(combined)
```

- **Success**:
  - Functions work with numpy arrays (model-agnostic)
  - Produces same weights as current ChemProp JointSampler
  - Can be used by classical models via sample_weight parameter

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 127-155) - Tier 2 Training Strategies
  - [src/admet/model/chemprop/joint_sampler.py](src/admet/model/chemprop/joint_sampler.py) - Current implementation

- **Dependencies**:
  - Task 2.1 complete

---

### Task 2.3: Extract CurriculumLearningConfig and implement scheduler

Create model-agnostic curriculum learning configuration and scheduler.

- **Files**:
  - `src/admet/training/curriculum.py` - New file

- **Implementation**:

```python
"""Model-agnostic curriculum learning."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List
import numpy as np


class CurriculumPhase(str, Enum):
    WARMUP = "warmup"
    EXPAND = "expand"
    ROBUST = "robust"
    POLISH = "polish"


@dataclass
class CurriculumConfig:
    """Quality-aware curriculum learning configuration.

    Model-agnostic: works with any model that can use sample weights.
    """
    enabled: bool = False
    quality_col: str = "Quality"
    qualities: list[str] = field(default_factory=lambda: ["high", "medium", "low"])

    # Phase progression
    patience: int = 5
    strategy: str = "sampled"  # "sampled" or "weighted"
    reset_early_stopping_on_phase_change: bool = False

    # Phase proportions [high, medium, low]
    warmup_proportions: list[float] | None = None
    expand_proportions: list[float] | None = None
    robust_proportions: list[float] | None = None
    polish_proportions: list[float] | None = None

    # Normalization
    count_normalize: bool = True
    min_high_quality_proportion: float = 0.25

    # Monitoring
    monitor_metric: str = "val_loss"
    log_per_quality_metrics: bool = True

    def get_default_proportions(self, n_qualities: int) -> dict[str, list[float]]:
        """Get default phase proportions based on number of quality levels."""
        if n_qualities == 2:
            return {
                "warmup": [0.90, 0.10],
                "expand": [0.70, 0.30],
                "robust": [0.60, 0.40],
                "polish": [0.80, 0.20],
            }
        else:  # 3 qualities
            return {
                "warmup": [0.80, 0.15, 0.05],
                "expand": [0.60, 0.30, 0.10],
                "robust": [0.50, 0.35, 0.15],
                "polish": [0.70, 0.20, 0.10],
            }


class CurriculumScheduler:
    """Manages curriculum phase progression.

    Model-agnostic scheduler that tracks phase state and determines
    when to advance based on validation metrics.
    """

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.phase = CurriculumPhase.WARMUP
        self.phase_order = [
            CurriculumPhase.WARMUP,
            CurriculumPhase.EXPAND,
            CurriculumPhase.ROBUST,
            CurriculumPhase.POLISH,
        ]
        self.phase_idx = 0
        self.best_metric = float("inf")
        self.epochs_without_improvement = 0
        self.phase_history: list[tuple[int, str]] = []  # (epoch, phase_name)

    def get_current_proportions(self) -> list[float]:
        """Get proportions for current phase."""
        defaults = self.config.get_default_proportions(len(self.config.qualities))
        phase_name = self.phase.value

        # Use custom proportions if provided
        custom = getattr(self.config, f"{phase_name}_proportions", None)
        if custom is not None:
            return custom
        return defaults.get(phase_name, defaults["warmup"])

    def step(self, epoch: int, metric_value: float) -> bool:
        """Update scheduler state after epoch.

        Returns True if phase advanced.
        """
        if metric_value < self.best_metric:
            self.best_metric = metric_value
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.config.patience:
            return self._advance_phase(epoch)
        return False

    def _advance_phase(self, epoch: int) -> bool:
        """Advance to next phase if possible."""
        if self.phase_idx < len(self.phase_order) - 1:
            self.phase_idx += 1
            self.phase = self.phase_order[self.phase_idx]
            self.phase_history.append((epoch, self.phase.value))
            self.epochs_without_improvement = 0
            if self.config.reset_early_stopping_on_phase_change:
                self.best_metric = float("inf")
            return True
        return False

    def get_state(self) -> dict:
        """Get serializable state for checkpointing."""
        return {
            "phase": self.phase.value,
            "phase_idx": self.phase_idx,
            "best_metric": self.best_metric,
            "epochs_without_improvement": self.epochs_without_improvement,
            "phase_history": self.phase_history,
        }

    def load_state(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self.phase = CurriculumPhase(state["phase"])
        self.phase_idx = state["phase_idx"]
        self.best_metric = state["best_metric"]
        self.epochs_without_improvement = state["epochs_without_improvement"]
        self.phase_history = state["phase_history"]
```

- **Success**:
  - Scheduler is stateless regarding model internals
  - Can be used by any model type
  - State can be serialized for checkpointing
  - Behavior matches current ChemProp curriculum

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 127-155) - Tier 2 Training Strategies
  - [src/admet/model/chemprop/curriculum.py](src/admet/model/chemprop/curriculum.py) - Current implementation

- **Dependencies**:
  - Task 2.2 complete

---

### Task 2.4: Extract TaskAffinityConfig and implement affinity computer

Create model-agnostic task affinity configuration.

- **Files**:
  - `src/admet/training/task_affinity.py` - New file

- **Implementation**:

```python
"""Model-agnostic task affinity computation."""
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class TaskAffinityConfig:
    """Task grouping via gradient affinity.

    Model-agnostic configuration for computing task affinity scores
    and clustering tasks into groups.
    """
    enabled: bool = False
    method: str = "pretraining"  # "pretraining" or "online"

    # Common settings
    n_groups: int = 3
    clustering_method: str = "agglomerative"  # or "spectral"
    seed: int = 42

    # Pre-training method specific
    affinity_epochs: int = 1
    affinity_batch_size: int = 64
    affinity_lr: float = 1e-3
    affinity_type: str = "cosine"  # or "dot_product"

    # Online method specific
    compute_every_n_steps: int = 1
    log_every_n_steps: int = 100
    log_epoch_summary: bool = True
    lookahead_lr: float = 0.001
    use_optimizer_lr: bool = True

    # Shared parameter patterns (for neural models)
    exclude_param_patterns: list[str] = field(
        default_factory=lambda: ["predictor", "ffn", "output", "head", "readout"]
    )

    # Logging
    log_to_mlflow: bool = True
    save_plots: bool = False


def cluster_tasks(
    affinity_matrix: np.ndarray,
    n_groups: int,
    method: str = "agglomerative",
    seed: int = 42,
) -> list[list[int]]:
    """Cluster tasks based on affinity matrix.

    Parameters
    ----------
    affinity_matrix : np.ndarray
        Symmetric affinity matrix of shape (n_tasks, n_tasks).
    n_groups : int
        Number of groups to create.
    method : str
        Clustering method: "agglomerative" or "spectral".
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[list[int]]
        List of task groups, each group is a list of task indices.
    """
    from sklearn.cluster import AgglomerativeClustering, SpectralClustering

    n_tasks = affinity_matrix.shape[0]
    n_groups = min(n_groups, n_tasks)

    # Convert affinity to distance
    distance_matrix = 1 - affinity_matrix

    if method == "agglomerative":
        clustering = AgglomerativeClustering(
            n_clusters=n_groups,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(distance_matrix)
    else:  # spectral
        clustering = SpectralClustering(
            n_clusters=n_groups,
            affinity="precomputed",
            random_state=seed,
        )
        labels = clustering.fit_predict(affinity_matrix)

    # Group tasks by label
    groups = [[] for _ in range(n_groups)]
    for task_idx, group_idx in enumerate(labels):
        groups[group_idx].append(task_idx)

    return [g for g in groups if g]  # Remove empty groups
```

- **Success**:
  - Config captures all settings from both TaskAffinityConfig and InterTaskAffinityConfig
  - Clustering function works with any affinity matrix
  - Can be used by any model that can compute gradients

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 127-155) - Tier 2 Training Strategies
  - [src/admet/model/chemprop/task_affinity.py](src/admet/model/chemprop/task_affinity.py) - Current implementation

- **Dependencies**:
  - Task 2.3 complete

---

### Task 2.5: Create CurriculumAwareModel protocol

Define protocol for models that support curriculum learning.

- **Files**:
  - `src/admet/training/protocols.py` - New file

- **Implementation**:

```python
"""Protocols for training-aware models."""
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class SamplingAwareModel(Protocol):
    """Protocol for models that support sample weighting."""

    def fit(
        self,
        smiles: list[str],
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        **kwargs,
    ) -> "SamplingAwareModel":
        """Train with optional sample weights."""
        ...


@runtime_checkable
class CurriculumAwareModel(Protocol):
    """Protocol for models that support curriculum learning."""

    def get_quality_labels(self) -> np.ndarray:
        """Get quality labels for training data."""
        ...

    def update_sample_weights(self, weights: np.ndarray) -> None:
        """Update sampling weights for next epoch."""
        ...

    def get_validation_metric(self, quality: str | None = None) -> float:
        """Get validation metric, optionally per-quality."""
        ...


@runtime_checkable
class TaskAffinityAwareModel(Protocol):
    """Protocol for models that support task affinity computation."""

    def compute_task_gradients(self, batch_indices: np.ndarray) -> dict[int, np.ndarray]:
        """Compute per-task gradients for given batch.

        Returns dict mapping task_idx -> gradient vector.
        """
        ...

    def get_encoder_parameters(self) -> list[np.ndarray]:
        """Get shared encoder parameters."""
        ...
```

- **Success**:
  - Protocols are runtime checkable
  - ChemPropModel can implement all protocols
  - Classical models can implement SamplingAwareModel

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 155-190) - Phase 2 design

- **Dependencies**:
  - Task 2.4 complete

---

### Task 2.6: Update ClassicalModelBase for curriculum support

Add curriculum learning support to classical models via sample weights.

- **Files**:
  - `src/admet/model/classical/base.py` - Modify ClassicalModelBase

- **Implementation**:

```python
# Add to ClassicalModelBase

from admet.training.sampling import compute_task_weights, compute_quality_weights
from admet.training.curriculum import CurriculumScheduler, CurriculumConfig
from admet.training.protocols import SamplingAwareModel, CurriculumAwareModel


class ClassicalModelBase(BaseModel, MLflowMixin, SamplingAwareModel):
    """Base class with curriculum support."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self._quality_labels: np.ndarray | None = None
        self._curriculum_scheduler: CurriculumScheduler | None = None
        self._current_sample_weights: np.ndarray | None = None

        # Initialize curriculum if configured
        curriculum_config = config.get("curriculum", {})
        if curriculum_config.get("enabled", False):
            self._curriculum_scheduler = CurriculumScheduler(
                CurriculumConfig(**curriculum_config)
            )

    def _compute_sample_weights(
        self,
        targets: np.ndarray,
        quality_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute sample weights for current epoch."""
        weights = np.ones(len(targets))

        # Task oversampling
        sampling_config = self.config.get("sampling", {})
        if sampling_config.get("enabled", False):
            alpha = sampling_config.get("task_oversampling_alpha", 0.5)
            task_weights = compute_task_weights(targets, alpha)
            weights = weights * task_weights

        # Curriculum weighting
        if self._curriculum_scheduler and quality_labels is not None:
            proportions = self._curriculum_scheduler.get_current_proportions()
            quality_counts = {
                q: (quality_labels == q).sum()
                for q in self._curriculum_scheduler.config.qualities
            }
            quality_weights = compute_quality_weights(
                quality_labels,
                self._curriculum_scheduler.config.qualities,
                proportions,
                quality_counts,
            )
            weights = weights * quality_weights

        # Normalize
        weights = weights / weights.sum() * len(weights)
        return weights

    def fit(
        self,
        smiles: list[str],
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        quality_labels: np.ndarray | None = None,
        **kwargs,
    ) -> "ClassicalModelBase":
        """Train with optional curriculum-aware sample weights."""
        self._quality_labels = quality_labels

        # Compute weights if not provided
        if sample_weight is None and (
            self.config.get("sampling", {}).get("enabled", False)
            or self._curriculum_scheduler is not None
        ):
            sample_weight = self._compute_sample_weights(y, quality_labels)

        self._current_sample_weights = sample_weight

        # Call parent fit with sample_weight
        return super().fit(smiles, y, sample_weight=sample_weight, **kwargs)
```

- **Success**:
  - Classical models can use curriculum learning
  - Sample weights computed identically to ChemProp
  - Backward compatible (works without curriculum config)

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 155-190) - Phase 2 design
  - [src/admet/model/classical/base.py](src/admet/model/classical/base.py) - Current implementation

- **Dependencies**:
  - Tasks 2.2-2.5 complete

---

## Phase 3: Unified Featurization

### Task 3.1: Create FeaturizationConfig dataclass hierarchy

Create configuration hierarchy for different featurization types.

- **Files**:
  - `src/admet/model/config.py` - Add featurization configs

- **Implementation**:

```python
@dataclass
class GraphFeaturizerConfig:
    """Configuration for molecular graph featurization (ChemProp)."""
    atom_features: list[str] = field(
        default_factory=lambda: ["atomic_num", "formal_charge", "chiral_tag"]
    )
    bond_features: list[str] = field(
        default_factory=lambda: ["bond_type", "is_conjugated"]
    )
    use_3d: bool = False


@dataclass
class FoundationFeaturizerConfig:
    """Configuration for foundation model featurization (Chemeleon)."""
    model_name: str = "chemeleon"
    checkpoint_path: str = "auto"
    freeze_encoder: bool = True
    cache_dir: str | None = None


@dataclass
class FeaturizationConfig:
    """Master configuration for molecular featurization.

    The `type` field determines which sub-config is used:
    - "auto": Auto-detect based on model type
    - "fingerprint": Use fingerprint features (for classical models)
    - "graph": Use molecular graph (for ChemProp)
    - "foundation": Use foundation model embeddings (for Chemeleon)
    """
    type: str = "auto"

    # Sub-configurations
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)
    graph: GraphFeaturizerConfig = field(default_factory=GraphFeaturizerConfig)
    foundation: FoundationFeaturizerConfig = field(default_factory=FoundationFeaturizerConfig)
```

- **Success**:
  - All featurization options configurable via YAML
  - Auto-detection works for each model type
  - Existing FingerprintConfig unchanged

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 165-200) - Tier 3 Featurization

- **Dependencies**:
  - Phase 2 complete

---

### Task 3.2: Create featurizer factory

Create factory function to instantiate appropriate featurizer.

- **Files**:
  - `src/admet/features/factory.py` - New file

- **Implementation**:

```python
"""Featurizer factory for model-agnostic feature generation."""
from typing import Protocol, runtime_checkable
import numpy as np

from admet.model.config import FeaturizationConfig, FingerprintConfig
from admet.features.fingerprints import FingerprintGenerator


@runtime_checkable
class Featurizer(Protocol):
    """Protocol for molecular featurizers."""

    def featurize(self, smiles: list[str]) -> np.ndarray:
        """Convert SMILES to feature representation."""
        ...


class FingerprintFeaturizer:
    """Wrapper for fingerprint-based featurization."""

    def __init__(self, config: FingerprintConfig):
        self.generator = FingerprintGenerator(config)

    def featurize(self, smiles: list[str]) -> np.ndarray:
        return self.generator.generate(smiles)


def create_featurizer(
    config: FeaturizationConfig,
    model_type: str = "chemprop",
) -> Featurizer | None:
    """Create appropriate featurizer based on config.

    Parameters
    ----------
    config : FeaturizationConfig
        Featurization configuration.
    model_type : str
        Model type for auto-detection.

    Returns
    -------
    Featurizer | None
        Featurizer instance, or None if model handles its own featurization.
    """
    feat_type = config.type

    if feat_type == "auto":
        # Auto-detect based on model type
        if model_type in ("xgboost", "lightgbm", "catboost"):
            feat_type = "fingerprint"
        elif model_type == "chemeleon":
            feat_type = "foundation"
        else:  # chemprop
            feat_type = "graph"

    if feat_type == "fingerprint":
        return FingerprintFeaturizer(config.fingerprint)
    elif feat_type == "foundation":
        # Chemeleon handles its own featurization
        return None
    else:  # graph
        # ChemProp handles its own graph featurization
        return None
```

- **Success**:
  - Factory works with all model types
  - Auto-detection correct for each model
  - Returns None when model handles featurization internally

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 165-200) - Tier 3 Featurization

- **Dependencies**:
  - Task 3.1 complete

---

### Task 3.3: Create graph featurizer wrapper

Create wrapper for ChemProp's graph featurization.

- **Files**:
  - `src/admet/features/graph.py` - New file

- **Implementation**:

```python
"""Graph featurization wrapper for ChemProp."""
from typing import Any
from dataclasses import dataclass

from admet.model.config import GraphFeaturizerConfig


@dataclass
class GraphFeaturizerWrapper:
    """Wrapper around ChemProp's graph featurization.

    Provides consistent interface while allowing configuration
    of atom and bond features.
    """
    config: GraphFeaturizerConfig

    def get_chemprop_featurizer(self) -> Any:
        """Get ChemProp featurizer instance.

        Returns
        -------
        chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer
            Configured featurizer for ChemProp.
        """
        from chemprop import featurizers

        # Use default ChemProp featurizer
        # Custom atom/bond features would require extending ChemProp
        return featurizers.SimpleMoleculeMolGraphFeaturizer()

    def get_atom_feature_dim(self) -> int:
        """Get dimension of atom features."""
        # Default ChemProp atom feature dimension
        return 133

    def get_bond_feature_dim(self) -> int:
        """Get dimension of bond features."""
        # Default ChemProp bond feature dimension
        return 14
```

- **Success**:
  - Wrapper provides consistent interface
  - Works with existing ChemProp code
  - Config allows future customization

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 165-200) - Tier 3 Featurization

- **Dependencies**:
  - Task 3.2 complete

---

### Task 3.4: Add featurization tests

Add tests for featurizer factory and wrappers.

- **Files**:
  - `tests/test_featurization.py` - New file

- **Implementation**:

```python
"""Tests for featurization system."""
import pytest
import numpy as np

from admet.model.config import FeaturizationConfig, FingerprintConfig
from admet.features.factory import create_featurizer, FingerprintFeaturizer


class TestFeaturizerFactory:
    def test_auto_detect_xgboost(self):
        config = FeaturizationConfig(type="auto")
        featurizer = create_featurizer(config, model_type="xgboost")
        assert isinstance(featurizer, FingerprintFeaturizer)

    def test_auto_detect_chemprop(self):
        config = FeaturizationConfig(type="auto")
        featurizer = create_featurizer(config, model_type="chemprop")
        assert featurizer is None  # ChemProp handles its own

    def test_explicit_fingerprint(self):
        config = FeaturizationConfig(type="fingerprint")
        featurizer = create_featurizer(config, model_type="any")
        assert isinstance(featurizer, FingerprintFeaturizer)


class TestFingerprintFeaturizer:
    def test_morgan_fingerprint(self):
        config = FingerprintConfig(type="morgan")
        featurizer = FingerprintFeaturizer(config)
        features = featurizer.featurize(["CCO", "c1ccccc1"])
        assert features.shape == (2, 2048)

    def test_handles_invalid_smiles(self):
        config = FingerprintConfig(type="morgan")
        featurizer = FingerprintFeaturizer(config)
        # Should handle gracefully
        features = featurizer.featurize(["CCO", "invalid_smiles", "c1ccccc1"])
        assert features.shape[0] == 3
```

- **Success**:
  - All tests pass
  - Coverage >90% for featurization code
  - Edge cases tested

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 320-340) - Testing Strategy

- **Dependencies**:
  - Tasks 3.1-3.3 complete

---

# PART 2: Phases 4-6

---

## Phase 4: Unified Model Factory

### Task 4.1: Create UnifiedModelConfig master dataclass

Create the master configuration class that combines all tiers.

- **Files**:
  - `src/admet/model/config.py` - Add UnifiedModelConfig at end of file

- **Implementation**:

```python
@dataclass
class ChempropParams:
    """ChemProp-specific architecture parameters."""
    depth: int = 5
    message_hidden_dim: int = 600
    aggregation: str = "mean"
    dropout: float = 0.1
    batch_norm: bool = True
    ffn_type: str = "regression"
    num_layers: int = 2
    hidden_dim: int = 600
    trunk_n_layers: int = 2
    trunk_hidden_dim: int = 600
    n_experts: int = 4


@dataclass
class XGBoostParams:
    """XGBoost-specific parameters."""
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    n_jobs: int = -1


@dataclass
class LightGBMParams:
    """LightGBM-specific parameters."""
    n_estimators: int = 100
    max_depth: int = -1
    learning_rate: float = 0.1
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 20
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0


@dataclass
class CatBoostParams:
    """CatBoost-specific parameters."""
    iterations: int = 100
    depth: int = 6
    learning_rate: float = 0.1
    l2_leaf_reg: float = 3.0
    verbose: bool = False


@dataclass
class ChemeleonParams:
    """Chemeleon-specific parameters."""
    checkpoint_path: str = "auto"
    freeze_encoder: bool = True
    unfreeze_encoder_epoch: int | None = None
    unfreeze_encoder_lr_multiplier: float = 0.1
    ffn_hidden_dim: int = 300
    ffn_num_layers: int = 2
    dropout: float = 0.0


@dataclass
class RayConfig:
    """Ray parallelization settings for ensemble training."""
    max_parallel: int = 1
    num_cpus: int | None = None
    num_gpus: int | None = None


@dataclass
class UnifiedModelConfig:
    """Complete unified model configuration.

    This is the master config class that combines all configuration tiers:
    - Core: data, mlflow, optimization
    - Training strategies: sampling, curriculum, task_affinity
    - Featurization: fingerprint, graph, foundation
    - Model-specific: chemprop, xgboost, lightgbm, catboost, chemeleon
    - Ensemble: ray parallelization

    The `model_type` field determines which model-specific params are used.
    """
    # Model type discriminator
    model_type: str = "chemprop"

    # Core configs (Tier 1)
    data: UnifiedDataConfig = field(default_factory=UnifiedDataConfig)
    mlflow: UnifiedMlflowConfig = field(default_factory=UnifiedMlflowConfig)
    optimization: UnifiedOptimizationConfig = field(default_factory=UnifiedOptimizationConfig)

    # Training strategies (Tier 2)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    task_affinity: TaskAffinityConfig = field(default_factory=TaskAffinityConfig)

    # Featurization (Tier 3)
    featurization: FeaturizationConfig = field(default_factory=FeaturizationConfig)

    # Model-specific params (Tier 4)
    chemprop: ChempropParams = field(default_factory=ChempropParams)
    xgboost: XGBoostParams = field(default_factory=XGBoostParams)
    lightgbm: LightGBMParams = field(default_factory=LightGBMParams)
    catboost: CatBoostParams = field(default_factory=CatBoostParams)
    chemeleon: ChemeleonParams = field(default_factory=ChemeleonParams)

    # Ensemble settings
    ray: RayConfig = field(default_factory=RayConfig)

    def get_model_params(self) -> Any:
        """Get model-specific parameters based on model_type."""
        return getattr(self, self.model_type)
```

- **Success**:
  - All model types have params accessible
  - Config can be loaded from YAML with OmegaConf
  - Backward compatible with model creation

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 200-240) - Tier 5 Unified Config

- **Dependencies**:
  - Phase 3 complete

---

### Task 4.2: Update ModelRegistry with unified config support

Add method to create models from unified config.

- **Files**:
  - `src/admet/model/registry.py` - Add create_from_unified method

- **Implementation**:

```python
class ModelRegistry:
    """Registry for ADMET models."""

    _registry: dict[str, type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class."""
        def decorator(model_cls: type[BaseModel]) -> type[BaseModel]:
            cls._registry[name] = model_cls
            model_cls.model_type = name
            return model_cls
        return decorator

    @classmethod
    def create(cls, config: DictConfig) -> BaseModel:
        """Create model from config (legacy method)."""
        model_type = config.get("model", {}).get("type", "chemprop")
        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._registry[model_type].from_config(config)

    @classmethod
    def create_from_unified(
        cls,
        config: "UnifiedModelConfig",
        df_train: pd.DataFrame | None = None,
        df_validation: pd.DataFrame | None = None,
    ) -> BaseModel:
        """Create model from unified configuration.

        Parameters
        ----------
        config : UnifiedModelConfig
            Unified model configuration.
        df_train : pd.DataFrame | None
            Training data (required for some models).
        df_validation : pd.DataFrame | None
            Validation data.

        Returns
        -------
        BaseModel
            Configured model instance.
        """
        model_type = config.model_type
        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type}")

        model_cls = cls._registry[model_type]

        # Check if model supports unified config
        if hasattr(model_cls, "from_unified_config"):
            return model_cls.from_unified_config(
                config,
                df_train=df_train,
                df_validation=df_validation,
            )
        else:
            # Fall back to converting to legacy config
            legacy_config = cls._convert_to_legacy(config, model_type)
            return model_cls.from_config(legacy_config)

    @classmethod
    def _convert_to_legacy(
        cls,
        unified: "UnifiedModelConfig",
        model_type: str,
    ) -> DictConfig:
        """Convert unified config to legacy format for backward compat."""
        if model_type == "chemprop":
            from admet.model.chemprop.config import ChempropConfig
            legacy = ChempropConfig.from_unified(unified)
            return OmegaConf.structured(legacy)
        else:
            # For classical models, create dict config
            return OmegaConf.create({
                "model": {
                    "type": model_type,
                    model_type: OmegaConf.to_container(
                        OmegaConf.structured(unified.get_model_params())
                    ),
                    "fingerprint": OmegaConf.to_container(
                        OmegaConf.structured(unified.featurization.fingerprint)
                    ),
                },
                "data": OmegaConf.to_container(
                    OmegaConf.structured(unified.data)
                ),
                "mlflow": OmegaConf.to_container(
                    OmegaConf.structured(unified.mlflow)
                ),
                "sampling": OmegaConf.to_container(
                    OmegaConf.structured(unified.sampling)
                ),
                "curriculum": OmegaConf.to_container(
                    OmegaConf.structured(unified.curriculum)
                ),
            })
```

- **Success**:
  - Works with all registered model types
  - Falls back to legacy conversion when needed
  - Unified config passed to models that support it

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 205-220) - Phase 4 design

- **Dependencies**:
  - Task 4.1 complete

---

### Task 4.3: Add from_unified_config to ChempropAdapter

Add unified config support to ChemProp.

- **Files**:
  - `src/admet/model/chemprop/adapter.py` - Add from_unified_config classmethod

- **Implementation**:

```python
@classmethod
def from_unified_config(
    cls,
    config: "UnifiedModelConfig",
    df_train: pd.DataFrame | None = None,
    df_validation: pd.DataFrame | None = None,
) -> "ChempropModelAdapter":
    """Create ChempropModelAdapter from unified configuration.

    Parameters
    ----------
    config : UnifiedModelConfig
        Unified model configuration.
    df_train : pd.DataFrame | None
        Training data.
    df_validation : pd.DataFrame | None
        Validation data.

    Returns
    -------
    ChempropModelAdapter
        Configured model adapter.
    """
    from admet.model.chemprop.config import ChempropConfig

    # Convert to legacy ChempropConfig
    legacy_config = ChempropConfig.from_unified(config)

    # Create model
    return cls.from_config(
        OmegaConf.structured(legacy_config),
        df_train=df_train,
        df_validation=df_validation,
    )
```

- **Success**:
  - ChemProp works with unified config
  - All features preserved (curriculum, sampling, etc.)
  - Tests pass

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 205-220) - Phase 4 design

- **Dependencies**:
  - Task 4.2 complete

---

### Task 4.4: Add from_unified_config to classical models

Add unified config support to XGBoost, LightGBM, CatBoost.

- **Files**:
  - `src/admet/model/classical/base.py` - Add from_unified_config to ClassicalModelBase
  - `src/admet/model/classical/xgboost_model.py` - Inherit support
  - `src/admet/model/classical/lightgbm_model.py` - Inherit support
  - `src/admet/model/classical/catboost_model.py` - Inherit support

- **Implementation**:

```python
# In ClassicalModelBase

@classmethod
def from_unified_config(
    cls,
    config: "UnifiedModelConfig",
    df_train: pd.DataFrame | None = None,
    df_validation: pd.DataFrame | None = None,
) -> "ClassicalModelBase":
    """Create model from unified configuration.

    Parameters
    ----------
    config : UnifiedModelConfig
        Unified model configuration.
    df_train : pd.DataFrame | None
        Training data (used for quality labels if curriculum enabled).
    df_validation : pd.DataFrame | None
        Validation data.

    Returns
    -------
    ClassicalModelBase
        Configured model instance.
    """
    from admet.model.registry import ModelRegistry

    # Convert unified config to DictConfig for legacy path
    legacy_config = ModelRegistry._convert_to_legacy(config, config.model_type)

    # Create instance
    instance = cls(legacy_config)

    # Store training data for curriculum if needed
    if df_train is not None and config.curriculum.enabled:
        quality_col = config.curriculum.quality_col
        if quality_col in df_train.columns:
            instance._quality_labels = df_train[quality_col].values

    return instance
```

- **Success**:
  - All classical models work with unified config
  - Curriculum learning enabled via sample weights
  - Fingerprint config correctly applied

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 205-220) - Phase 4 design

- **Dependencies**:
  - Tasks 4.2-4.3 complete

---

### Task 4.5: Add from_unified_config to ChemeleonModel

Add unified config support to Chemeleon.

- **Files**:
  - `src/admet/model/chemeleon/model.py` - Add from_unified_config classmethod

- **Implementation**:

```python
@classmethod
def from_unified_config(
    cls,
    config: "UnifiedModelConfig",
    df_train: pd.DataFrame | None = None,
    df_validation: pd.DataFrame | None = None,
) -> "ChemeleonModel":
    """Create ChemeleonModel from unified configuration.

    Parameters
    ----------
    config : UnifiedModelConfig
        Unified model configuration.
    df_train : pd.DataFrame | None
        Training data.
    df_validation : pd.DataFrame | None
        Validation data.

    Returns
    -------
    ChemeleonModel
        Configured model instance.
    """
    # Build Chemeleon-specific config
    chemeleon_config = OmegaConf.create({
        "model": {
            "type": "chemeleon",
            "chemeleon": OmegaConf.to_container(
                OmegaConf.structured(config.chemeleon)
            ),
        },
        "data": OmegaConf.to_container(
            OmegaConf.structured(config.data)
        ),
        "optimization": OmegaConf.to_container(
            OmegaConf.structured(config.optimization)
        ),
        "mlflow": OmegaConf.to_container(
            OmegaConf.structured(config.mlflow)
        ),
        "sampling": OmegaConf.to_container(
            OmegaConf.structured(config.sampling)
        ),
        "curriculum": OmegaConf.to_container(
            OmegaConf.structured(config.curriculum)
        ),
    })

    return cls.from_config(chemeleon_config)
```

- **Success**:
  - Chemeleon works with unified config
  - Sampling and curriculum config passed through
  - Foundation model settings preserved

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 205-220) - Phase 4 design

- **Dependencies**:
  - Task 4.4 complete

---

### Task 4.6: Update ModelEnsemble for unified configs

Update ensemble to use unified config for all model types.

- **Files**:
  - `src/admet/model/chemprop/ensemble.py` - Update _train_single_model method

- **Implementation**:

```python
def _create_model_config(
    self,
    split_info: SplitFoldInfo,
) -> DictConfig:
    """Create per-model config, handling both unified and legacy formats.

    Parameters
    ----------
    split_info : SplitFoldInfo
        Information about the split/fold.

    Returns
    -------
    DictConfig
        Configuration for this specific model.
    """
    # Check if using unified config
    if hasattr(self.config, "model_type"):
        # Unified config path
        model_config = OmegaConf.to_container(self.config, resolve=True)
        model_config["data"]["data_dir"] = str(split_info.data_dir)
        return OmegaConf.create(model_config)
    else:
        # Legacy config path (ChemProp-specific)
        return self._create_chemprop_config(split_info)

def _train_single_model(
    self,
    split_info: SplitFoldInfo,
) -> BaseModel:
    """Train a single model for the given split/fold.

    Handles both unified and legacy configs.
    """
    model_config = self._create_model_config(split_info)

    # Load data
    df_train = pd.read_csv(split_info.train_file)
    df_val = pd.read_csv(split_info.validation_file)

    # Create model using registry
    from admet.model.registry import ModelRegistry

    if hasattr(self.config, "model_type"):
        # Unified config
        unified = OmegaConf.merge(
            OmegaConf.structured(UnifiedModelConfig),
            model_config,
        )
        model = ModelRegistry.create_from_unified(
            unified,
            df_train=df_train,
            df_validation=df_val,
        )
    else:
        # Legacy path
        model = ModelRegistry.create(model_config)

    # Train
    smiles = df_train[self.config.data.smiles_col].tolist()
    targets = df_train[list(self.config.data.target_cols)].values

    model.fit(smiles, targets)
    return model
```

- **Success**:
  - Ensemble works with any model type
  - Unified config automatically detected
  - Falls back to legacy for ChemProp-specific configs

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 205-220) - Phase 4 design

- **Dependencies**:
  - Tasks 4.1-4.5 complete

---

## Phase 5: HPO Integration

### Task 5.1: Create UnifiedSearchSpace config

Create unified HPO search space that works across models.

- **Files**:
  - `src/admet/model/hpo/__init__.py` - Add UnifiedSearchSpace

- **Implementation**:

```python
@dataclass
class UnifiedSearchSpace:
    """Unified search space for all model types.

    The model_type determines which model-specific search space is used.
    Training strategy parameters are searched regardless of model type.
    """
    # Model type (not searched, determines which space to use)
    model_type: str = "chemprop"

    # Model-specific search spaces
    chemprop: ChempropSearchSpace | None = None
    xgboost: XGBoostSearchSpace | None = None
    lightgbm: LightGBMSearchSpace | None = None
    catboost: CatBoostSearchSpace | None = None
    chemeleon: ChemeleonSearchSpace | None = None

    # Fingerprint search space (for classical models)
    fingerprint: FingerprintSearchSpace | None = None

    # Training strategy search spaces (model-agnostic)
    sampling: SamplingSearchSpace | None = None
    curriculum: CurriculumSearchSpace | None = None

    # Optimization search space (mainly for neural models)
    optimization: OptimizationSearchSpace | None = None

    def get_model_search_space(self) -> Any:
        """Get search space for current model type."""
        return getattr(self, self.model_type)


@dataclass
class SamplingSearchSpace:
    """Search space for sampling parameters."""
    task_oversampling_alpha: ParameterSpace | None = None


@dataclass
class CurriculumSearchSpace:
    """Search space for curriculum parameters."""
    patience: ParameterSpace | None = None
    warmup_high_proportion: ParameterSpace | None = None
    min_high_quality_proportion: ParameterSpace | None = None


@dataclass
class OptimizationSearchSpace:
    """Search space for optimization parameters."""
    learning_rate: ParameterSpace | None = None
    batch_size: ParameterSpace | None = None
    max_epochs: ParameterSpace | None = None
    warmup_epochs: ParameterSpace | None = None
    weight_decay: ParameterSpace | None = None
```

- **Success**:
  - Search space works for all model types
  - Training strategy params searchable
  - Consistent with Ray Tune format

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 225-250) - Phase 5 design

- **Dependencies**:
  - Phase 4 complete

---

### Task 5.2: Add training strategy parameters to search space

Enable HPO over curriculum and sampling parameters.

- **Files**:
  - `src/admet/model/hpo/search_space.py` - New file with search space builders

- **Implementation**:

```python
"""Search space builders for unified HPO."""
from typing import Any, Dict
from ray import tune

from admet.model.hpo import (
    UnifiedSearchSpace,
    ParameterSpace,
    SamplingSearchSpace,
    CurriculumSearchSpace,
)


def build_ray_search_space(
    search_space: UnifiedSearchSpace,
) -> Dict[str, Any]:
    """Convert UnifiedSearchSpace to Ray Tune search space.

    Parameters
    ----------
    search_space : UnifiedSearchSpace
        Unified search space configuration.

    Returns
    -------
    Dict[str, Any]
        Ray Tune compatible search space.
    """
    result = {}

    # Add model-specific params
    model_space = search_space.get_model_search_space()
    if model_space:
        result.update(_convert_dataclass_to_tune(model_space, prefix="model"))

    # Add fingerprint params (for classical models)
    if search_space.fingerprint:
        result.update(_convert_dataclass_to_tune(
            search_space.fingerprint, prefix="fingerprint"
        ))

    # Add training strategy params
    if search_space.sampling:
        result.update(_convert_dataclass_to_tune(
            search_space.sampling, prefix="sampling"
        ))

    if search_space.curriculum:
        result.update(_convert_dataclass_to_tune(
            search_space.curriculum, prefix="curriculum"
        ))

    if search_space.optimization:
        result.update(_convert_dataclass_to_tune(
            search_space.optimization, prefix="optimization"
        ))

    return result


def _convert_param_space(param: ParameterSpace) -> Any:
    """Convert ParameterSpace to Ray Tune distribution."""
    if param.type == "uniform":
        return tune.uniform(param.low, param.high)
    elif param.type == "loguniform":
        return tune.loguniform(param.low, param.high)
    elif param.type == "choice":
        return tune.choice(param.values)
    elif param.type == "quniform":
        return tune.quniform(param.low, param.high, param.q)
    elif param.type == "randint":
        return tune.randint(int(param.low), int(param.high))
    else:
        raise ValueError(f"Unknown parameter type: {param.type}")


def _convert_dataclass_to_tune(
    obj: Any,
    prefix: str = "",
) -> Dict[str, Any]:
    """Convert dataclass with ParameterSpace fields to Ray Tune format."""
    result = {}
    for field_name, value in obj.__dict__.items():
        if value is None:
            continue
        key = f"{prefix}.{field_name}" if prefix else field_name
        if isinstance(value, ParameterSpace):
            result[key] = _convert_param_space(value)
    return result
```

- **Success**:
  - Training strategy params searchable via HPO
  - Works with Ray Tune
  - Consistent parameter naming

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 225-250) - Phase 5 design

- **Dependencies**:
  - Task 5.1 complete

---

### Task 5.3: Update HPO trainable for unified configs

Update the trainable function to use unified configs.

- **Files**:
  - `src/admet/model/chemprop/hpo_trainable.py` - Update to use unified config

- **Implementation**:

```python
def unified_trainable(
    config: Dict[str, Any],
    checkpoint_dir: str | None = None,
) -> None:
    """Trainable function for unified HPO.

    Parameters
    ----------
    config : Dict[str, Any]
        Hyperparameter configuration from Ray Tune.
    checkpoint_dir : str | None
        Directory for checkpointing.
    """
    from ray import train
    from admet.model.config import UnifiedModelConfig
    from admet.model.registry import ModelRegistry

    # Extract base config
    base_config = config.pop("_base_config")
    model_type = config.pop("_model_type", "chemprop")

    # Merge hyperparameters into base config
    merged = _merge_hpo_params(base_config, config)

    # Create unified config
    unified = OmegaConf.merge(
        OmegaConf.structured(UnifiedModelConfig),
        merged,
    )

    # Load data
    df_train = pd.read_csv(unified.data.data_dir + "/train.csv")
    df_val = pd.read_csv(unified.data.data_dir + "/validation.csv")

    # Create and train model
    model = ModelRegistry.create_from_unified(
        unified,
        df_train=df_train,
        df_validation=df_val,
    )

    smiles = df_train[unified.data.smiles_col].tolist()
    targets = df_train[list(unified.data.target_cols)].values

    val_smiles = df_val[unified.data.smiles_col].tolist()
    val_targets = df_val[list(unified.data.target_cols)].values

    model.fit(smiles, targets, val_smiles=val_smiles, val_y=val_targets)

    # Evaluate
    predictions = model.predict(val_smiles)
    mae = np.mean(np.abs(predictions - val_targets))

    # Report to Ray
    train.report({"val_mae": mae})


def _merge_hpo_params(
    base_config: Dict[str, Any],
    hpo_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge HPO parameters into base config.

    Parameters like "model.depth" are nested into the config structure.
    """
    result = OmegaConf.to_container(base_config, resolve=True)

    for key, value in hpo_params.items():
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    return result
```

- **Success**:
  - Works with any model type
  - Training strategy params applied correctly
  - Reports metrics to Ray Tune

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 225-250) - Phase 5 design

- **Dependencies**:
  - Task 5.2 complete

---

### Task 5.4: Create HPO integration tests

Add tests for unified HPO.

- **Files**:
  - `tests/test_hpo_unified.py` - New test file

- **Implementation**:

```python
"""Tests for unified HPO system."""
import pytest
from omegaconf import OmegaConf

from admet.model.hpo import (
    UnifiedSearchSpace,
    ParameterSpace,
    SamplingSearchSpace,
    CurriculumSearchSpace,
)
from admet.model.hpo.search_space import build_ray_search_space


class TestUnifiedSearchSpace:
    def test_get_model_search_space(self):
        space = UnifiedSearchSpace(model_type="xgboost")
        assert space.get_model_search_space() is None  # Not set

    def test_with_model_space(self):
        from admet.model.hpo import XGBoostSearchSpace
        space = UnifiedSearchSpace(
            model_type="xgboost",
            xgboost=XGBoostSearchSpace(
                max_depth=ParameterSpace(type="randint", low=3, high=10),
            ),
        )
        assert space.get_model_search_space() is not None


class TestBuildRaySearchSpace:
    def test_sampling_params(self):
        space = UnifiedSearchSpace(
            sampling=SamplingSearchSpace(
                task_oversampling_alpha=ParameterSpace(
                    type="uniform", low=0.0, high=1.0
                ),
            ),
        )
        ray_space = build_ray_search_space(space)
        assert "sampling.task_oversampling_alpha" in ray_space

    def test_curriculum_params(self):
        space = UnifiedSearchSpace(
            curriculum=CurriculumSearchSpace(
                patience=ParameterSpace(type="randint", low=3, high=10),
            ),
        )
        ray_space = build_ray_search_space(space)
        assert "curriculum.patience" in ray_space

    def test_combined_spaces(self):
        from admet.model.hpo import XGBoostSearchSpace
        space = UnifiedSearchSpace(
            model_type="xgboost",
            xgboost=XGBoostSearchSpace(
                max_depth=ParameterSpace(type="randint", low=3, high=10),
            ),
            sampling=SamplingSearchSpace(
                task_oversampling_alpha=ParameterSpace(
                    type="uniform", low=0.0, high=1.0
                ),
            ),
        )
        ray_space = build_ray_search_space(space)
        assert "model.max_depth" in ray_space
        assert "sampling.task_oversampling_alpha" in ray_space
```

- **Success**:
  - All tests pass
  - Coverage >90% for HPO code
  - Integration with Ray Tune verified

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 320-340) - Testing Strategy

- **Dependencies**:
  - Tasks 5.1-5.3 complete

---

## Phase 6: Migration and Documentation

### Task 6.1: Create config migration script

Create script to migrate existing configs to unified format.

- **Files**:
  - `scripts/migrate_configs_v2.py` - New migration script

- **Implementation**:

```python
#!/usr/bin/env python3
"""Migrate existing configs to unified format."""
import argparse
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf

from admet.model.config import UnifiedModelConfig


def detect_config_type(config: Dict[str, Any]) -> str:
    """Detect if config is ChemProp, classical, or unified."""
    if "model_type" in config:
        return "unified"

    model_section = config.get("model", {})
    model_type = model_section.get("type", "chemprop")

    if model_type == "chemprop":
        # Check for ChemProp-specific structure
        if "chemprop" in model_section or "depth" in model_section:
            return "chemprop"
    return model_type


def migrate_chemprop_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ChemProp config to unified format."""
    model_section = config.get("model", {})
    chemprop_params = model_section.get("chemprop", model_section)

    unified = {
        "model_type": "chemprop",
        "data": config.get("data", {}),
        "optimization": config.get("optimization", {}),
        "mlflow": _migrate_mlflow(config.get("mlflow", {})),
        "chemprop": {
            "depth": chemprop_params.get("depth", 5),
            "message_hidden_dim": chemprop_params.get("message_hidden_dim", 600),
            "hidden_dim": chemprop_params.get("hidden_dim", 600),
            "num_layers": chemprop_params.get("num_layers", 2),
            "dropout": chemprop_params.get("dropout", 0.1),
            "batch_norm": chemprop_params.get("batch_norm", True),
            "ffn_type": chemprop_params.get("ffn_type", "regression"),
            "aggregation": chemprop_params.get("aggregation", "mean"),
        },
        "sampling": _migrate_sampling(config),
        "curriculum": _migrate_curriculum(config),
        "task_affinity": config.get("task_affinity", {}),
        "ray": config.get("ray", {}),
    }

    return unified


def migrate_classical_config(
    config: Dict[str, Any],
    model_type: str,
) -> Dict[str, Any]:
    """Convert classical model config to unified format."""
    model_section = config.get("model", {})

    unified = {
        "model_type": model_type,
        "data": config.get("data", {}),
        "mlflow": _migrate_mlflow(config.get("mlflow", {})),
        model_type: model_section.get(model_type, {}),
        "featurization": {
            "type": "fingerprint",
            "fingerprint": model_section.get("fingerprint", {}),
        },
        "sampling": config.get("sampling", {"enabled": False}),
        "curriculum": config.get("curriculum", {"enabled": False}),
        "ray": config.get("ray", {}),
    }

    return unified


def _migrate_mlflow(mlflow_config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize MLflow config field names."""
    return {
        "enabled": mlflow_config.get("tracking", mlflow_config.get("enabled", True)),
        "tracking_uri": mlflow_config.get("tracking_uri"),
        "experiment_name": mlflow_config.get("experiment_name", "admet"),
        "run_name": mlflow_config.get("run_name"),
        "nested": mlflow_config.get("nested", False),
        "log_model": mlflow_config.get("log_model", True),
    }


def _migrate_sampling(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract sampling config from joint_sampling or standalone."""
    joint = config.get("joint_sampling", {})
    if joint.get("enabled", False):
        return {
            "enabled": True,
            "task_oversampling_alpha": joint.get(
                "task_oversampling", {}
            ).get("alpha", 0.5),
            "seed": joint.get("seed", 42),
        }
    return {"enabled": False}


def _migrate_curriculum(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract curriculum config from joint_sampling or standalone."""
    joint = config.get("joint_sampling", {})
    curriculum = joint.get("curriculum", config.get("curriculum", {}))
    return curriculum


def migrate_file(input_path: Path, output_path: Path) -> None:
    """Migrate a single config file."""
    config = OmegaConf.to_container(OmegaConf.load(input_path))
    config_type = detect_config_type(config)

    if config_type == "unified":
        print(f"  {input_path.name}: Already unified, skipping")
        return

    if config_type == "chemprop":
        unified = migrate_chemprop_config(config)
    else:
        unified = migrate_classical_config(config, config_type)

    # Validate against schema
    try:
        OmegaConf.merge(
            OmegaConf.structured(UnifiedModelConfig),
            OmegaConf.create(unified),
        )
    except Exception as e:
        print(f"  {input_path.name}: Validation failed: {e}")
        return

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"# Migrated from {input_path.name}\n")
        f.write(OmegaConf.to_yaml(OmegaConf.create(unified)))

    print(f"  {input_path.name} -> {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Migrate configs to unified format")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configs = list(args.input_dir.glob("**/*.yaml"))
    print(f"Found {len(configs)} config files")

    for config_path in configs:
        rel_path = config_path.relative_to(args.input_dir)
        output_path = args.output_dir / rel_path

        if args.dry_run:
            print(f"  Would migrate: {rel_path}")
        else:
            migrate_file(config_path, output_path)


if __name__ == "__main__":
    main()
```

- **Success**:
  - Script handles all config types
  - Validates migrated configs
  - Dry-run mode available

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 280-310) - Migration Strategy

- **Dependencies**:
  - Phase 5 complete

---

### Task 6.2: Migrate example configs

Migrate configs in configs/4-more-models/ as examples.

- **Files**:
  - `configs/4-more-models/chemprop_unified.yaml` - Migrated example
  - `configs/4-more-models/xgboost_unified.yaml` - Migrated example

- **Implementation**:

Run migration script on example configs and verify output.

```bash
python scripts/migrate_configs_v2.py \
    --input-dir configs/4-more-models \
    --output-dir configs/unified-examples
```

- **Success**:
  - Example configs migrated successfully
  - Migrated configs load without errors
  - Models train with migrated configs

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 280-310) - Migration Strategy

- **Dependencies**:
  - Task 6.1 complete

---

### Task 6.3: Add config validation with helpful errors

Add validation that provides helpful error messages.

- **Files**:
  - `src/admet/model/config_validation.py` - New validation module

- **Implementation**:

```python
"""Configuration validation with helpful error messages."""
from typing import Any, Dict, List
from omegaconf import OmegaConf, DictConfig

from admet.model.config import UnifiedModelConfig


class ConfigValidationError(ValueError):
    """Error with helpful context for fixing config issues."""

    def __init__(self, message: str, suggestions: List[str] | None = None):
        self.suggestions = suggestions or []
        full_message = message
        if self.suggestions:
            full_message += "\n\nSuggestions:\n" + "\n".join(
                f"  - {s}" for s in self.suggestions
            )
        super().__init__(full_message)


def validate_unified_config(config: DictConfig) -> UnifiedModelConfig:
    """Validate config and return typed UnifiedModelConfig.

    Parameters
    ----------
    config : DictConfig
        Configuration to validate.

    Returns
    -------
    UnifiedModelConfig
        Validated configuration.

    Raises
    ------
    ConfigValidationError
        If validation fails with helpful suggestions.
    """
    errors = []
    suggestions = []

    # Check model_type
    model_type = config.get("model_type", "chemprop")
    valid_types = ["chemprop", "xgboost", "lightgbm", "catboost", "chemeleon"]
    if model_type not in valid_types:
        errors.append(f"Invalid model_type: '{model_type}'")
        suggestions.append(f"Valid model types: {valid_types}")

    # Check data config
    data = config.get("data", {})
    if not data.get("data_dir") and not data.get("train_file"):
        errors.append("Missing data path: need either 'data.data_dir' or 'data.train_file'")
        suggestions.append("Set data.data_dir to directory containing train.csv and validation.csv")

    if not data.get("target_cols"):
        errors.append("Missing target_cols: no prediction targets specified")
        suggestions.append("Set data.target_cols to list of column names, e.g., ['LogD', 'KSOL']")

    # Check curriculum requires quality_col
    curriculum = config.get("curriculum", {})
    if curriculum.get("enabled", False):
        quality_col = curriculum.get("quality_col", "Quality")
        if not quality_col:
            errors.append("Curriculum enabled but no quality_col specified")
            suggestions.append("Set curriculum.quality_col to column with quality labels")

    # Check featurization for classical models
    if model_type in ["xgboost", "lightgbm", "catboost"]:
        feat = config.get("featurization", {})
        feat_type = feat.get("type", "auto")
        if feat_type not in ["auto", "fingerprint"]:
            errors.append(f"Classical models require fingerprint features, got: {feat_type}")
            suggestions.append("Set featurization.type to 'fingerprint' or 'auto'")

    if errors:
        raise ConfigValidationError(
            f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors),
            suggestions=suggestions,
        )

    # Merge with schema and return
    try:
        merged = OmegaConf.merge(
            OmegaConf.structured(UnifiedModelConfig),
            config,
        )
        return OmegaConf.to_object(merged)
    except Exception as e:
        raise ConfigValidationError(
            f"Failed to parse config: {e}",
            suggestions=["Check YAML syntax and field names"],
        )
```

- **Success**:
  - Helpful error messages for common mistakes
  - Suggests fixes for each error
  - Validates all required fields

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 320-340) - Testing Strategy

- **Dependencies**:
  - Task 6.2 complete

---

### Task 6.4: Update documentation

Update docs with unified config reference.

- **Files**:
  - `docs/guide/unified_config.rst` - New documentation page
  - `docs/guide/config_reference.rst` - Update with links to unified docs

- **Implementation**:

Create comprehensive documentation covering:
1. Unified config schema overview
2. Migration guide from legacy configs
3. Examples for each model type
4. Training strategy configuration
5. HPO with unified search spaces

- **Success**:
  - Documentation builds without errors
  - Examples are runnable
  - Migration path clearly documented

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 280-310) - Migration Strategy

- **Dependencies**:
  - Task 6.3 complete

---

### Task 6.5: Create integration test suite

Create comprehensive integration tests.

- **Files**:
  - `tests/test_unified_integration.py` - New integration test file

- **Implementation**:

```python
"""Integration tests for unified configuration system."""
import pytest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from admet.model.config import UnifiedModelConfig
from admet.model.registry import ModelRegistry


@pytest.fixture
def sample_data():
    """Create sample training data."""
    n_samples = 100
    return pd.DataFrame({
        "SMILES": ["CCO"] * n_samples,
        "LogD": np.random.randn(n_samples),
        "KSOL": np.random.randn(n_samples),
        "Quality": np.random.choice(["high", "medium", "low"], n_samples),
    })


@pytest.fixture
def unified_config():
    """Create base unified config."""
    return OmegaConf.structured(UnifiedModelConfig)


class TestUnifiedChemprop:
    def test_train_chemprop(self, sample_data, unified_config):
        unified_config.model_type = "chemprop"
        unified_config.data.target_cols = ["LogD", "KSOL"]
        unified_config.optimization.max_epochs = 1

        model = ModelRegistry.create_from_unified(
            unified_config,
            df_train=sample_data,
            df_validation=sample_data,
        )

        smiles = sample_data["SMILES"].tolist()
        targets = sample_data[["LogD", "KSOL"]].values

        model.fit(smiles, targets)
        predictions = model.predict(smiles)

        assert predictions.shape == targets.shape


class TestUnifiedClassical:
    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
    def test_train_classical(self, sample_data, unified_config, model_type):
        unified_config.model_type = model_type
        unified_config.data.target_cols = ["LogD", "KSOL"]

        model = ModelRegistry.create_from_unified(
            unified_config,
            df_train=sample_data,
        )

        smiles = sample_data["SMILES"].tolist()
        targets = sample_data[["LogD", "KSOL"]].values

        model.fit(smiles, targets)
        predictions = model.predict(smiles)

        assert predictions.shape == targets.shape


class TestCurriculumLearning:
    def test_curriculum_xgboost(self, sample_data, unified_config):
        unified_config.model_type = "xgboost"
        unified_config.data.target_cols = ["LogD"]
        unified_config.curriculum.enabled = True
        unified_config.curriculum.quality_col = "Quality"

        model = ModelRegistry.create_from_unified(
            unified_config,
            df_train=sample_data,
        )

        smiles = sample_data["SMILES"].tolist()
        targets = sample_data[["LogD"]].values
        quality = sample_data["Quality"].values

        model.fit(smiles, targets, quality_labels=quality)

        assert model.is_fitted


class TestConfigMigration:
    def test_chemprop_migration(self):
        from scripts.migrate_configs_v2 import migrate_chemprop_config

        legacy = {
            "model": {"type": "chemprop", "depth": 5, "hidden_dim": 600},
            "data": {"data_dir": "test", "target_cols": ["LogD"]},
            "optimization": {"max_epochs": 100},
        }

        unified = migrate_chemprop_config(legacy)

        assert unified["model_type"] == "chemprop"
        assert unified["chemprop"]["depth"] == 5
```

- **Success**:
  - All integration tests pass
  - Tests cover all model types
  - Curriculum learning tested with classical models

- **Research References**:
  - #file:../research/config-harmonization-plan.md (Lines 320-340) - Testing Strategy

- **Dependencies**:
  - Tasks 6.1-6.4 complete

---

## Dependencies Summary

- Python 3.10+
- OmegaConf for config management
- dataclasses for structured configs
- pytest for testing
- Ray Tune for HPO
- scikit-learn for clustering (task affinity)
- Existing model implementations

## Success Criteria

- [ ] Single YAML schema works for all model types
- [ ] Curriculum learning available for classical models via sample_weight
- [ ] Task affinity grouping works across neural model types
- [ ] HPO can search training strategy parameters for any model
- [ ] All existing configs continue to work (backward compatibility)
- [ ] Test coverage >90% on new config/training modules
- [ ] Documentation complete with migration guide
