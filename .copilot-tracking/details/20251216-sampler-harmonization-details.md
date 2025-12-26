<!-- markdownlint-disable-file -->

# Task Details: Sampler Harmonization

## Research Reference

**Source Research**: #file:../research/20251216-sampler-harmonization-research.md

---

## Phase 1: Configuration Schema

### Task 1.1: Create JointSamplingConfig dataclass

Create a new configuration dataclass that unifies both sampling strategies under a single schema.

- **Files**:
  - `src/admet/model/chemprop/config.py` - Add new dataclass after CurriculumConfig

- **Implementation**:
```python
@dataclass
class TaskOversamplingConfig:
    """Configuration for task-aware oversampling of sparse tasks."""

    enabled: bool = False
    alpha: float = 0.5
    # alpha=0: uniform task sampling
    # alpha=1: full inverse-proportional (rare tasks sampled much more)


@dataclass
class JointSamplingConfig:
    """
    Unified configuration for combined task and curriculum sampling.

    Combines task-aware oversampling (for imbalanced multi-task data) with
    quality-aware curriculum learning (for data quality progression).

    When both strategies are enabled, weights are composed multiplicatively:
        w_joint[i] = w_task[i] × w_quality[i]

    For multi-task samples, the "primary" task (used for weight computation)
    is the rarest task among those the sample has labels for.

    Parameters
    ----------
    enabled : bool, default=False
        Master switch to enable joint sampling. When False, falls back to
        legacy behavior (task_sampling_alpha or curriculum.enabled separately).
    task_oversampling : TaskOversamplingConfig
        Configuration for task-aware oversampling.
    curriculum : CurriculumConfig
        Configuration for quality-aware curriculum learning.
    composition_mode : str, default="multiplicative"
        How to combine task and curriculum weights. Only "multiplicative" is
        supported: w_joint = w_task × w_quality
    seed : int, default=42
        Base random seed for reproducible sampling.
    increment_seed_per_epoch : bool, default=True
        If True, seed = base_seed + epoch_num for variety across epochs.
        If False, same sampling order every epoch (fully deterministic).
    log_weight_statistics : bool, default=True
        Log weight distribution statistics (min, max, entropy, effective samples)
        at the start of each epoch.
    log_to_mlflow : bool, default=True
        Log sampling statistics to MLflow for experiment tracking.
    """

    enabled: bool = False
    task_oversampling: TaskOversamplingConfig = field(default_factory=TaskOversamplingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    composition_mode: str = "multiplicative"
    seed: int = 42
    increment_seed_per_epoch: bool = True
    log_weight_statistics: bool = True
    log_to_mlflow: bool = True
```

- **Success**:
  - Dataclass passes OmegaConf structured config validation
  - Default values are sensible and match existing behavior
  - Type hints are complete and accurate

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md (Lines 145-175) - Proposed config structure

- **Dependencies**:
  - Existing `CurriculumConfig` dataclass must be defined first

### Task 1.2: Create TaskOversamplingConfig sub-dataclass

Define the sub-configuration for task-aware oversampling parameters.

- **Files**:
  - `src/admet/model/chemprop/config.py` - Add before JointSamplingConfig

- **Implementation**:
```python
@dataclass
class TaskOversamplingConfig:
    """
    Configuration for task-aware oversampling of sparse tasks.

    Uses inverse-power sampling schedule: p_i ∝ n_i^(-α) where n_i is the
    number of labels for task i.

    For samples with multiple task labels, the "primary" task is the rarest
    task (smallest label count) among the sample's valid tasks.

    Parameters
    ----------
    enabled : bool, default=False
        Whether to enable task-aware oversampling.
    alpha : float, default=0.5
        Power law exponent for task sampling probability.
        - α = 0: Uniform sampling across tasks
        - α = 0.5: Moderate rebalancing (recommended starting point)
        - α = 1: Full inverse proportional (rare tasks sampled much more)

        Values outside [0, 1] will trigger a warning but still function.
        Values > 1 cause extreme over-sampling of rare tasks.
    """

    enabled: bool = False
    alpha: float = 0.5

    def __post_init__(self):
        if self.alpha < 0 or self.alpha > 1:
            import warnings
            warnings.warn(
                f"TaskOversamplingConfig.alpha={self.alpha} is outside "
                f"recommended range [0, 1]. Values > 1 cause extreme "
                f"over-sampling of rare tasks.",
                UserWarning,
                stacklevel=2,
            )
```

- **Success**:
  - Alpha parameter has clear documentation of valid range [0, 1]
  - Default alpha=0.5 provides moderate rebalancing

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md (Lines 50-65) - TaskAwareSampler alpha semantics

- **Dependencies**:
  - None

### Task 1.3: Update ChempropConfig to include JointSamplingConfig

Add the new joint sampling config to the main configuration schema.

- **Files**:
  - `src/admet/model/chemprop/config.py` - Add field to ChempropConfig

- **Implementation**:
```python
@dataclass
class ChempropConfig:
    """Complete configuration for ChempropModel training."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)

    # Legacy curriculum config (for backward compatibility)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    # New unified sampling config
    joint_sampling: JointSamplingConfig = field(default_factory=JointSamplingConfig)

    # ... rest of existing fields
```

- **Success**:
  - ChempropConfig includes joint_sampling field
  - Legacy curriculum field preserved for backward compatibility
  - OmegaConf can load both old and new config formats

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md (Lines 170-180) - Config integration

- **Dependencies**:
  - Task 1.1: JointSamplingConfig must be defined

---

## Phase 2: JointSampler Implementation

### Task 2.1: Create JointSampler class skeleton

Create the new sampler module with class structure and docstrings.

- **Files**:
  - `src/admet/model/chemprop/joint_sampler.py` - NEW FILE

- **Implementation**:
```python
"""
Joint sampler for combined task-aware and curriculum-aware sampling.

This module provides a unified sampler that combines:
1. Task-aware oversampling: Rebalances sampling across tasks with different
   label counts using inverse-power weighting.
2. Curriculum-aware sampling: Adjusts sampling based on data quality labels
   that change with curriculum phase progression.

The two strategies are combined via multiplicative weight composition:
    w_joint[i] = w_task[i] × w_quality[i]

For multi-task samples, the "primary" task (used for weight computation)
is the rarest task among those the sample has labels for.

Examples
--------
>>> from admet.model.chemprop.joint_sampler import JointSampler
>>> from admet.model.chemprop.curriculum import CurriculumState
>>>
>>> # Create joint sampler with both strategies
>>> sampler = JointSampler(
...     targets=target_array,           # (N, T) array with NaN for missing
...     quality_labels=quality_list,    # ["high", "medium", "low", ...]
...     curriculum_state=curr_state,    # CurriculumState object
...     task_alpha=0.3,                 # Task rebalancing strength
...     num_samples=1000,
...     seed=42,
...     increment_seed_per_epoch=True,  # Vary sampling across epochs
... )
>>>
>>> # Use with DataLoader
>>> loader = DataLoader(dataset, sampler=sampler, batch_size=32)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler

if TYPE_CHECKING:
    from admet.model.chemprop.curriculum import CurriculumState

logger = logging.getLogger("admet.model.chemprop.joint_sampler")


class JointSampler(Sampler[int]):
    """
    Unified sampler combining task-aware and curriculum-aware sampling.

    Computes joint sampling weights as the product of task weights and
    curriculum quality weights, enabling simultaneous optimization for
    both task imbalance and data quality progression.

    Parameters
    ----------
    targets : np.ndarray
        Target matrix of shape (N, T) where N is number of molecules and
        T is number of tasks. Missing labels should be NaN.
    quality_labels : Sequence[str] | None
        Quality label for each sample. Required if curriculum_state is provided.
    curriculum_state : CurriculumState | None
        Curriculum state for quality-aware sampling. If None, only task
        weighting is applied.
    task_alpha : float | None
        Exponent for task-aware inverse-power sampling. If None, only
        curriculum weighting is applied. Values outside [0, 1] trigger warning.
    composition_mode : str, default="multiplicative"
        How to compose weights. Only "multiplicative" is supported.
    num_samples : int | None
        Number of samples per epoch. Defaults to len(targets).
    seed : int | None
        Base random seed for reproducibility.
    increment_seed_per_epoch : bool, default=True
        If True, seed = base_seed + epoch_num for variety across epochs.
    log_weight_statistics : bool, default=True
        Log weight distribution statistics at epoch start.

    Attributes
    ----------
    task_counts : np.ndarray
        Number of valid labels per task.
    task_indices : list[np.ndarray]
        Valid sample indices for each task.
    _epoch : int
        Current epoch number (for seed incrementing).

    Raises
    ------
    ValueError
        If curriculum_state is provided but quality_labels is None.
        If both task_alpha and curriculum_state are None (no sampling strategy).
    """

    def __init__(
        self,
        targets: np.ndarray,
        quality_labels: Sequence[str] | None = None,
        curriculum_state: "CurriculumState | None" = None,
        task_alpha: float | None = None,
        composition_mode: str = "multiplicative",
        num_samples: int | None = None,
        seed: int | None = None,
        increment_seed_per_epoch: bool = True,
        log_weight_statistics: bool = True,
    ) -> None:
        super().__init__(None)  # type: ignore[arg-type]
        # Implementation in subsequent tasks
        self._epoch = 0
        ...

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for seed incrementing."""
        self._epoch = epoch

    def _compute_task_weights(self) -> np.ndarray:
        """Compute per-sample weights from task label counts."""
        ...

    def _compute_curriculum_weights(self) -> np.ndarray:
        """Compute per-sample weights from curriculum phase."""
        ...

    def _compute_joint_weights(self) -> np.ndarray:
        """Compose task and curriculum weights."""
        ...

    def _log_weight_statistics(self, weights: np.ndarray) -> None:
        """Log weight distribution statistics."""
        ...

    def __iter__(self) -> Iterator[int]:
        """Generate sample indices for one epoch."""
        ...

    def __len__(self) -> int:
        """Return number of samples per epoch."""
        ...
```

- **Success**:
  - Module created with complete docstrings
  - Class signature matches design specification
  - Type hints are accurate and complete
  - Imports are minimal and conditional where appropriate

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md (Lines 180-220) - API design

- **Dependencies**:
  - None (skeleton only)

### Task 2.2: Implement task-aware weight computation

Implement the `_compute_task_weights` method using inverse-power scheduling with rarest task selection.

- **Files**:
  - `src/admet/model/chemprop/joint_sampler.py` - Add method implementation

- **Implementation**:
```python
def _compute_task_weights(self) -> np.ndarray:
    """
    Compute per-sample weights from task label counts.

    For each sample, determines its "primary" task (the task with fewest
    labels among those the sample has) and assigns weight based on that
    task's inverse frequency. This ensures multi-label samples are weighted
    by their rarest task contribution.

    Weight formula: w[i] = n_primary_task^(-α)

    Returns
    -------
    np.ndarray
        Array of shape (N,) with task-based weights for each sample.
    """
    if self.task_alpha is None or self.task_alpha == 0:
        return np.ones(len(self.targets))

    # Validate alpha range
    if self.task_alpha < 0 or self.task_alpha > 1:
        logger.warning(
            "task_alpha=%.2f is outside recommended range [0, 1]. "
            "Values > 1 cause extreme over-sampling of rare tasks.",
            self.task_alpha,
        )

    weights = np.zeros(len(self.targets))

    for i in range(len(self.targets)):
        # Find tasks this sample has labels for
        valid_tasks = np.where(~np.isnan(self.targets[i]))[0]

        if len(valid_tasks) == 0:
            weights[i] = 1.0  # Fallback for samples with no labels
            continue

        # Use the rarest task (smallest count) as primary
        task_counts_for_sample = self.task_counts[valid_tasks]
        primary_task_idx = valid_tasks[np.argmin(task_counts_for_sample)]
        primary_task_count = self.task_counts[primary_task_idx]

        # Inverse-power weight
        weights[i] = (primary_task_count + 1e-6) ** (-self.task_alpha)

    return weights
```

- **Success**:
  - Samples with rare tasks get higher weights
  - Alpha=0 produces uniform weights
  - Alpha=1 produces full inverse-proportional weights
  - Multi-label samples use rarest task for weight
  - Handles samples with no labels gracefully
  - Alpha validation warning logged for out-of-range values

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md (Lines 50-65) - Task weight formula

- **Dependencies**:
  - Task 2.1: Class skeleton must exist

### Task 2.3: Implement curriculum-aware weight computation

Implement the `_compute_curriculum_weights` method using phase probabilities.

- **Files**:
  - `src/admet/model/chemprop/joint_sampler.py` - Add method implementation

- **Implementation**:
```python
def _compute_curriculum_weights(self) -> np.ndarray:
    """
    Compute per-sample weights from curriculum phase.

    Reads current phase weights from curriculum_state and maps each
    sample's quality label to its corresponding weight.

    Returns
    -------
    np.ndarray
        Array of shape (N,) with curriculum-based weights for each sample.
    """
    if self.curriculum_state is None:
        return np.ones(len(self.targets))

    probs = self.curriculum_state.sampling_probs()
    weights = np.zeros(len(self.quality_labels), dtype=np.float64)

    for i, label in enumerate(self.quality_labels):
        weights[i] = probs.get(label, 0.0)

    # Handle all-zero weights (unknown labels or edge cases)
    if weights.sum() == 0:
        logger.warning(
            "All curriculum weights are zero. Using uniform sampling. "
            "Check that quality_labels match curriculum qualities."
        )
        return np.ones(len(self.targets))

    return weights
```

- **Success**:
  - Weights match current curriculum phase
  - Unknown quality labels get zero weight (never sampled)
  - Graceful fallback when all weights are zero
  - Logs warning for debugging zero-weight scenarios

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md (Lines 70-90) - Curriculum weight computation

- **Dependencies**:
  - Task 2.1: Class skeleton must exist

### Task 2.4: Implement multiplicative weight composition

Implement the `_compute_joint_weights` method that combines both strategies.

- **Files**:
  - `src/admet/model/chemprop/joint_sampler.py` - Add method implementation

- **Implementation**:
```python
def _compute_joint_weights(self) -> np.ndarray:
    """
    Compose task and curriculum weights via multiplication.

    Joint weight formula:
        w_joint[i] = w_task[i] × w_quality[i]

    The result is normalized to sum to 1 for valid probability distribution.

    Returns
    -------
    np.ndarray
        Normalized joint weights of shape (N,).
    """
    task_weights = self._compute_task_weights()
    curriculum_weights = self._compute_curriculum_weights()

    if self.composition_mode == "multiplicative":
        joint_weights = task_weights * curriculum_weights
    else:
        # Default to multiplicative if unknown mode
        logger.warning(
            f"Unknown composition_mode '{self.composition_mode}', "
            "using multiplicative."
        )
        joint_weights = task_weights * curriculum_weights

    # Handle edge case: all zero weights
    total = joint_weights.sum()
    if total == 0:
        logger.warning(
            "All joint weights are zero after composition. "
            "Falling back to uniform sampling."
        )
        return np.ones(len(self.targets)) / len(self.targets)

    # Normalize to probability distribution
    return joint_weights / total
```

- **Success**:
  - Multiplication correctly combines both weight schemes
  - Result sums to 1.0 (valid probability distribution)
  - Zero-weight edge case handled gracefully
  - Logs informative warnings for debugging

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md (Lines 155-165) - Multiplicative composition

- **Dependencies**:
  - Task 2.2: `_compute_task_weights` must be implemented
  - Task 2.3: `_compute_curriculum_weights` must be implemented

### Task 2.5: Implement __iter__ and __len__ methods

Complete the sampler interface with iteration and length methods.

- **Files**:
  - `src/admet/model/chemprop/joint_sampler.py` - Add method implementations

- **Implementation**:
```python
def __iter__(self) -> Iterator[int]:
    """
    Generate sample indices for one epoch.

    Recomputes weights on each iteration to capture curriculum phase
    changes that may have occurred during training.

    Yields
    ------
    int
        Sample indices according to joint probability distribution.
    """
    # Log phase for debugging
    if self.curriculum_state is not None:
        current_phase = self.curriculum_state.phase
        if self._last_phase is not None and current_phase != self._last_phase:
            logger.info(
                "JointSampler: curriculum phase changed %s -> %s",
                self._last_phase,
                current_phase,
            )
        self._last_phase = current_phase

    # Recompute weights (curriculum weights may have changed)
    weights = self._compute_joint_weights()

    # Sample indices
    if self.seed is not None:
        rng = np.random.default_rng(self.seed)
    else:
        rng = np.random.default_rng()

    indices = rng.choice(
        len(self.targets),
        size=self._num_samples,
        replace=True,
        p=weights,
    )

    return iter(indices.tolist())


def __len__(self) -> int:
    """Return number of samples per epoch."""
    return self._num_samples
```

- **Success**:
  - Iteration yields correct number of samples
  - Weights recomputed each iteration (captures phase changes)
  - Phase transitions are logged for debugging
  - Seed produces reproducible sampling

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md (Lines 100-130) - Existing sampler iteration patterns

- **Dependencies**:
  - Task 2.4: `_compute_joint_weights` must be implemented

---

## Phase 3: Model Integration

### Task 3.1: Update ChempropModel.__init__ for JointSamplingConfig

Add joint_sampling_config parameter and initialization logic.

- **Files**:
  - `src/admet/model/chemprop/model.py` - Update __init__ method

- **Implementation**:
```python
def __init__(
    self,
    # ... existing parameters ...
    joint_sampling_config: JointSamplingConfig | None = None,
) -> None:
    # ... existing initialization ...

    # Joint sampling configuration
    self.joint_sampling_config = joint_sampling_config

    # Initialize curriculum state if joint sampling with curriculum is enabled
    if (
        self.joint_sampling_config is not None
        and self.joint_sampling_config.enabled
        and self.joint_sampling_config.curriculum.enabled
    ):
        if self.curriculum_state is None:
            self.curriculum_state = CurriculumState(
                qualities=list(self.joint_sampling_config.curriculum.qualities),
                patience=self.joint_sampling_config.curriculum.patience,
            )
        # Set quality column for data preparation
        self.quality_col = self.joint_sampling_config.curriculum.quality_col
```

- **Success**:
  - New parameter added without breaking existing signatures
  - Curriculum state initialized correctly when needed
  - Quality column set for data loading

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md (Lines 200-230) - Model integration

- **Dependencies**:
  - Phase 1 complete: JointSamplingConfig must be defined

### Task 3.2: Refactor _prepare_data() sampler selection logic

Update the dataloader creation to use JointSampler when appropriate.

- **Files**:
  - `src/admet/model/chemprop/model.py` - Update _prepare_data method (lines 808-860)

- **Implementation**:
```python
# In _prepare_data(), replace the sampler selection block:

sampler: Any = None
if split == "train":
    # Priority 1: Joint sampling (new unified approach)
    if (
        self.joint_sampling_config is not None
        and self.joint_sampling_config.enabled
    ):
        from admet.model.chemprop.joint_sampler import JointSampler

        task_alpha = (
            self.joint_sampling_config.task_oversampling.alpha
            if self.joint_sampling_config.task_oversampling.enabled
            else None
        )

        sampler = JointSampler(
            targets=ys,
            quality_labels=self._quality_labels.get("train"),
            curriculum_state=self.curriculum_state,
            task_alpha=task_alpha,
            composition_mode=self.joint_sampling_config.composition_mode,
            num_samples=len(datasets[split]),
            seed=self.joint_sampling_config.seed,
        )
        logger.info(
            "Using JointSampler: task_alpha=%s, curriculum=%s, mode=%s",
            task_alpha,
            self.curriculum_state is not None,
            self.joint_sampling_config.composition_mode,
        )

    # Priority 2: Legacy curriculum sampling (backward compatibility)
    elif (
        self.curriculum_state is not None
        and self._quality_labels.get("train") is not None
    ):
        sampler = DynamicCurriculumSampler(
            quality_labels=self._quality_labels["train"],
            curriculum_state=self.curriculum_state,
            num_samples=len(datasets[split]),
            seed=self.hyperparams.seed,
        )
        logger.info("Using legacy DynamicCurriculumSampler")

    # Priority 3: Legacy task sampling (backward compatibility)
    elif self.hyperparams.task_sampling_alpha is not None:
        sampler = TaskAwareSampler(
            targets=ys,
            alpha=self.hyperparams.task_sampling_alpha,
            seed=self.hyperparams.seed,
        )
        logger.info("Using legacy TaskAwareSampler with alpha=%s",
                    self.hyperparams.task_sampling_alpha)
```

- **Success**:
  - JointSampler used when joint_sampling_config.enabled is True
  - Legacy samplers still work when joint_sampling is not enabled
  - Appropriate logging for debugging sampler selection
  - No changes to validation/test dataloader creation

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md (Lines 240-280) - Integration strategy

- **Dependencies**:
  - Phase 2 complete: JointSampler must be implemented
  - Task 3.1: __init__ must accept joint_sampling_config

### Task 3.3: Update from_config() factory method

Update the factory method to pass JointSamplingConfig to __init__.

- **Files**:
  - `src/admet/model/chemprop/model.py` - Update from_config class method

- **Implementation**:
```python
@classmethod
def from_config(cls, config: ChempropConfig) -> "ChempropModel":
    """Create ChempropModel from configuration object."""

    # ... existing parameter extraction ...

    # Extract joint sampling config if present
    joint_sampling_config = getattr(config, 'joint_sampling', None)

    return cls(
        # ... existing parameters ...
        joint_sampling_config=joint_sampling_config,
    )
```

- **Success**:
  - Config-driven model creation works with joint sampling
  - Missing joint_sampling field handled gracefully (None)
  - All existing config-based tests still pass

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md (Lines 280-300) - Factory method

- **Dependencies**:
  - Task 3.1: __init__ must accept joint_sampling_config

---

## Phase 4: Testing

### Task 4.1: Create test fixtures for joint sampling scenarios

Add pytest fixtures for common test scenarios.

- **Files**:
  - `tests/test_joint_sampler.py` - NEW FILE

- **Implementation**:
```python
"""Unit tests for admet.model.chemprop.joint_sampler module."""

import numpy as np
import pytest

from admet.model.chemprop.curriculum import CurriculumState
from admet.model.chemprop.joint_sampler import JointSampler


@pytest.fixture
def balanced_targets() -> np.ndarray:
    """Target array with balanced task labels."""
    # 100 samples, 3 tasks, ~33% labels per task
    targets = np.full((100, 3), np.nan)
    for i in range(100):
        targets[i, i % 3] = float(i % 10)  # Assign to one task each
    return targets


@pytest.fixture
def imbalanced_targets() -> np.ndarray:
    """Target array with highly imbalanced task labels."""
    # 100 samples, 3 tasks: task0=80 labels, task1=15 labels, task2=5 labels
    targets = np.full((100, 3), np.nan)
    # Task 0: samples 0-79
    targets[:80, 0] = np.arange(80) % 10
    # Task 1: samples 80-94
    targets[80:95, 1] = np.arange(15) % 10
    # Task 2: samples 95-99
    targets[95:, 2] = np.arange(5) % 10
    return targets


@pytest.fixture
def quality_labels() -> list[str]:
    """Quality labels for 100 samples."""
    # 50 high, 30 medium, 20 low
    return ["high"] * 50 + ["medium"] * 30 + ["low"] * 20


@pytest.fixture
def curriculum_state() -> CurriculumState:
    """Curriculum state in warmup phase."""
    state = CurriculumState(qualities=["high", "medium", "low"])
    state.phase = "warmup"
    state.weights = state._weights_for_phase("warmup")
    return state
```

- **Success**:
  - Fixtures cover balanced and imbalanced task scenarios
  - Fixtures cover various quality distributions
  - Fixtures are reusable across multiple test functions

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md (Lines 300-320) - Test scenarios

- **Dependencies**:
  - Phase 2 complete: JointSampler must be implemented

### Task 4.2: Test JointSampler weight computation

Test that weights are computed correctly for various scenarios.

- **Files**:
  - `tests/test_joint_sampler.py` - Add test functions

- **Implementation**:
```python
class TestJointSamplerWeights:
    """Tests for weight computation methods."""

    def test_task_weights_uniform_with_alpha_zero(
        self, balanced_targets: np.ndarray
    ) -> None:
        """Alpha=0 should produce uniform task weights."""
        sampler = JointSampler(
            targets=balanced_targets,
            task_alpha=0.0,
            seed=42,
        )
        weights = sampler._compute_task_weights()
        np.testing.assert_array_almost_equal(
            weights, np.ones(len(balanced_targets))
        )

    def test_task_weights_favor_rare_tasks(
        self, imbalanced_targets: np.ndarray
    ) -> None:
        """Higher alpha should give rare tasks higher weights."""
        sampler = JointSampler(
            targets=imbalanced_targets,
            task_alpha=0.5,
            seed=42,
        )
        weights = sampler._compute_task_weights()

        # Samples for task 2 (5 labels) should have higher weight than task 0 (80 labels)
        task2_weights = weights[95:]  # Last 5 samples
        task0_weights = weights[:80]  # First 80 samples

        assert task2_weights.mean() > task0_weights.mean()

    def test_curriculum_weights_match_phase(
        self,
        balanced_targets: np.ndarray,
        quality_labels: list[str],
        curriculum_state: CurriculumState,
    ) -> None:
        """Curriculum weights should match phase probabilities."""
        sampler = JointSampler(
            targets=balanced_targets,
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            seed=42,
        )
        weights = sampler._compute_curriculum_weights()

        # In warmup: high=0.9, medium=0.1, low=0.0
        # First 50 samples are "high"
        assert weights[:50].mean() == pytest.approx(0.9, rel=1e-6)
        # Next 30 are "medium"
        assert weights[50:80].mean() == pytest.approx(0.1, rel=1e-6)
        # Last 20 are "low"
        assert weights[80:].mean() == pytest.approx(0.0, rel=1e-6)

    def test_joint_weights_are_normalized(
        self,
        imbalanced_targets: np.ndarray,
        quality_labels: list[str],
        curriculum_state: CurriculumState,
    ) -> None:
        """Joint weights should sum to 1."""
        sampler = JointSampler(
            targets=imbalanced_targets,
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            task_alpha=0.3,
            seed=42,
        )
        weights = sampler._compute_joint_weights()

        assert weights.sum() == pytest.approx(1.0, rel=1e-9)
        assert (weights >= 0).all()
```

- **Success**:
  - All weight computation tests pass
  - Edge cases (alpha=0, all zero weights) are covered
  - Normalization is verified

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md (Lines 50-90) - Weight formulas

- **Dependencies**:
  - Task 4.1: Fixtures must be defined

### Task 4.3: Test curriculum phase transitions with joint sampling

Test that phase changes correctly update joint weights.

- **Files**:
  - `tests/test_joint_sampler.py` - Add test functions

- **Implementation**:
```python
class TestJointSamplerPhaseTransitions:
    """Tests for curriculum phase transition handling."""

    def test_weights_change_with_phase(
        self,
        imbalanced_targets: np.ndarray,
        quality_labels: list[str],
    ) -> None:
        """Joint weights should change when curriculum phase advances."""
        state = CurriculumState(qualities=["high", "medium", "low"])

        sampler = JointSampler(
            targets=imbalanced_targets,
            quality_labels=quality_labels,
            curriculum_state=state,
            task_alpha=0.3,
            seed=42,
        )

        # Get weights in warmup phase
        state.phase = "warmup"
        state.weights = state._weights_for_phase("warmup")
        warmup_weights = sampler._compute_joint_weights()

        # Get weights in robust phase
        state.phase = "robust"
        state.weights = state._weights_for_phase("robust")
        robust_weights = sampler._compute_joint_weights()

        # Weights should differ (low quality now has non-zero weight)
        assert not np.allclose(warmup_weights, robust_weights)

        # In robust phase, low quality samples should have non-zero weight
        low_quality_indices = [i for i, q in enumerate(quality_labels) if q == "low"]
        assert robust_weights[low_quality_indices].sum() > 0

    def test_iteration_recomputes_weights(
        self,
        balanced_targets: np.ndarray,
        quality_labels: list[str],
    ) -> None:
        """Each iteration should recompute weights to capture phase changes."""
        state = CurriculumState(qualities=["high", "medium", "low"])
        state.phase = "warmup"

        sampler = JointSampler(
            targets=balanced_targets,
            quality_labels=quality_labels,
            curriculum_state=state,
            task_alpha=0.3,
            num_samples=50,
            seed=42,
        )

        # First iteration in warmup
        indices_warmup = list(sampler)

        # Change phase
        state.phase = "robust"
        state.weights = state._weights_for_phase("robust")

        # Second iteration should reflect phase change
        indices_robust = list(sampler)

        # Distributions should be different (though may overlap due to randomness)
        # Count low-quality samples in each
        low_indices = set(i for i, q in enumerate(quality_labels) if q == "low")

        low_in_warmup = sum(1 for i in indices_warmup if i in low_indices)
        low_in_robust = sum(1 for i in indices_robust if i in low_indices)

        # Robust phase should sample more low-quality
        assert low_in_robust > low_in_warmup
```

- **Success**:
  - Phase transitions produce different weight distributions
  - Low-quality samples only sampled in appropriate phases
  - Iteration correctly reflects current phase

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md (Lines 120-140) - Phase transition behavior

- **Dependencies**:
  - Task 4.1: Fixtures must be defined
  - Task 4.2: Basic weight tests should pass

### Task 4.4: Test backward compatibility scenarios

Test that existing configurations continue to work.

- **Files**:
  - `tests/test_joint_sampler.py` - Add test functions

- **Implementation**:
```python
class TestBackwardCompatibility:
    """Tests for backward compatibility with legacy configurations."""

    def test_task_only_sampling(self, imbalanced_targets: np.ndarray) -> None:
        """JointSampler with only task_alpha should behave like TaskAwareSampler."""
        from admet.model.chemprop.task_sampler import TaskAwareSampler

        joint = JointSampler(
            targets=imbalanced_targets,
            task_alpha=0.5,
            num_samples=1000,
            seed=42,
        )

        legacy = TaskAwareSampler(
            targets=imbalanced_targets,
            alpha=0.5,
            num_samples=1000,
            seed=42,
        )

        # Sample distributions should be similar
        joint_indices = list(joint)
        legacy_indices = list(legacy)

        # Count samples per task range
        def count_by_task(indices):
            task0 = sum(1 for i in indices if i < 80)
            task1 = sum(1 for i in indices if 80 <= i < 95)
            task2 = sum(1 for i in indices if i >= 95)
            return task0, task1, task2

        joint_counts = count_by_task(joint_indices)
        legacy_counts = count_by_task(legacy_indices)

        # Should have similar distribution (within statistical variance)
        for j, l in zip(joint_counts, legacy_counts):
            assert abs(j - l) < 100  # Allow some variance

    def test_curriculum_only_sampling(
        self,
        balanced_targets: np.ndarray,
        quality_labels: list[str],
        curriculum_state: CurriculumState,
    ) -> None:
        """JointSampler with only curriculum should behave like DynamicCurriculumSampler."""
        from admet.model.chemprop.curriculum_sampler import DynamicCurriculumSampler

        joint = JointSampler(
            targets=balanced_targets,
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            num_samples=1000,
            seed=42,
        )

        legacy = DynamicCurriculumSampler(
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            num_samples=1000,
            seed=42,
        )

        joint_indices = list(joint)
        legacy_indices = list(legacy)

        # Count by quality
        def count_by_quality(indices):
            high = sum(1 for i in indices if quality_labels[i] == "high")
            med = sum(1 for i in indices if quality_labels[i] == "medium")
            low = sum(1 for i in indices if quality_labels[i] == "low")
            return high, med, low

        joint_counts = count_by_quality(joint_indices)
        legacy_counts = count_by_quality(legacy_indices)

        # Similar distribution
        for j, l in zip(joint_counts, legacy_counts):
            assert abs(j - l) < 100
```

- **Success**:
  - Task-only mode produces similar distribution to TaskAwareSampler
  - Curriculum-only mode produces similar distribution to DynamicCurriculumSampler
  - No regressions in existing behavior

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md (Lines 180-200) - Backward compatibility requirements

- **Dependencies**:
  - Task 4.1: Fixtures must be defined

### Task 4.5: Test edge cases and error handling

Test error conditions and edge cases.

- **Files**:
  - `tests/test_joint_sampler.py` - Add test functions

- **Implementation**:
```python
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_error_without_any_strategy(self, balanced_targets: np.ndarray) -> None:
        """Should raise error if neither task_alpha nor curriculum_state provided."""
        with pytest.raises(ValueError, match="at least one sampling strategy"):
            JointSampler(targets=balanced_targets)

    def test_error_curriculum_without_quality_labels(
        self,
        balanced_targets: np.ndarray,
        curriculum_state: CurriculumState,
    ) -> None:
        """Should raise error if curriculum_state but no quality_labels."""
        with pytest.raises(ValueError, match="quality_labels required"):
            JointSampler(
                targets=balanced_targets,
                curriculum_state=curriculum_state,
            )

    def test_unknown_quality_labels_get_zero_weight(
        self,
        balanced_targets: np.ndarray,
        curriculum_state: CurriculumState,
    ) -> None:
        """Unknown quality labels should have zero sampling probability."""
        quality_labels = ["high"] * 50 + ["unknown"] * 50

        with pytest.warns(UserWarning, match="unknown"):
            sampler = JointSampler(
                targets=balanced_targets,
                quality_labels=quality_labels,
                curriculum_state=curriculum_state,
                seed=42,
            )

        # Only "high" samples should be sampled
        indices = list(sampler)
        for i in indices:
            assert quality_labels[i] == "high"

    def test_all_samples_missing_labels(self) -> None:
        """Should handle targets with all NaN gracefully."""
        targets = np.full((10, 3), np.nan)

        sampler = JointSampler(
            targets=targets,
            task_alpha=0.5,
            seed=42,
        )

        # Should fall back to uniform sampling
        indices = list(sampler)
        assert len(indices) == len(targets)

    def test_reproducibility_with_seed(
        self,
        imbalanced_targets: np.ndarray,
        quality_labels: list[str],
        curriculum_state: CurriculumState,
    ) -> None:
        """Same seed should produce identical sampling."""
        sampler1 = JointSampler(
            targets=imbalanced_targets,
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            task_alpha=0.3,
            num_samples=100,
            seed=12345,
        )

        sampler2 = JointSampler(
            targets=imbalanced_targets,
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            task_alpha=0.3,
            num_samples=100,
            seed=12345,
        )

        assert list(sampler1) == list(sampler2)

    def test_seed_increment_per_epoch_changes_sampling(
        self,
        imbalanced_targets: np.ndarray,
    ) -> None:
        """increment_seed_per_epoch=True should produce different samples each epoch."""
        sampler = JointSampler(
            targets=imbalanced_targets,
            task_alpha=0.5,
            num_samples=100,
            seed=42,
            increment_seed_per_epoch=True,
        )

        # Epoch 0
        sampler.set_epoch(0)
        epoch0_indices = list(sampler)

        # Epoch 1
        sampler.set_epoch(1)
        epoch1_indices = list(sampler)

        # Epoch 2
        sampler.set_epoch(2)
        epoch2_indices = list(sampler)

        # Different epochs should produce different orderings
        assert epoch0_indices != epoch1_indices
        assert epoch1_indices != epoch2_indices
        assert epoch0_indices != epoch2_indices

    def test_seed_increment_disabled_same_sampling(
        self,
        imbalanced_targets: np.ndarray,
    ) -> None:
        """increment_seed_per_epoch=False should produce same samples each epoch."""
        sampler = JointSampler(
            targets=imbalanced_targets,
            task_alpha=0.5,
            num_samples=100,
            seed=42,
            increment_seed_per_epoch=False,
        )

        sampler.set_epoch(0)
        epoch0_indices = list(sampler)

        sampler.set_epoch(1)
        epoch1_indices = list(sampler)

        # Same seed = same ordering
        assert epoch0_indices == epoch1_indices

    def test_alpha_out_of_range_warning(
        self,
        balanced_targets: np.ndarray,
    ) -> None:
        """Alpha outside [0, 1] should trigger warning."""
        with pytest.warns(UserWarning, match="outside recommended range"):
            JointSampler(
                targets=balanced_targets,
                task_alpha=1.5,
                seed=42,
            )

    def test_multi_label_sample_uses_rarest_task(self) -> None:
        """Multi-label samples should use rarest task for weight computation."""
        # Create targets where some samples have multiple task labels
        # Task 0: 100 labels, Task 1: 10 labels
        targets = np.full((110, 2), np.nan)
        targets[:100, 0] = np.arange(100)  # Task 0: samples 0-99
        targets[100:, 1] = np.arange(10)   # Task 1: samples 100-109

        # Add multi-label sample: has labels for both tasks
        targets[0, 1] = 0.5  # Sample 0 now has both task 0 and task 1 labels

        sampler = JointSampler(
            targets=targets,
            task_alpha=0.5,
            seed=42,
        )

        weights = sampler._compute_task_weights()

        # Sample 0 has labels for both tasks (100 labels vs 10 labels)
        # Should use rarest task (task 1 with 10 labels) for weight
        sample_0_weight = weights[0]

        # Compare to pure task 0 sample (sample 1)
        sample_1_weight = weights[1]

        # Sample 0 should have HIGHER weight because it uses rarest task
        assert sample_0_weight > sample_1_weight
```

- **Success**:
  - Appropriate errors raised for invalid configurations
  - Unknown quality labels handled with warning
  - All-NaN targets don't crash
  - Seeded sampling is reproducible
  - Seed increment per epoch produces varied sampling
  - Seed increment disabled produces deterministic sampling
  - Alpha validation warns for out-of-range values
  - Multi-label samples correctly use rarest task

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md (Lines 250-280) - Error handling

- **Dependencies**:
  - Task 4.1: Fixtures must be defined

### Task 4.6: Test weight statistics logging

Test that weight statistics are correctly computed and logged.

- **Files**:
  - `tests/test_joint_sampler.py` - Add test functions

- **Implementation**:
```python
class TestWeightStatisticsLogging:
    """Tests for weight statistics logging functionality."""

    def test_log_weight_statistics_computes_metrics(
        self,
        imbalanced_targets: np.ndarray,
        quality_labels: list[str],
        curriculum_state: CurriculumState,
    ) -> None:
        """_log_weight_statistics should compute correct metrics."""
        sampler = JointSampler(
            targets=imbalanced_targets,
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            task_alpha=0.3,
            seed=42,
            log_weight_statistics=True,
        )

        weights = sampler._compute_joint_weights()

        # Verify weights are valid
        assert weights.min() >= 0
        assert weights.max() <= 1.0
        assert weights.sum() == pytest.approx(1.0, rel=1e-9)

        # Verify effective samples calculation
        w_sum = weights.sum()
        expected_eff_samples = (w_sum ** 2) / np.sum(weights ** 2)

        # Effective samples should be > 0 and <= num samples
        assert expected_eff_samples > 0
        assert expected_eff_samples <= len(weights)

    def test_log_weight_statistics_captures_logger_output(
        self,
        imbalanced_targets: np.ndarray,
        caplog,
    ) -> None:
        """Logger should receive weight statistics."""
        import logging

        sampler = JointSampler(
            targets=imbalanced_targets,
            task_alpha=0.5,
            seed=42,
            log_weight_statistics=True,
        )

        with caplog.at_level(logging.INFO, logger="admet.model.chemprop.joint_sampler"):
            weights = sampler._compute_joint_weights()
            sampler._log_weight_statistics(weights)

        # Check log message contains expected metrics
        assert "Sampling weights" in caplog.text
        assert "entropy" in caplog.text
        assert "effective_samples" in caplog.text

    def test_mlflow_logging_graceful_when_not_active(
        self,
        imbalanced_targets: np.ndarray,
    ) -> None:
        """MLflow logging should not fail when no active run."""
        sampler = JointSampler(
            targets=imbalanced_targets,
            task_alpha=0.5,
            seed=42,
            log_weight_statistics=True,
            log_to_mlflow=True,  # Enabled but no active run
        )

        weights = sampler._compute_joint_weights()
        # Should not raise
        sampler._log_weight_statistics(weights)
```

- **Success**:
  - Weight statistics computed correctly
  - Logger receives expected output
  - MLflow logging graceful when not active
  - Metrics include entropy and effective samples

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md - Logging requirements

- **Dependencies**:
  - Task 4.1: Fixtures must be defined
  - Task 2.4: Weight computation must work

---

## Phase 5: Documentation and Configuration

### Task 5.1: Create example YAML configuration file

Create a complete example configuration demonstrating joint sampling.

- **Files**:
  - `configs/0-experiment/ensemble_joint_sampling.yaml` - NEW FILE

- **Implementation**:
```yaml
# Example configuration for joint task + curriculum sampling
# This config demonstrates the unified sampling approach

data:
  data_dir: "assets/dataset/splits/scaffold/v1/high/cluster/split_0.6_0.2_0.2/split_0/fold_0"
  smiles_col: "SMILES"
  target_cols:
    - "Solubility"
    - "Lipophilicity"
    - "PAMPA"
    - "hERG"
    - "Clearance"
  target_weights: []

model:
  depth: 5
  message_hidden_dim: 600
  num_layers: 2
  hidden_dim: 600
  dropout: 0.25
  batch_norm: true
  ffn_type: "regression"

optimization:
  criterion: "MAE"
  init_lr: 1.04e-5
  max_lr: 1.04e-4
  final_lr: 2.09e-6
  warmup_epochs: 5
  max_epochs: 150
  patience: 15
  batch_size: 128
  num_workers: 0
  seed: 12345
  progress_bar: false
  # Legacy field - ignored when joint_sampling.enabled=true
  task_sampling_alpha: null

mlflow:
  tracking: true
  tracking_uri: "http://127.0.0.1:8084"
  experiment_name: "joint_sampling_experiment"

# ============================================================================
# Joint Sampling Configuration (NEW)
# ============================================================================
# Combines task-aware oversampling with quality-aware curriculum learning
# for optimal training on imbalanced, multi-quality datasets.
joint_sampling:
  enabled: true

  # Task Oversampling: Rebalance sampling across tasks with different label counts
  task_oversampling:
    enabled: true
    alpha: 0.3  # Moderate rebalancing (0=uniform, 1=full inverse)

  # Curriculum Learning: Progress from high to low quality data
  curriculum:
    enabled: true
    quality_col: "Quality"
    qualities:
      - "high"
      - "medium"
      - "low"
    patience: 5
    strategy: "sampled"
    reset_early_stopping_on_phase_change: false
    log_per_quality_metrics: true

  # How to combine task and curriculum weights
  composition_mode: "multiplicative"  # w_joint = w_task × w_quality

  seed: 42

# Legacy curriculum config - ignored when joint_sampling.enabled=true
curriculum:
  enabled: false
```

- **Success**:
  - Valid YAML that loads with OmegaConf
  - All fields documented with comments
  - Demonstrates recommended settings
  - Explains relationship to legacy configs

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md (Lines 145-175) - Config structure

- **Dependencies**:
  - Phase 1 complete: Config schema must be defined

### Task 5.2: Update module docstrings and API documentation

Ensure all public APIs have complete documentation.

- **Files**:
  - `src/admet/model/chemprop/joint_sampler.py` - Update docstrings
  - `src/admet/model/chemprop/config.py` - Update docstrings

- **Implementation**:
Ensure all classes and public methods have:
- One-line summary
- Extended description where needed
- Parameters section with types and descriptions
- Returns section with types and descriptions
- Raises section listing exceptions
- Examples section with working code
- Notes section for important caveats

- **Success**:
  - All public classes/methods have complete docstrings
  - Examples are executable and correct
  - Parameter types match actual implementation

- **Research References**:
  - #file:../.github/instructions/python.instructions.md - Documentation standards

- **Dependencies**:
  - Phase 2 complete: Implementation must be finished

### Task 5.3: Add usage examples to module docstring

Add comprehensive examples to the module-level docstring.

- **Files**:
  - `src/admet/model/chemprop/joint_sampler.py` - Update module docstring

- **Implementation**:
```python
"""
Joint sampler for combined task-aware and curriculum-aware sampling.

...existing description...

Examples
--------
Basic usage with both strategies:

>>> import numpy as np
>>> from admet.model.chemprop.joint_sampler import JointSampler
>>> from admet.model.chemprop.curriculum import CurriculumState
>>>
>>> # Create sample data
>>> targets = np.random.randn(100, 5)  # 100 samples, 5 tasks
>>> targets[targets < 0] = np.nan  # ~50% missing labels
>>> quality_labels = ["high"] * 40 + ["medium"] * 35 + ["low"] * 25
>>>
>>> # Initialize curriculum
>>> state = CurriculumState(qualities=["high", "medium", "low"])
>>>
>>> # Create joint sampler
>>> sampler = JointSampler(
...     targets=targets,
...     quality_labels=quality_labels,
...     curriculum_state=state,
...     task_alpha=0.3,
...     seed=42,
... )
>>>
>>> # Use with PyTorch DataLoader
>>> from torch.utils.data import DataLoader
>>> loader = DataLoader(dataset, sampler=sampler, batch_size=32)

Task-only sampling (no curriculum):

>>> sampler = JointSampler(targets=targets, task_alpha=0.5, seed=42)

Curriculum-only sampling (no task rebalancing):

>>> sampler = JointSampler(
...     targets=targets,
...     quality_labels=quality_labels,
...     curriculum_state=state,
...     seed=42,
... )
"""
```

- **Success**:
  - Examples cover all major use cases
  - Examples are syntactically correct
  - Examples demonstrate integration with DataLoader

- **Research References**:
  - #file:../.github/instructions/python.instructions.md - Example formatting

- **Dependencies**:
  - Task 5.2: Base documentation must be complete

---

## Phase 6: HPO Integration and MLflow Logging

### Task 6.1: Add JointSampling parameters to HPO search space

Add the new JointSampling configuration parameters to the hyperparameter optimization search space.

- **Files**:
  - `src/admet/hpo/search_space.py` - Add joint sampling parameters (or equivalent HPO config file)

- **Implementation**:
```python
# Add to the hyperparameter search space definition
joint_sampling_search_space = {
    # Task oversampling parameters
    "joint_sampling.task_oversampling.enabled": tune.choice([True, False]),
    "joint_sampling.task_oversampling.alpha": tune.uniform(0.0, 1.0),

    # Curriculum parameters (if not already present)
    "joint_sampling.curriculum.enabled": tune.choice([True, False]),
    "joint_sampling.curriculum.warmup_epochs": tune.choice([3, 5, 7, 10]),
    "joint_sampling.curriculum.patience": tune.choice([2, 3, 5]),

    # Seed behavior
    "joint_sampling.increment_seed_per_epoch": tune.choice([True, False]),
}

# Combined search space for experiments
def get_joint_sampling_space(include_curriculum: bool = True) -> dict:
    """
    Get HPO search space for joint sampling parameters.

    Parameters
    ----------
    include_curriculum : bool, default=True
        Whether to include curriculum-specific parameters.

    Returns
    -------
    dict
        Ray Tune search space configuration.
    """
    space = {
        "joint_sampling.enabled": tune.choice([True, False]),
        "joint_sampling.task_oversampling.enabled": tune.choice([True, False]),
        "joint_sampling.task_oversampling.alpha": tune.uniform(0.0, 1.0),
        "joint_sampling.seed": tune.randint(0, 10000),
        "joint_sampling.increment_seed_per_epoch": tune.choice([True, False]),
    }

    if include_curriculum:
        space.update({
            "joint_sampling.curriculum.enabled": tune.choice([True, False]),
            "joint_sampling.curriculum.warmup_epochs": tune.choice([3, 5, 7, 10]),
            "joint_sampling.curriculum.patience": tune.choice([2, 3, 5]),
            "joint_sampling.curriculum.quality_weights.high": tune.uniform(0.5, 1.0),
            "joint_sampling.curriculum.quality_weights.medium": tune.uniform(0.3, 0.8),
            "joint_sampling.curriculum.quality_weights.low": tune.uniform(0.0, 0.5),
        })

    return space
```

- **Success**:
  - HPO can search over alpha values in [0, 1]
  - HPO can toggle task oversampling and curriculum independently
  - HPO can search seed increment behavior
  - Search space integrates with existing Ray Tune configuration

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md - HPO integration requirements

- **Dependencies**:
  - Phase 1: JointSamplingConfig must exist with all parameters
  - Existing HPO infrastructure (Ray Tune integration)

### Task 6.2: Implement MLflow weight statistics logging

Implement logging of sampling weight statistics to MLflow for experiment tracking.

- **Files**:
  - `src/admet/model/chemprop/joint_sampler.py` - Add `_log_weight_statistics` implementation
  - `src/admet/model/chemprop/model.py` - Add MLflow integration in training loop

- **Implementation** (joint_sampler.py):
```python
def _log_weight_statistics(self, weights: np.ndarray) -> None:
    """
    Log weight distribution statistics.

    Logs to both local logger and MLflow (if enabled and available).

    Metrics logged:
    - weight_min, weight_max: Range of weights
    - weight_mean, weight_std: Central tendency and spread
    - weight_entropy: Shannon entropy of normalized weights (higher = more uniform)
    - effective_samples: Effective sample size = (sum(w))^2 / sum(w^2)

    Parameters
    ----------
    weights : np.ndarray
        Array of sampling weights.
    """
    if len(weights) == 0:
        return

    # Basic statistics
    w_min, w_max = float(weights.min()), float(weights.max())
    w_mean, w_std = float(weights.mean()), float(weights.std())

    # Normalize to probability distribution
    w_sum = weights.sum()
    if w_sum > 0:
        p = weights / w_sum
        # Shannon entropy (higher = more uniform)
        entropy = float(-np.sum(p * np.log(p + 1e-10)))
        # Effective sample size
        eff_samples = float((w_sum ** 2) / np.sum(weights ** 2))
    else:
        entropy = 0.0
        eff_samples = 0.0

    stats = {
        "sampler/weight_min": w_min,
        "sampler/weight_max": w_max,
        "sampler/weight_mean": w_mean,
        "sampler/weight_std": w_std,
        "sampler/weight_entropy": entropy,
        "sampler/effective_samples": eff_samples,
        "sampler/actual_samples": len(weights),
        "sampler/epoch": self._epoch,
    }

    # Add task and curriculum info if available
    if self.task_alpha is not None:
        stats["sampler/task_alpha"] = self.task_alpha
    if self.curriculum_state is not None:
        stats["sampler/curriculum_phase"] = self.curriculum_state.current_phase.value

    logger.info(
        "Sampling weights: min=%.4f, max=%.4f, entropy=%.4f, effective_samples=%.1f",
        w_min, w_max, entropy, eff_samples,
    )

    # Log to MLflow if enabled and available
    if self._log_to_mlflow:
        try:
            import mlflow
            if mlflow.active_run():
                mlflow.log_metrics(stats, step=self._epoch)
        except ImportError:
            pass  # MLflow not installed
        except Exception as e:
            logger.debug("Failed to log to MLflow: %s", e)
```

- **Implementation** (model.py integration):
```python
# In the training epoch callback or on_train_epoch_start:
def on_train_epoch_start(self) -> None:
    """Called at the start of each training epoch."""
    if hasattr(self, '_joint_sampler') and self._joint_sampler is not None:
        # Update epoch for seed incrementing
        self._joint_sampler.set_epoch(self.current_epoch)

        # Trigger weight statistics logging
        if self._joint_sampler._log_weight_statistics:
            weights = self._joint_sampler._compute_joint_weights()
            self._joint_sampler._log_weight_statistics(weights)
```

- **Success**:
  - Weight statistics appear in MLflow metrics UI
  - Metrics are logged at each epoch with correct step number
  - Entropy metric shows sampling uniformity (higher = more uniform)
  - Effective samples metric shows how many "independent" samples the weights represent
  - Graceful handling when MLflow not installed or not active

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md - Logging requirements

- **Dependencies**:
  - Task 2.4: Weight computation must work
  - MLflow dependency (optional, gracefully handled)

---

## Phase 7: Configuration Migration

### Task 7.1: Create migration script for YAML config files

Create a Python script to automatically migrate existing config files to the new joint_sampling schema.

- **Files**:
  - `scripts/lib/migrate_sampling_configs.py` - NEW FILE

- **Implementation**:
```python
"""
Migrate YAML config files from legacy sampling schema to joint_sampling.

This script migrates:
- optimization.task_sampling_alpha → joint_sampling.task_oversampling.alpha
- curriculum.enabled → joint_sampling.curriculum.enabled
- curriculum.* → joint_sampling.curriculum.*

Usage:
    python scripts/lib/migrate_sampling_configs.py --config-dir configs/0-experiment
    python scripts/lib/migrate_sampling_configs.py --config-file configs/0-experiment/chemprop.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def migrate_config(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """
    Migrate a config dict from legacy to joint_sampling schema.

    Parameters
    ----------
    config : dict
        Original config dictionary.

    Returns
    -------
    tuple[dict, bool]
        (migrated_config, was_modified)
    """
    was_modified = False
    migrated = config.copy()

    # Extract legacy fields
    task_alpha = migrated.get("optimization", {}).get("task_sampling_alpha")
    curriculum_config = migrated.get("curriculum")

    # Check if migration needed
    if task_alpha is None and (curriculum_config is None or not curriculum_config.get("enabled")):
        return migrated, False  # Nothing to migrate

    # Create joint_sampling config
    joint_sampling: dict[str, Any] = {"enabled": False}

    # Migrate task_sampling_alpha
    if task_alpha is not None:
        joint_sampling["enabled"] = True
        joint_sampling["task_oversampling"] = {
            "enabled": True,
            "alpha": task_alpha,
        }
        # Remove from optimization
        if "optimization" in migrated and "task_sampling_alpha" in migrated["optimization"]:
            del migrated["optimization"]["task_sampling_alpha"]
            was_modified = True
            logger.info(f"Migrated task_sampling_alpha={task_alpha}")

    # Migrate curriculum config
    if curriculum_config and curriculum_config.get("enabled"):
        joint_sampling["enabled"] = True
        joint_sampling["curriculum"] = curriculum_config.copy()
        # Keep curriculum at top level for backward compatibility initially
        # Users can remove it after verifying migration
        was_modified = True
        logger.info("Migrated curriculum config to joint_sampling")

    # Add joint_sampling to config
    if joint_sampling["enabled"]:
        migrated["joint_sampling"] = joint_sampling
        was_modified = True

    return migrated, was_modified


def migrate_file(
    config_path: Path,
    dry_run: bool = False,
    backup: bool = True,
) -> None:
    """
    Migrate a single YAML config file.

    Parameters
    ----------
    config_path : Path
        Path to config file.
    dry_run : bool, default=False
        If True, only print changes without writing.
    backup : bool, default=True
        If True, create .bak backup before overwriting.
    """
    logger.info(f"Processing {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    migrated, was_modified = migrate_config(config)

    if not was_modified:
        logger.info(f"  No changes needed")
        return

    if dry_run:
        logger.info(f"  [DRY RUN] Would migrate:")
        logger.info(yaml.dump({"joint_sampling": migrated.get("joint_sampling")}, default_flow_style=False))
        return

    # Backup
    if backup:
        backup_path = config_path.with_suffix(".yaml.bak")
        backup_path.write_text(config_path.read_text())
        logger.info(f"  Created backup: {backup_path}")

    # Write migrated config
    with open(config_path, "w") as f:
        yaml.dump(migrated, f, default_flow_style=False, sort_keys=False)

    logger.info(f"  Migrated successfully")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate sampling configs to joint_sampling schema")
    parser.add_argument("--config-file", type=Path, help="Single config file to migrate")
    parser.add_argument("--config-dir", type=Path, help="Directory of config files to migrate")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup creation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if args.config_file:
        migrate_file(args.config_file, dry_run=args.dry_run, backup=not args.no_backup)
    elif args.config_dir:
        for config_path in sorted(args.config_dir.rglob("*.yaml")):
            migrate_file(config_path, dry_run=args.dry_run, backup=not args.no_backup)
    else:
        parser.error("Must specify --config-file or --config-dir")


if __name__ == "__main__":
    main()
```

- **Success**:
  - Script correctly identifies legacy fields
  - Migrates task_sampling_alpha to joint_sampling.task_oversampling.alpha
  - Migrates curriculum config to joint_sampling.curriculum
  - Creates backups before overwriting
  - Dry-run mode shows what would change
  - Handles files with no migration needed

- **Research References**:
  - #file:../research/20251216-sampler-harmonization-research.md - Config structure

- **Dependencies**:
  - Phase 1: JointSamplingConfig schema must be defined

### Task 7.2: Migrate all configs in 0-experiment/ directory

Run migration script on experiment configs.

- **Files**:
  - `configs/0-experiment/chemprop.yaml` - UPDATE
  - `configs/0-experiment/ensemble_chemprop_production.yaml` - UPDATE

- **Command**:
```bash
python scripts/lib/migrate_sampling_configs.py --config-dir configs/0-experiment
```

- **Expected Changes**:
  - `ensemble_chemprop_production.yaml`: `task_sampling_alpha: 0.1` → `joint_sampling.task_oversampling.alpha: 0.1`
  - `chemprop.yaml`: `task_sampling_alpha: 0.5` → `joint_sampling.task_oversampling.alpha: 0.5`
  - All curriculum configs nested under `joint_sampling.curriculum`

- **Success**:
  - All files migrated successfully
  - Backups created (.yaml.bak)
  - Configs validate with OmegaConf
  - Training runs work with migrated configs

- **Dependencies**:
  - Task 7.1: Migration script must exist

### Task 7.3: Migrate all configs in 2-hpo-ensemble/ directory

Run migration script on HPO ensemble configs (100 files).

- **Files**:
  - `configs/2-hpo-ensemble/ensemble_chemprop_hpo_*.yaml` - UPDATE 100 files

- **Command**:
```bash
python scripts/lib/migrate_sampling_configs.py --config-dir configs/2-hpo-ensemble --verbose
```

- **Expected Changes**:
  - Migrate varying `task_sampling_alpha` values (0.0 to 1.0)
  - Migrate curriculum configs where enabled
  - Create 100 backup files

- **Success**:
  - All 100 files processed
  - Different alpha values correctly migrated
  - HPO sweep still covers expected search space
  - No validation errors

- **Dependencies**:
  - Task 7.1: Migration script must exist

### Task 7.4: Migrate all configs in 3-production/ directory

Run migration script on production configs.

- **Files**:
  - `configs/3-production/ensemble_chemprop_hpo_*.yaml` - UPDATE multiple files

- **Command**:
```bash
python scripts/lib/migrate_sampling_configs.py --config-dir configs/3-production
```

- **Expected Changes**:
  - Migrate production-tuned alpha values
  - Preserve all other production settings

- **Success**:
  - Production configs migrated
  - No functionality regressions
  - Configs load correctly

- **Dependencies**:
  - Task 7.1: Migration script must exist

### Task 7.5: Update config documentation and comments

Update YAML comments to reflect new schema.

- **Files**:
  - `configs/0-experiment/ensemble_chemprop_production.yaml` - UPDATE comments

- **Implementation**:
```yaml
# Joint Sampling Configuration
# ============================
# Unified configuration for task-aware oversampling and curriculum learning.
# Both strategies can be enabled simultaneously with multiplicative weight composition.
joint_sampling:
  enabled: true

  # Task oversampling: Rebalance sampling for imbalanced multi-task data
  task_oversampling:
    enabled: true
    alpha: 0.1  # Power law exponent (0=uniform, 1=full inverse proportional)

  # Curriculum learning: Quality-aware progressive training
  curriculum:
    enabled: false
    quality_col: "Quality"
    qualities: ["high", "medium", "low"]
    patience: 5
    strategy: "sampled"
    reset_early_stopping_on_phase_change: false
    log_per_quality_metrics: true

  # Seed management
  seed: 42
  increment_seed_per_epoch: true  # Vary sampling order across epochs

  # Logging options
  log_weight_statistics: true
  log_to_mlflow: true

# Legacy curriculum config (deprecated, use joint_sampling.curriculum instead)
# This field is kept for backward compatibility but will be removed in future versions
curriculum:
  enabled: false
  quality_col: "Quality"
  qualities: ["high", "medium", "low"]
```

- **Success**:
  - Comments explain new schema
  - Legacy fields marked as deprecated
  - Examples show both task and curriculum enabled
  - Migration path documented

- **Dependencies**:
  - Task 7.2, 7.3, 7.4: Configs must be migrated

---

## Dependencies Summary

- Python 3.10+
- NumPy >= 1.20
- PyTorch >= 2.0
- OmegaConf >= 2.3
- pytest >= 7.0
- Ray Tune >= 2.0 (for HPO)
- MLflow >= 2.0 (optional, for logging)
- PyYAML >= 6.0 (for config migration)
- Existing modules: `curriculum.py`, `curriculum_sampler.py`, `task_sampler.py`

## Success Criteria Summary

1. **Correctness**: JointSampler produces valid probability distributions
2. **Composition**: Multiplicative weights correctly combine both strategies
3. **Dynamic Updates**: Curriculum phase changes reflected in sampling
4. **Task Rebalancing**: Rare tasks sampled more frequently with alpha > 0
5. **Multi-label Handling**: Rarest task selection for multi-label samples
6. **Seed Behavior**: Seed increments per epoch when enabled
7. **Backward Compatibility**: Legacy configs work unchanged
8. **Test Coverage**: >90% coverage for new code
9. **Documentation**: Complete API docs and examples
10. **Configuration**: Valid YAML schema with OmegaConf
11. **HPO Integration**: Parameters searchable in hyperparameter optimization
12. **MLflow Logging**: Weight statistics logged for experiment tracking
13. **Config Migration**: All YAML configs successfully migrated to new schema
14. **Migration Script**: Automated migration with dry-run and backup support
