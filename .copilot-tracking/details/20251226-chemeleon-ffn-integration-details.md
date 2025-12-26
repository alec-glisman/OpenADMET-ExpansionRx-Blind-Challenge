<!-- markdownlint-disable-file -->

# Task Details: CheMeleon-Chemprop FFN Integration

## Research Reference

**Source Research**: #file:../research/20251226-chemeleon-ffn-integration-plan.md

---

## Phase 1: Shared FFN Factory

### Task 1.1: Create FFN Factory Module

Create `src/admet/model/ffn_factory.py` with a unified FFN creation function.

- **Files**:
  - `src/admet/model/ffn_factory.py` - NEW: Shared factory module

- **Implementation**:

```python
"""Shared FFN factory for Chemprop and CheMeleon models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chemprop import nn

from admet.model.chemprop.ffn import BranchedFFN, MixtureOfExpertsRegressionFFN

if TYPE_CHECKING:
    from chemprop.nn.metrics import ChempropMetric
    from torch import Tensor


def create_ffn_predictor(
    ffn_type: str,
    input_dim: int,
    n_tasks: int,
    hidden_dim: int = 300,
    n_layers: int = 2,
    dropout: float = 0.0,
    n_experts: int | None = None,
    trunk_n_layers: int | None = None,
    trunk_hidden_dim: int | None = None,
    task_groups: list[list[int]] | None = None,
    criterion: ChempropMetric | None = None,
    task_weights: Tensor | None = None,
) -> nn.Predictor:
    """Create an FFN predictor based on type."""
    if ffn_type == "regression":
        return nn.RegressionFFN(...)
    elif ffn_type == "mixture_of_experts":
        return MixtureOfExpertsRegressionFFN(...)
    elif ffn_type == "branched":
        return BranchedFFN(...)
    else:
        raise ValueError(f"Unknown ffn_type: {ffn_type}")
```

- **Success**:
  - Factory function handles all 3 FFN types
  - Proper type hints and docstrings
  - Imports work correctly

- **Research References**:
  - #file:../research/20251226-chemeleon-ffn-integration-plan.md (Lines 60-120)

- **Dependencies**: None (first task)

---

### Task 1.2: Add FFN Factory Tests

Create `tests/test_ffn_factory.py` with unit tests.

- **Files**:
  - `tests/test_ffn_factory.py` - NEW: Factory tests

- **Implementation**:

```python
"""Tests for shared FFN factory."""

import pytest
from chemprop import nn

from admet.model.ffn_factory import create_ffn_predictor
from admet.model.chemprop.ffn import BranchedFFN, MixtureOfExpertsRegressionFFN


class TestFFNFactory:
    @pytest.mark.parametrize("ffn_type,expected_class", [
        ("regression", nn.RegressionFFN),
        ("mixture_of_experts", MixtureOfExpertsRegressionFFN),
        ("branched", BranchedFFN),
    ])
    def test_create_ffn_predictor(self, ffn_type, expected_class):
        ffn = create_ffn_predictor(
            ffn_type=ffn_type,
            input_dim=300,
            n_tasks=2,
        )
        assert isinstance(ffn, expected_class)
```

- **Success**:
  - All 3 FFN types create correct class instances
  - Invalid ffn_type raises ValueError

- **Research References**:
  - #file:../research/20251226-chemeleon-ffn-integration-plan.md (Lines 280-310)

- **Dependencies**: Task 1.1

---

## Phase 2: Config Updates

### Task 2.1: Update ChemeleonModelParams

Add FFN architecture parameters to `ChemeleonModelParams` in `src/admet/model/config.py`.

- **Files**:
  - `src/admet/model/config.py` - MODIFY: Add FFN fields to ChemeleonModelParams

- **Current** (approx line 276):

```python
@dataclass
class ChemeleonModelParams:
    checkpoint_path: str = "auto"
    unfreeze_schedule: UnfreezeScheduleConfig = field(default_factory=UnfreezeScheduleConfig)
    ffn_hidden_dim: int = 300
    ffn_num_layers: int = 2
    dropout: float = 0.0
```

- **New**:

```python
@dataclass
class ChemeleonModelParams:
    checkpoint_path: str = "auto"
    unfreeze_schedule: UnfreezeScheduleConfig = field(default_factory=UnfreezeScheduleConfig)
    ffn_type: str = "regression"  # NEW: 'regression', 'mixture_of_experts', 'branched'
    ffn_hidden_dim: int = 300
    ffn_num_layers: int = 2
    dropout: float = 0.0
    batch_norm: bool = False  # NEW
    n_experts: int | None = None  # NEW: MoE only
    trunk_n_layers: int | None = None  # NEW: Branched only
    trunk_hidden_dim: int | None = None  # NEW: Branched only
```

- **Success**:
  - New fields added with correct types and defaults
  - Docstring updated to document new parameters

- **Research References**:
  - #file:../research/20251226-chemeleon-ffn-integration-plan.md (Lines 125-165)

- **Dependencies**: None

---

## Phase 3: CheMeleon Model Update

### Task 3.1: Update ChemeleonModel._init_model()

Modify `_init_model()` to use the shared FFN factory.

- **Files**:
  - `src/admet/model/chemeleon/model.py` - MODIFY: Use FFN factory

- **Current** (line 187):

```python
self.ffn = nn.RegressionFFN(
    input_dim=self.mp.output_dim,
    hidden_dim=self._get_model_param("ffn_hidden_dim", 300),
    n_layers=self._get_model_param("ffn_num_layers", 2),
    dropout=self._get_model_param("dropout", 0.0),
    n_tasks=n_tasks,
)
```

- **New**:

```python
from admet.model.ffn_factory import create_ffn_predictor

# In _init_model():
ffn_type = self._get_model_param("ffn_type", "regression")
self.ffn = create_ffn_predictor(
    ffn_type=ffn_type,
    input_dim=self.mp.output_dim,
    n_tasks=n_tasks,
    hidden_dim=self._get_model_param("ffn_hidden_dim", 300),
    n_layers=self._get_model_param("ffn_num_layers", 2),
    dropout=self._get_model_param("dropout", 0.0),
    n_experts=self._get_model_param("n_experts", None),
    trunk_n_layers=self._get_model_param("trunk_n_layers", None),
    trunk_hidden_dim=self._get_model_param("trunk_hidden_dim", None),
)
```

- **Success**:
  - FFN factory used instead of hardcoded RegressionFFN
  - All FFN params passed through from config

- **Research References**:
  - #file:../research/20251226-chemeleon-ffn-integration-plan.md (Lines 170-200)

- **Dependencies**: Task 1.1, Task 2.1

---

### Task 3.2: Add CheMeleon FFN Type Tests

Add tests for CheMeleon with different FFN types.

- **Files**:
  - `tests/test_chemeleon_model.py` - MODIFY: Add FFN type tests

- **Implementation**:

```python
class TestChemeleonFFNTypes:
    """Tests for CheMeleon FFN architecture support."""

    @pytest.mark.parametrize("ffn_type", ["regression", "mixture_of_experts", "branched"])
    def test_ffn_type_creation(self, ffn_type, mock_checkpoint):
        """Test model creation with each FFN type."""
        config = OmegaConf.create({
            "model": {
                "type": "chemeleon",
                "chemeleon": {
                    "checkpoint_path": mock_checkpoint,
                    "ffn_type": ffn_type,
                    "n_experts": 4 if ffn_type == "mixture_of_experts" else None,
                    "trunk_n_layers": 2 if ffn_type == "branched" else None,
                },
            },
            "data": {"target_cols": ["target_0", "target_1"]},
            "mlflow": {"enabled": False},
        })
        model = ChemeleonModel(config)
        assert model.ffn is not None
```

- **Success**:
  - Tests pass for all 3 FFN types
  - Model correctly initializes with MoE and Branched FFN

- **Research References**:
  - #file:../research/20251226-chemeleon-ffn-integration-plan.md (Lines 280-310)

- **Dependencies**: Task 3.1

---

## Phase 4: Chemprop Refactor

### Task 4.1: Refactor ChempropModel to Use Factory

Update `ChempropModel` to use the shared FFN factory for consistency.

- **Files**:
  - `src/admet/model/chemprop/model.py` - MODIFY: Use FFN factory

- **Current Pattern** (multiple locations):

```python
# Direct instantiation of FFN classes
if self.hyperparams.ffn_type == "regression":
    ffn = nn.RegressionFFN(...)
elif self.hyperparams.ffn_type == "mixture_of_experts":
    ffn = MixtureOfExpertsRegressionFFN(...)
# etc.
```

- **New Pattern**:

```python
from admet.model.ffn_factory import create_ffn_predictor

ffn = create_ffn_predictor(
    ffn_type=self.hyperparams.ffn_type,
    input_dim=message_hidden_dim,
    n_tasks=len(target_cols),
    hidden_dim=self.hyperparams.hidden_dim,
    n_layers=self.hyperparams.num_layers,
    dropout=self.hyperparams.dropout,
    n_experts=self.hyperparams.n_experts,
    trunk_n_layers=self.hyperparams.trunk_n_layers,
    trunk_hidden_dim=self.hyperparams.trunk_hidden_dim,
    criterion=criterion,
    task_weights=task_weights_tensor,
)
```

- **Success**:
  - ChempropModel uses shared factory
  - All existing Chemprop tests pass
  - No behavior change

- **Research References**:
  - #file:../research/20251226-chemeleon-ffn-integration-plan.md (Lines 205-230)

- **Dependencies**: Task 1.1

---

## Phase 5: HPO Support

### Task 5.1: Create CheMeleon HPO Config

Create `src/admet/model/chemeleon/hpo_config.py` with HPO configuration classes.

- **Files**:
  - `src/admet/model/chemeleon/hpo_config.py` - NEW: HPO config classes

- **Implementation**:

```python
"""HPO configuration for CheMeleon model."""

from dataclasses import dataclass, field


@dataclass
class ChemeleonSearchSpaceConfig:
    """Search space configuration for CheMeleon HPO."""

    # FFN architecture
    ffn_type: list[str] = field(
        default_factory=lambda: ["regression", "mixture_of_experts", "branched"]
    )
    ffn_hidden_dim: tuple[int, int] = (128, 512)
    ffn_num_layers: tuple[int, int] = (1, 4)
    dropout: tuple[float, float] = (0.0, 0.3)

    # MoE-specific (conditional)
    n_experts: tuple[int, int] = (2, 8)

    # Branched-specific (conditional)
    trunk_n_layers: tuple[int, int] = (1, 3)
    trunk_hidden_dim: tuple[int, int] = (128, 512)

    # Training
    learning_rate: tuple[float, float] = (1e-5, 1e-3)
    batch_size: list[int] = field(default_factory=lambda: [16, 32, 64])


@dataclass
class ChemeleonHPOConfig:
    """HPO configuration for CheMeleon model."""

    num_samples: int = 50
    max_epochs: int = 100
    grace_period: int = 10
    reduction_factor: int = 3
    search_space: ChemeleonSearchSpaceConfig = field(
        default_factory=ChemeleonSearchSpaceConfig
    )
```

- **Success**:
  - Config classes defined with sensible defaults
  - Conditional params (MoE, Branched) documented

- **Research References**:
  - #file:../research/20251226-chemeleon-ffn-integration-plan.md (Lines 235-260)
  - `src/admet/model/chemprop/hpo_config.py` - Pattern reference

- **Dependencies**: None

---

### Task 5.2: Create CheMeleon HPO Search Space Builder

Create `src/admet/model/chemeleon/hpo_search_space.py` with Ray Tune search space builder.

- **Files**:
  - `src/admet/model/chemeleon/hpo_search_space.py` - NEW: Search space builder

- **Implementation**:

```python
"""Build Ray Tune search space for CheMeleon HPO."""

from ray import tune

from admet.model.chemeleon.hpo_config import ChemeleonSearchSpaceConfig


def build_chemeleon_search_space(config: ChemeleonSearchSpaceConfig) -> dict:
    """Build Ray Tune search space from config."""
    search_space = {
        "ffn_type": tune.choice(config.ffn_type),
        "ffn_hidden_dim": tune.randint(*config.ffn_hidden_dim),
        "ffn_num_layers": tune.randint(*config.ffn_num_layers),
        "dropout": tune.uniform(*config.dropout),
        "learning_rate": tune.loguniform(*config.learning_rate),
        "batch_size": tune.choice(config.batch_size),
    }

    # Conditional MoE params
    search_space["n_experts"] = tune.sample_from(
        lambda spec: tune.randint(*config.n_experts).sample()
        if spec.config.get("ffn_type") == "mixture_of_experts"
        else None
    )

    # Conditional Branched params
    search_space["trunk_n_layers"] = tune.sample_from(
        lambda spec: tune.randint(*config.trunk_n_layers).sample()
        if spec.config.get("ffn_type") == "branched"
        else None
    )

    return search_space
```

- **Success**:
  - Search space builds correctly
  - Conditional params only sampled when relevant FFN type selected

- **Research References**:
  - #file:../research/20251226-chemeleon-ffn-integration-plan.md (Lines 235-260)
  - `src/admet/model/chemprop/hpo_search_space.py` - Pattern reference

- **Dependencies**: Task 5.1

---

### Task 5.3: Create CheMeleon HPO Runner

Create `src/admet/model/chemeleon/hpo.py` with HPO runner class.

- **Files**:
  - `src/admet/model/chemeleon/hpo.py` - NEW: HPO runner

- **Implementation**:

```python
"""HPO runner for CheMeleon model."""

from ray import tune
from ray.tune.schedulers import ASHAScheduler

from admet.model.chemeleon.hpo_config import ChemeleonHPOConfig
from admet.model.chemeleon.hpo_search_space import build_chemeleon_search_space


class ChemeleonHPO:
    """Hyperparameter optimization for CheMeleon model."""

    def __init__(self, config: ChemeleonHPOConfig, base_config: dict):
        self.hpo_config = config
        self.base_config = base_config

    def run(self) -> tune.ResultGrid:
        """Run HPO search."""
        search_space = build_chemeleon_search_space(self.hpo_config.search_space)

        scheduler = ASHAScheduler(
            max_t=self.hpo_config.max_epochs,
            grace_period=self.hpo_config.grace_period,
            reduction_factor=self.hpo_config.reduction_factor,
        )

        tuner = tune.Tuner(
            trainable=self._create_trainable(),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                num_samples=self.hpo_config.num_samples,
                scheduler=scheduler,
                metric="val_loss",
                mode="min",
            ),
        )

        return tuner.fit()
```

- **Success**:
  - HPO runner integrates with Ray Tune
  - ASHA scheduler configured correctly
  - Can search over FFN architectures

- **Research References**:
  - #file:../research/20251226-chemeleon-ffn-integration-plan.md (Lines 235-260)
  - `src/admet/model/chemprop/hpo.py` - Pattern reference

- **Dependencies**: Task 5.1, Task 5.2

---

## Phase 6: Config Files

### Task 6.1: Create CheMeleon Example Config

Create `configs/0-experiment/chemeleon.yaml` with documented FFN options.

- **Files**:
  - `configs/0-experiment/chemeleon.yaml` - NEW: Example config

- **Implementation**:

```yaml
# CheMeleon model configuration
# Supports all FFN architectures: regression, mixture_of_experts, branched

model:
  type: chemeleon
  chemeleon:
    checkpoint_path: auto  # Downloads from Zenodo
    freeze_encoder: true

    # FFN Architecture (choose one)
    ffn_type: regression  # Options: regression, mixture_of_experts, branched
    ffn_hidden_dim: 300
    ffn_num_layers: 2
    dropout: 0.0
    batch_norm: false

    # MoE-specific (when ffn_type: mixture_of_experts)
    # n_experts: 4

    # Branched-specific (when ffn_type: branched)
    # trunk_n_layers: 2
    # trunk_hidden_dim: 300

    unfreeze_schedule:
      freeze_encoder: true
      unfreeze_encoder_epoch: null

data:
  data_dir: assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data/split_0/fold_0
  smiles_col: SMILES
  target_cols:
    - LogD
    - Log KSOL

optimization:
  max_epochs: 100
  batch_size: 32
  patience: 15

mlflow:
  enabled: true
  experiment_name: chemeleon
```

- **Success**:
  - Config file is valid YAML
  - All FFN options documented in comments
  - Works with CLI: `admet-cli model train -c configs/0-experiment/chemeleon.yaml`

- **Research References**:
  - #file:../research/20251226-chemeleon-ffn-integration-plan.md (Lines 265-280)

- **Dependencies**: Task 2.1

---

### Task 6.2: Create CheMeleon HPO Config

Create `configs/1-hpo-single/hpo_chemeleon.yaml` for HPO.

- **Files**:
  - `configs/1-hpo-single/hpo_chemeleon.yaml` - NEW: HPO config

- **Implementation**:

```yaml
# HPO configuration for CheMeleon model
# Searches over FFN architectures

hpo:
  num_samples: 50
  max_epochs: 100
  grace_period: 10
  reduction_factor: 3

  search_space:
    # FFN architecture search
    ffn_type:
      - regression
      - mixture_of_experts
      - branched

    ffn_hidden_dim: [128, 512]
    ffn_num_layers: [1, 4]
    dropout: [0.0, 0.3]

    # MoE params (conditional)
    n_experts: [2, 8]

    # Branched params (conditional)
    trunk_n_layers: [1, 3]
    trunk_hidden_dim: [128, 512]

    # Training
    learning_rate: [1e-5, 1e-3]
    batch_size: [16, 32, 64]

model:
  type: chemeleon
  chemeleon:
    checkpoint_path: auto

data:
  data_dir: assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data/split_0/fold_0
  smiles_col: SMILES
  target_cols:
    - LogD

mlflow:
  enabled: true
  experiment_name: chemeleon_hpo
```

- **Success**:
  - HPO config is valid YAML
  - Search space includes FFN architecture
  - Works with HPO CLI

- **Research References**:
  - #file:../research/20251226-chemeleon-ffn-integration-plan.md (Lines 265-280)

- **Dependencies**: Task 5.1

---

## Phase 7: Documentation

### Task 7.1: Update README.md

Update model table and add CheMeleon FFN examples.

- **Files**:
  - `README.md` - MODIFY: Update model table

- **Changes**:

1. Update Models table to show CheMeleon FFN support:

| Model | FFN Types | Description |
|-------|-----------|-------------|
| chemprop | regression, MoE, branched | D-MPNN with multiple FFN heads |
| chemeleon | regression, MoE, branched | Pre-trained encoder + FFN |

2. Add CheMeleon example in Quick Start section

- **Success**:
  - README accurately reflects CheMeleon capabilities
  - Example config shown

- **Research References**:
  - #file:../research/20251226-chemeleon-ffn-integration-plan.md (Lines 240-250)

- **Dependencies**: Task 3.1

---

### Task 7.2: Update docs/guide/modeling.rst

Add CheMeleon FFN configuration examples.

- **Files**:
  - `docs/guide/modeling.rst` - MODIFY: Add CheMeleon section

- **Changes**:

Add new section "CheMeleon FFN Architectures":

```rst
CheMeleon FFN Architectures
---------------------------

CheMeleon supports the same FFN architectures as Chemprop:

.. code-block:: yaml

   model:
     type: chemeleon
     chemeleon:
       ffn_type: mixture_of_experts
       n_experts: 4

See :doc:`/guide/architecture` for details on each FFN type.
```

- **Success**:
  - CheMeleon FFN options documented
  - Examples for each FFN type

- **Research References**:
  - #file:../research/20251226-chemeleon-ffn-integration-plan.md (Lines 240-250)

- **Dependencies**: Task 3.1

---

### Task 7.3: Update MODEL_CARD.md

Update architecture section with CheMeleon FFN info.

- **Files**:
  - `MODEL_CARD.md` - MODIFY: Update architecture

- **Changes**:

Add to Architecture section:

```markdown
### CheMeleon FFN Architectures

CheMeleon uses a frozen pre-trained message passing encoder with a trainable
FFN head. The FFN head can be configured as:

- **Regression**: Standard multi-layer FFN (default)
- **Mixture of Experts (MoE)**: Gated expert networks for multi-task learning
- **Branched**: Shared trunk with task-specific branches
```

- **Success**:
  - MODEL_CARD accurately describes CheMeleon architecture
  - FFN options documented

- **Research References**:
  - #file:../research/20251226-chemeleon-ffn-integration-plan.md (Lines 240-250)

- **Dependencies**: Task 3.1

---

## Dependencies Summary

| Phase | Depends On |
|-------|------------|
| Phase 1 | None |
| Phase 2 | None |
| Phase 3 | Phase 1, Phase 2 |
| Phase 4 | Phase 1 |
| Phase 5 | None (parallel) |
| Phase 6 | Phase 2, Phase 5 |
| Phase 7 | Phase 3 |

## Success Criteria

- [ ] CheMeleon model supports `ffn_type` parameter with all 3 options
- [ ] Shared FFN factory used by both Chemprop and CheMeleon
- [ ] CheMeleon HPO can search over FFN architectures
- [ ] All existing tests pass
- [ ] New tests for CheMeleon FFN types pass
- [ ] Documentation updated with examples
- [ ] Example config files created and validated
