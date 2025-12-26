<!-- markdownlint-disable-file -->

# Task Research Notes: CheMeleon-Chemprop FFN Integration

## Research Executed

### File Analysis

- `src/admet/model/chemeleon/model.py`
  - Currently hardcodes `nn.RegressionFFN` at line 187
  - No `ffn_type` parameter in config
  - Uses `ChemeleonModelParams` from `src/admet/model/config.py`

- `src/admet/model/chemprop/model.py`
  - Supports 3 FFN types: `regression`, `mixture_of_experts`, `branched`
  - FFN creation logic at lines 300-340 in `ChempropHyperparams`
  - Uses `MixtureOfExpertsRegressionFFN` and `BranchedFFN` from `ffn.py`

- `src/admet/model/chemprop/ffn.py`
  - `MixtureOfExpertsRegressionFFN` registered as `"regression-moe"`
  - `BranchedFFN` registered as `"regression-branched"`
  - Both inherit from `Predictor` and `HyperparametersMixin`

- `src/admet/model/config.py`
  - `ChemeleonModelParams` at line 276: only has `ffn_hidden_dim`, `ffn_num_layers`, `dropout`
  - Missing: `ffn_type`, `n_experts`, `trunk_n_layers`, `trunk_hidden_dim`, `batch_norm`

- `src/admet/model/chemprop/hpo_config.py`
  - `SearchSpaceConfig` has `ffn_type`, `n_experts`, `trunk_depth`, `trunk_hidden_dim`
  - Conditional sampling logic in `hpo_search_space.py`

- `src/admet/model/chemprop/hpo_search_space.py`
  - Builds Ray Tune search space from config
  - Handles conditional params (MoE-specific, Branched-specific)

### Code Search Results

- FFN creation in ChempropModel:
  - `ffn_type: str = "regression"` (line 331)
  - FFN instantiation uses `RegressionFFN`, `MixtureOfExpertsRegressionFFN`, or `BranchedFFN`

- CheMeleon model initialization:
  - `_init_model()` creates `nn.RegressionFFN` directly
  - No factory pattern, no FFN type switching

### Project Conventions

- Standards referenced: `python.instructions.md`
- Guidelines followed: OmegaConf dataclasses for config, type hints, docstrings

## Key Discoveries

### Current Architecture Gap

CheMeleon model (`ChemeleonModel`) uses a pre-trained message passing encoder from Zenodo but only supports the standard `RegressionFFN` for the prediction head. Chemprop model (`ChempropModel`) supports 3 FFN architectures.

### FFN Factory Pattern in Chemprop

Chemprop creates FFNs in `ChempropModel._build_ffn()` (implied from hyperparams):

```python
# From ChempropHyperparams dataclass
ffn_type: str = "regression"  # options: 'regression', 'mixture_of_experts', 'branched'
trunk_n_layers: Optional[int] = None
trunk_hidden_dim: Optional[int] = None
n_experts: Optional[int] = None
```

### Integration Points

1. **Config**: `ChemeleonModelParams` needs FFN architecture params
2. **Model**: `ChemeleonModel._init_model()` needs FFN factory logic
3. **HPO**: Need new `ChemeleonHPOConfig` or extend existing
4. **Tests**: `test_chemeleon_model.py` needs FFN type tests

## Recommended Approach

### Phase 1: Shared FFN Factory (Code)

Create `src/admet/model/ffn_factory.py` with a reusable factory function:

```python
"""Shared FFN factory for Chemprop and CheMeleon models."""

from typing import Sequence
from chemprop import nn
from chemprop.nn.metrics import ChempropMetric
from torch import Tensor

from admet.model.chemprop.ffn import BranchedFFN, MixtureOfExpertsRegressionFFN


def create_ffn_predictor(
    ffn_type: str,
    input_dim: int,
    n_tasks: int,
    hidden_dim: int = 300,
    n_layers: int = 2,
    dropout: float = 0.0,
    # MoE-specific
    n_experts: int | None = None,
    # Branched-specific
    trunk_n_layers: int | None = None,
    trunk_hidden_dim: int | None = None,
    task_groups: list[list[int]] | None = None,
    # Loss configuration
    criterion: ChempropMetric | None = None,
    task_weights: Tensor | None = None,
) -> nn.Predictor:
    """Create an FFN predictor based on type.

    Parameters
    ----------
    ffn_type : str
        One of: 'regression', 'mixture_of_experts', 'branched'
    input_dim : int
        Input dimension from message passing encoder
    n_tasks : int
        Number of prediction tasks
    hidden_dim : int
        Hidden dimension for FFN layers
    n_layers : int
        Number of FFN layers
    dropout : float
        Dropout probability
    n_experts : int | None
        Number of experts (MoE only)
    trunk_n_layers : int | None
        Trunk layers (Branched only)
    trunk_hidden_dim : int | None
        Trunk hidden dim (Branched only)
    task_groups : list[list[int]] | None
        Task groupings (Branched only, defaults to one group per task)
    criterion : ChempropMetric | None
        Loss criterion
    task_weights : Tensor | None
        Per-task loss weights

    Returns
    -------
    nn.Predictor
        Configured FFN predictor
    """
    if ffn_type == "regression":
        return nn.RegressionFFN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            n_tasks=n_tasks,
            criterion=criterion,
            task_weights=task_weights,
        )

    elif ffn_type == "mixture_of_experts":
        return MixtureOfExpertsRegressionFFN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            n_tasks=n_tasks,
            n_experts=n_experts or 4,
            criterion=criterion,
            task_weights=task_weights,
        )

    elif ffn_type == "branched":
        # Default task groups: one task per branch
        if task_groups is None:
            task_groups = [[i] for i in range(n_tasks)]

        return BranchedFFN(
            task_groups=task_groups,
            n_tasks=n_tasks,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            trunk_n_layers=trunk_n_layers or 1,
            trunk_hidden_dim=trunk_hidden_dim or hidden_dim,
            criterion=criterion,
            task_weights=task_weights,
        )

    else:
        raise ValueError(f"Unknown ffn_type: {ffn_type}")
```

### Phase 2: Update ChemeleonModelParams (Config)

Update `src/admet/model/config.py`:

```python
@dataclass
class ChemeleonModelParams:
    """Chemeleon-specific model parameters.

    Parameters
    ----------
    checkpoint_path : str
        Path to pretrained checkpoint or "auto" for download.
    unfreeze_schedule : UnfreezeScheduleConfig
        Gradual unfreezing configuration.
    ffn_type : str
        FFN architecture: 'regression', 'mixture_of_experts', 'branched'.
    ffn_hidden_dim : int
        Hidden dimension for FFN layers.
    ffn_num_layers : int
        Number of FFN layers.
    dropout : float
        Dropout probability.
    batch_norm : bool
        Whether to use batch normalization.
    n_experts : int | None
        Number of experts (MoE only).
    trunk_n_layers : int | None
        Trunk layers (Branched only).
    trunk_hidden_dim : int | None
        Trunk hidden dim (Branched only).
    """

    checkpoint_path: str = "auto"
    unfreeze_schedule: UnfreezeScheduleConfig = field(default_factory=UnfreezeScheduleConfig)
    ffn_type: str = "regression"
    ffn_hidden_dim: int = 300
    ffn_num_layers: int = 2
    dropout: float = 0.0
    batch_norm: bool = False
    n_experts: int | None = None
    trunk_n_layers: int | None = None
    trunk_hidden_dim: int | None = None
```

### Phase 3: Update ChemeleonModel._init_model() (Model)

Update `src/admet/model/chemeleon/model.py`:

```python
from admet.model.ffn_factory import create_ffn_predictor

def _init_model(self, n_tasks: int) -> None:
    """Initialize model components."""
    # Load pre-trained message passing
    checkpoint_path = self._get_model_param("checkpoint_path", "auto")
    self.mp = self._load_pretrained_mp(checkpoint_path)

    # Freeze encoder if configured
    if self._get_model_param("freeze_encoder", True):
        self._freeze_encoder()

    # Initialize featurizer and aggregation
    self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    self.agg = nn.MeanAggregation()

    # Initialize FFN using factory
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

    # Create MPNN
    self.mpnn = models.MPNN(
        message_passing=self.mp,
        agg=self.agg,
        predictor=self.ffn,
        batch_norm=self._get_model_param("batch_norm", False),
    )
```

### Phase 4: CheMeleon HPO Support (HPO)

Create `src/admet/model/chemeleon/hpo.py` with:

1. `ChemeleonHPOConfig` - HPO configuration for CheMeleon
2. `ChemeleonTrainable` - Ray Tune trainable for CheMeleon
3. `ChemeleonHPO` - HPO runner class

Key differences from Chemprop HPO:
- No message passing hyperparams (frozen encoder from checkpoint)
- FFN architecture search space (shared with Chemprop)
- Unfreeze schedule hyperparams (optional)

### Phase 5: Update ChempropModel to Use Factory (Refactor)

Update `src/admet/model/chemprop/model.py` to use the shared factory:

```python
from admet.model.ffn_factory import create_ffn_predictor

# In model initialization, replace direct FFN creation with:
self.ffn = create_ffn_predictor(
    ffn_type=hyperparams.ffn_type,
    input_dim=message_hidden_dim,
    n_tasks=len(target_cols),
    hidden_dim=hyperparams.hidden_dim,
    n_layers=hyperparams.num_layers,
    dropout=hyperparams.dropout,
    n_experts=hyperparams.n_experts,
    trunk_n_layers=hyperparams.trunk_n_layers,
    trunk_hidden_dim=hyperparams.trunk_hidden_dim,
    criterion=criterion,
    task_weights=task_weights_tensor,
)
```

### Phase 6: Documentation Updates

#### Files to Update

1. **README.md**
   - Update Models table to show CheMeleon supports all FFN types
   - Add CheMeleon example with FFN config

2. **MODEL_CARD.md**
   - Add CheMeleon FFN architecture section
   - Document shared factory pattern

3. **docs/guide/modeling.rst**
   - Add CheMeleon FFN configuration examples
   - Document MoE/Branched usage with CheMeleon

4. **docs/guide/configuration.rst**
   - Document `ChemeleonModelParams` FFN fields
   - Add CheMeleon config examples

5. **docs/guide/hpo.rst**
   - Add CheMeleon HPO section
   - Document FFN search space for CheMeleon

6. **docs/guide/architecture.rst**
   - Add `ffn_factory.py` to module list
   - Document shared architecture

### Phase 7: Config Files

Create `configs/0-experiment/chemeleon.yaml`:

```yaml
model:
  type: chemeleon
  chemeleon:
    checkpoint_path: auto
    freeze_encoder: true
    ffn_type: regression  # or mixture_of_experts, branched
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
    # ... other targets

optimization:
  max_epochs: 100
  batch_size: 32
  patience: 15

mlflow:
  enabled: true
  experiment_name: chemeleon
```

Create `configs/1-hpo-single/hpo_chemeleon.yaml` for HPO.

### Phase 8: Tests

Update `tests/test_chemeleon_model.py`:

```python
class TestChemeleonFFNTypes:
    """Tests for CheMeleon FFN architecture support."""

    @pytest.mark.parametrize("ffn_type", ["regression", "mixture_of_experts", "branched"])
    def test_ffn_type_creation(self, ffn_type):
        """Test model creation with each FFN type."""
        config = OmegaConf.create({
            "model": {
                "type": "chemeleon",
                "chemeleon": {
                    "ffn_type": ffn_type,
                    "n_experts": 4 if ffn_type == "mixture_of_experts" else None,
                    "trunk_n_layers": 2 if ffn_type == "branched" else None,
                },
            },
            "data": {"target_cols": ["target_0", "target_1"]},
            "mlflow": {"enabled": False},
        })
        model = ChemeleonModel(config)
        # ... assertions
```

## Implementation Guidance

### Objectives

1. Unify FFN creation across Chemprop and CheMeleon models
2. Enable all 3 FFN architectures for CheMeleon
3. Support HPO for CheMeleon FFN architecture selection
4. Maintain backward compatibility

### Key Tasks

| Phase | Task | Files | Priority |
|-------|------|-------|----------|
| 1 | Create shared FFN factory | `src/admet/model/ffn_factory.py` | P0 |
| 2 | Update ChemeleonModelParams | `src/admet/model/config.py` | P0 |
| 3 | Update ChemeleonModel._init_model() | `src/admet/model/chemeleon/model.py` | P0 |
| 4 | Create CheMeleon HPO | `src/admet/model/chemeleon/hpo.py`, `hpo_config.py` | P1 |
| 5 | Refactor ChempropModel to use factory | `src/admet/model/chemprop/model.py` | P1 |
| 6 | Update documentation | `README.md`, `MODEL_CARD.md`, `docs/guide/*.rst` | P1 |
| 7 | Create config examples | `configs/0-experiment/chemeleon.yaml` | P1 |
| 8 | Add/update tests | `tests/test_chemeleon_model.py`, `tests/test_ffn_factory.py` | P1 |

### Dependencies

- Phase 1 must complete before Phases 3 and 5
- Phase 2 must complete before Phase 3
- Phase 3 must complete before Phase 4
- Phases 6-8 can proceed in parallel after Phase 3

### Success Criteria

- [ ] CheMeleon model supports `ffn_type` parameter with all 3 options
- [ ] Shared FFN factory used by both Chemprop and CheMeleon
- [ ] CheMeleon HPO can search over FFN architectures
- [ ] All existing tests pass
- [ ] New tests for CheMeleon FFN types pass
- [ ] Documentation updated with examples
- [ ] Example config files created

### Estimated Effort

| Phase | Effort |
|-------|--------|
| Phase 1: FFN Factory | 1 hour |
| Phase 2: Config Update | 30 min |
| Phase 3: Model Update | 1 hour |
| Phase 4: HPO Support | 2-3 hours |
| Phase 5: Chemprop Refactor | 1 hour |
| Phase 6: Documentation | 1-2 hours |
| Phase 7: Config Files | 30 min |
| Phase 8: Tests | 1-2 hours |
| **Total** | **8-11 hours** |
