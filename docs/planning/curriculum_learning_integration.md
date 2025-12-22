# Curriculum Learning Integration Plan

## Status: ✅ IMPLEMENTED

**Last Updated:** December 2024

This document originally outlined a plan to integrate quality-aware curriculum learning into the Chemprop training pipeline. The implementation is now complete with count-normalized sampling.

## Implementation Summary

### Completed Features

1. **Count-Normalized Sampling** (NEW)
   - Target proportions are achieved regardless of dataset size imbalance
   - Formula: `per_sample_weight = target_proportion / count`
   - Example: With High=5k, Medium=100k, Low=15k, setting warmup=[0.8, 0.15, 0.05]
     actually results in 80% high-quality samples in batches

2. **Conservative Default Proportions**
   - Warmup: 80% high, 15% medium, 5% low
   - Expand: 60% high, 30% medium, 10% low
   - Robust: 50% high, 35% medium, 15% low
   - Polish: 70% high, 20% medium, 10% low (maintains diversity)

3. **HPO-Friendly Parameters**
   - Per-phase proportions configurable via YAML
   - `count_normalize` toggle for backward compatibility
   - `min_high_quality_proportion` safety floor (default: 0.25)

4. **Full Integration**
   - Single model training via `ChempropModel`
   - Ensemble training via `ChempropEnsemble`
   - HPO integration via Ray Tune

### Key Files

| File | Description |
|------|-------------|
| `curriculum.py` | `CurriculumPhaseConfig`, `CurriculumState`, `CurriculumCallback` |
| `curriculum_sampler.py` | `DynamicCurriculumSampler` with count normalization |
| `config.py` | `CurriculumConfig` with HPO-friendly parameters |
| `model.py` | `_build_phase_config()` helper for YAML → config conversion |

### Usage

```yaml
# configs/curriculum/chemprop_curriculum.yaml
joint_sampling:
  enabled: true
  curriculum:
    enabled: true
    quality_col: Quality
    qualities: [high, medium, low]
    patience: 5
    count_normalize: true
    min_high_quality_proportion: 0.25
    # Optional: override phase proportions for HPO
    # warmup_proportions: [0.80, 0.15, 0.05]
```

## Original State

### Existing Components

1. **`curriculum.py`** - Already implemented:
   - `CurriculumState`: Manages phase transitions (warmup → expand → robust → polish)
   - `CurriculumCallback`: PyTorch Lightning callback that monitors validation loss
   - Supports arbitrary quality levels (not just high/medium/low)
   - Phase-dependent weights for sampling/weighting data by quality

2. **`tune_trainable.py`** - Legacy implementation showing curriculum usage:
   - Uses `augment_quality()` to add sample weights based on quality
   - Integrates `CurriculumCallback` with PyTorch Lightning trainer
   - Missing: actual data module that implements weighted sampling

### Integration Points (NOW COMPLETE)

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Single Model | `model.py` | ✅ Integrated | Quality column support and curriculum callback |
| Ensemble | `ensemble.py` | ✅ Integrated | Curriculum config passed through to each model |
| HPO | `hpo_trainable.py` | ✅ Integrated | Curriculum hyperparameters in search space |
| Config | `config.py` | ✅ Integrated | `CurriculumConfig` dataclass with HPO params |

---

## Original Implementation Plan (COMPLETED)
    weights: np.ndarray,
    num_samples: int,
) -> WeightedRandomSampler:
    """Create a PyTorch WeightedRandomSampler for curriculum-aware batching."""
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=num_samples,
        replacement=True,
    )
```

#### 2.2 Strategy Options

**Option A: Weighted Loss (Recommended for initial implementation)**

- Add sample-level weights to the loss computation
- Simpler to implement, doesn't require dataloader changes
- Weight each sample's contribution to the batch loss

**Option B: Weighted Sampling**

- Use `WeightedRandomSampler` in DataLoader
- Samples are drawn proportionally to curriculum weights
- Requires rebuilding dataloader when curriculum phase changes

---

### Phase 3: Single Model Integration

#### 3.1 Update `ChempropModel.__init__`

Add curriculum-related parameters:

```python
def __init__(
    self,
    # ... existing params ...
    curriculum_config: Optional[CurriculumConfig] = None,
):
    # ...
    self.curriculum_config = curriculum_config
    self.curriculum_state: Optional[CurriculumState] = None
    
    if curriculum_config and curriculum_config.enabled:
        self._init_curriculum()
```

#### 3.2 Add curriculum initialization method

```python
def _init_curriculum(self) -> None:
    """Initialize curriculum learning state and validate quality data."""
    cfg = self.curriculum_config
    
    # Validate quality column exists
    if cfg.quality_col not in self.dataframes["train"].columns:
        raise ValueError(f"Quality column '{cfg.quality_col}' not found in training data")
    
    # Initialize curriculum state
    self.curriculum_state = CurriculumState(
        qualities=cfg.qualities,
        patience=cfg.patience,
    )
    
    # Compute initial sample weights
    self._update_sample_weights()
```

#### 3.3 Update `_prepare_trainer`

Add `CurriculumCallback` to trainer callbacks:

```python
def _prepare_trainer(self) -> None:
    callbacks_list = [
        # ... existing callbacks ...
    ]
    
    # Add curriculum callback if enabled
    if self.curriculum_state is not None:
        curriculum_callback = CurriculumCallback(
            self.curriculum_state,
            monitor_metric=f"val_{self.curriculum_config.qualities[0]}_loss",
        )
        callbacks_list.append(curriculum_callback)
```

#### 3.4 Per-Quality Validation Metrics

Update model to compute separate validation metrics per quality level:

```python
def _compute_quality_metrics(self, batch, quality_labels) -> Dict[str, float]:
    """Compute validation metrics split by quality level."""
    metrics = {}
    for quality in self.curriculum_config.qualities:
        mask = quality_labels == quality
        if mask.sum() > 0:
            quality_loss = self._compute_loss(batch[mask])
            metrics[f"val_{quality}_loss"] = quality_loss
    return metrics
```

---

### Phase 4: Ensemble Integration

#### 4.1 Update `EnsembleConfig`

```python
@dataclass
class EnsembleConfig:
    # ... existing fields ...
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
```

#### 4.2 Update `_create_single_model_config`

Pass curriculum config to each model:

```python
def _create_single_model_config(self, split_fold_info: SplitFoldInfo) -> ChempropConfig:
    return ChempropConfig(
        # ... existing fields ...
        curriculum=CurriculumConfig(
            enabled=self.config.curriculum.enabled,
            quality_col=self.config.curriculum.quality_col,
            qualities=list(self.config.curriculum.qualities),
            patience=self.config.curriculum.patience,
            strategy=self.config.curriculum.strategy,
        ),
    )
```

---

### Phase 5: HPO Integration

#### 5.1 Update `HPOConfig`

```python
@dataclass
class HPOConfig:
    # ... existing fields ...
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
```

#### 5.2 Add curriculum hyperparameters to search space

Update `hpo_config.py`:

```python
@dataclass
class SearchSpaceConfig:
    # ... existing fields ...
    
    # Curriculum learning parameters
    curriculum_patience: ParameterSpace | None = None
    curriculum_enabled: ParameterSpace | None = None  # choice: [True, False]
```

#### 5.3 Update `hpo_trainable.py`

```python
def train_chemprop_trial(config: dict[str, Any]) -> None:
    # ... existing code ...
    
    # Extract curriculum config
    curriculum_enabled = config.get("curriculum_enabled", False)
    curriculum_patience = config.get("curriculum_patience", 3)
    
    if curriculum_enabled:
        curriculum_config = CurriculumConfig(
            enabled=True,
            quality_col=config.get("quality_col", "Quality"),
            qualities=config.get("curriculum_qualities", ["high", "medium", "low"]),
            patience=curriculum_patience,
        )
    else:
        curriculum_config = None
    
    model = ChempropModel(
        # ... existing params ...
        curriculum_config=curriculum_config,
    )
```

---

### Phase 6: YAML Configuration Support

#### 6.1 Example single model config

```yaml
# configs/curriculum/chemprop_curriculum.yaml
data:
  train_file: "train.csv"
  # ...
  
curriculum:
  enabled: true
  quality_col: "Quality"
  qualities: ["high", "medium", "low"]
  patience: 5
  strategy: "weighted"

model:
  # ...
```

#### 6.2 Example HPO config

```yaml
# configs/curriculum/hpo_chemprop_curriculum.yaml
search_space:
  # ... existing params ...
  
  curriculum_enabled:
    type: choice
    values: [true, false]
    
  curriculum_patience:
    type: choice
    values: [3, 5, 7, 10]

# Fixed curriculum settings (not searched)
curriculum:
  quality_col: "Quality"
  qualities: ["high", "medium", "low"]
  strategy: "weighted"
```

---

## Implementation Order

| Step | Task | Estimated Effort | Dependencies |
|------|------|------------------|--------------|
| 1 | Add `CurriculumConfig` to `config.py` | 30 min | None |
| 2 | Create `curriculum_data.py` utilities | 1 hr | Step 1 |
| 3 | Update `ChempropModel` with curriculum support | 2 hr | Steps 1-2 |
| 4 | Add per-quality validation metrics | 1 hr | Step 3 |
| 5 | Update ensemble to pass curriculum config | 30 min | Step 3 |
| 6 | Add curriculum params to HPO search space | 1 hr | Steps 1, 3 |
| 7 | Update `hpo_trainable.py` | 1 hr | Steps 1, 6 |
| 8 | Write unit tests | 2 hr | All above |
| 9 | Update documentation | 1 hr | All above |

**Total estimated effort: ~10 hours**

---

## Testing Strategy

### Unit Tests

1. **`test_curriculum_config.py`**
   - Test `CurriculumConfig` initialization and validation
   - Test serialization/deserialization with OmegaConf

2. **`test_curriculum_data.py`**
   - Test `compute_quality_weights()` for various phase states
   - Test `create_weighted_sampler()` produces valid samples

3. **`test_curriculum_model.py`**
   - Test `ChempropModel` initializes curriculum correctly
   - Test `CurriculumCallback` is added to trainer
   - Test phase transitions during training (mock trainer metrics)

### Integration Tests

1. Test full training loop with curriculum enabled
2. Test HPO with curriculum parameters in search space
3. Test ensemble training with curriculum config propagation

---

## Open Questions

1. **Per-quality metrics**: Should we compute validation loss separately for each quality level? This requires the validation dataloader to include quality labels.

2. **Phase transitions during HPO**: Should ASHA early stopping consider curriculum phase? A model might look worse in "expand" phase but improve in "polish".

3. **Ensemble diversity**: Should different ensemble members use different curriculum settings? This could increase ensemble diversity.

4. **Curriculum for test/blind sets**: Should curriculum weights affect evaluation? Typically no, but worth clarifying.

---

## Files to Create/Modify

### New Files

- `src/admet/model/chemprop/curriculum_data.py`
- `tests/test_curriculum_config.py`
- `tests/test_curriculum_data.py`
- `tests/test_curriculum_model.py`
- `configs/curriculum/chemprop_curriculum.yaml`
- `configs/curriculum/hpo_chemprop_curriculum.yaml`

### Modified Files

- `src/admet/model/chemprop/config.py` - Add `CurriculumConfig`
- `src/admet/model/chemprop/model.py` - Integrate curriculum
- `src/admet/model/chemprop/ensemble.py` - Pass curriculum config
- `src/admet/model/chemprop/hpo_config.py` - Add curriculum params
- `src/admet/model/chemprop/hpo_search_space.py` - Handle curriculum params
- `src/admet/model/chemprop/hpo_trainable.py` - Build curriculum config
- `src/admet/model/chemprop/__init__.py` - Export new modules
