# Configuration Harmonization Plan

## Executive Summary

The current configuration system has separate implementations for Chemprop and classical models, causing feature fragmentation. Key capabilities like curriculum learning, task affinity grouping, and featurization are tightly coupled to specific models rather than being model-agnostic. This plan outlines a phased approach to harmonize configurations across all model types.

## Current State Analysis

### Configuration Files and Their Relationships

```
src/admet/model/
├── config.py                    # Base configs (partial, needs completion)
│   ├── BaseDataConfig           # Shared data config
│   ├── BaseMlflowConfig         # Shared MLflow config
│   ├── BaseModelConfig          # Base model config (incomplete)
│   ├── FingerprintConfig        # Featurization for classical models
│   └── Model-specific params    # XGBoost, LightGBM, CatBoost, Chemeleon
│
├── chemprop/config.py           # Chemprop-specific (too much here!)
│   ├── DataConfig               # Duplicates BaseDataConfig
│   ├── ModelConfig              # Chemprop architecture
│   ├── OptimizationConfig       # Training params
│   ├── MlflowConfig             # Duplicates BaseMlflowConfig
│   ├── JointSamplingConfig      # Curriculum + task oversampling
│   ├── CurriculumConfig         # Quality-aware curriculum
│   ├── TaskAffinityConfig       # Pre-training TAG
│   ├── InterTaskAffinityConfig  # In-training TAG
│   ├── ChempropConfig           # Full config
│   └── EnsembleConfig           # Ensemble-specific
│
├── classical/base.py            # Classical models use FingerprintConfig
└── hpo/__init__.py              # HPO search spaces (model-specific)
```

### Key Problems Identified

1. **Duplicated Configs**: `DataConfig`, `MlflowConfig` exist in both `model/config.py` and `chemprop/config.py`

2. **Chemprop-Only Features**: These should be model-agnostic:
   - `JointSamplingConfig` (task oversampling + curriculum)
   - `CurriculumConfig` (quality-aware training)
   - `TaskAffinityConfig` (task grouping)
   - `InterTaskAffinityConfig` (in-training affinity)

3. **Inconsistent Config Structure**:
   - Chemprop: `config.model.depth`, `config.model.hidden_dim`
   - Classical: `config.model.xgboost.n_estimators`, `config.model.fingerprint.type`
   - This inconsistency makes ensemble/HPO code complex

4. **Missing Shared Optimization Config**: Classical models don't have a unified optimization section

5. **Featurizer Support**:
   - Chemprop: Uses molecular graphs (built-in to chemprop library)
   - Classical: Uses `FingerprintConfig`
   - Chemeleon: Uses its own featurizer
   - No unified abstraction

## Proposed Unified Configuration Schema

### Tier 1: Core Configs (Model-Agnostic)

```python
# src/admet/model/config.py

@dataclass
class UnifiedDataConfig:
    """Universal data configuration for all models."""
    data_dir: str = MISSING
    train_file: str | None = None      # Override for single-file mode
    validation_file: str | None = None
    test_file: str | None = None
    blind_file: str | None = None
    smiles_col: str = "SMILES"
    target_cols: list[str] = field(default_factory=list)
    target_weights: list[float] = field(default_factory=list)
    quality_col: str | None = None     # For curriculum learning
    output_dir: str | None = None
    # Ensemble-specific
    splits: list[int] | None = None
    folds: list[int] | None = None


@dataclass
class UnifiedMlflowConfig:
    """Universal MLflow tracking configuration."""
    enabled: bool = True
    tracking_uri: str | None = None
    experiment_name: str = "admet"
    run_name: str | None = None
    run_id: str | None = None
    parent_run_id: str | None = None
    nested: bool = False
    log_model: bool = True


@dataclass
class UnifiedOptimizationConfig:
    """Universal optimization configuration."""
    # Common to all models
    max_epochs: int = 150          # For neural models, ignored by tree-based
    patience: int = 15             # Early stopping
    batch_size: int = 32           # For neural models
    seed: int = 42
    progress_bar: bool = False

    # Learning rate (neural models)
    learning_rate: float = 0.001
    init_lr: float | None = None   # OneCycle start
    max_lr: float | None = None    # OneCycle peak
    final_lr: float | None = None  # OneCycle end
    warmup_epochs: int = 5

    # Scheduler type
    scheduler: str = "onecycle"    # "onecycle", "cosine", "constant", "step"

    # Loss function (primarily for neural)
    criterion: str = "MAE"

    # Data loading
    num_workers: int = 0
```

### Tier 2: Training Strategy Configs (Model-Agnostic)

```python
@dataclass
class SamplingConfig:
    """Configuration for training data sampling strategies."""
    enabled: bool = False

    # Task-aware oversampling
    task_oversampling_alpha: float = 0.5  # 0=uniform, 1=full inverse

    # General sampling
    num_samples: int | None = None        # Samples per epoch
    seed: int = 42
    increment_seed_per_epoch: bool = True
    log_to_mlflow: bool = True


@dataclass
class CurriculumLearningConfig:
    """Quality-aware curriculum learning (model-agnostic)."""
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


@dataclass
class TaskAffinityConfig:
    """Task grouping via gradient affinity (model-agnostic)."""
    enabled: bool = False
    method: str = "pretraining"  # "pretraining" or "online"

    # Common settings
    n_groups: int = 3
    clustering_method: str = "agglomerative"
    seed: int = 42

    # Pre-training method specific
    affinity_epochs: int = 1
    affinity_batch_size: int = 64
    affinity_lr: float = 1e-3
    affinity_type: str = "cosine"

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
```

### Tier 3: Featurization Configs

```python
@dataclass
class FeaturizationConfig:
    """Configuration for molecular featurization."""
    type: str = "auto"  # "auto", "fingerprint", "graph", "foundation"

    # Fingerprint settings (for classical models)
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)

    # Graph settings (for Chemprop)
    graph: GraphFeaturizerConfig = field(default_factory=GraphFeaturizerConfig)

    # Foundation model settings (for Chemeleon, etc.)
    foundation: FoundationFeaturizerConfig = field(default_factory=FoundationFeaturizerConfig)


@dataclass
class GraphFeaturizerConfig:
    """Configuration for molecular graph featurization."""
    atom_features: list[str] = field(default_factory=lambda: ["atomic_num", "formal_charge", "chiral_tag"])
    bond_features: list[str] = field(default_factory=lambda: ["bond_type", "is_conjugated"])
    use_3d: bool = False


@dataclass
class FoundationFeaturizerConfig:
    """Configuration for foundation model featurization."""
    model_name: str = "chemeleon"
    checkpoint_path: str = "auto"
    freeze_encoder: bool = True
```

### Tier 4: Model-Specific Configs

```python
@dataclass
class ChempropParams:
    """Chemprop-specific architecture parameters."""
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
```

### Tier 5: Unified Model Config

```python
@dataclass
class UnifiedModelConfig:
    """Complete unified model configuration."""
    # Core settings
    data: UnifiedDataConfig = field(default_factory=UnifiedDataConfig)
    mlflow: UnifiedMlflowConfig = field(default_factory=UnifiedMlflowConfig)
    optimization: UnifiedOptimizationConfig = field(default_factory=UnifiedOptimizationConfig)

    # Model type and parameters
    model_type: str = "chemprop"
    chemprop: ChempropParams = field(default_factory=ChempropParams)
    xgboost: XGBoostParams = field(default_factory=XGBoostParams)
    lightgbm: LightGBMParams = field(default_factory=LightGBMParams)
    catboost: CatBoostParams = field(default_factory=CatBoostParams)
    chemeleon: ChemeleonParams = field(default_factory=ChemeleonParams)

    # Featurization
    featurization: FeaturizationConfig = field(default_factory=FeaturizationConfig)

    # Training strategies (model-agnostic)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    curriculum: CurriculumLearningConfig = field(default_factory=CurriculumLearningConfig)
    task_affinity: TaskAffinityConfig = field(default_factory=TaskAffinityConfig)

    # Ensemble/Ray settings
    ray: RayConfig = field(default_factory=RayConfig)
```

## Proposed YAML Schema

```yaml
# configs/example_unified.yaml

data:
  data_dir: assets/dataset/split_train_val/v3/quality_high/bitbirch
  test_file: assets/dataset/set/local_test.csv
  blind_file: assets/dataset/set/blind_test.csv
  smiles_col: SMILES
  quality_col: Quality  # NEW: unified quality column
  target_cols:
    - LogD
    - Log KSOL
    - Log HLM CLint
  target_weights: [1.0, 1.0, 1.0]
  splits: null
  folds: null

model_type: chemprop  # "chemprop", "xgboost", "lightgbm", "catboost", "chemeleon"

# Model-specific params accessed by model_type
chemprop:
  depth: 5
  message_hidden_dim: 600
  ffn_type: regression
  num_layers: 2
  hidden_dim: 600
  dropout: 0.1

# Featurization (primarily for classical models)
featurization:
  type: auto  # auto-detect based on model_type
  fingerprint:
    type: morgan
    morgan:
      radius: 2
      n_bits: 2048

# Universal optimization
optimization:
  criterion: MAE
  max_epochs: 150
  patience: 15
  batch_size: 128
  learning_rate: 0.001
  warmup_epochs: 5
  scheduler: onecycle
  seed: 12345

# Training strategies (model-agnostic!)
sampling:
  enabled: true
  task_oversampling_alpha: 0.5
  seed: 42

curriculum:
  enabled: true
  quality_col: Quality
  qualities: [high, medium, low]
  patience: 5
  strategy: sampled
  warmup_proportions: [0.8, 0.15, 0.05]

task_affinity:
  enabled: false
  method: pretraining
  n_groups: 3

# Ensemble parallelization
ray:
  max_parallel: 5

# Tracking
mlflow:
  enabled: true
  experiment_name: unified_experiment
```

## Implementation Phases

### Phase 1: Unified Base Configs (1-2 days)

**Goal**: Create unified base configs without breaking existing code.

1. Create `UnifiedDataConfig` in `src/admet/model/config.py`
2. Create `UnifiedMlflowConfig` in `src/admet/model/config.py`
3. Create `UnifiedOptimizationConfig` with superset of all model params
4. Add conversion helpers: `ChempropConfig.from_unified()`, `ChempropConfig.to_unified()`
5. Update tests to verify backward compatibility

**Files to modify**:

- `src/admet/model/config.py` (extend)
- `src/admet/model/chemprop/config.py` (add conversion methods)
- `tests/test_chemprop_config.py` (add conversion tests)

### Phase 2: Model-Agnostic Training Strategies (2-3 days)

**Goal**: Extract curriculum and sampling logic from Chemprop-specific code.

1. Create `src/admet/training/` module:
   - `sampling.py`: Task-aware sampler (model-agnostic)
   - `curriculum.py`: Curriculum scheduler (model-agnostic)
   - `task_affinity.py`: Task affinity computation (model-agnostic)

2. Refactor `JointSampler` to work with any model's dataloader

3. Create protocol/interface for curriculum-aware training:

   ```python
   class CurriculumAwareModel(Protocol):
       def get_quality_indices(self, quality: str) -> np.ndarray: ...
       def update_sampling_weights(self, weights: np.ndarray) -> None: ...
   ```

4. Update `ClassicalModelBase` to support curriculum learning

**Files to create**:

- `src/admet/training/__init__.py`
- `src/admet/training/sampling.py`
- `src/admet/training/curriculum.py`
- `src/admet/training/protocols.py`

**Files to modify**:

- `src/admet/model/chemprop/model.py` (use extracted modules)
- `src/admet/model/chemprop/joint_sampler.py` (extract core logic)
- `src/admet/model/classical/base.py` (add curriculum support)

### Phase 3: Unified Featurization (1-2 days)

**Goal**: Consistent featurization interface across models.

1. Create `FeaturizationConfig` in `src/admet/model/config.py`
2. Create `src/admet/features/factory.py` for featurizer creation
3. Ensure graph featurization can be configured (atom features, bond features)
4. Add featurization to Chemeleon config

**Files to create**:

- `src/admet/features/factory.py`
- `src/admet/features/graph.py` (wrapper for chemprop featurizers)

**Files to modify**:

- `src/admet/model/config.py` (add FeaturizationConfig)
- `src/admet/features/fingerprints.py` (ensure API consistency)

### Phase 4: Unified Model Factory (1-2 days)

**Goal**: Single entry point for model creation from unified config.

1. Update `ModelRegistry` to accept unified config:

   ```python
   model = ModelRegistry.create_from_unified(config: UnifiedModelConfig)
   ```

2. Each model type implements config extraction:

   ```python
   class ChempropAdapter:
       @classmethod
       def from_unified_config(cls, config: UnifiedModelConfig) -> ChempropModel:
           ...
   ```

3. Update `ModelEnsemble` to use unified config path

**Files to modify**:

- `src/admet/model/registry.py`
- `src/admet/model/chemprop/adapter.py`
- `src/admet/model/classical/*.py`
- `src/admet/model/chemeleon/model.py`
- `src/admet/model/chemprop/ensemble.py`

### Phase 5: HPO Integration (2-3 days)

**Goal**: Unified HPO search spaces that work across models.

1. Create unified search space config:

   ```python
   @dataclass
   class UnifiedSearchSpace:
       # Model-agnostic
       learning_rate: ParameterSpace | None
       batch_size: ParameterSpace | None

       # Model-specific (selected by model_type)
       chemprop: ChempropSearchSpace | None
       xgboost: XGBoostSearchSpace | None
       ...

       # Training strategies
       sampling_alpha: ParameterSpace | None
       curriculum_patience: ParameterSpace | None
   ```

2. Update HPO trainable to use unified config

3. Add curriculum/sampling HPO support

**Files to modify**:

- `src/admet/model/hpo/__init__.py`
- `src/admet/model/chemprop/hpo_config.py`
- `src/admet/model/chemprop/hpo_trainable.py`

### Phase 6: Config Migration & Documentation (1-2 days)

**Goal**: Migrate existing configs and document new schema.

1. Create migration script: `scripts/migrate_configs.py`
2. Migrate all configs in `configs/` directory
3. Update documentation in `docs/guide/config_reference.rst`
4. Add config validation with helpful error messages

**Files to create**:

- `scripts/migrate_configs_v2.py`
- `docs/guide/unified_config.rst`

**Files to modify**:

- All YAML files in `configs/`
- `docs/guide/config_reference.rst`

## Migration Strategy

### Backward Compatibility

1. **Dual Config Support**: Both old and new config formats work during transition
2. **Auto-Detection**: Config loader detects format and converts as needed
3. **Deprecation Warnings**: Old format logs warnings but still works
4. **Version Field**: Add `config_version: 2` to new configs

### Conversion Functions

```python
def convert_chemprop_to_unified(old_config: ChempropConfig) -> UnifiedModelConfig:
    """Convert legacy ChempropConfig to unified format."""
    ...

def convert_unified_to_chemprop(unified: UnifiedModelConfig) -> ChempropConfig:
    """Convert unified config to legacy ChempropConfig for backward compat."""
    ...
```

## Testing Strategy

### Unit Tests

1. Config conversion roundtrip tests
2. Each training strategy in isolation
3. Featurizer factory tests

### Integration Tests

1. Train Chemprop with unified config
2. Train XGBoost with curriculum learning
3. Train ensemble with mixed model types
4. HPO with unified search space

### Regression Tests

1. Ensure existing configs produce identical results
2. Performance benchmarks don't degrade

## Success Metrics

1. **Single Config Schema**: One YAML structure works for all models
2. **Feature Parity**: Curriculum/sampling/affinity available for all models
3. **Reduced Duplication**: No more duplicate config classes
4. **Test Coverage**: >90% coverage on config/training modules
5. **Documentation**: Complete config reference with examples

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing workflows | Dual config support, extensive testing |
| Performance regression | Benchmark suite, gradual rollout |
| Incomplete migration | Migration script with validation |
| Learning curve for users | Comprehensive docs, migration guide |

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Base Configs | 1-2 days | None |
| Phase 2: Training Strategies | 2-3 days | Phase 1 |
| Phase 3: Featurization | 1-2 days | Phase 1 |
| Phase 4: Model Factory | 1-2 days | Phases 1-3 |
| Phase 5: HPO Integration | 2-3 days | Phases 1-4 |
| Phase 6: Migration | 1-2 days | Phases 1-5 |

**Total: 8-14 days** (can be parallelized with multiple contributors)

## Next Steps

1. Review and approve this plan
2. Create feature branch: `feature/unified-config`
3. Begin Phase 1 implementation
4. Set up CI checks for config validation
5. Schedule weekly sync on progress
