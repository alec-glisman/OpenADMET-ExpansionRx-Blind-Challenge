<!-- markdownlint-disable-file -->

# Task Research Notes: Sampler Harmonization

## Research Executed

### File Analysis

- [src/admet/model/chemprop/task_sampler.py](../src/admet/model/chemprop/task_sampler.py)
  - **TaskAwareSampler**: Inverse-power sampling for imbalanced multi-task learning
  - Samples tasks with probability `p_i ∝ n_i^(-α)` where `n_i` is label count for task i
  - α=0: uniform task sampling, α=1: inverse proportional (rare tasks sampled more)
  - Maintains `task_indices` mapping tasks to valid molecule indices
  - After sampling a task, uniformly samples a molecule with that task's label

- [src/admet/model/chemprop/curriculum_sampler.py](../src/admet/model/chemprop/curriculum_sampler.py)
  - **DynamicCurriculumSampler**: Quality-aware sampling that updates with curriculum phases
  - Reads weights dynamically from `CurriculumState` on each iteration
  - Phases: warmup (high quality focus) → expand → robust → polish
  - Maintains `_label_indices` mapping quality labels to sample indices
  - Computes per-sample weights from curriculum state probabilities

- [src/admet/model/chemprop/curriculum.py](../src/admet/model/chemprop/curriculum.py)
  - **CurriculumState**: Manages phase transitions based on validation loss plateau
  - **CurriculumCallback**: PyTorch Lightning callback that advances phases
  - Phase weights are precomputed per phase (e.g., warmup: high=0.9, medium=0.1, low=0.0)

- [src/admet/model/chemprop/model.py#L780-860](../src/admet/model/chemprop/model.py)
  - Current implementation uses **mutually exclusive** sampler selection
  - Priority: CurriculumSampler (if enabled) > TaskAwareSampler (if alpha set) > Standard shuffle
  - No combined sampling is currently implemented

### Code Search Results

- `TaskAwareSampler` usage: [model.py#L838-849](../src/admet/model/chemprop/model.py)
  - Used when `self.hyperparams.task_sampling_alpha is not None`
  - Creates sampler with targets array, alpha, and seed

- `DynamicCurriculumSampler` usage: [model.py#L808-835](../src/admet/model/chemprop/model.py)
  - Used when `curriculum_state is not None and _quality_labels["train"] is not None`
  - Takes quality_labels, curriculum_state, num_samples, seed

- Configuration patterns in [configs/0-experiment/ensemble_chemprop_production.yaml](../configs/0-experiment/ensemble_chemprop_production.yaml):
  ```yaml
  optimization:
    task_sampling_alpha: 0.1  # Task-level oversampling

  curriculum:
    enabled: false  # Curriculum disabled in production config
    quality_col: "Quality"
    qualities: ["high", "medium", "low"]
  ```

### External Research

- PyTorch Sampler API (https://docs.pytorch.org/docs/stable/data.html)
  - Samplers implement `__iter__()` yielding indices and `__len__()`
  - `WeightedRandomSampler`: Fixed weights at construction time
  - Custom samplers can compose multiple sampling strategies
  - BatchSampler wraps another sampler to yield mini-batches

- TAG Paper (arXiv:2109.04617)
  - Multi-task learning benefits from strategic task sampling
  - Gradient-based task grouping can identify beneficial training combinations
  - Task imbalance significantly impacts multi-task model convergence

### Project Conventions

- Standards referenced: [python.instructions.md](../.github/instructions/python.instructions.md)
- Guidelines followed: PEP 8, type hints, docstrings with PEP 257

## Key Discoveries

### Project Structure

The sampling pipeline follows a clear hierarchy:

1. **Configuration Layer**: `ChempropConfig` / `CurriculumConfig` / `OptimizationConfig`
2. **State Management**: `CurriculumState` tracks phase progression
3. **Sampler Layer**: `TaskAwareSampler` or `DynamicCurriculumSampler`
4. **Training Integration**: PyTorch DataLoader with custom sampler

Current architecture **prevents combining both strategies** because the model uses an if-elif chain that selects one sampler exclusively.

### Implementation Patterns

**Existing Sampler Interface Pattern:**
```python
class Sampler(Sampler[int]):
    def __init__(
        self,
        data: np.ndarray,          # Task targets or quality labels
        num_samples: int | None,    # Samples per epoch
        seed: int | None,           # Reproducibility
    ) -> None:
        ...

    def __iter__(self) -> Iterator[int]:
        """Yield sample indices for one epoch."""
        ...

    def __len__(self) -> int:
        return self.num_samples
```

**Current Mutual Exclusivity (model.py lines 808-854):**
```python
if curriculum_state is not None and quality_labels is not None:
    sampler = DynamicCurriculumSampler(...)  # Quality-based only
elif task_sampling_alpha is not None:
    sampler = TaskAwareSampler(...)  # Task-based only
else:
    # Standard shuffle
```

### API and Schema Documentation

**TaskAwareSampler Parameters:**
- `targets: np.ndarray` - Shape (N, T), NaN for missing labels
- `alpha: float` - Power law exponent [0, 1]
- `num_samples: int | None` - Defaults to dataset length
- `seed: int | None` - For reproducibility

**DynamicCurriculumSampler Parameters:**
- `quality_labels: Sequence[str]` - Per-sample quality tags
- `curriculum_state: CurriculumState` - Shared state object
- `num_samples: int | None` - Defaults to dataset length
- `seed: int | None` - For reproducibility

### Configuration Examples

**Proposed Unified Sampling Config Structure:**
```yaml
sampling:
  # Master enable for unified sampling
  enabled: true

  # Task oversampling for imbalanced multi-task data
  task_oversampling:
    enabled: true
    alpha: 0.3  # Power law exponent (0=uniform, 1=full inverse), validated ∈ [0, 1]

  # Quality-aware curriculum learning
  curriculum:
    enabled: true
    quality_col: "Quality"
    qualities: ["high", "medium", "low"]
    patience: 5
    strategy: "sampled"  # "sampled" | "weighted"
    reset_early_stopping_on_phase_change: false

  # Composition mode (multiplicative is the only supported mode)
  composition_mode: "multiplicative"  # w_joint = w_task × w_quality

  # Seed management
  seed: 42
  increment_seed_per_epoch: true  # Increment seed each epoch for variety

  # Logging options
  log_weight_statistics: true  # Log min/max/entropy of weights each epoch
  log_to_mlflow: true  # Log sampling statistics to MLflow
```

### Technical Requirements

1. **Joint Weight Computation**: Both samplers compute weights independently
   - TaskAwareSampler: `w_task[i] = (task_count[primary_task_of_sample_i])^(-α)`
   - CurriculumSampler: `w_quality[i] = phase_weight[quality_of_sample_i]`
   - **Multi-task samples**: Use the **rarest task** (smallest count) as primary task

2. **Composition Mode**:
   - **Product (multiplicative)**: `w_joint[i] = w_task[i] × w_quality[i]` (selected approach)

3. **Dynamic Updates**:
   - Curriculum weights change with phase transitions
   - Task weights are static (computed once at initialization)
   - Seed increments per epoch for sampling variety

4. **Normalization**: Joint weights must sum to 1 for valid probability distribution

5. **Quality Labels**: All samples are expected to have a Quality column value (no missing allowed)

6. **Alpha Validation**: Validate `α ∈ [0, 1]` with warning if outside range

## Recommended Approach

### Hierarchical Composition with Multiplicative Weights

**Rationale:**
- Maintains simplicity of single sampler interface
- Allows both strategies to influence every sample
- Naturally handles dynamic curriculum phase changes
- Mathematically sound: product of probabilities remains valid distribution

**Implementation Strategy:**

1. **Create `JointSampler` class** that encapsulates both sampling strategies:
   ```python
   class JointSampler(Sampler[int]):
       def __init__(
           self,
           targets: np.ndarray,
           quality_labels: Sequence[str],
           curriculum_state: CurriculumState | None,
           task_alpha: float | None,
           composition_mode: str = "multiplicative",
           num_samples: int | None = None,
           seed: int | None = None,
       ) -> None:
           ...
   ```

2. **Weight computation** in `_compute_weights()`:
   ```python
   def _compute_weights(self) -> np.ndarray:
       weights = np.ones(len(self.targets))

       # Apply task-aware weighting (static)
       if self.task_alpha is not None:
           for i in range(len(weights)):
               task_idx = self._get_primary_task(i)
               weights[i] *= (self.task_counts[task_idx] + 1e-6) ** (-self.task_alpha)

       # Apply curriculum weighting (dynamic)
       if self.curriculum_state is not None:
           probs = self.curriculum_state.sampling_probs()
           for i, label in enumerate(self.quality_labels):
               weights[i] *= probs.get(label, 0.0)

       # Normalize
       return weights / weights.sum()
   ```

3. **Create `JointSamplingConfig` dataclass**:
   ```python
   @dataclass
   class JointSamplingConfig:
       enabled: bool = False

       # Task oversampling sub-config
       task_alpha: float | None = None

       # Curriculum sub-config (reference existing CurriculumConfig)
       curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

       # Composition settings
       composition_mode: str = "multiplicative"  # multiplicative | hierarchical

       seed: int = 42
   ```

4. **Update model.py** to use unified sampler:
   ```python
   if self.joint_sampling_config and self.joint_sampling_config.enabled:
       sampler = JointSampler(
           targets=ys,
           quality_labels=self._quality_labels["train"],
           curriculum_state=self.curriculum_state,
           task_alpha=self.joint_sampling_config.task_alpha,
           composition_mode=self.joint_sampling_config.composition_mode,
           seed=self.joint_sampling_config.seed,
       )
   ```

### Alternative Approaches (Not Selected)

**Interleaved Sampling**: Alternate between samplers per batch
- Pros: Clean separation, easy to understand
- Cons: Half the batches ignore one strategy, less sample-level control
- **Not selected**: Loses the benefit of combined optimization at sample level

**Sequential Pipeline**: Curriculum sampler wraps task sampler
- Pros: Composable, each sampler unchanged
- Cons: Complex initialization, harder to reason about weights
- **Not selected**: Over-engineering for the use case

**Loss Weighting Instead**: Apply both as loss weights rather than sampling
- Pros: All samples seen, differentiable
- Cons: Doesn't reduce rare-task variance, curriculum concept lost
- **Not selected**: Changes the fundamental approach

## Implementation Guidance

- **Objectives**:
  1. Create unified `JointSampler` class with multiplicative weight composition
  2. Define `JointSamplingConfig` dataclass with nested sub-configs
  3. Update `ChempropModel` to instantiate joint sampler when both strategies enabled
  4. Ensure backward compatibility: existing configs still work unchanged
  5. Add comprehensive unit tests for joint sampling behavior
  6. Implement epoch-varying seed for sampling variety
  7. Add weight statistics logging (min/max/entropy/effective samples)
  8. Integrate sampling metrics with MLflow
  9. Validate alpha range [0, 1] with warnings
  10. Add joint sampling parameters to HPO search space

- **Key Tasks**:
  1. Create `src/admet/model/chemprop/joint_sampler.py` with `JointSampler` class
  2. Add `JointSamplingConfig` to `src/admet/model/chemprop/config.py`
  3. Modify `_prepare_data()` in `model.py` to use joint sampler
  4. Update YAML config schema in `configs/` directory
  5. Add tests in `tests/test_joint_sampler.py`
  6. Update `hpo_search_space.py` with joint sampling parameters
  7. Add MLflow logging for sampling statistics

- **Dependencies**:
  - Existing `CurriculumState`, `CurriculumConfig` classes
  - NumPy for weight computation
  - PyTorch `Sampler` base class
  - Existing test fixtures from `conftest.py`

- **Success Criteria**:
  1. JointSampler produces valid probability distribution
  2. Weights update correctly when curriculum phase advances
  3. Task imbalance correction verified via sampling statistics
  4. Backward compatibility: setting only `task_sampling_alpha` works as before
  5. Backward compatibility: setting only `curriculum.enabled` works as before
  6. All existing tests pass
  7. New tests achieve >90% coverage for joint sampler module
  8. Seed increments per epoch producing different sampling orders
  9. Weight statistics logged at epoch start (min, max, entropy, effective samples)
  10. MLflow receives sampling distribution metrics per epoch
  11. Alpha values outside [0, 1] trigger warning but still function
  12. HPO can search over joint sampling parameters

## File Structure Recommendation

```
src/admet/model/chemprop/
├── config.py                    # Add JointSamplingConfig
├── joint_sampler.py             # NEW: JointSampler class
├── curriculum_sampler.py        # Unchanged
├── curriculum.py                # Unchanged
├── task_sampler.py              # Unchanged
└── model.py                     # Update _prepare_data()

tests/
├── test_joint_sampler.py        # NEW: Comprehensive tests
├── test_curriculum_sampler.py   # Unchanged
└── conftest.py                  # Add joint sampler fixtures

configs/
└── 0-experiment/
    └── ensemble_joint_sampling.yaml  # NEW: Example config
```
