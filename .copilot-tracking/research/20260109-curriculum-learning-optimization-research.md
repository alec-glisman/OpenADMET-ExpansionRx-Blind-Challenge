# Task Research Notes: Curriculum Learning Configuration Optimization

## Research Executed

### File Analysis

- [chemprop_curriculum.yaml](configs/0-experiment/curriculum-learning/chemprop_curriculum.yaml)
  - Current config uses `quality_high_medium_low` dataset with 95k training samples
  - Curriculum enabled but **missing critical parameters**: `count_normalize`, `monitor_metric`, `min_high_quality_proportion`
  - Data path points to single fold (split_0/fold_0)

- [ensemble_curriculum.yaml](configs/0-experiment/curriculum-learning/ensemble_curriculum.yaml)
  - Reference config with full curriculum features enabled
  - Uses `monitor_metric: val/mae/high` for metric alignment with test set

### Data Distribution Analysis

**Training Data (split_0/fold_0):**

| Quality | Count | Proportion |
|---------|-------|------------|
| high | 3,610 | 3.77% |
| medium | 80,005 | 83.62% |
| low | 12,066 | 12.61% |

**Validation Data:**

| Quality | Count | Proportion |
|---------|-------|------------|
| high | 916 | 3.93% |
| medium | 19,498 | 83.62% |
| low | 2,903 | 12.45% |

**Local Test (evaluation target):**

| Quality | Count | Proportion |
|---------|-------|------------|
| high | 798 | 100% |

### Target Coverage by Quality (Critical Finding)

| Target | HIGH | MEDIUM | LOW |
|--------|------|--------|-----|
| LogD | 93.2% | 13.1% | 4.1% |
| Log KSOL | 95.3% | 75.2% | 83.7% |
| Log HLM CLint | 78.9% | 7.4% | 21.8% |
| Log MLM CLint | 84.3% | 1.8% | 0.5% |
| Caco-2 Papp A>B | 43.5% | 6.5% | 0.5% |
| Caco-2 Efflux | 43.6% | 2.2% | 9.6% |
| Log MPPB | 26.5% | 0.1% | 10.0% |
| Log MBPB | 18.4% | 0% | 0% |
| Log MGMB | 2.0% | 0% | 0% |

**Key Insight**: For most targets, HIGH quality data dominates useful coverage. Medium/Low quality adds value primarily for Log KSOL.

### Local Test Coverage

| Target | Coverage |
|--------|----------|
| LogD | 99.9% |
| Log KSOL | 99.5% |
| Log HLM CLint | 4.4% |
| Log MLM CLint | 68.2% |
| Caco-2 Papp A>B | 21.4% |
| Caco-2 Efflux | 21.4% |
| Log MPPB | 19.2% |
| Log MBPB | 19.0% |
| Log MGMB | 11.4% |

## Key Discoveries

### Critical Issues with Current Configuration

1. **Missing `count_normalize: true`**
   - With only 3.77% high-quality data, without count normalization, even 80% warmup weights result in ~4% high-quality in actual batches
   - This defeats the purpose of curriculum learning

2. **Missing `monitor_metric: val/mae/high`**
   - Currently monitoring overall `val_loss` which is dominated by medium-quality (83.6%)
   - Test set is 100% high-quality, so this creates metric misalignment
   - Early stopping and phase transitions optimize for wrong distribution

3. **Missing `reset_early_stopping_on_phase_change: true`**
   - When phase changes (e.g., warmup→expand), the model sees different data distribution
   - Without resetting, early stopping may trigger prematurely on phase transition noise

4. **Data Source Concern**
   - Current config uses `quality_high_medium_low` dataset
   - Your ensemble hyperparameters use `quality_high` only dataset
   - Curriculum learning requires multi-quality data to function

5. **Seed Mismatch**
   - `optimization.seed: 12345` vs `joint_sampling.seed: 42`
   - Should be consistent for reproducibility

### Project Conventions

- Standards referenced: curriculum.rst documentation, ensemble_curriculum.yaml reference config
- Instructions followed: count_normalize, monitor_metric alignment for high-quality test evaluation

### Implementation Patterns

**Count Normalization Math:**

```python
# Without count_normalize (current broken behavior):
# Raw weights: [0.80, 0.15, 0.05] applied per sample
# Result: ~4% high in batches (dominated by medium's 22x count)

# With count_normalize=True:
weight_sample = target_proportion / quality_count
# high:   0.80 / 3610 = 0.000222 per sample
# medium: 0.15 / 80005 = 0.0000019 per sample
# low:    0.05 / 12066 = 0.0000041 per sample
# Result: ~80% high in actual batches (as intended)
```

### Complete Examples

**Recommended curriculum configuration from docs:**

```yaml
joint_sampling:
  enabled: true
  task_oversampling:
    alpha: 0.02
  curriculum:
    enabled: true
    quality_col: Quality
    qualities:
      - high
      - medium
      - low
    patience: 5
    strategy: sampled
    reset_early_stopping_on_phase_change: true  # CRITICAL for curriculum
    log_per_quality_metrics: true
    seed: 42
    count_normalize: true                        # CRITICAL for imbalanced data
    min_high_quality_proportion: 0.25           # Safety floor
    monitor_metric: val/mae/high                # CRITICAL for high-quality test
    early_stopping_metric: null                 # Uses monitor_metric
    # Optional: Enable if standard curriculum doesn't improve
    adaptive_enabled: false
    # Optional: Enable for stronger high-quality emphasis
    loss_weighting_enabled: false
    loss_weights:
      high: 1.0
      medium: 0.5
      low: 0.3
  num_samples: null
  seed: 42
  increment_seed_per_epoch: true
  log_to_mlflow: true
```

## Recommended Approach

### Configuration Changes (Priority Order)

**1. CRITICAL - Add Missing Curriculum Parameters:**

```yaml
curriculum:
  count_normalize: true              # Without this, curriculum is broken
  min_high_quality_proportion: 0.25  # Safety floor
  monitor_metric: val/mae/high       # Align with test set quality
  early_stopping_metric: null        # Use same as monitor_metric
  reset_early_stopping_on_phase_change: true  # Prevent premature stopping
```

**2. IMPORTANT - Fix Data Source:**

- Keep `quality_high_medium_low` for curriculum learning (correct)
- This differs from ensemble params which used `quality_high` only

**3. IMPORTANT - Seed Consistency:**

```yaml
optimization:
  seed: 42  # Match curriculum seed
```

**4. CONSIDER - Enable Loss Weighting:**

```yaml
curriculum:
  loss_weighting_enabled: true
  loss_weights:
    high: 1.0
    medium: 0.5
    low: 0.3
```

This provides additional high-quality emphasis on top of sampling weights.

**5. CONSIDER - Increase Patience:**

```yaml
curriculum:
  patience: 8-10  # Give phases more time to converge
```

Current patience=5 may cause premature phase transitions.

### Data Source Recommendation

**Use `quality_high_medium_low` dataset** (current) for curriculum learning because:

1. Curriculum requires multiple quality levels to progress through phases
2. High-quality data alone (3.77%) would be insufficient for robust training
3. Medium/Low quality helps regularization and prevents overfitting to small high-quality set

The curriculum mechanism ensures the model:

- Learns core patterns from high-quality first (warmup: 80% high)
- Gains robustness from medium/low quality (expand/robust phases)
- Fine-tunes back on high-quality (polish: 70% high)

### Target Weights Analysis

Current target weights from ensemble HPO:

```yaml
target_weights:
  - 1.5  # LogD - high test coverage
  - 0.7  # Log KSOL - high test coverage
  - 1.0  # Log HLM CLint - low test coverage (4.4%)
  - 1.1  # Log MLM CLint - medium test coverage
  - 1.4  # Caco-2 Papp A>B - medium test coverage
  - 1.8  # Caco-2 Efflux - medium test coverage (highest weight!)
  - 1.3  # Log MPPB
  - 1.4  # Log MBPB
  - 0.7  # Log MGMB - lowest test coverage (11.4%)
```

These weights seem reasonable but consider that HLM CLint has only 4.4% test coverage but weight=1.0. The ensemble HPO likely optimized these for overall performance.

## Implementation Guidance

### Objectives

- Maximize performance on high-quality held-out test set
- Use curriculum learning to leverage multi-quality training data effectively
- Ensure proper metric alignment between training monitoring and test evaluation

### Key Tasks

1. Add `count_normalize: true` (CRITICAL)
2. Add `monitor_metric: val/mae/high` (CRITICAL)
3. Add `min_high_quality_proportion: 0.25` (IMPORTANT)
4. Set `reset_early_stopping_on_phase_change: true` (IMPORTANT)
5. Align seeds to 42 (consistency)
6. Consider `loss_weighting_enabled: true` (enhancement)

### Dependencies

- Requires multi-quality dataset (`quality_high_medium_low`)
- MLflow tracking for per-quality metric logging
- `log_per_quality_metrics: true` for monitoring

### Success Criteria

- Training curves show proper phase transitions in MLflow
- `val/mae/high` improves through curriculum phases
- Test MAE on high-quality data improves vs non-curriculum baseline
- Per-quality metrics logged and tracked
