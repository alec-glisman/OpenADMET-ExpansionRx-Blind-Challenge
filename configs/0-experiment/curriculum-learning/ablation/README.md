# Curriculum Learning Ablation Study

This directory contains configuration files for controlled ablation experiments to understand the effect of curriculum learning on model performance.

## Background

Initial experiments showed curriculum learning degraded performance:

- **Base model** (high-quality only): Test MAE = 0.307, R² = 0.582
- **Curriculum model** (mixed quality): Test MAE = 0.405, R² = 0.215

The curriculum model showed catastrophic performance on several tasks (negative R² on Caco-2 Papp A>B, MGMB).

## Experiments

### 01: Baseline Curriculum

**Purpose**: Isolate the effect of curriculum learning by keeping all other settings identical to the base model.

**Key changes from base**:

- Enable curriculum learning
- Set `num_workers: 0` (required for curriculum)

**Hypothesis**: If performance degrades, the curriculum mechanism itself (or the external data) is the problem.

### 02: Two-Quality Only

**Purpose**: Test if low-quality data is the primary source of noise.

**Key changes**:

- Use only `[high, medium]` quality levels
- Automatically uses two-quality phase weights

**Hypothesis**: If performance improves vs 01, low-quality data is the main problem.

### 03: Selective Tasks (Aggressive Loss Weighting)

**Purpose**: Reduce the influence of medium/low quality data by aggressively down-weighting their loss contributions.

**Key changes**:

- `loss_weights: {high: 1.0, medium: 0.3, low: 0.1}`

**Hypothesis**: By heavily penalizing medium/low quality losses, the model will focus on high-quality patterns.

### 04: High-Quality Focus

**Purpose**: Use high-quality metrics for all training decisions (early stopping, phase transitions).

**Key changes**:

- `monitor_metric: val/mae/high`
- `early_stopping_metric: val/mae/high`
- `regression_threshold: 0.05` (stricter)
- `min_high_quality_proportion: 0.70`

**Hypothesis**: Optimizing directly for high-quality performance prevents catastrophic forgetting.

### 05: Fine-Tuning Approach

**Purpose**: Pre-train on all data, then fine-tune exclusively on high-quality data.

**Key changes**:

- Enable new `finetune` phase after `polish`
- `finetune_proportions: [1.0, 0.0, 0.0]` (100% high-quality)

**Hypothesis**: Pre-training extracts useful representations; fine-tuning removes harmful patterns.

## Running the Experiments

### Run All

```bash
./scripts/run_curriculum_ablation.sh
```

### Run Individually

```**bash**
admet model train -c configs/0-experiment/curriculum-learning/ablation/01_baseline_curriculum.yaml
admet model train -c configs/0-experiment/curriculum-learning/ablation/02_two_quality_only.yaml
admet model train -c configs/0-experiment/curriculum-learning/ablation/03_selective_tasks.yaml
admet model train -c configs/0-experiment/curriculum-learning/ablation/04_high_quality_focus.yaml
admet model train -c configs/0-experiment/curriculum-learning/ablation/05_finetune_approach.yaml
```

## Success Criteria

- Test MAE (mean) <= 0.307 (matches or beats base model)
- Test R² (mean) >= 0.50
- No tasks with R² < 0 on test set

## Results

| Experiment | Test MAE | Test R² | Notes |
|------------|----------|---------|-------|
| Base Model | 0.307 | 0.582 | Baseline |
| 01 Baseline | - | - | - |
| 02 Two-Quality | - | - | - |
| 03 Selective | - | - | - |
| 04 HQ Focus | - | - | - |
| 05 Finetune | - | - | - |

## MLflow Experiment

All experiments log to: `curriculum_ablation_study`
