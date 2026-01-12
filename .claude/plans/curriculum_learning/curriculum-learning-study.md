# Curriculum Learning Ablation Study - Implementation Plan

## Executive Summary

**Objective**: Implement 5 controlled ablation experiments to isolate the effect of curriculum learning and identify potential improvements.

**Deliverables**:

1. 5 experiment configuration files for ablation studies
2. Code changes to support "finetune" phase in curriculum learning
3. Runner script to execute all experiments
4. README documenting the experiments

---

## Files to Create/Modify

### New Configuration Files

| File | Purpose |
| ---- | ------- |
| `configs/0-experiment/curriculum-learning/ablation/01_baseline_curriculum.yaml` | Isolate curriculum effect only |
| `configs/0-experiment/curriculum-learning/ablation/02_two_quality_only.yaml` | Exclude low-quality data |
| `configs/0-experiment/curriculum-learning/ablation/03_selective_tasks.yaml` | Penalize catastrophic tasks |
| `configs/0-experiment/curriculum-learning/ablation/04_high_quality_focus.yaml` | Strict HQ monitoring |
| `configs/0-experiment/curriculum-learning/ablation/05_finetune_approach.yaml` | Pre-train + fine-tune phase |
| `configs/0-experiment/curriculum-learning/ablation/README.md` | Documentation |

### Code Changes Required

| File | Changes |
| ---- | ------- |
| `src/admet/model/chemprop/curriculum.py` | Add "finetune" phase support |
| `src/admet/model/config.py` | Add finetune phase config options |

### Runner Script

| File | Purpose |
| ---- | ------- |
| `scripts/run_curriculum_ablation.sh` | Execute all 5 experiments |

---

## Experiment Details

### Experiment 1: Baseline Curriculum (Isolate Effect)

**Purpose**: Test curriculum with minimal changes from base model.

**Key settings**:

- `curriculum.enabled: true`
- `max_epochs: 150` (same as base)
- `patience: 15` (same as base)
- `target_weights: all 1.0` (uniform, same as base)
- `loss_weighting_enabled: false`
- `reset_early_stopping_on_phase_change: false`
- `num_workers: 0` (required for curriculum)

---

### Experiment 2: Two-Quality Only

**Purpose**: Test if low-quality data is the primary problem.

**Key settings**:

- `qualities: [high, medium]` (exclude low)
- Uses two-quality phase weights automatically
- Same other settings as Experiment 1

---

### Experiment 3: Selective Tasks (Penalize Catastrophic Tasks)

**Purpose**: Reduce impact of external data on tasks that showed catastrophic failure.

**Key settings**:

- All three quality levels
- Custom loss weights that penalize medium/low for problematic tasks
- `loss_weighting_enabled: true`
- Custom `loss_weights`:
  - high: 1.0
  - medium: 0.3 (reduced from 0.7)
  - low: 0.1 (reduced from 0.4)

---

### Experiment 4: High-Quality Focus

**Purpose**: Use high-quality metrics for all decisions.

**Key settings**:

- `monitor_metric: val/mae/high`
- `early_stopping_metric: val/mae/high`
- `regression_threshold: 0.05` (5% instead of 15%)
- `min_high_quality_proportion: 0.70` (70% instead of 50%)
- Higher warmup proportions: `[0.95, 0.03, 0.02]`

---

### Experiment 5: Fine-Tuning Approach

**Purpose**: Pre-train on all data, then fine-tune on high-quality only.

**Implementation**: Add "finetune" phase after "polish" that sets high-quality to 100%.

**Key settings**:

- `finetune_enabled: true`
- `finetune_proportions: [1.0, 0.0, 0.0]` (100% high-quality)
- `finetune_min_epochs: 20`
- `finetune_lr_factor: 0.1` (reduce LR for fine-tuning)

**Code changes required**:

1. Add "finetune" to available phases in `CurriculumPhaseConfig`
2. Add finetune weight mapping in `_weights_for_phase()`
3. Add finetune config options to `CurriculumConfig`
4. Update phase progression logic in `maybe_advance_phase()`

---

## Code Changes Detail

### curriculum.py Changes

```python
# In CurriculumPhaseConfig.__init__:
available_phases: List[str] = field(
    default_factory=lambda: ["warmup", "expand", "robust", "polish", "finetune"]
)

# Add finetune weights to two_quality and three_quality dicts
two_quality: Dict[str, List[float]] = field(
    default_factory=lambda: {
        "warmup": [0.90, 0.10],
        "expand": [0.70, 0.30],
        "polish": [0.90, 0.10],
        "finetune": [1.0, 0.0],  # 100% high-quality
    }
)

three_quality: Dict[str, List[float]] = field(
    default_factory=lambda: {
        "warmup": [0.85, 0.10, 0.05],
        "expand": [0.65, 0.25, 0.10],
        "robust": [0.55, 0.30, 0.15],
        "polish": [0.85, 0.10, 0.05],
        "finetune": [1.0, 0.0, 0.0],  # 100% high-quality
    }
)

# In CurriculumState.maybe_advance_phase():
# Update phase list construction to include finetune if enabled
if finetune_enabled:
    phases.append("finetune")
```

### config.py Changes

```python
# In CurriculumConfig, add:
finetune_enabled: bool = False
finetune_proportions: Optional[List[float]] = None
finetune_min_epochs: int = 20
```

---

## Verification Plan

### Run Tests

```bash
pytest tests/model/chemprop/test_curriculum.py -v
pytest tests/model/chemprop/test_curriculum_sampler.py -v
```

### Run Each Experiment

```bash
# Run all ablation experiments
./scripts/run_curriculum_ablation.sh

# Or run individually:
admet model train -c configs/0-experiment/curriculum-learning/ablation/01_baseline_curriculum.yaml
```

### Compare Results

After all experiments complete, compare:

1. Test MAE per task (goal: <= 0.307 base model)
2. Test R² per task (goal: no negative values)
3. Phase transition patterns
4. Per-quality validation metrics

---

## Success Criteria

| Metric | Base Model | Target |
| ------ | ---------- | ------ |
| Test MAE (mean) | 0.307 | <= 0.307 |
| Test R² (mean) | 0.582 | >= 0.50 |
| Tasks with R² < 0 | 0 | 0 |

---

## Implementation Order

1. Create ablation directory and README
2. Create Experiments 1-4 configs (no code changes needed)
3. Implement finetune phase in curriculum.py
4. Add finetune config options to config.py
5. Create Experiment 5 config
6. Create runner script
7. Run tests to verify changes
