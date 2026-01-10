<!-- markdownlint-disable-file -->

# Task Details: Hybrid Ensemble Model for Minimum MA-RAE

## Research Reference

**Source Research**: #file:../research/20260109-hybrid-ensemble-strategy-research.md

---

## Phase 1: Analysis and Documentation

### Task 1.1: Extract Per-Task MAE Values from All 5 Submissions

Compile MAE values and ranks from SUBMISSIONS.md for all 9 tasks across 5 models.

**Per-Task MAE Comparison Table:**

| Task | Jan-05 MoE | Jan-06 Baseline | Jan-07 Large | Jan-08 Chemeleon | Jan-09 Weighted | **Best Model** | **Best MAE** |
|------|-----------|-----------------|--------------|------------------|-----------------|----------------|--------------|
| LogD | 0.31 (#15) | 0.35 (#63) | 0.32 (#28) | 0.38 (#88) | 0.35 (#61) | **Jan-05** | **0.31** |
| Log KSOL | 0.34 (#15) | 0.34 (#17) | 0.34 (#15) | 0.38 (#69) | 0.35 (#30) | **Jan-05/07** | **0.34** |
| Log MLM CLint | 0.34 (#12) | 0.36 (#39) | 0.35 (#16) | 0.40 (#146) | 0.35 (#27) | **Jan-05** | **0.34** |
| Log HLM CLint | 0.32 (#51) | 0.30 (#35) | 0.31 (#55) | 0.32 (#74) | 0.30 (#33) | **Jan-09** | **0.30** |
| Caco-2 Efflux | 0.35 (#94) | 0.35 (#98) | 0.33 (#53) | 0.36 (#116) | 0.33 (#50) | **Jan-09** | **0.33** |
| Caco-2 Papp A>B | 0.25 (#71) | 0.25 (#70) | 0.22 (#16) | 0.28 (#126) | 0.24 (#42) | **Jan-07** | **0.22** |
| Log MPPB | 0.22 (#100) | 0.19 (#47) | 0.22 (#105) | 0.17 (#24) | 0.17 (#24) | **Jan-08/09** | **0.17** |
| Log MBPB | 0.15 (#51) | 0.16 (#65) | 0.18 (#121) | 0.15 (#54) | 0.15 (#55) | **Jan-05** | **0.15** |
| Log MGMB | 0.17 (#36) | 0.17 (#18) | 0.20 (#96) | 0.15 (#5) | 0.16 (#12) | **Jan-08** | **0.15** |

**Files**:
- [SUBMISSIONS.md](../../SUBMISSIONS.md) - Source of all metrics

**Success**:
- All 9 tasks × 5 models = 45 MAE values extracted
- Best model identified for each task

---

### Task 1.2: Calculate Expected Performance Improvement Per Task

Calculate improvement from each submission's current best single-task performance to ensemble performance.

**Reference Submissions for Comparison:**

| Submission | Overall Rank | MA-RAE | Best Tasks |
|------------|-------------|--------|------------|
| Jan-05 MoE | 35/285 (12.3%) | 0.61 | LogD, KSOL, MLM CLint |
| Jan-06 Baseline | 31/289 (10.7%) | 0.61 | KSOL, HLM CLint |
| Jan-07 Large | 38/295 (12.9%) | 0.61 | KSOL, MLM CLint, Caco-2 Papp |
| Jan-08 Chemeleon | 55/302 (18.2%) | 0.63 | MPPB, MGMB |
| Jan-09 Weighted | 18/307 (5.9%) | 0.59 | HLM CLint, Caco-2 Efflux |

**Per-Task Improvement Analysis:**

| Task | Jan-09 MAE | Jan-09 Rank | Best MAE | Best Rank | **MAE Δ** | **Rank Δ** | **% Improvement** |
|------|-----------|-------------|----------|-----------|-----------|------------|-------------------|
| LogD | 0.35 | #61 | 0.31 (Jan-05) | #15 | -0.04 | +46 | **11.4%** |
| Log KSOL | 0.35 | #30 | 0.34 (Jan-05/07) | #15 | -0.01 | +15 | **2.9%** |
| Log MLM CLint | 0.35 | #27 | 0.34 (Jan-05) | #12 | -0.01 | +15 | **2.9%** |
| Log HLM CLint | 0.30 | #33 | 0.30 (Jan-09) | #33 | 0.00 | 0 | **0.0%** |
| Caco-2 Efflux | 0.33 | #50 | 0.33 (Jan-09) | #50 | 0.00 | 0 | **0.0%** |
| Caco-2 Papp A>B | 0.24 | #42 | 0.22 (Jan-07) | #16 | -0.02 | +26 | **8.3%** |
| Log MPPB | 0.17 | #24 | 0.17 (Jan-08/09) | #24 | 0.00 | 0 | **0.0%** |
| Log MBPB | 0.15 | #55 | 0.15 (Jan-05) | #51 | 0.00 | +4 | **0.0%** |
| Log MGMB | 0.16 | #12 | 0.15 (Jan-08) | #5 | -0.01 | +7 | **6.3%** |

**Key Improvements over Jan-09 (current best):**
- LogD: 11.4% improvement (rank 61 → 15)
- Caco-2 Papp A>B: 8.3% improvement (rank 42 → 16)
- Log KSOL: 2.9% improvement (rank 30 → 15)
- Log MLM CLint: 2.9% improvement (rank 27 → 12)
- Log MGMB: 6.3% improvement (rank 12 → 5)

**Files**:
- [20260109-hybrid-ensemble-strategy-research.md](../research/20260109-hybrid-ensemble-strategy-research.md) (Lines 30-65) - Per-task analysis

**Success**:
- MAE delta calculated for each task
- Improvement percentage documented
- Tasks with zero improvement identified (already using best model)

---

### Task 1.3: Calculate Expected MA-RAE Improvement

Calculate overall expected MA-RAE from hybrid ensemble.

**Calculation Method:**

MA-RAE is the mean of Relative Average Error (RAE) across all 9 tasks. Since we're selecting the best MAE per task, we expect:

1. **Jan-09 Current Performance:**
   - MA-RAE: 0.59
   - Average MAE: (0.35 + 0.35 + 0.35 + 0.30 + 0.33 + 0.24 + 0.17 + 0.15 + 0.16) / 9 = **0.267**

2. **Hybrid Ensemble Expected Performance:**
   - Best MAE per task: (0.31 + 0.34 + 0.34 + 0.30 + 0.33 + 0.22 + 0.17 + 0.15 + 0.15) / 9 = **0.257**

3. **Expected Improvement:**
   - MAE Improvement: (0.267 - 0.257) / 0.267 = **3.7%**
   - This translates to approximately 3-5% improvement in MA-RAE

**Expected MA-RAE Range:**

| Metric | Jan-09 Value | Hybrid Expected | Improvement |
|--------|--------------|-----------------|-------------|
| MA-RAE | 0.59 | **0.56-0.58** | **1.7-5.1%** |
| Overall Rank | 18/307 (5.9%) | **~12-15** | **Top 4-5%** |

**Per-Task Expected Ranks (Hybrid):**

| Task | Expected Rank | Rank Percentile |
|------|---------------|-----------------|
| LogD | #15 | Top 5% |
| Log KSOL | #15 | Top 5% |
| Log MLM CLint | #12 | Top 4% |
| Log HLM CLint | #33 | Top 11% |
| Caco-2 Efflux | #50 | Top 16% |
| Caco-2 Papp A>B | #16 | Top 5% |
| Log MPPB | #24 | Top 8% |
| Log MBPB | #51 | Top 17% |
| Log MGMB | #5 | Top 2% |
| **Average Rank** | **24.6** | **Top 8%** |

**Files**:
- [SUBMISSIONS.md](../../SUBMISSIONS.md) - MA-RAE calculation reference

**Success**:
- MA-RAE improvement estimate documented
- Expected rank improvement calculated
- Confidence range provided

---

## Phase 2: Implementation

### Task 2.1: Create Hybrid Ensemble Script

Create script that selects best model predictions for each task.

**Optimal Model Selection Mapping:**

```python
TASK_MODEL_MAPPING = {
    "LogD": "jan05_moe",           # Rank #15, MAE 0.31
    "Log KSOL": "jan05_moe",       # Rank #15, MAE 0.34 (tie with Jan-07)
    "Log MLM CLint": "jan05_moe",  # Rank #12, MAE 0.34
    "Log HLM CLint": "jan09_weighted",  # Rank #33, MAE 0.30
    "Log Caco-2 Permeability Efflux": "jan09_weighted",  # Rank #50, MAE 0.33
    "Log Caco-2 Permeability Papp A>B": "jan07_large",   # Rank #16, MAE 0.22
    "Log MPPB": "jan08_chemeleon",      # Rank #24, MAE 0.17 (tie with Jan-09)
    "Log MBPB": "jan05_moe",       # Rank #51, MAE 0.15
    "Log MGMB": "jan08_chemeleon", # Rank #5, MAE 0.15
}
```

**Model Prediction Paths:**

```python
MODEL_PATHS = {
    "jan05_moe": "assets/submissions/2026-01-05/mlflow-artifacts/6/c781fb7efe4a4b70a6fb6263dd3dd8e9/artifacts/predictions/blind_predictions.csv",
    "jan06_baseline": "assets/submissions/2026-01-06/mlflow-artifacts/6/ca2760b28f5945ee9b387915db9da875/artifacts/predictions/blind_predictions.csv",
    "jan07_large": "assets/submissions/2026-01-07/mlflow-artifacts/6/5ef1d4104f42489184188968ede410d6/artifacts/predictions/blind_predictions.csv",
    "jan08_chemeleon": "assets/submissions/2026-01-08/mlflow-artifacts/12/d7d51490fea9458e99e8e6677f425c37/artifacts/predictions/blind_predictions.csv",
    "jan09_weighted": "assets/submissions/2026-01-09.0/mlflow-artifacts/15/2d072c086c974a47b029f295a546497f/artifacts/predictions/blind_predictions.csv",
}
```

**Implementation Algorithm:**

```python
def create_hybrid_predictions():
    # 1. Load all model predictions
    predictions = {name: pd.read_csv(path) for name, path in MODEL_PATHS.items()}

    # 2. Start with any model as base (for SMILES column)
    hybrid = predictions["jan09_weighted"][["SMILES"]].copy()

    # 3. Select best prediction for each task
    for task, model_name in TASK_MODEL_MAPPING.items():
        hybrid[task] = predictions[model_name][task]

    return hybrid
```

**Files**:
- Create: `configs/3-hpo-ensemble-production/3_hybrid_ensemble/create_hybrid_predictions.py`
- Reference: [merge_task_weighted_predictions.py](../../configs/3-hpo-ensemble-production/2_task_weighted_ensemble/merge_task_weighted_predictions.py) (Lines 1-100)

**Success**:
- Script runs without errors
- Selects correct model for each task
- Output matches expected column structure

---

### Task 2.2: Generate Merged Blind Predictions

Run script to generate final blind_predictions.csv for submission.

**Output Requirements:**

```csv
SMILES,LogD,Log KSOL,Log MLM CLint,Log HLM CLint,Log Caco-2 Permeability Efflux,Log Caco-2 Permeability Papp A>B,Log MPPB,Log MBPB,Log MGMB
CC(=O)Nc1ccc(O)cc1,-0.45,1.23,...
...
```

**Validation Steps:**

1. Verify all 9 task columns present
2. Verify SMILES column matches submission format
3. Verify no NaN values in predictions
4. Verify row count matches expected blind set size

**Files**:
- Create: `assets/submissions/2026-01-10-hybrid/blind_predictions.csv`
- Create: `assets/submissions/2026-01-10-hybrid/submission_metadata.json`

**Success**:
- CSV file created with correct format
- All predictions are numeric (no NaN)
- Ready for HuggingFace submission

---

## Phase 3: Validation

### Task 3.1: Verify Prediction File Format Matches Submission Requirements

Validate that generated predictions match the challenge submission format.

**Validation Checks:**

```python
def validate_predictions(df):
    required_columns = [
        "SMILES", "LogD", "Log KSOL", "Log MLM CLint", "Log HLM CLint",
        "Log Caco-2 Permeability Efflux", "Log Caco-2 Permeability Papp A>B",
        "Log MPPB", "Log MBPB", "Log MGMB"
    ]

    # Check all columns present
    assert all(col in df.columns for col in required_columns)

    # Check no NaN values
    assert df[required_columns[1:]].isna().sum().sum() == 0

    # Check SMILES are strings
    assert df["SMILES"].dtype == object

    # Check predictions are numeric
    for col in required_columns[1:]:
        assert pd.api.types.is_numeric_dtype(df[col])
```

**Files**:
- Validate: `assets/submissions/2026-01-10-hybrid/blind_predictions.csv`

**Success**:
- All validation checks pass
- File ready for submission to HuggingFace

---

## Summary: Expected Performance

### Per-Task Expected Results

| Task | Current Best (Jan-09) | Hybrid Expected | Improvement |
|------|----------------------|-----------------|-------------|
| LogD | Rank #61, MAE 0.35 | Rank #15, MAE 0.31 | **+46 ranks** |
| Log KSOL | Rank #30, MAE 0.35 | Rank #15, MAE 0.34 | **+15 ranks** |
| Log MLM CLint | Rank #27, MAE 0.35 | Rank #12, MAE 0.34 | **+15 ranks** |
| Log HLM CLint | Rank #33, MAE 0.30 | Rank #33, MAE 0.30 | No change |
| Caco-2 Efflux | Rank #50, MAE 0.33 | Rank #50, MAE 0.33 | No change |
| Caco-2 Papp A>B | Rank #42, MAE 0.24 | Rank #16, MAE 0.22 | **+26 ranks** |
| Log MPPB | Rank #24, MAE 0.17 | Rank #24, MAE 0.17 | No change |
| Log MBPB | Rank #55, MAE 0.15 | Rank #51, MAE 0.15 | **+4 ranks** |
| Log MGMB | Rank #12, MAE 0.16 | Rank #5, MAE 0.15 | **+7 ranks** |

### Overall Expected Results

| Metric | Jan-09 (Current Best) | Hybrid Expected | Improvement |
|--------|----------------------|-----------------|-------------|
| MA-RAE | 0.59 | **0.56-0.58** | **1.7-5.1%** |
| Overall Rank | #18 (Top 5.9%) | **#12-15** | **Top 4-5%** |
| Average Task Rank | ~37 | **~24.6** | **12 rank improvement** |

### Model Utilization Summary

| Model | Tasks Using | % of Tasks |
|-------|-------------|------------|
| Jan-05 MoE | LogD, KSOL, MLM CLint, MBPB | 44% (4/9) |
| Jan-07 Large | Caco-2 Papp A>B | 11% (1/9) |
| Jan-08 Chemeleon | MPPB, MGMB | 22% (2/9) |
| Jan-09 Weighted | HLM CLint, Caco-2 Efflux | 22% (2/9) |
| Jan-06 Baseline | (none) | 0% |
