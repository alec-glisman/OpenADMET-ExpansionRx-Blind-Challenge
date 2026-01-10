<!-- markdownlint-disable-file -->

# Task Research Notes: Hybrid Ensemble Model Strategy

## Research Executed

### File Analysis

- `/home/aglisman/VSCodeProjects/OpenADMET-ExpansionRx-Blind-Challenge/SUBMISSIONS.md`
  - Comprehensive per-task metrics for 5 January 2026 submissions (excluding erroneous Jan-03)
  - Per-task ranks, MAE, and gaps to leader for 9 ADMET endpoints

- `configs/3-hpo-ensemble-production/2_task_weighted_ensemble/merge_task_weighted_predictions.py`
  - Existing task-weighted ensemble implementation selecting best model per task
  - Infrastructure for loading and merging predictions from multiple submissions

- `src/admet/model/chemprop/ensemble.py` and `src/admet/model/ensemble.py`
  - Ray-parallelized ensemble training with aggregation (mean/median)
  - Prediction caching and uncertainty estimation support

### Project Conventions

- Standards referenced: Python dataclasses for configuration, NumPy for numerical operations
- Guidelines followed: OmegaConf for config management, pandas for prediction handling

## Key Discoveries

### Per-Task Performance Analysis (January 2026 Submissions)

| Task | Jan-05 (MoE) | Jan-06 (Baseline) | Jan-07 (Large) | Jan-08 (Chemeleon) | Jan-09 (Weighted) | **Best Model** |
|------|-------------|-------------------|----------------|-------------------|-------------------|----------------|
| LogD | **#15** (0.31) | #63 (0.35) | #28 (0.32) | #88 (0.38) | #61 (0.35) | **Jan-05 MoE** |
| Log KSOL | #15 (0.34) | **#17** (0.34) | **#15** (0.34) | #69 (0.38) | #30 (0.35) | **Jan-06/07** |
| Log MLM CLint | **#12** (0.34) | #39 (0.36) | #16 (0.35) | #146 (0.40) | **#27** (0.35) | **Jan-05 MoE** |
| Log HLM CLint | #51 (0.32) | **#35** (0.30) | #55 (0.31) | #74 (0.32) | **#33** (0.30) | **Jan-09 Weighted** |
| Caco-2 Efflux | #94 (0.35) | #98 (0.35) | **#53** (0.33) | #116 (0.36) | **#50** (0.33) | **Jan-09 Weighted** |
| Caco-2 Papp A>B | #71 (0.25) | #70 (0.25) | **#16** (0.22) | #126 (0.28) | **#42** (0.24) | **Jan-07 Large** |
| Log MPPB | #100 (0.22) | #47 (0.19) | #105 (0.22) | **#24** (0.17) | **#24** (0.17) | **Jan-08/09** |
| Log MBPB | **#51** (0.15) | #65 (0.16) | #121 (0.18) | **#54** (0.15) | #55 (0.15) | **Jan-05 MoE** |
| Log MGMB | #36 (0.17) | #18 (0.17) | #96 (0.20) | **#5** (0.15) | **#12** (0.16) | **Jan-08 Chemeleon** |

### Model Specialization Patterns

1. **Jan-05 (Chemprop MoE):** Excels at LogD (#15), MLM CLint (#12), MBPB (#51)
2. **Jan-06 (Baseline):** Strong on KSOL (#17), HLM CLint (#35)
3. **Jan-07 (Large MPNN):** Best for KSOL (#15), Caco-2 Papp A>B (#16)
4. **Jan-08 (Chemeleon):** Dominates plasma binding - MPPB (#24), MGMB (#5)
5. **Jan-09 (Task-Weighted):** Strong balanced performance - HLM CLint (#33), Caco-2 Efflux (#50)

### Ensemble Strategy Options

#### Strategy 1: Task-Best Selection (Simple)

Select best-performing model's predictions for each task:

| Task | Selected Model | Expected Rank |
|------|----------------|---------------|
| LogD | Jan-05 (MoE) | 15 |
| Log KSOL | Jan-07 (Large) | 15 |
| Log MLM CLint | Jan-05 (MoE) | 12 |
| Log HLM CLint | Jan-09 (Weighted) | 33 |
| Caco-2 Efflux | Jan-09 (Weighted) | 50 |
| Caco-2 Papp A>B | Jan-07 (Large) | 16 |
| Log MPPB | Jan-08 (Chemeleon) | 24 |
| Log MBPB | Jan-05 (MoE) | 51 |
| Log MGMB | Jan-08 (Chemeleon) | 5 |

**Expected average rank:** (15+15+12+33+50+16+24+51+5)/9 ≈ **24.6**
**Potential MA-RAE improvement:** Significant - selects best performer per task

#### Strategy 2: Weighted Average Ensemble

Combine predictions from multiple models using task-specific weights derived from inverse rank:

```python
# Weight formula: w_i = 1 / (rank_i + epsilon) / sum(1 / (rank_j + epsilon))
# For LogD: Jan-05 gets highest weight (rank 15), Jan-08 lowest (rank 88)
```

**Pros:** Smooths predictions, reduces overfitting to single model's biases
**Cons:** May dilute best model's performance with weaker contributions

#### Strategy 3: Optimized Weighted Ensemble (Recommended)

Use differential evolution or grid search to find optimal per-task weights:

```python
from scipy.optimize import differential_evolution

def loss_function(weights, predictions_dict, task):
    """Minimize MAE on validation set for each task."""
    normalized_weights = weights / weights.sum()
    ensemble_pred = sum(w * pred for w, pred in zip(normalized_weights, predictions_dict.values()))
    return compute_mae(ensemble_pred, ground_truth)

# Find optimal weights for each task independently
for task in TASKS:
    bounds = [(0, 1) for _ in range(n_models)]
    result = differential_evolution(loss_function, bounds, args=(predictions, task))
    optimal_weights[task] = result.x / result.x.sum()
```

**Pros:** Data-driven optimization, task-specific blending
**Cons:** Requires held-out validation data (we have test set from splits)

#### Strategy 4: Stacking Meta-Learner

Train a lightweight model (Ridge regression) on base model predictions:

```python
from sklearn.linear_model import RidgeCV

# Stack predictions from all 5 models as features
X_meta = np.column_stack([model.predict(X_val) for model in models])
y_meta = y_val

# Train separate meta-learner for each task
meta_model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0])
meta_model.fit(X_meta, y_meta)
```

**Pros:** Learns non-linear combinations, handles model correlations
**Cons:** Risk of overfitting, requires more data

### Implementation Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Hybrid Ensemble Pipeline                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Jan-05    │  │   Jan-06    │  │   Jan-07    │  │   Jan-08    │    │
│  │ Chemprop    │  │  Baseline   │  │   Large     │  │  Chemeleon  │    │
│  │    MoE      │  │  Chemprop   │  │   MPNN      │  │    MoE      │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                │                │                │           │
│         └────────────────┴────────────────┴────────────────┘           │
│                                   │                                     │
│                    ┌──────────────▼──────────────┐                      │
│                    │      Per-Task Aggregator     │                      │
│                    │  ┌─────────────────────────┐ │                      │
│                    │  │ Task → Optimal Weights  │ │                      │
│                    │  │ LogD: [0.7, 0.1, 0.1, 0.1] │                    │
│                    │  │ KSOL: [0.2, 0.3, 0.4, 0.1] │                    │
│                    │  │ MPPB: [0.0, 0.1, 0.0, 0.9] │                    │
│                    │  │ ...                       │                      │
│                    │  └─────────────────────────┘ │                      │
│                    └──────────────┬──────────────┘                      │
│                                   │                                     │
│                    ┌──────────────▼──────────────┐                      │
│                    │   Final Blind Predictions    │                      │
│                    └─────────────────────────────┘                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Complete Implementation Example

```python
#!/usr/bin/env python3
"""Optimized Hybrid Ensemble for Minimum MA-RAE.

This script creates a hybrid ensemble by:
1. Loading predictions from multiple model submissions
2. Computing optimal per-task weights using grid search or differential evolution
3. Generating final blended predictions

The optimization uses the test split predictions (with known targets) to
find weights that minimize MAE for each task, then applies those weights
to the blind predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import differential_evolution
from typing import Dict, List, Tuple


# Task configuration
TASKS = [
    "LogD",
    "Log KSOL",
    "Log MLM CLint",
    "Log HLM CLint",
    "Log Caco-2 Permeability Efflux",
    "Log Caco-2 Permeability Papp A>B",
    "Log MPPB",
    "Log MBPB",
    "Log MGMB",
]

# Model submissions with prediction paths
MODELS = {
    "jan05_moe": {
        "test_path": "assets/submissions/2026-01-05/mlflow-artifacts/6/c781fb7efe4a4b70a6fb6263dd3dd8e9/artifacts/predictions/test_predictions.csv",
        "blind_path": "assets/submissions/2026-01-05/mlflow-artifacts/6/c781fb7efe4a4b70a6fb6263dd3dd8e9/artifacts/predictions/blind_predictions.csv",
    },
    "jan06_baseline": {
        "test_path": "assets/submissions/2026-01-06/mlflow-artifacts/6/ca2760b28f5945ee9b387915db9da875/artifacts/predictions/test_predictions.csv",
        "blind_path": "assets/submissions/2026-01-06/mlflow-artifacts/6/ca2760b28f5945ee9b387915db9da875/artifacts/predictions/blind_predictions.csv",
    },
    "jan07_large": {
        "test_path": "assets/submissions/2026-01-07/mlflow-artifacts/6/5ef1d4104f42489184188968ede410d6/artifacts/predictions/test_predictions.csv",
        "blind_path": "assets/submissions/2026-01-07/mlflow-artifacts/6/5ef1d4104f42489184188968ede410d6/artifacts/predictions/blind_predictions.csv",
    },
    "jan08_chemeleon": {
        "test_path": "assets/submissions/2026-01-08/mlflow-artifacts/12/d7d51490fea9458e99e8e6677f425c37/artifacts/predictions/test_predictions.csv",
        "blind_path": "assets/submissions/2026-01-08/mlflow-artifacts/12/d7d51490fea9458e99e8e6677f425c37/artifacts/predictions/blind_predictions.csv",
    },
    "jan09_weighted": {
        "test_path": "assets/submissions/2026-01-09.0/mlflow-artifacts/15/2d072c086c974a47b029f295a546497f/artifacts/predictions/test_predictions.csv",
        "blind_path": "assets/submissions/2026-01-09.0/mlflow-artifacts/15/2d072c086c974a47b029f295a546497f/artifacts/predictions/blind_predictions.csv",
    },
}


def load_predictions(
    base_path: Path,
    model_configs: Dict[str, Dict[str, str]],
    split: str = "test"
) -> Dict[str, pd.DataFrame]:
    """Load predictions from all models."""
    predictions = {}
    path_key = f"{split}_path"

    for model_name, paths in model_configs.items():
        pred_path = base_path / paths[path_key]
        if pred_path.exists():
            predictions[model_name] = pd.read_csv(pred_path)
            print(f"✓ Loaded {model_name}: {len(predictions[model_name])} samples")
        else:
            print(f"✗ Missing: {pred_path}")

    return predictions


def extract_ground_truth(test_predictions: pd.DataFrame, task: str) -> np.ndarray:
    """Extract ground truth values from test predictions (if available)."""
    actual_col = f"{task}_actual"
    if actual_col in test_predictions.columns:
        return test_predictions[actual_col].values
    return None


def compute_weighted_prediction(
    predictions: Dict[str, pd.DataFrame],
    weights: np.ndarray,
    task: str,
) -> np.ndarray:
    """Compute weighted average prediction for a task."""
    model_names = list(predictions.keys())
    normalized_weights = weights / weights.sum()

    pred_col = f"{task}_mean" if f"{task}_mean" in predictions[model_names[0]].columns else task

    weighted_pred = np.zeros(len(predictions[model_names[0]]))
    for i, model_name in enumerate(model_names):
        if pred_col in predictions[model_name].columns:
            weighted_pred += normalized_weights[i] * predictions[model_name][pred_col].values
        else:
            weighted_pred += normalized_weights[i] * predictions[model_name][task].values

    return weighted_pred


def optimize_weights_for_task(
    test_predictions: Dict[str, pd.DataFrame],
    task: str,
    ground_truth: np.ndarray,
    method: str = "differential_evolution",
) -> Tuple[np.ndarray, float]:
    """Find optimal weights for a single task."""
    n_models = len(test_predictions)

    def loss_function(weights):
        pred = compute_weighted_prediction(test_predictions, weights, task)
        return np.mean(np.abs(pred - ground_truth))  # MAE

    if method == "differential_evolution":
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        result = differential_evolution(
            loss_function,
            bounds,
            maxiter=500,
            tol=1e-7,
            seed=42,
        )
        optimal_weights = result.x / result.x.sum()
        optimal_mae = result.fun

    elif method == "grid_search":
        # Coarse grid search with normalized weights
        best_mae = float("inf")
        best_weights = np.ones(n_models) / n_models

        from itertools import product
        grid_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        for weights in product(grid_values, repeat=n_models):
            weights = np.array(weights)
            if weights.sum() == 0:
                continue
            weights = weights / weights.sum()
            mae = loss_function(weights)
            if mae < best_mae:
                best_mae = mae
                best_weights = weights

        optimal_weights = best_weights
        optimal_mae = best_mae

    return optimal_weights, optimal_mae


def create_hybrid_ensemble(
    base_path: Path,
    output_path: Path,
    optimization_method: str = "differential_evolution",
) -> pd.DataFrame:
    """Create optimized hybrid ensemble predictions."""
    print("=" * 70)
    print("Hybrid Ensemble Generation")
    print("=" * 70)

    # Load test predictions for weight optimization
    test_preds = load_predictions(base_path, MODELS, split="test")
    if not test_preds:
        raise ValueError("No test predictions found")

    # Load blind predictions for final output
    blind_preds = load_predictions(base_path, MODELS, split="blind")
    if not blind_preds:
        raise ValueError("No blind predictions found")

    # Optimize weights for each task
    optimal_weights = {}
    print("\nOptimizing per-task weights...")
    print("-" * 70)

    for task in TASKS:
        # Get ground truth from test predictions
        first_model = list(test_preds.keys())[0]
        ground_truth = extract_ground_truth(test_preds[first_model], task)

        if ground_truth is not None:
            weights, mae = optimize_weights_for_task(
                test_preds, task, ground_truth, method=optimization_method
            )
            optimal_weights[task] = weights
            print(f"  {task:35s}: MAE={mae:.4f}, weights={weights.round(3)}")
        else:
            # Fall back to equal weights if no ground truth
            n_models = len(test_preds)
            optimal_weights[task] = np.ones(n_models) / n_models
            print(f"  {task:35s}: No ground truth, using equal weights")

    # Generate final predictions
    print("\nGenerating hybrid ensemble predictions...")
    first_model = list(blind_preds.keys())[0]
    result = pd.DataFrame()
    result["SMILES"] = blind_preds[first_model]["SMILES"].copy()

    for task in TASKS:
        weights = optimal_weights[task]
        result[task] = compute_weighted_prediction(blind_preds, weights, task)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"\n✓ Saved hybrid ensemble to: {output_path}")
    print(f"  Shape: {result.shape}")

    # Print summary
    print("\n" + "=" * 70)
    print("Optimal Weight Summary (per task)")
    print("=" * 70)
    model_names = list(test_preds.keys())
    header = f"{'Task':35s} | " + " | ".join(f"{m[:10]:>10s}" for m in model_names)
    print(header)
    print("-" * len(header))
    for task in TASKS:
        w = optimal_weights[task]
        row = f"{task:35s} | " + " | ".join(f"{w[i]:10.3f}" for i in range(len(w)))
        print(row)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate optimized hybrid ensemble")
    parser.add_argument("--base-path", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("assets/submissions/hybrid_ensemble/blind_predictions.csv"))
    parser.add_argument("--method", choices=["differential_evolution", "grid_search"], default="differential_evolution")
    args = parser.parse_args()

    create_hybrid_ensemble(args.base_path, args.output, args.method)
```

### Expected Performance

Based on per-task best performance analysis:

| Metric | Current Best (Jan-09) | Task-Best Selection | Optimized Weighted |
|--------|----------------------|---------------------|-------------------|
| Expected Rank | 18/307 (5.9%) | ~15/307 (4.9%) | ~12/307 (3.9%) |
| Expected MA-RAE | 0.59 | ~0.56 | ~0.54 |
| LogD Rank | 61 | 15 | ~12 |
| KSOL Rank | 30 | 15 | ~14 |
| Caco-2 Efflux Rank | 50 | 50 | ~45 |
| MGMB Rank | 12 | 5 | ~5 |

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Overfitting weights to test split | Medium | Use cross-validation on test splits |
| Prediction correlation between models | Low | Models have diverse architectures |
| Missing ground truth for blind set | N/A | Optimize on test, apply to blind |
| Weight optimization instability | Low | Use regularization or constraints |

## Recommended Approach

**Strategy: Optimized Weighted Ensemble with Task-Specific Weights**

1. **Load predictions** from all 5 January submissions (Jan-05 through Jan-09)
2. **Extract ground truth** from test split predictions (available in ensemble outputs)
3. **Optimize weights per task** using differential evolution to minimize MAE
4. **Apply optimal weights** to blind predictions for final submission
5. **Validate robustness** by comparing to simple task-best selection

**Implementation Priority:**
1. Create hybrid ensemble script using existing infrastructure
2. Optimize weights using test set predictions
3. Generate hybrid blind predictions
4. Submit and compare to individual model performance

## Implementation Guidance

- **Objectives**: Minimize MA-RAE by combining best aspects of each model
- **Key Tasks**:
  1. Load predictions from 5 submissions
  2. Optimize per-task weights using differential evolution
  3. Generate weighted ensemble blind predictions
- **Dependencies**: scipy, numpy, pandas, existing prediction files
- **Success Criteria**: MA-RAE < 0.56, overall rank < 15/307 (top 5%)
