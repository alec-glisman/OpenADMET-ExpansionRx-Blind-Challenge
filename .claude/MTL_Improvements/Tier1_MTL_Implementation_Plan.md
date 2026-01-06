# Tier 1 Multi-Task Learning Implementation Plan

**Project**: OpenADMET ExpansionRx Challenge
**Date**: January 3, 2026 (Revised)
**Version**: 2.0
**Target**: Implement uncertainty weighting, enhanced task affinity, and PCGrad for ADMET prediction
**Expected Improvement**: 10-15% MA-RAE reduction (0.60 → 0.52-0.54)

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 3, 2026 | Initial plan |
| 2.0 | Jan 3, 2026 | **Major revisions**: Fixed uncertainty loss formula, added Phase 0 diagnostics, addressed integration issues, corrected config placement |

---

## Table of Contents

1. [Overview](#overview)
2. [**Phase 0: Diagnostics (NEW)**](#phase-0-diagnostics)
3. [Phase 1: Uncertainty-Weighted Loss](#phase-1-uncertainty-weighted-loss)
4. [Phase 2: Enhanced Task Affinity Grouping](#phase-2-enhanced-task-affinity-grouping)
5. [Phase 3: Gradient Surgery (PCGrad)](#phase-3-gradient-surgery-pcgrad)
6. [Phase 4: Integration & Testing](#phase-4-integration--testing)
7. [Phase 5: Experiments & Ablation Studies](#phase-5-experiments--ablation-studies)
8. [Configuration Best Practices](#configuration-best-practices)
9. [Success Criteria](#success-criteria)
10. [**Known Issues & Mitigations**](#known-issues--mitigations)

---

## Overview

### Timeline (Revised)

| Phase | Duration | Description | Dependencies |
|-------|----------|-------------|--------------|
| **Phase 0** | **0.5 day** | **Diagnostic scripts (gradient conflicts, task correlations)** | None |
| Phase 1 | 1 day | Uncertainty-weighted loss implementation | Phase 0 |
| Phase 2 | 2-3 days | Enhanced task affinity with optimal grouping | Phase 0 |
| Phase 3 | 3-5 days | PCGrad gradient surgery | **Phase 0 must show conflicts > 0.1** |
| Phase 4 | 2-3 days | Integration testing | Phases 1-3 |
| Phase 5 | 7-10 days | Full experimental evaluation | Phase 4 |
| **Total** | **16-23 days** | Complete implementation to leaderboard | |

### Decision Gate

> **IMPORTANT**: Phase 3 (PCGrad) should ONLY proceed if Phase 0 diagnostics show:
>
> - Gradient conflict frequency > 0.1 per step
> - Mean cosine similarity between task gradients < 0.5
>
> If tasks are already well-aligned, PCGrad adds overhead without benefit.

### Architecture Stack

```
Existing Components (Keep):
├── ChempropModel (PyTorch Lightning)
├── Task Affinity Module (InterTaskAffinityCallback) - 1611 lines, well-tested
├── Curriculum Learning (CurriculumCallback)
├── Joint Sampler (JointSampler)
└── MLflow Tracking

New Components (Add):
├── GradientConflictDiagnostic (Phase 0) - NEW
├── UncertaintyWeightedLoss (Phase 1) - CORRECTED FORMULA
├── OptimalTaskGrouper (Phase 2) - Reuse existing infrastructure
└── PCGradCallback (Phase 3) - Lightning callback approach
```

### Integration Considerations

**Existing Infrastructure to Leverage:**

- `InterTaskAffinityCallback` already computes gradient-based affinity during training
- `TaskAffinityComputer` and `TaskGrouper` exist and are well-tested (953 lines of tests)
- Phase 2 should ENHANCE these, not replace them

**Potential Conflicts to Address:**

1. **PCGrad vs InterTaskAffinity**: Both manipulate gradients - cannot run simultaneously
2. **Uncertainty weights vs target_weights**: Need clear precedence rules
3. **Curriculum phases**: Uncertainty weighting behavior during phase transitions

---

## Phase 0: Diagnostics

**Goal**: Measure baseline gradient conflicts and task correlations BEFORE implementing complex solutions.
**Expected Output**: Quantified evidence whether PCGrad (Phase 3) is needed.
**Effort**: 0.5 days

### 0.1 Rationale

> **Why Diagnose First?**
>
> PCGrad adds significant complexity and computational overhead (~2x training time).
> If task gradients are already well-aligned (cosine > 0.5), PCGrad provides no benefit.
> This diagnostic phase produces evidence for a go/no-go decision on Phase 3.

### 0.2 Implementation

#### File: `scripts/analysis/gradient_conflict_diagnostic.py`

```python
#!/usr/bin/env python3
"""Gradient conflict diagnostic for multi-task ADMET prediction.

Measures pairwise gradient conflicts between tasks to determine if gradient
surgery (PCGrad) would provide benefit.

Usage:
    python scripts/analysis/gradient_conflict_diagnostic.py \
        -c configs/3-production/ensemble_chemprop_hpo_001.yaml \
        --num-batches 100 \
        --output-dir assets/analysis/gradient_conflicts

Output:
    - conflict_matrix.csv: NxN matrix of pairwise conflict frequencies
    - cosine_similarity_matrix.csv: NxN matrix of mean cosine similarities
    - summary_report.md: Human-readable analysis with recommendation
    - gradient_analysis.png: Visualization of conflict patterns
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

from admet.data import load_and_prepare_data
from admet.model.chemprop.model import ChempropModel

logger = logging.getLogger(__name__)

# ADMET endpoint names for labeling
TASK_NAMES = [
    "LogD", "Log_KSOL", "Log_HLM_CLint", "Log_MLM_CLint",
    "Log_Caco2_Papp", "Log_Caco2_Efflux", "Log_MPPB", "Log_MBPB", "Log_MGMB"
]


class GradientConflictAnalyzer:
    """Analyzes pairwise gradient conflicts between MTL tasks.

    For each batch, computes per-task gradients and measures:
    1. Cosine similarity between task gradient vectors
    2. Conflict frequency (cosine < 0) between task pairs
    3. Conflict magnitude when conflicts occur
    """

    def __init__(
        self,
        model: nn.Module,
        shared_params: List[nn.Parameter],
        n_tasks: int = 9,
    ):
        self.model = model
        self.shared_params = shared_params
        self.n_tasks = n_tasks

        # Storage for analysis
        self.cosine_similarities: List[np.ndarray] = []
        self.conflict_counts = np.zeros((n_tasks, n_tasks))
        self.total_counts = np.zeros((n_tasks, n_tasks))

    def compute_per_task_gradients(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """Compute gradient for each task separately.

        Returns
        -------
        Dict mapping task_idx to flattened gradient vector.
        """
        task_gradients = {}
        criterion = nn.MSELoss(reduction='none')

        for task_idx in range(self.n_tasks):
            # Get mask for this task
            task_mask = mask[:, task_idx]
            if task_mask.sum() == 0:
                continue

            # Compute task-specific loss
            task_preds = predictions[:, task_idx][task_mask]
            task_targets = targets[:, task_idx][task_mask]
            task_loss = criterion(task_preds, task_targets).mean()

            # Compute gradients w.r.t. shared parameters
            self.model.zero_grad()
            task_loss.backward(retain_graph=True)

            # Flatten gradients into single vector
            grad_vec = torch.cat([
                p.grad.flatten() if p.grad is not None
                else torch.zeros_like(p).flatten()
                for p in self.shared_params
            ])
            task_gradients[task_idx] = grad_vec.detach()

        return task_gradients

    def analyze_batch(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Analyze gradient conflicts for a single batch."""
        task_gradients = self.compute_per_task_gradients(predictions, targets, mask)

        # Compute pairwise cosine similarities
        task_indices = list(task_gradients.keys())
        cosine_matrix = np.full((self.n_tasks, self.n_tasks), np.nan)

        for i, idx_i in enumerate(task_indices):
            for j, idx_j in enumerate(task_indices):
                if idx_i >= idx_j:
                    continue

                g_i = task_gradients[idx_i]
                g_j = task_gradients[idx_j]

                # Cosine similarity
                norm_i = torch.norm(g_i)
                norm_j = torch.norm(g_j)

                if norm_i > 1e-8 and norm_j > 1e-8:
                    cosine = torch.dot(g_i, g_j) / (norm_i * norm_j)
                    cosine_val = cosine.item()

                    cosine_matrix[idx_i, idx_j] = cosine_val
                    cosine_matrix[idx_j, idx_i] = cosine_val

                    # Track conflicts (cosine < 0)
                    self.total_counts[idx_i, idx_j] += 1
                    self.total_counts[idx_j, idx_i] += 1

                    if cosine_val < 0:
                        self.conflict_counts[idx_i, idx_j] += 1
                        self.conflict_counts[idx_j, idx_i] += 1

        self.cosine_similarities.append(cosine_matrix)

    def generate_report(self, output_dir: Path) -> Dict:
        """Generate comprehensive analysis report."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compute summary statistics
        cosine_stack = np.stack(self.cosine_similarities, axis=0)
        mean_cosine = np.nanmean(cosine_stack, axis=0)
        std_cosine = np.nanstd(cosine_stack, axis=0)

        # Conflict frequency matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            conflict_freq = self.conflict_counts / self.total_counts
            conflict_freq = np.nan_to_num(conflict_freq, nan=0.0)

        # Save matrices
        pd.DataFrame(conflict_freq, index=TASK_NAMES, columns=TASK_NAMES).to_csv(
            output_dir / "conflict_matrix.csv"
        )
        pd.DataFrame(mean_cosine, index=TASK_NAMES, columns=TASK_NAMES).to_csv(
            output_dir / "cosine_similarity_matrix.csv"
        )

        # Compute summary metrics
        # Only look at upper triangle (unique pairs)
        upper_mask = np.triu(np.ones_like(conflict_freq), k=1).astype(bool)

        summary = {
            "mean_conflict_frequency": float(conflict_freq[upper_mask].mean()),
            "max_conflict_frequency": float(conflict_freq[upper_mask].max()),
            "mean_cosine_similarity": float(np.nanmean(mean_cosine[upper_mask])),
            "min_cosine_similarity": float(np.nanmin(mean_cosine[upper_mask])),
            "num_batches_analyzed": len(self.cosine_similarities),
            "pcgrad_recommended": conflict_freq[upper_mask].mean() > 0.1,
        }

        # Generate markdown report
        self._write_markdown_report(output_dir, summary, conflict_freq, mean_cosine)

        # Create visualization
        self._create_visualization(output_dir, conflict_freq, mean_cosine)

        return summary

    def _write_markdown_report(
        self,
        output_dir: Path,
        summary: Dict,
        conflict_freq: np.ndarray,
        mean_cosine: np.ndarray,
    ):
        """Write human-readable markdown report."""
        recommendation = (
            "✅ **PROCEED with Phase 3 (PCGrad)** - Significant gradient conflicts detected."
            if summary["pcgrad_recommended"]
            else "⏭️ **SKIP Phase 3 (PCGrad)** - Tasks are well-aligned, PCGrad overhead not justified."
        )

        # Find most conflicting pair
        upper_mask = np.triu(np.ones_like(conflict_freq), k=1).astype(bool)
        conflict_freq_upper = conflict_freq.copy()
        conflict_freq_upper[~upper_mask] = 0
        max_idx = np.unravel_index(conflict_freq_upper.argmax(), conflict_freq_upper.shape)

        report = f"""# Gradient Conflict Analysis Report

Generated: {pd.Timestamp.now().isoformat()}

## Summary

| Metric | Value | Threshold |
|--------|-------|-----------|
| Mean Conflict Frequency | {summary['mean_conflict_frequency']:.3f} | > 0.1 |
| Max Conflict Frequency | {summary['max_conflict_frequency']:.3f} | - |
| Mean Cosine Similarity | {summary['mean_cosine_similarity']:.3f} | < 0.5 |
| Min Cosine Similarity | {summary['min_cosine_similarity']:.3f} | - |
| Batches Analyzed | {summary['num_batches_analyzed']} | - |

## Recommendation

{recommendation}

## Most Conflicting Task Pair

**{TASK_NAMES[max_idx[0]]}** vs **{TASK_NAMES[max_idx[1]]}**
- Conflict Frequency: {conflict_freq[max_idx]:.3f}
- Mean Cosine Similarity: {mean_cosine[max_idx]:.3f}

## Interpretation

- **Conflict Frequency > 0.1**: Gradients point in opposite directions more than 10% of the time.
  This indicates tasks have competing objectives that PCGrad can resolve.

- **Cosine Similarity < 0.5**: Gradients are poorly aligned on average.
  Even when not strictly conflicting, tasks may benefit from gradient surgery.

## Next Steps

1. If PCGrad recommended: Proceed to Phase 3 implementation
2. If PCGrad NOT recommended: Focus on Phases 1-2, skip Phase 3
3. Re-run this analysis after Phase 1 (uncertainty weighting may change gradient dynamics)
"""

        (output_dir / "summary_report.md").write_text(report)
        logger.info(f"Report written to {output_dir / 'summary_report.md'}")

    def _create_visualization(
        self,
        output_dir: Path,
        conflict_freq: np.ndarray,
        mean_cosine: np.ndarray,
    ):
        """Create heatmap visualization."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Conflict frequency heatmap
            sns.heatmap(
                conflict_freq,
                xticklabels=TASK_NAMES,
                yticklabels=TASK_NAMES,
                annot=True,
                fmt=".2f",
                cmap="Reds",
                ax=axes[0],
            )
            axes[0].set_title("Gradient Conflict Frequency\n(fraction of batches with cosine < 0)")

            # Cosine similarity heatmap
            sns.heatmap(
                mean_cosine,
                xticklabels=TASK_NAMES,
                yticklabels=TASK_NAMES,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                center=0,
                ax=axes[1],
            )
            axes[1].set_title("Mean Gradient Cosine Similarity")

            plt.tight_layout()
            plt.savefig(output_dir / "gradient_analysis.png", dpi=150)
            plt.close()

            logger.info(f"Visualization saved to {output_dir / 'gradient_analysis.png'}")

        except ImportError:
            logger.warning("matplotlib/seaborn not available, skipping visualization")


def main():
    parser = argparse.ArgumentParser(description="Gradient conflict diagnostic")
    parser.add_argument("-c", "--config", required=True, help="Training config YAML")
    parser.add_argument("--num-batches", type=int, default=100, help="Batches to analyze")
    parser.add_argument("--output-dir", default="assets/analysis/gradient_conflicts")
    parser.add_argument("--split", type=int, default=0, help="Data split to use")
    parser.add_argument("--fold", type=int, default=0, help="Fold within split")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load config
    cfg = OmegaConf.load(args.config)
    output_dir = Path(args.output_dir)

    logger.info(f"Running gradient conflict analysis with config: {args.config}")

    # Initialize model (without training)
    model_wrapper = ChempropModel(cfg)
    model, _ = model_wrapper._build_model()
    model.eval()

    # Get shared parameters (encoder, not FFN heads)
    shared_params = list(model.message_passing.parameters())
    logger.info(f"Analyzing gradients for {len(shared_params)} shared parameter groups")

    # Load data
    data = load_and_prepare_data(cfg)
    train_loader = model_wrapper._create_dataloader(data, "train")

    # Initialize analyzer
    analyzer = GradientConflictAnalyzer(
        model=model,
        shared_params=shared_params,
        n_tasks=9,
    )

    # Analyze batches
    model.train()  # Enable gradients
    for batch_idx, batch in enumerate(tqdm(train_loader, total=args.num_batches)):
        if batch_idx >= args.num_batches:
            break

        # Forward pass
        with torch.enable_grad():
            predictions = model(batch)
            analyzer.analyze_batch(
                predictions=predictions,
                targets=batch['targets'],
                mask=~torch.isnan(batch['targets']),
            )

    # Generate report
    summary = analyzer.generate_report(output_dir)

    # Print recommendation
    print("\n" + "="*60)
    if summary["pcgrad_recommended"]:
        print("📊 RESULT: Significant gradient conflicts detected!")
        print(f"   Mean conflict frequency: {summary['mean_conflict_frequency']:.3f}")
        print("   → PROCEED with Phase 3 (PCGrad)")
    else:
        print("📊 RESULT: Tasks are well-aligned, minimal gradient conflicts.")
        print(f"   Mean conflict frequency: {summary['mean_conflict_frequency']:.3f}")
        print("   → SKIP Phase 3, focus on Phases 1-2")
    print("="*60)


if __name__ == "__main__":
    main()
```

### 0.3 Acceptance Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Script runs without error | Exit code | 0 |
| Outputs conflict matrix | File exists | conflict_matrix.csv |
| Outputs similarity matrix | File exists | cosine_similarity_matrix.csv |
| Outputs summary report | File exists | summary_report.md |
| Report includes recommendation | Contains "PROCEED" or "SKIP" | ✓ |
| Visualization generated | File exists | gradient_analysis.png |

### 0.4 Unit Tests

#### File: `tests/test_gradient_diagnostic.py`

```python
"""Unit tests for gradient conflict diagnostic."""

import numpy as np
import pytest
import torch
import torch.nn as nn

# Mock import for testing without full model
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "analysis"))


class TestGradientConflictAnalyzer:
    """Tests for GradientConflictAnalyzer class."""

    def test_conflicting_gradients_detected(self):
        """Verify conflicts detected when gradients oppose."""
        # Create mock model with simple linear layer
        model = nn.Linear(10, 9)
        shared_params = list(model.parameters())

        # Create analyzer
        from gradient_conflict_diagnostic import GradientConflictAnalyzer
        analyzer = GradientConflictAnalyzer(model, shared_params, n_tasks=9)

        # Create predictions and targets that will cause opposing gradients
        # Task 0: wants weights to increase (pred < target)
        # Task 1: wants weights to decrease (pred > target)
        predictions = torch.zeros(32, 9)
        targets = torch.zeros(32, 9)

        # Engineer gradient conflict between task 0 and 1
        predictions[:, 0] = -1.0  # pred < target → gradient pushes weights up
        targets[:, 0] = 1.0
        predictions[:, 1] = 1.0   # pred > target → gradient pushes weights down
        targets[:, 1] = -1.0

        mask = torch.ones(32, 9, dtype=torch.bool)

        analyzer.analyze_batch(predictions, targets, mask)

        # Should detect conflict between task 0 and 1
        assert analyzer.conflict_counts[0, 1] > 0 or analyzer.conflict_counts[1, 0] > 0

    def test_aligned_gradients_no_conflict(self):
        """Verify no conflicts when gradients align."""
        model = nn.Linear(10, 9)
        shared_params = list(model.parameters())

        from gradient_conflict_diagnostic import GradientConflictAnalyzer
        analyzer = GradientConflictAnalyzer(model, shared_params, n_tasks=9)

        # All tasks have same gradient direction
        predictions = torch.zeros(32, 9)
        targets = torch.ones(32, 9)  # All tasks: pred < target
        mask = torch.ones(32, 9, dtype=torch.bool)

        analyzer.analyze_batch(predictions, targets, mask)

        # Generate report
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = analyzer.generate_report(Path(tmpdir))

            # Mean conflict should be low for aligned gradients
            assert summary["mean_conflict_frequency"] < 0.5

    def test_report_generation(self):
        """Verify all output files created."""
        model = nn.Linear(10, 9)
        shared_params = list(model.parameters())

        from gradient_conflict_diagnostic import GradientConflictAnalyzer
        analyzer = GradientConflictAnalyzer(model, shared_params, n_tasks=9)

        # Add some dummy data
        predictions = torch.randn(32, 9)
        targets = torch.randn(32, 9)
        mask = torch.ones(32, 9, dtype=torch.bool)

        for _ in range(5):
            analyzer.analyze_batch(predictions, targets, mask)

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            summary = analyzer.generate_report(output_dir)

            # Check all files created
            assert (output_dir / "conflict_matrix.csv").exists()
            assert (output_dir / "cosine_similarity_matrix.csv").exists()
            assert (output_dir / "summary_report.md").exists()

            # Check summary has expected keys
            assert "mean_conflict_frequency" in summary
            assert "pcgrad_recommended" in summary
```

### 0.5 Decision Criteria

After running the diagnostic:

| Outcome | Mean Conflict Freq | Action |
|---------|-------------------|--------|
| High Conflicts | > 0.15 | Phase 3 is HIGH priority |
| Moderate Conflicts | 0.10 - 0.15 | Phase 3 is MEDIUM priority |
| Low Conflicts | < 0.10 | SKIP Phase 3, focus on Phase 1-2 |

---

## Phase 1: Uncertainty-Weighted Loss

**Goal**: Automatically balance task contributions based on learned uncertainty parameters.
**Expected Impact**: 5-7% MA-RAE improvement, MGMB R² +0.06
**Effort**: 1 day

### 1.1 Implementation

#### File: `src/admet/model/chemprop/uncertainty_loss.py`

```python
"""Uncertainty-weighted multi-task loss.

Implements automatic task weighting via learned uncertainty parameters.
Based on Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to
Weigh Losses for Scene Geometry and Semantics" (CVPR).

References:
    - Original paper: https://arxiv.org/abs/1705.07115
    - Industry usage: Boehringer Ingelheim (2024), Bayer (2023)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class UncertaintyWeightedLoss(nn.Module):
    """Multi-task loss with learned uncertainty-based task weighting.

    Learns per-task uncertainty parameters σ_i that automatically balance
    task contributions. Tasks with high intrinsic noise get lower weight,
    while clean tasks drive training.

    Mathematical Derivation (Kendall et al. 2018):
    -----------------------------------------------
    For homoscedastic regression, model predictive uncertainty via:
        p(y|f(x)) = N(f(x), σ²)

    Log likelihood for task i:
        log p(y_i|f_i) = -1/(2σ_i²) * ||y_i - f_i||² - log(σ_i) + const

    Negative log likelihood (loss) summed over tasks:
        L = Σ_i [L_i/(2σ_i²) + log(σ_i)]

    Parameterization for numerical stability:
        Let s_i = log(σ_i²)  (learnable parameter)
        Then σ_i² = exp(s_i), and log(σ_i) = s_i/2

        L = Σ_i [L_i * exp(-s_i) / 2 + s_i / 2]

    Implementation Notes:
    - `log_vars` stores s_i = log(σ²), NOT log(σ)
    - Regularization term is `0.5 * log_vars` = `log(σ)`
    - Precision term is `exp(-log_vars)` = `1/σ²`

    Parameters
    ----------
    n_tasks : int
        Number of tasks (ADMET endpoints).
    base_criterion : nn.Module, optional
        Base loss function (default: MSELoss).
    init_log_vars : float, optional
        Initial value for log(σ²). Default 0.0 (σ=1.0).
    min_log_var : float, optional
        Minimum allowed log(σ²) to prevent numerical issues. Default -10.0.
    max_log_var : float, optional
        Maximum allowed log(σ²) to prevent degenerate solutions. Default 10.0.

    Attributes
    ----------
    log_vars : nn.Parameter
        Learned log(σ²) parameters, shape [n_tasks]. NOT log(σ)!
    n_tasks : int
        Number of tasks.

    Examples
    --------
    >>> criterion = UncertaintyWeightedLoss(n_tasks=9)
    >>> predictions = model(batch)  # [batch_size, 9]
    >>> targets = batch['targets']   # [batch_size, 9]
    >>> mask = ~torch.isnan(targets) # [batch_size, 9]
    >>> loss = criterion(predictions, targets, mask)
    >>> loss.backward()

    Notes
    -----
    - Sparse tasks (MGMB, MBPB) will learn higher σ → lower weight
    - Clean tasks (LogD, KSOL) will learn lower σ → higher weight
    - Automatic balancing requires no manual tuning
    - σ values can be inspected via `get_task_weights()` for interpretability

    References
    ----------
    .. [1] Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning
           using uncertainty to weigh losses for scene geometry and semantics.
           CVPR. https://arxiv.org/abs/1705.07115

    .. [2] Reference implementation: yaringal/multi-task-learning-example
           https://github.com/yaringal/multi-task-learning-example
    """

    def __init__(
        self,
        n_tasks: int,
        base_criterion: Optional[nn.Module] = None,
        init_log_vars: float = 0.0,
        min_log_var: float = -10.0,
        max_log_var: float = 10.0,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.base_criterion = base_criterion or nn.MSELoss(reduction='none')
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        # Initialize log(σ²) for numerical stability
        # σ²=1 initially (neutral weighting)
        self.log_vars = nn.Parameter(torch.full((n_tasks,), init_log_vars))

        logger.info(
            f"Initialized UncertaintyWeightedLoss with {n_tasks} tasks, "
            f"init_log_var={init_log_vars:.2f}"
        )

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """Compute uncertainty-weighted loss.

        Parameters
        ----------
        predictions : Tensor
            Model predictions, shape [batch_size, n_tasks].
        targets : Tensor
            Ground truth targets, shape [batch_size, n_tasks].
            Can contain NaN for missing values.
        mask : Tensor, optional
            Boolean mask indicating valid targets, shape [batch_size, n_tasks].
            If None, computed from targets (True where not NaN).

        Returns
        -------
        Tensor
            Scalar weighted loss.
        """
        if mask is None:
            mask = ~torch.isnan(targets)

        # Replace NaN with 0 for computation (masked out anyway)
        targets_clean = torch.nan_to_num(targets, nan=0.0)

        # Compute per-sample, per-task losses
        # Shape: [batch_size, n_tasks]
        task_losses = self.base_criterion(predictions, targets_clean)

        # Mask out invalid targets
        task_losses = task_losses * mask

        # Compute mean loss per task (averaging over batch and valid samples)
        # Shape: [n_tasks]
        counts = mask.sum(dim=0).clamp(min=1.0)  # Avoid division by zero
        mean_task_losses = task_losses.sum(dim=0) / counts

        # Clamp log_vars to prevent numerical issues
        log_vars_clamped = torch.clamp(
            self.log_vars,
            min=self.min_log_var,
            max=self.max_log_var
        )

        # Apply uncertainty weighting: L = Σ_i [L_i * exp(-s_i) / 2 + s_i / 2]
        # where s_i = log(σ²), so:
        #   - exp(-s_i) = 1/σ² (precision)
        #   - s_i / 2 = log(σ²) / 2 = log(σ) (regularization)
        precision = torch.exp(-log_vars_clamped)  # 1/σ²
        weighted_losses = 0.5 * precision * mean_task_losses + 0.5 * log_vars_clamped

        # Sum across tasks
        total_loss = weighted_losses.sum()

        return total_loss

    def get_task_weights(self) -> Dict[int, float]:
        """Get current task weights (precision = 1/σ²).

        Returns
        -------
        Dict[int, float]
            Mapping from task index to weight (precision).
            Higher weight = model is more confident in this task.
        """
        with torch.no_grad():
            precisions = torch.exp(-self.log_vars).cpu().numpy()
        return {i: float(p) for i, p in enumerate(precisions)}

    def get_task_uncertainties(self) -> Dict[int, float]:
        """Get current task uncertainties (σ).

        Returns
        -------
        Dict[int, float]
            Mapping from task index to uncertainty σ.
            Higher σ = task is noisier / more uncertain.
        """
        with torch.no_grad():
            sigmas = torch.exp(0.5 * self.log_vars).cpu().numpy()
        return {i: float(s) for i, s in enumerate(sigmas)}

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"n_tasks={self.n_tasks}, init_log_vars={self.log_vars.mean().item():.3f}"
```

#### File: `src/admet/model/chemprop/config.py` (additions)

```python
# Add to existing ModelConfig dataclass

@dataclass
class ModelConfig:
    # ... existing fields ...

    # Uncertainty weighting
    use_uncertainty_weighting: bool = False
    uncertainty_init_log_vars: float = 0.0
    uncertainty_min_log_var: float = -10.0
    uncertainty_max_log_var: float = 10.0
```

#### File: `src/admet/model/chemprop/model.py` (modifications)

```python
# Add import
from admet.model.chemprop.uncertainty_loss import UncertaintyWeightedLoss

class ChempropModel:
    def __init__(self, ...):
        # ... existing code ...

        # Initialize uncertainty-weighted loss if enabled
        if self.config.model.use_uncertainty_weighting:
            logger.info("Using uncertainty-weighted loss")
            self.criterion = UncertaintyWeightedLoss(
                n_tasks=len(self.config.data.target_cols),
                init_log_vars=self.config.model.uncertainty_init_log_vars,
                min_log_var=self.config.model.uncertainty_min_log_var,
                max_log_var=self.config.model.uncertainty_max_log_var,
            )
        else:
            # Standard MSE loss (existing behavior)
            self.criterion = nn.MSELoss()

    def _log_metrics(self, trainer, phase="val"):
        """Log metrics to MLflow."""
        # ... existing metric logging ...

        # Log task uncertainties if using uncertainty weighting
        if isinstance(self.criterion, UncertaintyWeightedLoss):
            uncertainties = self.criterion.get_task_uncertainties()
            weights = self.criterion.get_task_weights()

            for task_idx, task_name in enumerate(self.config.data.target_cols):
                mlflow.log_metric(
                    f"uncertainty/{task_name}",
                    uncertainties[task_idx],
                    step=trainer.current_epoch
                )
                mlflow.log_metric(
                    f"task_weight/{task_name}",
                    weights[task_idx],
                    step=trainer.current_epoch
                )
```

### 1.2 Unit Tests

#### File: `tests/unit/test_uncertainty_loss.py`

```python
"""Unit tests for uncertainty-weighted loss."""

import pytest
import torch
import torch.nn as nn

from admet.model.chemprop.uncertainty_loss import UncertaintyWeightedLoss


class TestUncertaintyWeightedLoss:
    """Tests for UncertaintyWeightedLoss."""

    @pytest.fixture
    def n_tasks(self):
        return 9

    @pytest.fixture
    def batch_size(self):
        return 32

    @pytest.fixture
    def loss_fn(self, n_tasks):
        return UncertaintyWeightedLoss(n_tasks=n_tasks)

    def test_initialization(self, loss_fn, n_tasks):
        """Test loss function initializes correctly."""
        assert loss_fn.n_tasks == n_tasks
        assert loss_fn.log_vars.shape == (n_tasks,)
        assert torch.allclose(loss_fn.log_vars, torch.zeros(n_tasks))

    def test_forward_shape(self, loss_fn, batch_size, n_tasks):
        """Test forward pass returns scalar loss."""
        predictions = torch.randn(batch_size, n_tasks)
        targets = torch.randn(batch_size, n_tasks)

        loss = loss_fn(predictions, targets)

        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0  # Positive loss

    def test_nan_handling(self, loss_fn, batch_size, n_tasks):
        """Test loss correctly handles NaN targets."""
        predictions = torch.randn(batch_size, n_tasks)
        targets = torch.randn(batch_size, n_tasks)

        # Introduce 50% NaN sparsity in tasks 7, 8 (MBPB, MGMB)
        targets[:, 7] = torch.where(
            torch.rand(batch_size) > 0.5,
            targets[:, 7],
            torch.tensor(float('nan'))
        )
        targets[:, 8] = torch.where(
            torch.rand(batch_size) > 0.5,
            targets[:, 8],
            torch.tensor(float('nan'))
        )

        loss = loss_fn(predictions, targets)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flow(self, loss_fn, batch_size, n_tasks):
        """Test gradients flow to both model and log_vars."""
        predictions = torch.randn(batch_size, n_tasks, requires_grad=True)
        targets = torch.randn(batch_size, n_tasks)

        loss = loss_fn(predictions, targets)
        loss.backward()

        # Check gradients exist
        assert predictions.grad is not None
        assert loss_fn.log_vars.grad is not None

        # Check gradients are non-zero
        assert not torch.allclose(predictions.grad, torch.zeros_like(predictions.grad))
        assert not torch.allclose(loss_fn.log_vars.grad, torch.zeros_like(loss_fn.log_vars.grad))

    def test_uncertainty_adaptation(self, loss_fn, batch_size, n_tasks):
        """Test that uncertainties adapt during training."""
        # Simulate noisy task (high loss)
        predictions = torch.zeros(batch_size, n_tasks)
        targets = torch.zeros(batch_size, n_tasks)
        targets[:, 0] = 10.0  # Task 0 has large error

        optimizer = torch.optim.Adam(loss_fn.parameters(), lr=0.01)

        initial_log_var_0 = loss_fn.log_vars[0].item()

        # Train for a few steps
        for _ in range(50):
            optimizer.zero_grad()
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

        final_log_var_0 = loss_fn.log_vars[0].item()

        # Task 0 should have higher uncertainty (larger log_var)
        assert final_log_var_0 > initial_log_var_0

    def test_get_task_weights(self, loss_fn, n_tasks):
        """Test task weight extraction."""
        weights = loss_fn.get_task_weights()

        assert len(weights) == n_tasks
        assert all(isinstance(k, int) for k in weights.keys())
        assert all(isinstance(v, float) for v in weights.values())
        assert all(v > 0 for v in weights.values())

    def test_get_task_uncertainties(self, loss_fn, n_tasks):
        """Test uncertainty extraction."""
        uncertainties = loss_fn.get_task_uncertainties()

        assert len(uncertainties) == n_tasks
        assert all(isinstance(v, float) for v in uncertainties.values())
        assert all(v > 0 for v in uncertainties.values())

    def test_clamping(self, n_tasks):
        """Test log_var clamping prevents numerical issues."""
        loss_fn = UncertaintyWeightedLoss(
            n_tasks=n_tasks,
            min_log_var=-5.0,
            max_log_var=5.0
        )

        # Force extreme values
        loss_fn.log_vars.data = torch.tensor([-100.0] * n_tasks)

        predictions = torch.randn(16, n_tasks)
        targets = torch.randn(16, n_tasks)

        loss = loss_fn(predictions, targets)

        # Should not explode
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() < 1e6
```

### 1.3 Configuration

#### File: `configs/tier1/uncertainty_weighting.yaml`

```yaml
# Tier 1 - Uncertainty Weighting Only
# Expected: 5-7% MA-RAE improvement

model:
  # MPNN architecture (current best)
  depth: 3
  message_hidden_dim: 700

  # FFN
  ffn_type: regression
  num_layers: 4
  hidden_dim: 200
  dropout: 0.15
  batch_norm: true

  # Uncertainty weighting (NEW)
  use_uncertainty_weighting: true
  uncertainty_init_log_vars: 0.0  # Start with σ=1.0 (neutral)
  uncertainty_min_log_var: -10.0  # Prevent numerical issues
  uncertainty_max_log_var: 10.0   # Prevent degenerate solutions

  # Optimization
  criterion: MSE  # Base criterion (wrapped by uncertainty weighting)
  init_lr: 0.00113
  max_lr: 0.000227
  final_lr: 0.000113
  warmup_epochs: 5
  patience: 15
  max_epochs: 150
  batch_size: 128

  # Task sampling
  task_sampling_alpha: 0.02

data:
  data_dir: "data/expansionrx_challenge"
  smiles_col: "SMILES"
  target_cols:
    - "LogD"
    - "KSOL"
    - "HLM CLint"
    - "MLM CLint"
    - "Caco-2 Papp"
    - "Caco-2 Efflux"
    - "MPPB"
    - "MBPB"
    - "MGMB"

mlflow:
  enabled: true
  experiment_name: "tier1_uncertainty_weighting"
  run_name: "uncertainty_baseline"
  tracking_uri: "http://127.0.0.1:8084"
```

### 1.4 Acceptance Test

#### File: `tests/acceptance/test_uncertainty_weighting.py`

```python
"""Acceptance tests for uncertainty weighting on real data."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

from admet.model.chemprop.model import ChempropModel


class TestUncertaintyWeightingAcceptance:
    """Acceptance tests using ExpansionRx subset."""

    @pytest.fixture(scope="class")
    def expansionrx_subset(self):
        """Load 1000-compound subset for fast testing."""
        data_path = Path("data/expansionrx_challenge/train.csv")
        if not data_path.exists():
            pytest.skip("ExpansionRx data not available")

        df = pd.read_csv(data_path)
        # Take first 1000 compounds (alphabetical, similar to temporal split)
        df_sorted = df.sort_values('Molecule Name')
        return df_sorted.iloc[:1000]

    @pytest.fixture
    def config_uncertainty(self):
        """Config with uncertainty weighting."""
        return OmegaConf.create({
            'model': {
                'depth': 3,
                'message_hidden_dim': 500,
                'num_layers': 2,
                'hidden_dim': 300,
                'dropout': 0.1,
                'batch_norm': True,
                'ffn_type': 'regression',
                'use_uncertainty_weighting': True,
                'uncertainty_init_log_vars': 0.0,
                'criterion': 'MSE',
                'max_epochs': 20,  # Short for testing
                'patience': 5,
                'batch_size': 64,
                'init_lr': 1e-3,
                'max_lr': 1e-3,
                'final_lr': 1e-4,
                'warmup_epochs': 2,
            },
            'data': {
                'data_dir': 'data/expansionrx_challenge',
                'smiles_col': 'SMILES',
                'target_cols': [
                    'LogD', 'KSOL', 'HLM CLint', 'MLM CLint',
                    'Caco-2 Papp', 'Caco-2 Efflux', 'MPPB', 'MBPB', 'MGMB'
                ],
            },
            'mlflow': {
                'enabled': False  # Disable for testing
            }
        })

    @pytest.fixture
    def config_baseline(self, config_uncertainty):
        """Config without uncertainty weighting (baseline)."""
        config = OmegaConf.create(config_uncertainty)
        config.model.use_uncertainty_weighting = False
        return config

    def test_uncertainty_trains_successfully(self, expansionrx_subset, config_uncertainty):
        """Test uncertainty weighting trains without errors."""
        train_df = expansionrx_subset.iloc[:800]
        val_df = expansionrx_subset.iloc[800:]

        model = ChempropModel(
            df_train=train_df,
            df_validation=val_df,
            **config_uncertainty.model,
            **config_uncertainty.data
        )

        import time
        start = time.time()
        model.fit()
        elapsed = time.time() - start

        # Should complete within reasonable time (2 min on 3080)
        assert elapsed < 180

        # Should have trained
        assert model.trainer.current_epoch > 0

    def test_uncertainty_improves_sparse_tasks(
        self,
        expansionrx_subset,
        config_uncertainty,
        config_baseline
    ):
        """Test uncertainty weighting improves sparse task performance."""
        train_df = expansionrx_subset.iloc[:800]
        val_df = expansionrx_subset.iloc[800:]

        # Train baseline
        model_baseline = ChempropModel(
            df_train=train_df,
            df_validation=val_df,
            **config_baseline.model,
            **config_baseline.data
        )
        model_baseline.fit()
        baseline_metrics = model_baseline.evaluate(val_df)

        # Train with uncertainty weighting
        model_uncertainty = ChempropModel(
            df_train=train_df,
            df_validation=val_df,
            **config_uncertainty.model,
            **config_uncertainty.data
        )
        model_uncertainty.fit()
        uncertainty_metrics = model_uncertainty.evaluate(val_df)

        # Check sparse tasks improved (MBPB index 7, MGMB index 8)
        sparse_tasks = [7, 8]

        for task_idx in sparse_tasks:
            baseline_r2 = baseline_metrics['r2_per_task'][task_idx]
            uncertainty_r2 = uncertainty_metrics['r2_per_task'][task_idx]

            # Expect improvement or at worst minor degradation
            assert uncertainty_r2 >= baseline_r2 - 0.05

    def test_learned_uncertainties_reflect_sparsity(
        self,
        expansionrx_subset,
        config_uncertainty
    ):
        """Test learned uncertainties are higher for sparse tasks."""
        train_df = expansionrx_subset.iloc[:800]
        val_df = expansionrx_subset.iloc[800:]

        model = ChempropModel(
            df_train=train_df,
            df_validation=val_df,
            **config_uncertainty.model,
            **config_uncertainty.data
        )
        model.fit()

        # Get learned uncertainties
        uncertainties = model.criterion.get_task_uncertainties()

        # Sparse tasks (MBPB, MGMB) should have higher uncertainty
        sparse_uncertainties = [uncertainties[7], uncertainties[8]]
        abundant_uncertainties = [uncertainties[0], uncertainties[1]]  # LogD, KSOL

        avg_sparse = np.mean(sparse_uncertainties)
        avg_abundant = np.mean(abundant_uncertainties)

        # Sparse should be more uncertain
        assert avg_sparse > avg_abundant * 1.1  # At least 10% higher
```

---

## Phase 2: Enhanced Task Affinity Grouping

**Goal**: Optimize task grouping via data-driven affinity analysis.
**Expected Impact**: 3-5% MA-RAE improvement
**Effort**: 2-3 days

### 2.1 Implementation

#### File: `src/admet/model/chemprop/optimal_grouper.py`

```python
"""Optimal task grouping via affinity analysis.

Automatically determines optimal number of task groups and assignments
using silhouette analysis and domain knowledge validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


class ClusteringMethod(Enum):
    """Supported clustering methods."""
    AGGLOMERATIVE = "agglomerative"
    SPECTRAL = "spectral"
    KMEANS = "kmeans"


@dataclass
class TaskGroup:
    """Represents a group of related tasks."""
    group_id: int
    task_indices: List[int]
    task_names: Optional[List[str]] = None
    avg_affinity: float = 0.0  # Average within-group affinity

    def __repr__(self):
        if self.task_names:
            tasks_str = ", ".join(self.task_names)
        else:
            tasks_str = str(self.task_indices)
        return f"Group{self.group_id}({tasks_str}, aff={self.avg_affinity:.3f})"


class OptimalTaskGrouper:
    """Automatically find optimal task grouping via affinity analysis.

    Uses silhouette coefficient to determine optimal number of groups,
    then performs clustering with the chosen method.

    Parameters
    ----------
    affinity_matrix : np.ndarray or torch.Tensor
        Task affinity matrix [n_tasks, n_tasks].
        Higher values = more similar tasks.
    task_names : List[str], optional
        Names of tasks for interpretability.
    min_groups : int, default=2
        Minimum number of groups to consider.
    max_groups : int, default=6
        Maximum number of groups to consider.
    clustering_method : ClusteringMethod or str, default='agglomerative'
        Clustering method to use.
    allow_singleton : bool, default=True
        Allow groups with single task (for sparse/isolated tasks).
    domain_constraints : Dict[str, List[int]], optional
        Domain knowledge constraints, e.g.,
        {'lipophilicity': [0, 1], 'clearance': [2, 3]}.
        Tasks in same constraint must be in same group.

    Attributes
    ----------
    optimal_n_groups : int
        Optimal number of groups determined by silhouette analysis.
    task_groups : List[TaskGroup]
        Final task groupings.
    silhouette_scores : Dict[int, float]
        Silhouette scores for each n_groups tried.

    Examples
    --------
    >>> affinity = TaskAffinityComputer.compute_task_affinity(model, loader)
    >>> grouper = OptimalTaskGrouper(
    ...     affinity_matrix=affinity,
    ...     task_names=['LogD', 'KSOL', 'HLM', 'MLM', ...]
    ... )
    >>> grouper.find_optimal_groups()
    >>> print(f"Optimal n_groups: {grouper.optimal_n_groups}")
    >>> print(f"Groups: {grouper.task_groups}")
    """

    def __init__(
        self,
        affinity_matrix: np.ndarray | torch.Tensor,
        task_names: Optional[List[str]] = None,
        min_groups: int = 2,
        max_groups: int = 6,
        clustering_method: ClusteringMethod | str = ClusteringMethod.AGGLOMERATIVE,
        allow_singleton: bool = True,
        domain_constraints: Optional[Dict[str, List[int]]] = None,
    ):
        # Convert to numpy if needed
        if isinstance(affinity_matrix, torch.Tensor):
            affinity_matrix = affinity_matrix.cpu().numpy()

        self.affinity_matrix = affinity_matrix
        self.n_tasks = affinity_matrix.shape[0]
        self.task_names = task_names or [f"Task_{i}" for i in range(self.n_tasks)]
        self.min_groups = min_groups
        self.max_groups = min(max_groups, self.n_tasks)
        self.clustering_method = (
            ClusteringMethod(clustering_method)
            if isinstance(clustering_method, str)
            else clustering_method
        )
        self.allow_singleton = allow_singleton
        self.domain_constraints = domain_constraints or {}

        self.optimal_n_groups: Optional[int] = None
        self.task_groups: Optional[List[TaskGroup]] = None
        self.silhouette_scores: Dict[int, float] = {}

    def find_optimal_groups(self, verbose: bool = True) -> List[TaskGroup]:
        """Find optimal number of groups and perform clustering.

        Parameters
        ----------
        verbose : bool, default=True
            Print progress and results.

        Returns
        -------
        List[TaskGroup]
            Optimal task groupings.
        """
        if verbose:
            logger.info(f"Finding optimal task grouping ({self.min_groups}-{self.max_groups} groups)...")

        best_score = -1.0
        best_n = self.min_groups

        for n_groups in range(self.min_groups, self.max_groups + 1):
            labels = self._cluster(n_groups)

            # Skip if we got degenerate clustering (all in one group)
            if len(np.unique(labels)) < 2:
                continue

            # Compute silhouette score
            # Use distance = 1 - affinity for metric space
            distance_matrix = 1.0 - self.affinity_matrix
            score = silhouette_score(distance_matrix, labels, metric='precomputed')

            self.silhouette_scores[n_groups] = score

            if verbose:
                logger.info(f"  n_groups={n_groups}: silhouette={score:.4f}")

            if score > best_score:
                best_score = score
                best_n = n_groups

        self.optimal_n_groups = best_n

        if verbose:
            logger.info(f"Optimal n_groups: {best_n} (silhouette={best_score:.4f})")

        # Perform final clustering with optimal n_groups
        labels = self._cluster(best_n)
        self.task_groups = self._labels_to_groups(labels)

        # Validate domain constraints if provided
        if self.domain_constraints:
            self._validate_domain_constraints()

        if verbose:
            logger.info("Task groups:")
            for group in self.task_groups:
                logger.info(f"  {group}")

        return self.task_groups

    def _cluster(self, n_groups: int) -> np.ndarray:
        """Perform clustering with specified number of groups.

        Parameters
        ----------
        n_groups : int
            Number of groups to create.

        Returns
        -------
        np.ndarray
            Cluster labels, shape [n_tasks].
        """
        distance_matrix = 1.0 - self.affinity_matrix

        if self.clustering_method == ClusteringMethod.AGGLOMERATIVE:
            clusterer = AgglomerativeClustering(
                n_clusters=n_groups,
                metric='precomputed',
                linkage='average'
            )
            labels = clusterer.fit_predict(distance_matrix)

        elif self.clustering_method == ClusteringMethod.SPECTRAL:
            clusterer = SpectralClustering(
                n_clusters=n_groups,
                affinity='precomputed',
                assign_labels='kmeans',
                random_state=42
            )
            labels = clusterer.fit_predict(self.affinity_matrix)

        elif self.clustering_method == ClusteringMethod.KMEANS:
            # KMeans on affinity matrix rows as features
            clusterer = KMeans(n_clusters=n_groups, random_state=42)
            labels = clusterer.fit_predict(self.affinity_matrix)

        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")

        return labels

    def _labels_to_groups(self, labels: np.ndarray) -> List[TaskGroup]:
        """Convert cluster labels to TaskGroup objects.

        Parameters
        ----------
        labels : np.ndarray
            Cluster labels for each task.

        Returns
        -------
        List[TaskGroup]
            List of TaskGroup objects.
        """
        unique_labels = np.unique(labels)
        groups = []

        for group_id in unique_labels:
            task_indices = np.where(labels == group_id)[0].tolist()
            task_names = [self.task_names[i] for i in task_indices]

            # Compute average within-group affinity
            if len(task_indices) > 1:
                group_affinities = []
                for i in range(len(task_indices)):
                    for j in range(i + 1, len(task_indices)):
                        idx_i, idx_j = task_indices[i], task_indices[j]
                        group_affinities.append(self.affinity_matrix[idx_i, idx_j])
                avg_affinity = np.mean(group_affinities)
            else:
                # Singleton group
                avg_affinity = 1.0

            group = TaskGroup(
                group_id=int(group_id),
                task_indices=task_indices,
                task_names=task_names,
                avg_affinity=float(avg_affinity)
            )
            groups.append(group)

        # Sort by group ID
        groups.sort(key=lambda g: g.group_id)

        # Filter out singletons if not allowed
        if not self.allow_singleton:
            groups = [g for g in groups if len(g.task_indices) > 1]

        return groups

    def _validate_domain_constraints(self):
        """Validate that domain constraints are satisfied.

        Warns if tasks that should be together are in different groups.
        """
        for constraint_name, task_indices in self.domain_constraints.items():
            # Find which groups these tasks are in
            groups_for_constraint = set()
            for task_idx in task_indices:
                for group in self.task_groups:
                    if task_idx in group.task_indices:
                        groups_for_constraint.add(group.group_id)
                        break

            if len(groups_for_constraint) > 1:
                logger.warning(
                    f"Domain constraint '{constraint_name}' violated: "
                    f"tasks {task_indices} split across groups {groups_for_constraint}"
                )

    def get_task_to_group_mapping(self) -> Dict[int, int]:
        """Get mapping from task index to group ID.

        Returns
        -------
        Dict[int, int]
            Mapping {task_idx: group_id}.
        """
        if self.task_groups is None:
            raise ValueError("Must call find_optimal_groups() first")

        mapping = {}
        for group in self.task_groups:
            for task_idx in group.task_indices:
                mapping[task_idx] = group.group_id

        return mapping

    def plot_silhouette_analysis(self, save_path: Optional[str] = None):
        """Plot silhouette scores vs. number of groups.

        Parameters
        ----------
        save_path : str, optional
            Path to save figure. If None, displays interactively.
        """
        import matplotlib.pyplot as plt

        n_groups_list = sorted(self.silhouette_scores.keys())
        scores = [self.silhouette_scores[n] for n in n_groups_list]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(n_groups_list, scores, 'o-', linewidth=2, markersize=8)

        # Highlight optimal
        if self.optimal_n_groups is not None:
            optimal_score = self.silhouette_scores[self.optimal_n_groups]
            ax.plot(self.optimal_n_groups, optimal_score, 'r*', markersize=15,
                   label=f'Optimal (n={self.optimal_n_groups})')
            ax.legend()

        ax.set_xlabel('Number of Task Groups', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontsize=12)
        ax.set_title('Task Grouping: Silhouette Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved silhouette plot to {save_path}")
        else:
            plt.show()

    def plot_affinity_matrix_grouped(self, save_path: Optional[str] = None):
        """Plot affinity matrix with tasks reordered by group.

        Parameters
        ----------
        save_path : str, optional
            Path to save figure.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.task_groups is None:
            raise ValueError("Must call find_optimal_groups() first")

        # Reorder tasks by group
        ordered_indices = []
        for group in self.task_groups:
            ordered_indices.extend(group.task_indices)

        ordered_affinity = self.affinity_matrix[ordered_indices, :][:, ordered_indices]
        ordered_names = [self.task_names[i] for i in ordered_indices]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            ordered_affinity,
            xticklabels=ordered_names,
            yticklabels=ordered_names,
            cmap='RdYlGn',
            vmin=0, vmax=1,
            annot=True,
            fmt='.2f',
            ax=ax,
            cbar_kws={'label': 'Task Affinity'}
        )

        # Draw group boundaries
        group_sizes = [len(g.task_indices) for g in self.task_groups]
        cumsum = np.cumsum([0] + group_sizes)
        for pos in cumsum[1:-1]:
            ax.axhline(pos, color='blue', linewidth=2)
            ax.axvline(pos, color='blue', linewidth=2)

        ax.set_title('Task Affinity Matrix (Grouped)', fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved grouped affinity matrix to {save_path}")
        else:
            plt.show()
```

#### File: `src/admet/model/chemprop/config.py` (additions)

```python
@dataclass
class TaskAffinityConfig:
    # ... existing fields ...

    # Optimal grouping
    auto_find_optimal_groups: bool = False
    min_groups: int = 2
    max_groups: int = 6
    allow_singleton_groups: bool = True

    # Domain constraints (optional manual override)
    domain_constraints: Optional[Dict[str, List[int]]] = None
    # Example:
    # domain_constraints = {
    #     'lipophilicity': [0, 1],  # LogD, KSOL must be together
    #     'clearance': [2, 3],      # HLM, MLM must be together
    # }
```

### 2.2 Integration with Training

#### File: `src/admet/model/chemprop/model.py` (modifications)

```python
from admet.model.chemprop.optimal_grouper import OptimalTaskGrouper

class ChempropModel:
    def _setup_task_affinity(self):
        """Setup task affinity and grouping if enabled."""
        if not self.config.task_affinity.enabled:
            return

        logger.info("Computing task affinity...")

        # Compute affinity matrix (existing code)
        affinity_computer = TaskAffinityComputer(...)
        affinity_matrix = affinity_computer.compute_task_affinity(...)

        # Find optimal groups if requested
        if self.config.task_affinity.auto_find_optimal_groups:
            grouper = OptimalTaskGrouper(
                affinity_matrix=affinity_matrix,
                task_names=self.config.data.target_cols,
                min_groups=self.config.task_affinity.min_groups,
                max_groups=self.config.task_affinity.max_groups,
                clustering_method=self.config.task_affinity.clustering_method,
                allow_singleton=self.config.task_affinity.allow_singleton_groups,
                domain_constraints=self.config.task_affinity.domain_constraints,
            )

            task_groups = grouper.find_optimal_groups(verbose=True)

            # Log results to MLflow
            if self.config.mlflow.enabled:
                mlflow.log_param("optimal_n_groups", grouper.optimal_n_groups)
                for n, score in grouper.silhouette_scores.items():
                    mlflow.log_metric(f"silhouette_n{n}", score)

                # Save plots
                grouper.plot_silhouette_analysis(
                    save_path="outputs/silhouette_analysis.png"
                )
                mlflow.log_artifact("outputs/silhouette_analysis.png")

                grouper.plot_affinity_matrix_grouped(
                    save_path="outputs/affinity_matrix_grouped.png"
                )
                mlflow.log_artifact("outputs/affinity_matrix_grouped.png")

            # Use discovered groups for grouped FFN if applicable
            if self.config.model.ffn_type == "grouped_multihead":
                self.config.model.task_groups = [
                    g.task_indices for g in task_groups
                ]

        else:
            # Use manual n_groups (existing behavior)
            pass
```

### 2.3 Unit Tests

#### File: `tests/unit/test_optimal_grouper.py`

```python
"""Unit tests for optimal task grouping."""

import pytest
import numpy as np
from admet.model.chemprop.optimal_grouper import OptimalTaskGrouper, ClusteringMethod


class TestOptimalTaskGrouper:
    """Tests for OptimalTaskGrouper."""

    @pytest.fixture
    def mock_affinity_matrix(self):
        """Create mock affinity matrix with clear groups.

        Group 1: Tasks 0, 1 (high affinity)
        Group 2: Tasks 2, 3, 4 (high affinity)
        Group 3: Tasks 5, 6, 7, 8 (high affinity)
        """
        affinity = np.eye(9) * 1.0

        # Group 1
        affinity[0, 1] = affinity[1, 0] = 0.9

        # Group 2
        for i in [2, 3, 4]:
            for j in [2, 3, 4]:
                if i != j:
                    affinity[i, j] = 0.8

        # Group 3
        for i in [5, 6, 7, 8]:
            for j in [5, 6, 7, 8]:
                if i != j:
                    affinity[i, j] = 0.7

        return affinity

    @pytest.fixture
    def task_names(self):
        return [
            'LogD', 'KSOL', 'HLM CLint', 'MLM CLint', 'Caco-2 Papp',
            'Caco-2 Efflux', 'MPPB', 'MBPB', 'MGMB'
        ]

    def test_initialization(self, mock_affinity_matrix, task_names):
        """Test grouper initializes correctly."""
        grouper = OptimalTaskGrouper(
            affinity_matrix=mock_affinity_matrix,
            task_names=task_names
        )

        assert grouper.n_tasks == 9
        assert len(grouper.task_names) == 9
        assert grouper.optimal_n_groups is None

    def test_find_optimal_groups(self, mock_affinity_matrix, task_names):
        """Test optimal grouping discovery."""
        grouper = OptimalTaskGrouper(
            affinity_matrix=mock_affinity_matrix,
            task_names=task_names,
            min_groups=2,
            max_groups=5
        )

        groups = grouper.find_optimal_groups(verbose=False)

        # Should find 3 groups (matching our mock structure)
        assert grouper.optimal_n_groups == 3
        assert len(groups) == 3

        # Check silhouette scores were computed
        assert len(grouper.silhouette_scores) > 0

    def test_clustering_methods(self, mock_affinity_matrix):
        """Test all clustering methods work."""
        methods = [
            ClusteringMethod.AGGLOMERATIVE,
            ClusteringMethod.SPECTRAL,
            ClusteringMethod.KMEANS
        ]

        for method in methods:
            grouper = OptimalTaskGrouper(
                affinity_matrix=mock_affinity_matrix,
                clustering_method=method
            )
            groups = grouper.find_optimal_groups(verbose=False)

            # Should produce valid grouping
            assert len(groups) > 0
            assert grouper.optimal_n_groups is not None

    def test_task_to_group_mapping(self, mock_affinity_matrix):
        """Test task-to-group mapping generation."""
        grouper = OptimalTaskGrouper(affinity_matrix=mock_affinity_matrix)
        grouper.find_optimal_groups(verbose=False)

        mapping = grouper.get_task_to_group_mapping()

        assert len(mapping) == 9
        assert all(isinstance(k, int) for k in mapping.keys())
        assert all(isinstance(v, int) for v in mapping.values())

    def test_domain_constraints_validation(self, mock_affinity_matrix, task_names):
        """Test domain constraints are validated."""
        # Force tasks that should be together to be in same group
        domain_constraints = {
            'lipophilicity': [0, 1],  # LogD, KSOL
            'clearance': [2, 3],      # HLM, MLM
        }

        grouper = OptimalTaskGrouper(
            affinity_matrix=mock_affinity_matrix,
            task_names=task_names,
            domain_constraints=domain_constraints
        )

        groups = grouper.find_optimal_groups(verbose=False)

        # Check constraints are satisfied
        mapping = grouper.get_task_to_group_mapping()

        # LogD and KSOL should be in same group
        assert mapping[0] == mapping[1]

        # HLM and MLM should be in same group
        assert mapping[2] == mapping[3]

    def test_singleton_handling(self, mock_affinity_matrix):
        """Test singleton group handling."""
        # Test with singletons allowed
        grouper_allow = OptimalTaskGrouper(
            affinity_matrix=mock_affinity_matrix,
            allow_singleton=True
        )
        groups_allow = grouper_allow.find_optimal_groups(verbose=False)

        # Test with singletons not allowed
        grouper_no_single = OptimalTaskGrouper(
            affinity_matrix=mock_affinity_matrix,
            allow_singleton=False
        )
        groups_no_single = grouper_no_single.find_optimal_groups(verbose=False)

        # Should produce different results if singletons existed
        # (May be same if no singletons in optimal solution)
        assert isinstance(groups_allow, list)
        assert isinstance(groups_no_single, list)
```

### 2.4 Configuration

#### File: `configs/tier1/task_affinity_optimized.yaml`

```yaml
# Tier 1 - Optimized Task Affinity Grouping
# Expected: 3-5% MA-RAE improvement

model:
  # MPNN
  depth: 3
  message_hidden_dim: 700

  # Grouped FFN (NEW)
  ffn_type: grouped_multihead
  num_layers: 2
  hidden_dim: 600
  dropout: 0.1
  batch_norm: true

  # Task groups will be auto-determined
  # (set by optimal grouper)

  # Optimization
  criterion: MSE
  init_lr: 0.00113
  max_lr: 0.000227
  final_lr: 0.000113
  max_epochs: 150
  patience: 15
  batch_size: 128

# Task affinity configuration
task_affinity:
  enabled: true

  # Optimal grouping (NEW)
  auto_find_optimal_groups: true
  min_groups: 2
  max_groups: 6
  clustering_method: agglomerative
  allow_singleton_groups: true

  # Domain constraints (optional)
  domain_constraints:
    lipophilicity: [0, 1]      # LogD, KSOL
    clearance: [2, 3]          # HLM CLint, MLM CLint
    permeability: [4, 5]       # Caco-2 Papp, Efflux

  # Affinity computation
  affinity_epochs: 2
  affinity_linkage: ward

data:
  data_dir: "data/expansionrx_challenge"
  smiles_col: "SMILES"
  target_cols:
    - "LogD"           # 0
    - "KSOL"           # 1
    - "HLM CLint"      # 2
    - "MLM CLint"      # 3
    - "Caco-2 Papp"    # 4
    - "Caco-2 Efflux"  # 5
    - "MPPB"           # 6
    - "MBPB"           # 7
    - "MGMB"           # 8

mlflow:
  enabled: true
  experiment_name: "tier1_task_affinity"
  run_name: "optimal_grouping"
```

---

## Phase 3: Gradient Surgery (PCGrad)

**Goal**: Resolve gradient conflicts via projection.
**Expected Impact**: 3-5% MA-RAE improvement
**Effort**: 3-5 days

### 3.1 Implementation

#### File: `src/admet/model/chemprop/pcgrad.py`

```python
"""Projecting Conflicting Gradients (PCGrad) for multi-task learning.

Implements gradient surgery to resolve conflicting gradients between tasks.
Based on Yu et al. (2020) "Gradient Surgery for Multi-Task Learning" (NeurIPS).

Also includes FetterGrad variant from DeepDTAGen (Nature Comms, 2025).

References:
    - PCGrad: https://arxiv.org/abs/2001.06782
    - DeepDTAGen: https://www.nature.com/articles/s41467-025-59917-6
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class GradientSurgeryMethod(Enum):
    """Supported gradient surgery methods."""
    PCGRAD = "pcgrad"
    FETTER_GRAD = "fetter_grad"  # Variant from DeepDTAGen


class PCGrad:
    """Projecting Conflicting Gradients optimizer wrapper.

    Wraps an existing optimizer and modifies gradients before the optimization
    step to resolve conflicts between tasks.

    When two tasks have conflicting gradients (negative cosine similarity),
    projects one task's gradient onto the normal plane of the other to
    eliminate the conflict.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Base optimizer (e.g., Adam).
    method : GradientSurgeryMethod or str, default='pcgrad'
        Gradient surgery method to use.
    reduction : str, default='mean'
        How to combine task gradients after surgery:
        - 'mean': Average gradients
        - 'sum': Sum gradients
    cpu_offload : bool, default=False
        Offload gradient computation to CPU to save GPU memory.

    Attributes
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    method : GradientSurgeryMethod
        Gradient surgery method.
    n_conflicts : int
        Running count of gradient conflicts resolved.
    total_steps : int
        Running count of optimization steps.

    Examples
    --------
    >>> model = ChempropModel(...)
    >>> base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> optimizer = PCGrad(base_optimizer)
    >>>
    >>> # In training loop
    >>> for batch in loader:
    ...     predictions = model(batch)
    ...     task_losses = [criterion(pred[:, i], target[:, i]) for i in range(n_tasks)]
    ...
    ...     optimizer.pc_backward(task_losses)  # Instead of loss.backward()
    ...     optimizer.step()
    ...     optimizer.zero_grad()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        method: GradientSurgeryMethod | str = GradientSurgeryMethod.PCGRAD,
        reduction: str = 'mean',
        cpu_offload: bool = False,
    ):
        self.optimizer = optimizer
        self.method = (
            GradientSurgeryMethod(method)
            if isinstance(method, str)
            else method
        )
        self.reduction = reduction
        self.cpu_offload = cpu_offload

        self.n_conflicts = 0
        self.total_steps = 0

        logger.info(f"Initialized PCGrad with method={self.method.value}, reduction={reduction}")

    def zero_grad(self):
        """Zero out gradients (delegates to wrapped optimizer)."""
        self.optimizer.zero_grad()

    def step(self):
        """Perform optimization step (delegates to wrapped optimizer)."""
        self.optimizer.step()
        self.total_steps += 1

    def state_dict(self):
        """Get optimizer state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict)

    def pc_backward(self, task_losses: List[Tensor]):
        """Compute gradients with conflict resolution.

        Parameters
        ----------
        task_losses : List[Tensor]
            List of scalar loss tensors, one per task.

        Notes
        -----
        This method replaces the usual `loss.backward()` call.
        It computes per-task gradients, resolves conflicts, and
        sets the final gradients on model parameters.
        """
        # Get model parameters
        # (Assumes all parameters are shared, which is true for multi-task models)
        param_groups = self.optimizer.param_groups
        params = [p for group in param_groups for p in group['params'] if p.requires_grad]

        n_tasks = len(task_losses)

        # Compute per-task gradients
        task_grads = []
        for task_idx, loss in enumerate(task_losses):
            # Zero out previous gradients
            self.optimizer.zero_grad()

            # Compute gradients for this task
            loss.backward(retain_graph=(task_idx < n_tasks - 1))

            # Store gradients
            grads = []
            for param in params:
                if param.grad is not None:
                    if self.cpu_offload:
                        grads.append(param.grad.cpu().clone())
                    else:
                        grads.append(param.grad.clone())
                else:
                    grads.append(None)

            task_grads.append(grads)

        # Apply gradient surgery
        if self.method == GradientSurgeryMethod.PCGRAD:
            modified_grads = self._pcgrad(task_grads)
        elif self.method == GradientSurgeryMethod.FETTER_GRAD:
            modified_grads = self._fetter_grad(task_grads)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Set final gradients on parameters
        for param, grad in zip(params, modified_grads):
            if grad is not None:
                if self.cpu_offload:
                    param.grad = grad.to(param.device)
                else:
                    param.grad = grad

    def _pcgrad(self, task_grads: List[List[Optional[Tensor]]]) -> List[Optional[Tensor]]:
        """Apply PCGrad: project conflicting gradients.

        Parameters
        ----------
        task_grads : List[List[Optional[Tensor]]]
            Per-task gradients. Shape: [n_tasks, n_params].

        Returns
        -------
        List[Optional[Tensor]]
            Modified gradients after conflict resolution.
            Shape: [n_params].
        """
        n_tasks = len(task_grads)
        n_params = len(task_grads[0])

        # Project conflicting gradients for each task
        projected_task_grads = []

        for i in range(n_tasks):
            # Start with original gradient for task i
            grads_i = task_grads[i]

            # Project away from conflicting tasks
            for j in range(n_tasks):
                if i == j:
                    continue

                grads_j = task_grads[j]

                # Compute conflict for each parameter
                for param_idx in range(n_params):
                    g_i = grads_i[param_idx]
                    g_j = grads_j[param_idx]

                    if g_i is None or g_j is None:
                        continue

                    # Flatten for dot product
                    g_i_flat = g_i.flatten()
                    g_j_flat = g_j.flatten()

                    # Compute cosine similarity
                    dot = torch.dot(g_i_flat, g_j_flat)
                    norm_i = g_i_flat.norm()
                    norm_j = g_j_flat.norm()

                    if norm_i == 0 or norm_j == 0:
                        continue

                    cos_sim = dot / (norm_i * norm_j)

                    # If conflict (cos < 0), project g_i onto normal plane of g_j
                    if cos_sim < 0:
                        self.n_conflicts += 1

                        # Projection: g_i = g_i - (g_i · g_j / ||g_j||²) * g_j
                        proj_coef = dot / (norm_j ** 2)
                        proj = proj_coef * g_j_flat
                        g_i_flat = g_i_flat - proj

                        # Reshape back to original shape
                        grads_i[param_idx] = g_i_flat.reshape_as(g_i)

            projected_task_grads.append(grads_i)

        # Combine task gradients (mean or sum)
        final_grads = []
        for param_idx in range(n_params):
            param_grads = [
                task_grads[task_idx][param_idx]
                for task_idx in range(n_tasks)
            ]

            # Filter out None
            param_grads = [g for g in param_grads if g is not None]

            if len(param_grads) == 0:
                final_grads.append(None)
            else:
                if self.reduction == 'mean':
                    final_grads.append(torch.stack(param_grads).mean(dim=0))
                elif self.reduction == 'sum':
                    final_grads.append(torch.stack(param_grads).sum(dim=0))
                else:
                    raise ValueError(f"Unknown reduction: {self.reduction}")

        return final_grads

    def _fetter_grad(self, task_grads: List[List[Optional[Tensor]]]) -> List[Optional[Tensor]]:
        """Apply FetterGrad: variant from DeepDTAGen.

        Similar to PCGrad but uses average gradient as reference.

        Parameters
        ----------
        task_grads : List[List[Optional[Tensor]]]
            Per-task gradients.

        Returns
        -------
        List[Optional[Tensor]]
            Modified gradients.
        """
        n_tasks = len(task_grads)
        n_params = len(task_grads[0])

        # Compute average gradient
        avg_grads = []
        for param_idx in range(n_params):
            param_grads = [
                task_grads[task_idx][param_idx]
                for task_idx in range(n_tasks)
            ]
            param_grads = [g for g in param_grads if g is not None]

            if len(param_grads) == 0:
                avg_grads.append(None)
            else:
                avg_grads.append(torch.stack(param_grads).mean(dim=0))

        # Project each task gradient if it conflicts with average
        modified_task_grads = []

        for i in range(n_tasks):
            grads_i = task_grads[i]

            for param_idx in range(n_params):
                g_i = grads_i[param_idx]
                g_avg = avg_grads[param_idx]

                if g_i is None or g_avg is None:
                    continue

                g_i_flat = g_i.flatten()
                g_avg_flat = g_avg.flatten()

                # Check conflict with average
                dot = torch.dot(g_i_flat, g_avg_flat)
                norm_i = g_i_flat.norm()
                norm_avg = g_avg_flat.norm()

                if norm_i == 0 or norm_avg == 0:
                    continue

                cos_sim = dot / (norm_i * norm_avg)

                # If conflict, project
                if cos_sim < 0:
                    self.n_conflicts += 1
                    proj_coef = dot / (norm_avg ** 2)
                    proj = proj_coef * g_avg_flat
                    g_i_flat = g_i_flat - proj
                    grads_i[param_idx] = g_i_flat.reshape_as(g_i)

            modified_task_grads.append(grads_i)

        # Take mean
        final_grads = []
        for param_idx in range(n_params):
            param_grads = [
                modified_task_grads[task_idx][param_idx]
                for task_idx in range(n_tasks)
            ]
            param_grads = [g for g in param_grads if g is not None]

            if len(param_grads) == 0:
                final_grads.append(None)
            else:
                final_grads.append(torch.stack(param_grads).mean(dim=0))

        return final_grads

    def get_conflict_stats(self) -> Dict[str, float]:
        """Get gradient conflict statistics.

        Returns
        -------
        Dict[str, float]
            Statistics including:
            - 'n_conflicts': Total conflicts resolved
            - 'conflicts_per_step': Average conflicts per step
            - 'total_steps': Total optimization steps
        """
        return {
            'n_conflicts': self.n_conflicts,
            'conflicts_per_step': self.n_conflicts / max(self.total_steps, 1),
            'total_steps': self.total_steps,
        }
```

#### File: `src/admet/model/chemprop/config.py` (additions)

```python
@dataclass
class OptimizationConfig:
    # ... existing fields ...

    # Gradient surgery
    use_pcgrad: bool = False
    pcgrad_method: str = "pcgrad"  # or "fetter_grad"
    pcgrad_reduction: str = "mean"
    pcgrad_cpu_offload: bool = False
```

### 3.2 Integration with PyTorch Lightning

#### File: `src/admet/model/chemprop/lightning_module.py` (new or modify existing)

```python
"""PyTorch Lightning module with PCGrad support."""

import pytorch_lightning as pl
import torch
from typing import Dict, List

from admet.model.chemprop.pcgrad import PCGrad


class ChempropLightningModule(pl.LightningModule):
    """Lightning module for Chemprop with optional PCGrad."""

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = model.criterion

        # Track per-task losses for PCGrad
        self.automatic_optimization = not config.model.use_pcgrad

    def configure_optimizers(self):
        """Setup optimizer with optional PCGrad wrapper."""
        base_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.model.init_lr
        )

        if self.config.model.use_pcgrad:
            optimizer = PCGrad(
                base_optimizer,
                method=self.config.model.pcgrad_method,
                reduction=self.config.model.pcgrad_reduction,
                cpu_offload=self.config.model.pcgrad_cpu_offload,
            )
        else:
            optimizer = base_optimizer

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            base_optimizer,  # Note: use base optimizer for scheduler
            T_max=self.config.model.max_epochs,
            eta_min=self.config.model.final_lr
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def training_step(self, batch, batch_idx):
        """Training step with PCGrad support."""
        predictions = self.model(batch)
        targets = batch['targets']
        mask = ~torch.isnan(targets)

        if self.config.model.use_pcgrad:
            # Manual optimization with PCGrad
            optimizer = self.optimizers()

            # Compute per-task losses
            task_losses = []
            for task_idx in range(targets.shape[1]):
                task_mask = mask[:, task_idx]
                if task_mask.sum() == 0:
                    continue

                pred_task = predictions[task_mask, task_idx]
                target_task = targets[task_mask, task_idx]

                loss_task = torch.nn.functional.mse_loss(pred_task, target_task)
                task_losses.append(loss_task)

            # Apply PCGrad
            optimizer.pc_backward(task_losses)
            optimizer.step()
            optimizer.zero_grad()

            # Log total loss (for monitoring)
            total_loss = sum(task_losses) / len(task_losses)
            self.log('train_loss', total_loss, prog_bar=True)

            return total_loss

        else:
            # Standard optimization
            if isinstance(self.criterion, UncertaintyWeightedLoss):
                loss = self.criterion(predictions, targets, mask)
            else:
                # Standard MSE
                predictions_masked = predictions * mask
                targets_clean = torch.nan_to_num(targets, nan=0.0)
                targets_masked = targets_clean * mask

                loss = torch.nn.functional.mse_loss(
                    predictions_masked,
                    targets_masked,
                    reduction='sum'
                ) / mask.sum()

            self.log('train_loss', loss, prog_bar=True)
            return loss

    def on_train_epoch_end(self):
        """Log PCGrad statistics at end of epoch."""
        if self.config.model.use_pcgrad:
            optimizer = self.optimizers()
            if isinstance(optimizer, PCGrad):
                stats = optimizer.get_conflict_stats()
                self.log('pcgrad/conflicts_per_step', stats['conflicts_per_step'])
                self.log('pcgrad/total_conflicts', stats['n_conflicts'])
```

### 3.3 Unit Tests

#### File: `tests/unit/test_pcgrad.py`

```python
"""Unit tests for PCGrad optimizer."""

import pytest
import torch
import torch.nn as nn

from admet.model.chemprop.pcgrad import PCGrad, GradientSurgeryMethod


class TestPCGrad:
    """Tests for PCGrad optimizer wrapper."""

    @pytest.fixture
    def simple_model(self):
        """Simple model for testing."""
        return nn.Linear(10, 3)  # 3 tasks

    @pytest.fixture
    def base_optimizer(self, simple_model):
        return torch.optim.Adam(simple_model.parameters(), lr=0.01)

    @pytest.fixture
    def pcgrad_optimizer(self, base_optimizer):
        return PCGrad(base_optimizer)

    def test_initialization(self, pcgrad_optimizer):
        """Test PCGrad initializes correctly."""
        assert isinstance(pcgrad_optimizer.optimizer, torch.optim.Adam)
        assert pcgrad_optimizer.method == GradientSurgeryMethod.PCGRAD
        assert pcgrad_optimizer.n_conflicts == 0
        assert pcgrad_optimizer.total_steps == 0

    def test_pc_backward_conflicting_tasks(self, simple_model, pcgrad_optimizer):
        """Test PCGrad resolves conflicting gradients."""
        # Create inputs
        x = torch.randn(32, 10)
        targets = torch.randn(32, 3)

        # Forward pass
        predictions = simple_model(x)

        # Create conflicting losses
        # Task 0: minimize predictions
        # Task 1: maximize predictions (conflict!)
        # Task 2: neutral
        task_losses = [
            torch.nn.functional.mse_loss(predictions[:, 0], targets[:, 0]),
            torch.nn.functional.mse_loss(predictions[:, 1], -targets[:, 1]),  # Flip target
            torch.nn.functional.mse_loss(predictions[:, 2], targets[:, 2]),
        ]

        # Apply PCGrad
        pcgrad_optimizer.pc_backward(task_losses)

        # Check gradients exist
        for param in simple_model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

        # Check conflicts were detected
        assert pcgrad_optimizer.n_conflicts > 0

    def test_optimization_step(self, simple_model, pcgrad_optimizer):
        """Test full optimization step works."""
        x = torch.randn(32, 10)
        targets = torch.randn(32, 3)

        # Store initial parameters
        initial_params = [p.clone() for p in simple_model.parameters()]

        # Forward + backward
        predictions = simple_model(x)
        task_losses = [
            torch.nn.functional.mse_loss(predictions[:, i], targets[:, i])
            for i in range(3)
        ]

        pcgrad_optimizer.pc_backward(task_losses)
        pcgrad_optimizer.step()
        pcgrad_optimizer.zero_grad()

        # Check parameters changed
        for initial, current in zip(initial_params, simple_model.parameters()):
            assert not torch.allclose(initial, current)

        # Check step counter incremented
        assert pcgrad_optimizer.total_steps == 1

    def test_fetter_grad_method(self, base_optimizer):
        """Test FetterGrad method works."""
        pcgrad = PCGrad(base_optimizer, method=GradientSurgeryMethod.FETTER_GRAD)
        assert pcgrad.method == GradientSurgeryMethod.FETTER_GRAD

    def test_gradient_reduction_methods(self, simple_model):
        """Test different gradient reduction methods."""
        for reduction in ['mean', 'sum']:
            optimizer = torch.optim.Adam(simple_model.parameters())
            pcgrad = PCGrad(optimizer, reduction=reduction)

            x = torch.randn(16, 10)
            targets = torch.randn(16, 3)
            predictions = simple_model(x)

            task_losses = [
                torch.nn.functional.mse_loss(predictions[:, i], targets[:, i])
                for i in range(3)
            ]

            pcgrad.pc_backward(task_losses)

            # Should work without errors
            assert all(p.grad is not None for p in simple_model.parameters())

    def test_state_dict(self, pcgrad_optimizer):
        """Test state dict save/load."""
        state = pcgrad_optimizer.state_dict()
        assert isinstance(state, dict)

        # Should be able to load
        pcgrad_optimizer.load_state_dict(state)

    def test_get_conflict_stats(self, simple_model, pcgrad_optimizer):
        """Test conflict statistics retrieval."""
        # Run a few steps
        for _ in range(5):
            x = torch.randn(16, 10)
            targets = torch.randn(16, 3)
            predictions = simple_model(x)

            task_losses = [
                torch.nn.functional.mse_loss(predictions[:, i], targets[:, i])
                for i in range(3)
            ]

            pcgrad_optimizer.pc_backward(task_losses)
            pcgrad_optimizer.step()
            pcgrad_optimizer.zero_grad()

        stats = pcgrad_optimizer.get_conflict_stats()

        assert 'n_conflicts' in stats
        assert 'conflicts_per_step' in stats
        assert 'total_steps' in stats
        assert stats['total_steps'] == 5
```

### 3.4 Configuration

#### File: `configs/tier1/pcgrad.yaml`

```yaml
# Tier 1 - PCGrad Gradient Surgery
# Expected: 3-5% MA-RAE improvement

model:
  # MPNN
  depth: 3
  message_hidden_dim: 700

  # FFN
  ffn_type: regression
  num_layers: 4
  hidden_dim: 200
  dropout: 0.15
  batch_norm: true

  # PCGrad (NEW)
  use_pcgrad: true
  pcgrad_method: "pcgrad"  # or "fetter_grad"
  pcgrad_reduction: "mean"
  pcgrad_cpu_offload: false

  # Optimization
  criterion: MSE
  init_lr: 0.00113
  max_lr: 0.000227
  final_lr: 0.000113
  max_epochs: 150
  patience: 15
  batch_size: 128

data:
  data_dir: "data/expansionrx_challenge"
  smiles_col: "SMILES"
  target_cols:
    - "LogD"
    - "KSOL"
    - "HLM CLint"
    - "MLM CLint"
    - "Caco-2 Papp"
    - "Caco-2 Efflux"
    - "MPPB"
    - "MBPB"
    - "MGMB"

mlflow:
  enabled: true
  experiment_name: "tier1_pcgrad"
  run_name: "gradient_surgery"
```

---

## Phase 4: Integration & Testing

**Goal**: Combine all three strategies and validate system-wide.
**Effort**: 2-3 days

### 4.1 Combined Configuration

#### File: `configs/tier1/combined.yaml`

```yaml
# Tier 1 - All Strategies Combined
# Expected: 10-15% MA-RAE improvement

model:
  # MPNN
  depth: 3
  message_hidden_dim: 700

  # Grouped FFN with optimal grouping
  ffn_type: grouped_multihead
  num_layers: 2
  hidden_dim: 600
  dropout: 0.1
  batch_norm: true

  # Uncertainty weighting
  use_uncertainty_weighting: true
  uncertainty_init_log_vars: 0.0
  uncertainty_min_log_var: -10.0
  uncertainty_max_log_var: 10.0

  # PCGrad
  use_pcgrad: true
  pcgrad_method: "pcgrad"
  pcgrad_reduction: "mean"

  # Optimization
  criterion: MSE  # Wrapped by uncertainty weighting
  init_lr: 0.00113
  max_lr: 0.000227
  final_lr: 0.000113
  warmup_epochs: 5
  patience: 15
  max_epochs: 150
  batch_size: 128

  # Task sampling
  task_sampling_alpha: 0.02

# Task affinity
task_affinity:
  enabled: true
  auto_find_optimal_groups: true
  min_groups: 2
  max_groups: 6
  clustering_method: agglomerative
  allow_singleton_groups: true

  domain_constraints:
    lipophilicity: [0, 1]
    clearance: [2, 3]
    permeability: [4, 5]

  affinity_epochs: 2

data:
  data_dir: "data/expansionrx_challenge"
  smiles_col: "SMILES"
  target_cols:
    - "LogD"
    - "KSOL"
    - "HLM CLint"
    - "MLM CLint"
    - "Caco-2 Papp"
    - "Caco-2 Efflux"
    - "MPPB"
    - "MBPB"
    - "MGMB"
  output_dir: "outputs/tier1_combined"

mlflow:
  enabled: true
  experiment_name: "tier1_combined"
  run_name: "all_strategies"
  tracking_uri: "http://127.0.0.1:8084"
```

### 4.2 Integration Test

#### File: `tests/integration/test_tier1_combined.py`

```python
"""Integration test for combined Tier 1 strategies."""

import pytest
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

from admet.model.chemprop.model import ChempropModel


class TestTier1Integration:
    """Test all three Tier 1 strategies work together."""

    @pytest.fixture(scope="class")
    def config(self):
        """Load combined config."""
        config_path = Path("configs/tier1/combined.yaml")
        return OmegaConf.load(config_path)

    @pytest.fixture(scope="class")
    def train_data(self):
        """Load training subset."""
        data_path = Path("data/expansionrx_challenge/train.csv")
        if not data_path.exists():
            pytest.skip("Data not available")

        df = pd.read_csv(data_path)
        return df.sort_values('Molecule Name').iloc[:1000]

    def test_all_components_initialize(self, config):
        """Test all components initialize without errors."""
        # This test just checks config is valid
        assert config.model.use_uncertainty_weighting
        assert config.model.use_pcgrad
        assert config.task_affinity.enabled
        assert config.task_affinity.auto_find_optimal_groups

    def test_combined_training(self, config, train_data):
        """Test combined system trains successfully."""
        train_df = train_data.iloc[:800]
        val_df = train_data.iloc[800:]

        # Reduce epochs for testing
        config_test = OmegaConf.create(config)
        config_test.model.max_epochs = 10
        config_test.model.patience = 3
        config_test.mlflow.enabled = False

        model = ChempropModel(
            df_train=train_df,
            df_validation=val_df,
            **config_test.model,
            **config_test.data
        )

        # Should train without errors
        model.fit()

        # Check all components are active
        # 1. Uncertainty weighting
        from admet.model.chemprop.uncertainty_loss import UncertaintyWeightedLoss
        assert isinstance(model.criterion, UncertaintyWeightedLoss)

        # 2. PCGrad
        from admet.model.chemprop.pcgrad import PCGrad
        # (Check in optimizer - implementation dependent)

        # 3. Task affinity grouping
        # (Check groups were created - implementation dependent)

    def test_combined_improves_baseline(self, config, train_data):
        """Test combined approach improves over baseline."""
        train_df = train_data.iloc[:800]
        val_df = train_data.iloc[800:]

        # Baseline config (no Tier 1 features)
        config_baseline = OmegaConf.create(config)
        config_baseline.model.use_uncertainty_weighting = False
        config_baseline.model.use_pcgrad = False
        config_baseline.model.ffn_type = 'regression'
        config_baseline.model.max_epochs = 15
        config_baseline.mlflow.enabled = False

        # Combined config
        config_combined = OmegaConf.create(config)
        config_combined.model.max_epochs = 15
        config_combined.mlflow.enabled = False

        # Train baseline
        model_baseline = ChempropModel(
            df_train=train_df,
            df_validation=val_df,
            **config_baseline.model,
            **config_baseline.data
        )
        model_baseline.fit()
        baseline_metrics = model_baseline.evaluate(val_df)

        # Train combined
        model_combined = ChempropModel(
            df_train=train_df,
            df_validation=val_df,
            **config_combined.model,
            **config_combined.data
        )
        model_combined.fit()
        combined_metrics = model_combined.evaluate(val_df)

        # Check improvement (at least not worse)
        # MAE should decrease (or at worst stay same)
        baseline_mae = baseline_metrics['mae']
        combined_mae = combined_metrics['mae']

        assert combined_mae <= baseline_mae * 1.05  # Allow 5% tolerance

        # Check sparse tasks specifically
        sparse_indices = [7, 8]  # MBPB, MGMB
        for idx in sparse_indices:
            baseline_r2 = baseline_metrics['r2_per_task'][idx]
            combined_r2 = combined_metrics['r2_per_task'][idx]

            # Should improve or at worst minor degradation
            assert combined_r2 >= baseline_r2 - 0.05
```

---

## Phase 5: Experiments & Ablation Studies

**Goal**: Systematically evaluate each component and combinations.
**Effort**: 7-10 days

### 5.1 Experiment Design

#### Experiment Matrix

| Exp ID | Uncertainty | Task Affinity | PCGrad | Expected MA-RAE | Notes |
|--------|-------------|---------------|--------|-----------------|-------|
| E0 | ❌ | ❌ | ❌ | 0.60 | Baseline (Dec 16) |
| E1 | ✅ | ❌ | ❌ | 0.57 | Uncertainty only |
| E2 | ❌ | ✅ | ❌ | 0.58 | Task affinity only |
| E3 | ❌ | ❌ | ✅ | 0.58 | PCGrad only |
| E4 | ✅ | ✅ | ❌ | 0.55 | U + A |
| E5 | ✅ | ❌ | ✅ | 0.55 | U + P |
| E6 | ❌ | ✅ | ✅ | 0.56 | A + P |
| E7 | ✅ | ✅ | ✅ | 0.52-0.54 | All (target) |

#### Evaluation Protocol

**Single-Fold Rapid Evaluation** (1-2 days):

- Split: Butina split 0, fold 0
- Epochs: 50 (with early stopping)
- Metrics: Validation MAE, R², per-task R²
- Purpose: Quick screening

**Full 5×5 Ensemble** (5-7 days):

- 25 models: 5 Butina splits × 5 CV folds
- Epochs: 150 (with early stopping)
- Metrics: MA-RAE, R², MAE, RAE per endpoint
- Purpose: Leaderboard submission

### 5.2 Experiment Scripts

#### File: `scripts/experiments/run_tier1_ablation.py`

```python
"""Run Tier 1 ablation study."""

import argparse
import json
import mlflow
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

from admet.model.chemprop.model import ChempropModel


EXPERIMENTS = {
    'E0_baseline': {
        'use_uncertainty_weighting': False,
        'use_pcgrad': False,
        'ffn_type': 'regression',
        'task_affinity': {'enabled': False},
    },
    'E1_uncertainty': {
        'use_uncertainty_weighting': True,
        'use_pcgrad': False,
        'ffn_type': 'regression',
        'task_affinity': {'enabled': False},
    },
    'E2_affinity': {
        'use_uncertainty_weighting': False,
        'use_pcgrad': False,
        'ffn_type': 'grouped_multihead',
        'task_affinity': {'enabled': True, 'auto_find_optimal_groups': True},
    },
    'E3_pcgrad': {
        'use_uncertainty_weighting': False,
        'use_pcgrad': True,
        'ffn_type': 'regression',
        'task_affinity': {'enabled': False},
    },
    'E4_uncertainty_affinity': {
        'use_uncertainty_weighting': True,
        'use_pcgrad': False,
        'ffn_type': 'grouped_multihead',
        'task_affinity': {'enabled': True, 'auto_find_optimal_groups': True},
    },
    'E5_uncertainty_pcgrad': {
        'use_uncertainty_weighting': True,
        'use_pcgrad': True,
        'ffn_type': 'regression',
        'task_affinity': {'enabled': False},
    },
    'E6_affinity_pcgrad': {
        'use_uncertainty_weighting': False,
        'use_pcgrad': True,
        'ffn_type': 'grouped_multihead',
        'task_affinity': {'enabled': True, 'auto_find_optimal_groups': True},
    },
    'E7_combined': {
        'use_uncertainty_weighting': True,
        'use_pcgrad': True,
        'ffn_type': 'grouped_multihead',
        'task_affinity': {'enabled': True, 'auto_find_optimal_groups': True},
    },
}


def run_experiment(exp_id, exp_config, base_config, mode='single_fold'):
    """Run single experiment.

    Parameters
    ----------
    exp_id : str
        Experiment identifier.
    exp_config : dict
        Experiment-specific config overrides.
    base_config : OmegaConf
        Base configuration.
    mode : str
        'single_fold' or 'full_ensemble'.
    """
    # Merge configs
    config = OmegaConf.create(base_config)
    for key, value in exp_config.items():
        if key == 'task_affinity':
            config.task_affinity.update(value)
        else:
            config.model[key] = value

    # Set MLflow experiment
    config.mlflow.experiment_name = f"tier1_ablation_{mode}"
    config.mlflow.run_name = exp_id

    if mode == 'single_fold':
        # Quick single-fold evaluation
        train_results = run_single_fold(config, exp_id)
    else:
        # Full 5×5 ensemble
        train_results = run_full_ensemble(config, exp_id)

    return train_results


def run_single_fold(config, exp_id):
    """Run single-fold evaluation."""
    # Load data
    data_dir = Path(config.data.data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")

    # Take first fold only (Butina split 0, fold 0)
    # (Assume data has 'split' and 'fold' columns)
    train_df = train_df[
        (train_df['split'] == 0) & (train_df['fold'].isin([1, 2, 3, 4]))
    ]
    val_df = val_df[
        (val_df['split'] == 0) & (val_df['fold'] == 0)
    ]

    # Train
    model = ChempropModel(
        df_train=train_df,
        df_validation=val_df,
        **config.model,
        **config.data
    )

    with mlflow.start_run(run_name=exp_id):
        mlflow.log_params({
            'experiment_id': exp_id,
            'mode': 'single_fold',
            **exp_config
        })

        model.fit()

        # Evaluate
        metrics = model.evaluate(val_df)

        mlflow.log_metrics(metrics)

    return metrics


def run_full_ensemble(config, exp_id):
    """Run full 5×5 ensemble."""
    # Load data
    data_dir = Path(config.data.data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")

    all_predictions = []

    with mlflow.start_run(run_name=exp_id) as parent_run:
        mlflow.log_params({
            'experiment_id': exp_id,
            'mode': 'full_ensemble',
            **exp_config
        })

        # Train 25 models
        for split in range(5):
            for fold in range(5):
                with mlflow.start_run(
                    run_name=f"{exp_id}_s{split}_f{fold}",
                    nested=True
                ) as child_run:

                    # Split data
                    train_mask = (
                        (train_df['split'] != split) |
                        ((train_df['split'] == split) & (train_df['fold'] != fold))
                    )
                    val_mask = (train_df['split'] == split) & (train_df['fold'] == fold)

                    train_subset = train_df[train_mask]
                    val_subset = train_df[val_mask]

                    # Train
                    model = ChempropModel(
                        df_train=train_subset,
                        df_validation=val_subset,
                        **config.model,
                        **config.data
                    )
                    model.fit()

                    # Save predictions
                    # (For ensemble averaging)
                    val_preds = model.predict(val_subset)
                    all_predictions.append({
                        'split': split,
                        'fold': fold,
                        'predictions': val_preds,
                        'targets': val_subset[config.data.target_cols].values,
                    })

        # Ensemble evaluation
        # (Average predictions, compute metrics)
        ensemble_metrics = compute_ensemble_metrics(all_predictions)

        mlflow.log_metrics(ensemble_metrics)

    return ensemble_metrics


def compute_ensemble_metrics(all_predictions):
    """Compute metrics for ensemble."""
    # This is a simplified version
    # Full implementation would handle proper averaging
    # and compute all challenge metrics

    # ... implementation ...

    return {
        'ensemble_ma_rae': 0.0,  # Placeholder
        'ensemble_r2': 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Tier 1 ablation study")
    parser.add_argument(
        "--mode",
        choices=['single_fold', 'full_ensemble'],
        default='single_fold',
        help="Evaluation mode"
    )
    parser.add_argument(
        "--experiments",
        nargs='+',
        choices=list(EXPERIMENTS.keys()),
        default=list(EXPERIMENTS.keys()),
        help="Experiments to run"
    )
    parser.add_argument(
        "--config",
        default="configs/tier1/combined.yaml",
        help="Base config file"
    )

    args = parser.parse_args()

    # Load base config
    base_config = OmegaConf.load(args.config)

    # Run experiments
    results = {}
    for exp_id in args.experiments:
        print(f"\n{'='*60}")
        print(f"Running {exp_id} in {args.mode} mode")
        print(f"{'='*60}\n")

        exp_config = EXPERIMENTS[exp_id]
        metrics = run_experiment(exp_id, exp_config, base_config, args.mode)

        results[exp_id] = metrics

        print(f"\n{exp_id} Results:")
        print(json.dumps(metrics, indent=2))

    # Save summary
    summary_path = Path(f"results/tier1_ablation_{args.mode}_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()
```

#### Usage

```bash
# Single-fold rapid screening (1-2 days)
python scripts/experiments/run_tier1_ablation.py --mode single_fold

# Full ensemble evaluation (5-7 days)
python scripts/experiments/run_tier1_ablation.py --mode full_ensemble

# Run specific experiments only
python scripts/experiments/run_tier1_ablation.py \
    --experiments E0_baseline E7_combined \
    --mode single_fold
```

### 5.3 Results Analysis

#### File: `scripts/analysis/analyze_tier1_results.py`

```python
"""Analyze Tier 1 ablation results."""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(results_path):
    """Load experiment results."""
    with open(results_path) as f:
        return json.load(f)


def create_comparison_table(results):
    """Create comparison table."""
    df = pd.DataFrame(results).T
    df = df.sort_values('ensemble_ma_rae')

    # Compute improvements vs baseline
    baseline_mae = df.loc['E0_baseline', 'ensemble_ma_rae']
    df['improvement_pct'] = (
        (baseline_mae - df['ensemble_ma_rae']) / baseline_mae * 100
    )

    return df


def plot_ablation_results(df, save_path=None):
    """Plot ablation study results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: MA-RAE comparison
    ax = axes[0]
    df_sorted = df.sort_values('ensemble_ma_rae')
    colors = ['red' if idx == 'E0_baseline' else 'blue' for idx in df_sorted.index]

    ax.barh(range(len(df_sorted)), df_sorted['ensemble_ma_rae'], color=colors)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted.index)
    ax.set_xlabel('MA-RAE', fontsize=12)
    ax.set_title('Ablation Study: MA-RAE Comparison', fontsize=14, fontweight='bold')
    ax.axvline(df.loc['E0_baseline', 'ensemble_ma_rae'],
               color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    # Plot 2: Improvement percentage
    ax = axes[1]
    df_sorted = df.sort_values('improvement_pct', ascending=False)
    colors = ['green' if x > 0 else 'gray' for x in df_sorted['improvement_pct']]

    ax.barh(range(len(df_sorted)), df_sorted['improvement_pct'], color=colors)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted.index)
    ax.set_xlabel('Improvement (%)', fontsize=12)
    ax.set_title('Ablation Study: Improvement vs Baseline', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    results_path = Path("results/tier1_ablation_full_ensemble_summary.json")
    results = load_results(results_path)

    df = create_comparison_table(results)

    print("\nTier 1 Ablation Study Results")
    print("="*60)
    print(df)

    # Save table
    table_path = Path("results/tier1_ablation_table.csv")
    df.to_csv(table_path)
    print(f"\nTable saved to {table_path}")

    # Plot
    plot_path = Path("results/tier1_ablation_plots.png")
    plot_ablation_results(df, save_path=plot_path)


if __name__ == '__main__':
    main()
```

---

## Configuration Best Practices

### Hyperparameter Ranges

Based on literature and pharma industry practice:

#### Uncertainty Weighting

```yaml
# Conservative (stable)
uncertainty_init_log_vars: 0.0      # σ=1.0, neutral start
uncertainty_min_log_var: -5.0       # Prevents precision explosion
uncertainty_max_log_var: 5.0        # Prevents degenerate solutions

# Aggressive (more adaptation)
uncertainty_init_log_vars: -1.0     # σ=0.6, start with higher precision
uncertainty_min_log_var: -10.0      # Allow more extreme precisions
uncertainty_max_log_var: 10.0       # Allow more uncertainty
```

**Recommendation**: Start conservative, inspect learned σ values, then adjust if needed.

#### Task Affinity Grouping

```yaml
# Conservative
min_groups: 3
max_groups: 5
clustering_method: agglomerative
allow_singleton_groups: false  # Force tasks to group

# Aggressive
min_groups: 2
max_groups: 7
clustering_method: spectral  # More flexible
allow_singleton_groups: true  # Isolate problematic tasks
```

**Recommendation**: Use domain constraints to guide grouping, allow singletons for MGMB.

#### PCGrad

```yaml
# Standard
pcgrad_method: "pcgrad"
pcgrad_reduction: "mean"
pcgrad_cpu_offload: false  # Use GPU

# Memory-constrained
pcgrad_method: "pcgrad"
pcgrad_reduction: "mean"
pcgrad_cpu_offload: true  # Offload to CPU
```

**Recommendation**: Monitor conflict frequency. If conflicts_per_step > 10, gradients are very conflicting (expected). If < 0.1, tasks are already aligned (PCGrad may not help much).

### HPO Search Spaces

For Phase 5 hyperparameter optimization:

```python
# Ray Tune search space for Tier 1
search_space = {
    # Core architecture
    "depth": tune.choice([3, 4, 5]),
    "message_hidden_dim": tune.choice([600, 700, 900]),
    "hidden_dim": tune.choice([200, 300, 600]),
    "num_layers": tune.choice([2, 3, 4]),
    "dropout": tune.uniform(0.0, 0.2),

    # Uncertainty weighting
    "uncertainty_init_log_vars": tune.uniform(-2.0, 2.0),

    # Task affinity
    "min_groups": tune.randint(2, 4),
    "max_groups": tune.randint(4, 8),

    # Optimization
    "init_lr": tune.loguniform(1e-4, 1e-2),
    "batch_size": tune.choice([64, 128, 256]),
}
```

---

## Known Issues & Mitigations

This section documents issues identified during plan review and their resolutions.

### Issue 1: Uncertainty Loss Formula Documentation

**Problem**: Original documentation showed formula `L = Σ_i (L_i / (2σ_i²) + log(σ_i))` but did not clearly explain the parameterization used in code.

**Resolution**: Updated docstring with full mathematical derivation showing:

- Parameterization: `log_vars = s_i = log(σ²)`, NOT `log(σ)`
- Implementation: `loss = 0.5 * exp(-s) * L_i + 0.5 * s` where `0.5 * s = log(σ)`
- Added reference to yaringal/multi-task-learning-example for verification

**Impact**: Documentation now matches implementation; no code changes needed.

### Issue 2: PCGrad + Lightning Integration Complexity

**Problem**: Chemprop's `MPNN.training_step()` computes a single aggregated loss, but PCGrad requires per-task gradients for gradient surgery.

**Resolution**: Implement PCGrad as a **Lightning Callback** (see Phase 3.1) that:

1. Hooks into `on_before_backward()` to intercept the aggregate loss
2. Temporarily splits computation to get per-task gradients
3. Applies gradient surgery
4. Restores gradients before optimizer step

**Alternative Approach**: Modify `UncertaintyWeightedLoss.forward()` to return both total loss and per-task losses, enabling PCGrad to operate on task-specific gradients.

**Trade-offs**:

- Callback approach: Cleaner separation of concerns, but ~2x compute per step
- Modified loss approach: More efficient, but couples loss and gradient surgery

**Decision**: Start with callback approach for clarity; optimize later if needed.

### Issue 3: Feature Interaction Matrix

**Problem**: Plan did not document how new features interact with existing callbacks.

**Resolution**: Added this interaction matrix:

| Feature | InterTaskAffinity | Curriculum | Joint Sampler | target_weights |
|---------|-------------------|------------|---------------|----------------|
| Uncertainty Weighting | ✅ Compatible | ⚠️ See Note 1 | ✅ Compatible | ⚠️ See Note 2 |
| Optimal Grouper | ✅ Enhances | ✅ Compatible | ✅ Compatible | ✅ Compatible |
| PCGrad | ❌ Conflict | ✅ Compatible | ✅ Compatible | ⚠️ See Note 3 |

**Notes**:

1. **Uncertainty + Curriculum**: Uncertainty weights should be reset or scaled when curriculum phase changes (new data distribution)
2. **Uncertainty + target_weights**: These serve similar purposes. When both enabled, apply `target_weights` FIRST as fixed priors, then let uncertainty learn residual weights
3. **PCGrad + target_weights**: PCGrad modifies gradients, not losses. Apply target_weights before PCGrad

**Action**: Add configuration option `uncertainty_target_weight_mode: ["multiply", "replace", "additive"]`

### Issue 4: Missing Integration Tests

**Problem**: Unit tests cover individual components but no integration tests verify combined behavior.

**Resolution**: Added Phase 4 integration test specifications:

```python
# tests/integration/test_tier1_mtl_integration.py

class TestTier1Integration:
    """Integration tests for Tier 1 MTL features."""

    def test_uncertainty_plus_curriculum(self, mini_dataset):
        """Verify uncertainty weights reset on curriculum phase transition."""
        # Train with curriculum, verify uncertainty params change appropriately

    def test_pcgrad_skip_when_low_conflicts(self, mini_dataset):
        """PCGrad should be no-op when gradients align."""

    def test_all_features_combined(self, mini_dataset):
        """Smoke test: all features enabled, training completes without error."""

    def test_determinism_with_all_features(self, mini_dataset):
        """Same seed produces same results with all features enabled."""
```

### Issue 5: Config Schema Placement

**Problem**: Uncertainty weighting config was placed in `ModelConfig`, but these are optimization/loss parameters.

**Resolution**: Keep in `ModelConfig` for consistency with existing patterns. The `ModelConfig` dataclass already contains `criterion_name` which is loss-related. Added comment in config explaining rationale:

```python
@dataclass
class ModelConfig:
    # NOTE: Uncertainty weighting is technically loss/optimization related,
    # but kept here alongside criterion_name for consistency.
    # Future refactor could move to OptimizationConfig.
    use_uncertainty_weighting: bool = False
```

### Issue 6: Determinism Testing

**Problem**: No explicit tests for reproducibility when MTL features are enabled.

**Resolution**: Added determinism test to Phase 1:

```python
def test_uncertainty_loss_deterministic():
    """Verify same seed produces identical loss values."""
    torch.manual_seed(42)
    loss1 = UncertaintyWeightedLoss(n_tasks=9)

    torch.manual_seed(42)
    loss2 = UncertaintyWeightedLoss(n_tasks=9)

    # Same initialization
    assert torch.allclose(loss1.log_vars, loss2.log_vars)

    # Same forward output
    preds = torch.randn(32, 9)
    targets = torch.randn(32, 9)
    assert loss1(preds, targets) == loss2(preds, targets)
```

---

## Success Criteria

### Phase 0 Success (Diagnostics) - NEW

- ✅ Diagnostic script runs without error
- ✅ Outputs gradient conflict matrix and cosine similarity matrix
- ✅ Report clearly recommends PROCEED or SKIP for Phase 3
- ✅ Decision documented with quantitative evidence

### Phase 1 Success (Uncertainty Weighting)

- ✅ Unit tests pass (100% coverage)
- ✅ Acceptance test: MA-RAE ≤ 0.57 (5% improvement)
- ✅ MGMB R² ≥ 0.30 (+25% vs. baseline 0.24)
- ✅ Learned uncertainties: sparse tasks have σ > abundant tasks

### Phase 2 Success (Task Affinity)

- ✅ Unit tests pass
- ✅ Silhouette analysis identifies 3-5 optimal groups
- ✅ Groups match domain knowledge (lipophilicity, clearance, etc.)
- ✅ MA-RAE ≤ 0.58 (3% improvement)

### Phase 3 Success (PCGrad)

- ✅ Unit tests pass
- ✅ Gradient conflicts detected (conflicts_per_step > 0.1)
- ✅ MA-RAE ≤ 0.58 (3% improvement)
- ✅ No training instability

### Phase 4 Success (Integration)

- ✅ All components work together
- ✅ Integration tests pass
- ✅ Combined system trains faster than sequential additions

### Phase 5 Success (Experiments)

- ✅ All 8 experiments complete (E0-E7)
- ✅ E7 (combined) achieves MA-RAE ≤ 0.54 (10% improvement)
- ✅ Sparse tasks: MGMB R² ≥ 0.35, MBPB R² ≥ 0.45
- ✅ Leaderboard rank ≤ 10 (from current 17)
- ✅ Statistical significance confirmed (paired t-test, p < 0.05)

### Final Leaderboard Success

- ✅ MA-RAE ≤ 0.54 (competitive with top 10)
- ✅ All endpoints R² ≥ 0.40
- ✅ LogD rank improves to ≤ 30 (from 46)
- ✅ Overall rank ≤ 10

---

## Appendix: Quick Start Guide

### For Opus 4.5

**Phase 0 (Half day)**: Diagnostics - NEW

```bash
# Run gradient conflict diagnostic FIRST
python scripts/analysis/gradient_conflict_diagnostic.py \
    -c configs/3-production/ensemble_chemprop_hpo_001.yaml \
    --num-batches 100 \
    --output-dir assets/analysis/gradient_conflicts

# Review output to decide on Phase 3
cat assets/analysis/gradient_conflicts/summary_report.md
```

**Phase 1 (Day 1)**: Uncertainty weighting

```bash
# 1. Implement uncertainty_loss.py
# 2. Add config fields
# 3. Write unit tests
# 4. Run acceptance test
python tests/acceptance/test_uncertainty_weighting.py
```

**Phase 2 (Days 2-4)**: Task affinity

```bash
# 1. Implement optimal_grouper.py
# 2. Integrate with model
# 3. Write unit tests
# 4. Run on subset
python scripts/experiments/test_optimal_grouping.py
```

**Phase 3 (Days 5-9)**: PCGrad (CONDITIONAL on Phase 0 results)

```bash
# SKIP if Phase 0 recommended SKIP!
# 1. Implement pcgrad.py
# 2. Modify Lightning module
# 3. Write unit tests
# 4. Run acceptance test
python tests/acceptance/test_pcgrad.py
```

**Phase 4 (Days 10-12)**: Integration

```bash
# 1. Create combined config
# 2. Run integration test
python tests/integration/test_tier1_combined.py
```

**Phase 5 (Days 13-22)**: Experiments

```bash
# Single-fold screening
python scripts/experiments/run_tier1_ablation.py --mode single_fold

# Full ensemble (if promising)
python scripts/experiments/run_tier1_ablation.py --mode full_ensemble

# Analyze results
python scripts/analysis/analyze_tier1_results.py
```

---

## END OF IMPLEMENTATION PLAN
