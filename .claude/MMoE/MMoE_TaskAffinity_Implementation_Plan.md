# Multi-Gate Mixture-of-Experts (MMoE) and Task Affinity Implementation Plan

**Project**: OpenADMET ExpansionRx ADMET Property Prediction
**Date**: January 3, 2026
**Author**: Implementation planning for Opus 4.5
**Target**: Address performance gaps in sparse endpoints (MGMB, MBPB) and improve LogD predictions

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Background & Motivation](#background--motivation)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Phases](#implementation-phases)
5. [Configuration Schema](#configuration-schema)
6. [Python Implementation Details](#python-implementation-details)
7. [Testing Strategy](#testing-strategy)
8. [Experiment Design](#experiment-design)
9. [Key Concepts (Plain English)](#key-concepts-plain-english)
10. [Success Criteria](#success-criteria)
11. [References](#references)

---

## Executive Summary

This document outlines the implementation plan for three new FFN architectures to improve multi-task ADMET prediction:

1. **MMoE (Multi-Gate Mixture-of-Experts)**: Per-task gating networks for dynamic expert routing
2. **Grouped Multi-Head**: Task affinity-based architecture with task-group-specific decoders
3. **MMoE-Grouped**: Hybrid architecture combining both approaches with hierarchical experts

**Primary Goals**:

- Improve sparse endpoint performance (MGMB: R²=0.24 → target 0.40+, MBPB: R²=0.35 → target 0.50+)
- Boost LogD performance (current rank 46/234 → target top 20)
- Maintain/improve overall MA-RAE (current 0.60, target 0.57)

**Implementation Approach**: Option C - Sequential implementation with systematic ablation studies

---

## ⚠️ Critical Updates and Corrections (January 2026)

> **Review Status**: This plan has been validated against the codebase and web resources. The following corrections and additions address critical issues identified during review.

### GPU Memory Budget

**Constraint**: ~10 GB GPU memory budget

**Memory Estimates** (approximate, for batch_size=64, 9 tasks):

| Architecture | Estimated Memory | Notes |
|--------------|------------------|-------|
| Baseline MLP | ~2-3 GB | Single FFN head |
| MMoE (6 experts) | ~4-5 GB | Shared experts + 9 gates |
| MMoE (12 experts) | ~6-8 GB | At upper limit |
| Grouped (4 groups) | ~3-4 GB | Smaller than MMoE |
| MMoE-Grouped | ~5-7 GB | Depends on experts_per_group |

**Recommendations for 10GB budget**:

- Limit `n_experts` to 8 maximum
- Limit `expert_hidden_dim` to 600 maximum
- Use gradient checkpointing if needed
- Consider mixed precision training (fp16)

### CheMeleon Compatibility

**Requirement**: Must work with both Chemprop and CheMeleon backends.

**Implementation Notes**:

- Use `create_ffn_predictor()` factory from `ffn_factory.py`
- Register new FFN types via `PredictorRegistry.register()`
- CheMeleon's encoder outputs compatible `hidden_dim` (typically 256-512)
- Adjust `input_dim` parameter based on encoder backbone

**Factory Pattern** (existing code path):

```python
# In ffn_factory.py - extend for new types
def create_ffn_predictor(
    ffn_type: str,
    n_tasks: int,
    input_dim: int,
    **kwargs
) -> Predictor:
    if ffn_type == "mmoe":
        return MMoERegressionFFN(n_tasks=n_tasks, input_dim=input_dim, **kwargs)
    elif ffn_type == "grouped":
        return GroupedMultiHeadFFN(n_tasks=n_tasks, input_dim=input_dim, **kwargs)
    elif ffn_type == "mmoe_grouped":
        return MMoEGroupedFFN(n_tasks=n_tasks, input_dim=input_dim, **kwargs)
    # ... existing types
```

### Load Balancing Loss (CRITICAL ADDITION)

**Issue**: Original plan mentions entropy regularization but lacks load balancing loss.

**Why Load Balancing Matters**:

- Prevents "expert collapse" where 1-2 experts handle all tasks
- Industry standard from Google's Switch Transformer and subsequent MoE work
- Essential for training stability with many experts

**Implementation** (add to `MMoERegressionFFN`):

```python
def _compute_load_balance_loss(self, gate_weights_list: list[Tensor]) -> Tensor:
    """
    Compute load balancing loss (industry best practice).

    Loss = n_experts * sum_e(f_e * P_e)
    where f_e = fraction of routing to expert e
          P_e = average gate probability for expert e
    """
    all_gates = torch.stack(gate_weights_list, dim=0)  # [n_tasks, batch, experts]
    avg_gate_prob = all_gates.mean(dim=(0, 1))  # [experts]
    routing_fraction = avg_gate_prob
    load_balance = self.n_experts * (routing_fraction * avg_gate_prob).sum()
    return load_balance
```

**Add to `__init__`**:

```python
load_balance_weight: float = 0.01,  # Typical range: 0.001 - 0.1
```

**Add to training**:

```python
# In training_step
loss = main_loss + model.load_balance_weight * load_balance_loss
```

### Expert Utilization Tracking (MLflow)

**Requirement**: Track expert utilization with metrics AND plots logged as artifacts.

**Metrics to Log** (per epoch):

```python
# Log to MLflow during training
mlflow.log_metrics({
    f"expert_{i}_utilization": usage_pct
    for i, usage_pct in enumerate(expert_utilization)
})
mlflow.log_metric("gate_entropy_avg", entropy_value)
mlflow.log_metric("load_balance_loss", lb_loss.item())
```

**Plots to Generate** (log as artifacts):

```python
import matplotlib.pyplot as plt

def log_expert_utilization_plots(gate_weights_per_task: dict, epoch: int):
    """Generate and log expert utilization heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap: rows=tasks, cols=experts
    data = np.array([gate_weights_per_task[t].mean(0).cpu().numpy()
                     for t in range(9)])
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels([f'E{i}' for i in range(data.shape[1])])
    ax.set_yticks(range(9))
    ax.set_yticklabels(['LogD', 'KSOL', 'HLM', 'MLM', 'Caco-2 Papp',
                        'Caco-2 Efflux', 'MPPB', 'MBPB', 'MGMB'])
    ax.set_xlabel('Expert')
    ax.set_ylabel('Task')
    ax.set_title(f'Expert Utilization (Epoch {epoch})')
    plt.colorbar(im)

    plt.tight_layout()
    fig.savefig(f'/tmp/expert_utilization_epoch_{epoch}.png')
    mlflow.log_artifact(f'/tmp/expert_utilization_epoch_{epoch}.png',
                        artifact_path='expert_utilization')
    plt.close()
```

**Gate Weight Distribution Plot**:

```python
def log_gate_distribution_plot(gate_weights: Tensor, task_name: str, epoch: int):
    """Log histogram of gate weights for a specific task."""
    fig, ax = plt.subplots()
    weights = gate_weights.mean(0).cpu().numpy()  # Average over batch
    ax.bar(range(len(weights)), weights)
    ax.set_xlabel('Expert Index')
    ax.set_ylabel('Average Gate Weight')
    ax.set_title(f'{task_name} Gate Distribution (Epoch {epoch})')

    fig.savefig(f'/tmp/gate_dist_{task_name}_epoch_{epoch}.png')
    mlflow.log_artifact(f'/tmp/gate_dist_{task_name}_epoch_{epoch}.png',
                        artifact_path='gate_distributions')
    plt.close()
```

### Updated HPO Search Space (Memory-Constrained)

**Original ranges were too aggressive for 10GB budget. Updated ranges:**

```yaml
# MMoE HPO (memory-safe for 10GB)
mmoe_hpo:
  n_experts:
    type: choice
    values: [4, 6, 8]  # Reduced from [4, 6, 8, 10, 12]
  expert_hidden_dim:
    type: choice
    values: [300, 450, 600]  # Reduced from [300, 600, 900, 1200]
  expert_n_layers:
    type: choice
    values: [1, 2, 3]  # Unchanged
  gate_hidden_dim:
    type: choice
    values: [100, 200, 300]  # Reduced from [100, 200, 300, 400, 600]
  load_balance_weight:
    type: loguniform
    low: 0.001
    high: 0.1
  entropy_regularization:
    type: choice
    values: [0.0, 0.001, 0.01]
```

### Architecture Correction: Task Towers

**Issue Identified**: The original forward pass used simple `nn.Linear(expert_hidden_dim, 1)` for output layers.

**Correction**: True MMoE architecture requires task-specific **towers** (small MLPs), not just linear projections:

```python
# Original (too simple):
self.output_layers = nn.ModuleList([
    nn.Linear(expert_hidden_dim, 1)
    for _ in range(n_tasks)
])

# Corrected (proper task towers):
self.task_towers = nn.ModuleList([
    nn.Sequential(
        nn.Linear(expert_hidden_dim, gate_hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(gate_hidden_dim, 1)
    )
    for _ in range(n_tasks)
])
```

**Why This Matters**: Task towers allow each task to learn task-specific transformations of the weighted expert features. Without towers, all tasks are forced to directly regress from the same expert representation, reducing the benefit of per-task gating.

---

## Background & Motivation

### Current State

**Performance Analysis** (December 16, 2025 submission):

- Overall Rank: 17/234 (top 7.3%)
- MA-RAE: 0.60 ± 0.03
- R² (overall): 0.53 ± 0.04

**Problematic Endpoints**:

| Endpoint | Rank | R² | Issue |
|----------|------|-----|-------|
| LogD | 46/234 | 0.73 | Underperforming despite high data coverage |
| MGMB | — | 0.24 | Sparse data, poor generalization |
| MBPB | — | 0.35 | Sparse data, high variance |

**Current Architecture Limitations**:

1. **Standard MoE**: Single gating network assumes all tasks benefit from same expert weighting
2. **Hard Parameter Sharing**: Shared MPNN encoder may create task conflicts
3. **No Task Relationship Modeling**: Treats all 9 endpoints as independent despite clear chemical relationships

### Why MMoE?

**Core Problem**: Multi-task learning with heterogeneous tasks suffers from negative transfer when tasks have conflicting optimization landscapes.

**MMoE Solution**:

- Each task gets its own gating network → task-specific expert routing
- Experts can specialize for task subsets
- Reduces gradient conflicts between unrelated tasks

**Expected Benefits for ADMET**:

- **Sparse Tasks**: MGMB/MBPB can use dedicated experts without interference from abundant tasks
- **Task Conflicts**: LogD (lipophilicity) won't be forced to share features with MGMB (microbiome binding)
- **Selective Sharing**: Related tasks (HLM/MLM clearance) can still share experts

### Why Task Affinity Grouping?

**Core Problem**: With 9 tasks, MMoE has to learn task relationships from scratch during training.

**Task Affinity Solution**:

- Pre-compute gradient-based task similarity (already implemented in codebase)
- Cluster tasks into groups (e.g., [LogD, KSOL], [HLM, MLM], [Caco-2 Papp, Efflux], ...)
- Create group-specific architectures

**Expected Benefits**:

- **Reduced Search Space**: Experts pre-assigned to task groups
- **Faster Convergence**: Warm start with known task relationships
- **Interpretability**: Explicit task groupings aid in debugging

### Why MMoE-Grouped (Hybrid)?

**Synergy**: Combine pre-computed structure (task groups) with learned routing (per-task gates)

**Architecture**: Hierarchical experts

- Level 1: Task groups get dedicated expert pools
- Level 2: Within each group, per-task gates route to group's experts

**Expected Benefits**:

- **Best of Both**: Structure + flexibility
- **Scalability**: Adding new endpoints easier (assign to existing group)
- **Robustness**: Falls back to group-level sharing if task-specific routing fails

---

## Architecture Overview

### Phase 1: MMoE (Multi-Gate Mixture-of-Experts)

```
MPNN Encoder → [h] → ┌─ Gate_task1 → softmax → w1 ┐
                      ├─ Gate_task2 → softmax → w2 ├→ Σ(wi * Expert_i(h)) → [pred1, pred2, ..., pred9]
                      ├─ ...                        │
                      └─ Gate_task9 → softmax → w9 ┘

                      Expert_1(h) → e1 ────────────┐
                      Expert_2(h) → e2 ────────────┤
                      ...                          │
                      Expert_K(h) → eK ────────────┘
```

**Key Components**:

- **Shared Experts**: K expert networks (e.g., K=4-8), each is an MLP
- **Task-Specific Gates**: 9 gating networks, one per endpoint
- **Weighted Combination**: Each task's output = Σ(gate_weights × expert_outputs)

**Differences from Current MoE**:

| Current MoE | MMoE |
|-------------|------|
| 1 gate network (shared) | 9 gate networks (per-task) |
| All tasks use same expert weights | Each task learns custom expert weights |
| K outputs (one per expert) | 9 outputs (one per task) |

### Phase 2: Grouped Multi-Head Architecture

```
MPNN Encoder → [h] → Task Affinity Clustering
                      ↓
                      Group 1 [LogD, KSOL] ──→ Decoder_group1 → [pred_LogD, pred_KSOL]
                      Group 2 [HLM, MLM] ────→ Decoder_group2 → [pred_HLM, pred_MLM]
                      Group 3 [Caco-2 Papp, Caco-2 Efflux] → Decoder_group3 → [...]
                      Group 4 [MPPB, MBPB, MGMB] → Decoder_group4 → [...]
```

**Key Components**:

- **Task Affinity Module**: Compute gradient-based task similarity (already exists in codebase)
- **Clustering**: Group tasks using agglomerative/spectral/kmeans (already exists)
- **Group-Specific Decoders**: Each group gets its own MLP decoder
- **Optional Shared Trunk**: Common layers before group-specific branches

**Configuration Options**:

1. **Automatic Grouping**: Use gradient affinity + clustering
2. **Manual Grouping**: User specifies groups in YAML
3. **Hybrid**: Auto-compute, allow manual override

### Phase 3: MMoE-Grouped (Hybrid)

```
MPNN Encoder → [h] → Task Affinity Clustering
                      ↓
        Group 1 ──→ ┌─ Expert_1_1 ┐ ←── Gate_LogD (within group 1) ──→ pred_LogD
                    ├─ Expert_1_2 ├ ←── Gate_KSOL (within group 1) ──→ pred_KSOL
                    └─ Expert_1_3 ┘

        Group 2 ──→ ┌─ Expert_2_1 ┐ ←── Gate_HLM (within group 2) ──→ pred_HLM
                    ├─ Expert_2_2 ├ ←── Gate_MLM (within group 2) ──→ pred_MLM
                    └─ Expert_2_3 ┘
        ...
```

**Key Components**:

- **Hierarchical Experts**: Each task group has its own expert pool (K_group experts)
- **Per-Task Gates within Groups**: Tasks in same group share experts but have independent gates
- **Cross-Group Isolation**: Group 1 tasks cannot access Group 2 experts (reduces interference)

**Benefits**:

- **Modularity**: Sparse tasks (Group 4) get dedicated resources without competing with abundant tasks
- **Efficiency**: Fewer experts per group (e.g., 3-4) vs. global pool (8-12)
- **Graceful Degradation**: If grouping is suboptimal, per-task gates can still adapt

---

## Implementation Phases

### Phase 1: MMoE Implementation (Week 1-2)

**Deliverables**:

1. `MMoERegressionFFN` class in `src/admet/model/chemprop/ffn.py`
2. Configuration schema for `ffn_type: mmoe`
3. Unit tests for forward/backward pass
4. Regression tests on synthetic data
5. Acceptance test on ExpansionRx subset

**Tasks**:

- [ ] Implement `MMoERegressionFFN` class with per-task gating networks
- [ ] Add YAML configuration fields (`n_experts`, `gate_hidden_dim`, `gate_n_layers`)
- [ ] Register predictor as `"regression-mmoe"`
- [ ] Write unit tests for tensor shapes and gradient flow
- [ ] Write regression test comparing to baseline MLP on small dataset
- [ ] Write acceptance test on 1000-compound ExpansionRx subset (target: <2 min on 3080)
- [ ] Document hyperparameter ranges for HPO

### Phase 2: Grouped Multi-Head Implementation (Week 3-4)

**Deliverables**:

1. `GroupedMultiHeadFFN` class in `src/admet/model/chemprop/ffn.py`
2. Configuration schema for `ffn_type: grouped_multihead`
3. Integration with existing task affinity module
4. Unit tests for group assignment and decoder routing
5. Regression tests on grouped vs. ungrouped performance
6. Acceptance test with manual and automatic grouping modes

**Tasks**:

- [ ] Implement `GroupedMultiHeadFFN` with task-group-specific decoders
- [ ] Add YAML fields (`task_groups`, `auto_group`, `group_clustering_method`)
- [ ] Integrate with `TaskAffinityModule.compute_task_affinity()`
- [ ] Implement manual group specification (list of task indices)
- [ ] Write unit tests for group assignment correctness
- [ ] Write regression test: verify grouped arch improves sparse tasks vs. baseline
- [ ] Write acceptance test with both auto and manual grouping
- [ ] Document expected task groupings based on domain knowledge

### Phase 3: MMoE-Grouped Hybrid (Week 5-6)

**Deliverables**:

1. `MMoEGroupedFFN` class combining both architectures
2. Configuration schema for `ffn_type: mmoe_grouped`
3. Hierarchical expert-gate structure
4. Unit tests for hierarchical routing
5. Regression tests on task group isolation
6. Acceptance test comparing to Phase 1 and Phase 2 architectures

**Tasks**:

- [ ] Implement `MMoEGroupedFFN` with group-specific expert pools
- [ ] Add YAML fields (`experts_per_group`, `share_gates_within_group`)
- [ ] Implement expert-to-group assignment logic
- [ ] Write unit tests for cross-group isolation (Group 1 doesn't access Group 2 experts)
- [ ] Write regression test: verify no negative transfer between groups
- [ ] Write acceptance test on full 5-fold CV subset
- [ ] Document hierarchical gating behavior

### Phase 4: Hyperparameter Optimization (Week 7-8)

**Deliverables**:

1. HPO search spaces for all three architectures
2. Ray Tune integration with MLflow logging
3. Performance comparison across architectures
4. Recommendations for default configurations

**Tasks**:

- [ ] Define HPO search spaces (see [Configuration Schema](#configuration-schema))
- [ ] Run Ray Tune ASHA with ~500 trials per architecture
- [ ] Log experiments to MLflow with architecture tags
- [ ] Analyze top-10 configurations per architecture
- [ ] Generate performance reports (MA-RAE, per-endpoint R², rank improvement)
- [ ] Select best configuration per architecture for Phase 5

### Phase 5: Full Ensemble Evaluation (Week 9-10)

**Deliverables**:

1. Full 5×5 CV ensemble for each architecture
2. Leaderboard submissions
3. Ablation study results
4. Final model selection and documentation

**Tasks**:

- [ ] Train full 25-model ensembles (5 splits × 5 folds) for:
  - Baseline (current best MLP)
  - MMoE
  - Grouped Multi-Head
  - MMoE-Grouped
- [ ] Submit to leaderboard and track rankings
- [ ] Perform ablation studies (see [Experiment Design](#experiment-design))
- [ ] Generate comparison reports with statistical significance tests
- [ ] Update MODEL_CARD.md with best architecture
- [ ] Document lessons learned and future directions

---

## Configuration Schema

### Phase 1: MMoE Configuration

```yaml
# configs/mmoe/base.yaml

model:
  ffn_type: mmoe  # New FFN type

  # Shared with other FFN types
  hidden_dim: 600
  num_layers: 2
  dropout: 0.1

  # MMoE-specific parameters
  mmoe:
    n_experts: 6  # Number of expert networks (HPO range: 4-8, limited for 10GB GPU)
    expert_hidden_dim: 600  # Hidden dim for expert MLPs (default: same as hidden_dim)
    expert_n_layers: 2  # Layers per expert (HPO range: 1-4)

    # Per-task gating networks
    gate_hidden_dim: 300  # Hidden dim for gate networks (HPO range: 100-600)
    gate_n_layers: 1  # Layers per gate (HPO range: 1-3)
    gate_activation: relu  # Activation for gates (options: relu, tanh, gelu)

    # Regularization
    gate_dropout: 0.0  # Dropout for gates (HPO range: 0.0-0.3)
    expert_dropout: 0.1  # Dropout for experts (use global dropout if null)

    # Optional: Expert specialization encouragement
    entropy_regularization: 0.0  # Encourage diverse expert usage (HPO range: 0.0-0.01)
    load_balancing: false  # Force equal expert usage (can hurt performance)
```

**HPO Search Space**:

```python
{
    "ffn_type": "mmoe",
    "mmoe.n_experts": tune.choice([4, 6, 8]),  # 4-8 (memory-constrained for 10GB GPU)
    "mmoe.expert_hidden_dim": tune.choice([300, 450, 600]),  # Reduced for 10GB GPU
    "mmoe.expert_n_layers": tune.randint(1, 5),  # 1-4
    "mmoe.gate_hidden_dim": tune.choice([100, 200, 300]),  # Reduced for 10GB GPU
    "mmoe.gate_n_layers": tune.randint(1, 4),  # 1-3
    "mmoe.gate_activation": tune.choice(["relu", "gelu"]),
    "mmoe.gate_dropout": tune.uniform(0.0, 0.3),
    "mmoe.entropy_regularization": tune.loguniform(1e-4, 1e-2),
    "mmoe.load_balance_weight": tune.loguniform(0.001, 0.1),  # Industry best practice
}
```

### Phase 2: Grouped Multi-Head Configuration

```yaml
# configs/grouped/base.yaml

model:
  ffn_type: grouped_multihead

  # Shared parameters
  hidden_dim: 600
  num_layers: 2
  dropout: 0.1

  # Grouped architecture parameters
  grouped:
    # Task grouping strategy
    auto_group: true  # Compute groups from task affinity (true) or use manual (false)
    group_clustering_method: agglomerative  # options: agglomerative, spectral, kmeans
    n_groups: 4  # Number of task groups (HPO range: 2-6, ignored if manual groups)

    # Manual group specification (used if auto_group: false)
    # task_groups:
    #   - [0, 1]  # Group 1: LogD, KSOL
    #   - [2, 3]  # Group 2: HLM CLint, MLM CLint
    #   - [4, 5]  # Group 3: Caco-2 Papp, Caco-2 Efflux
    #   - [6, 7, 8]  # Group 4: MPPB, MBPB, MGMB

    # Task affinity computation (for auto_group: true)
    affinity_epochs: 2  # Epochs to compute gradients for affinity matrix
    affinity_linkage: ward  # For agglomerative clustering (ward, average, complete)

    # Shared trunk (optional)
    use_shared_trunk: false  # Add shared layers before group-specific decoders
    trunk_n_layers: 1  # Layers in shared trunk (HPO range: 1-3)
    trunk_hidden_dim: 600  # Hidden dim for trunk
    trunk_dropout: 0.1

    # Group-specific decoders
    decoder_n_layers: 2  # Layers per group decoder (HPO range: 1-4)
    decoder_hidden_dim: 600  # Hidden dim per group (or list for per-group control)
    decoder_dropout: 0.1

    # Output layer strategy
    shared_output_layer: false  # Share final linear layer across groups (usually false)
```

**HPO Search Space**:

```python
{
    "ffn_type": "grouped_multihead",
    "grouped.n_groups": tune.randint(2, 7),  # 2-6
    "grouped.group_clustering_method": tune.choice(["agglomerative", "spectral"]),
    "grouped.use_shared_trunk": tune.choice([True, False]),
    "grouped.trunk_n_layers": tune.randint(1, 4) if use_shared_trunk else 0,
    "grouped.decoder_n_layers": tune.randint(1, 5),  # 1-4
    "grouped.decoder_hidden_dim": tune.choice([300, 600, 900, 1200]),
}
```

### Phase 3: MMoE-Grouped Configuration

```yaml
# configs/mmoe_grouped/base.yaml

model:
  ffn_type: mmoe_grouped

  # Shared parameters
  hidden_dim: 600
  num_layers: 2
  dropout: 0.1

  # Hierarchical MMoE + Grouping parameters
  mmoe_grouped:
    # Task grouping (same as grouped architecture)
    auto_group: true
    group_clustering_method: agglomerative
    n_groups: 4

    # Expert pools per group
    experts_per_group: 4  # Number of experts in each group's pool (HPO range: 2-6)
    expert_hidden_dim: 600
    expert_n_layers: 2
    expert_dropout: 0.1

    # Per-task gates within each group
    gate_hidden_dim: 300
    gate_n_layers: 1
    gate_dropout: 0.0

    # Cross-group sharing (advanced)
    allow_cross_group_experts: false  # Allow tasks to access other groups' experts
    cross_group_penalty: 0.01  # Regularization if cross_group enabled

    # Optional shared trunk
    use_shared_trunk: false
    trunk_n_layers: 1
    trunk_hidden_dim: 600
```

**HPO Search Space**:

```python
{
    "ffn_type": "mmoe_grouped",
    "mmoe_grouped.n_groups": tune.randint(2, 7),
    "mmoe_grouped.experts_per_group": tune.randint(2, 7),  # 2-6
    "mmoe_grouped.expert_hidden_dim": tune.choice([300, 600, 900]),
    "mmoe_grouped.expert_n_layers": tune.randint(1, 5),
    "mmoe_grouped.gate_hidden_dim": tune.choice([100, 200, 300, 400]),
    "mmoe_grouped.gate_n_layers": tune.randint(1, 4),
    "mmoe_grouped.use_shared_trunk": tune.choice([True, False]),
}
```

---

## Python Implementation Details

### Phase 1: MMoE Implementation

**File**: `src/admet/model/chemprop/ffn.py`

#### Class Structure

```python
@PredictorRegistry.register("regression-mmoe")
class MMoERegressionFFN(Predictor, HyperparametersMixin):
    """
    Multi-Gate Mixture-of-Experts for multi-task regression.

    Each task has its own gating network that learns task-specific
    weightings over a shared pool of expert networks.

    References:
        Ma et al. (2018). "Modeling Task Relationships in Multi-task Learning
        with Multi-gate Mixture-of-Experts." KDD 2018.
    """

    n_targets = 1
    _T_default_criterion = MSE
    _T_default_metric = MSE

    def __init__(
        self,
        n_tasks: int,
        n_experts: int = 6,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        expert_hidden_dim: int = 600,
        expert_n_layers: int = 2,
        gate_hidden_dim: int = 300,
        gate_n_layers: int = 1,
        gate_activation: str = "relu",
        dropout: float = 0.0,
        gate_dropout: float = 0.0,
        expert_dropout: float | None = None,
        entropy_regularization: float = 0.0,
        criterion: ChempropMetric | None = None,
        output_transform: UnscaleTransform | None = None,
    ):
        """
        Initialize MMoE regression predictor.

        Args:
            n_tasks: Number of prediction tasks (9 for ADMET)
            n_experts: Number of expert networks
            input_dim: Input dimension from MPNN encoder
            expert_hidden_dim: Hidden dimension for expert MLPs
            expert_n_layers: Number of layers per expert
            gate_hidden_dim: Hidden dimension for gating networks
            gate_n_layers: Number of layers per gate
            gate_activation: Activation for gate networks
            dropout: Global dropout rate
            gate_dropout: Dropout specifically for gates
            expert_dropout: Dropout for experts (uses dropout if None)
            entropy_regularization: Coefficient for entropy loss on gate distributions
            criterion: Loss criterion
            output_transform: Output scaling transform
        """
        super().__init__(criterion, output_transform)

        self.n_tasks = n_tasks
        self.n_experts = n_experts
        self.entropy_reg = entropy_regularization

        # Expert networks (shared across all tasks)
        self.experts = nn.ModuleList([
            self._build_expert(
                input_dim,
                expert_hidden_dim,
                expert_n_layers,
                expert_dropout if expert_dropout is not None else dropout
            )
            for _ in range(n_experts)
        ])

        # Per-task gating networks
        self.gates = nn.ModuleList([
            self._build_gate(
                input_dim,
                gate_hidden_dim,
                gate_n_layers,
                n_experts,  # Output dimension = number of experts
                gate_activation,
                gate_dropout
            )
            for _ in range(n_tasks)
        ])

        # Task-specific output layers (1D for regression)
        self.output_layers = nn.ModuleList([
            nn.Linear(expert_hidden_dim, 1)
            for _ in range(n_tasks)
        ])

    def _build_expert(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float
    ) -> nn.Module:
        """Build a single expert MLP."""
        layers = []
        current_dim = input_dim

        for i in range(n_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        return nn.Sequential(*layers)

    def _build_gate(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_experts: int,
        activation: str,
        dropout: float
    ) -> nn.Module:
        """Build a single gating network."""
        layers = []
        current_dim = input_dim

        act_fn = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh
        }[activation]

        for i in range(n_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        # Final layer outputs gate weights (one per expert)
        layers.append(nn.Linear(current_dim, n_experts))

        return nn.Sequential(*layers)

    def forward(self, h: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Forward pass through MMoE.

        Args:
            h: Encoded molecular representation [batch_size, input_dim]
            mask: Task mask for missing labels [batch_size, n_tasks]

        Returns:
            predictions: [batch_size, n_tasks]
        """
        batch_size = h.size(0)

        # Compute all expert outputs
        # expert_outputs: [n_experts, batch_size, expert_hidden_dim]
        expert_outputs = torch.stack([
            expert(h) for expert in self.experts
        ], dim=0)

        # Compute per-task gate weights and weighted expert combinations
        task_outputs = []
        gate_entropies = []

        for task_idx in range(self.n_tasks):
            # Gate logits: [batch_size, n_experts]
            gate_logits = self.gates[task_idx](h)
            gate_weights = F.softmax(gate_logits, dim=1)  # [batch_size, n_experts]

            # Store entropy for regularization (encourage diverse expert usage)
            if self.entropy_reg > 0:
                gate_dist = gate_weights.mean(dim=0)  # Average over batch
                entropy = -(gate_dist * torch.log(gate_dist + 1e-10)).sum()
                gate_entropies.append(entropy)

            # Weighted combination of experts
            # gate_weights: [batch_size, n_experts, 1]
            # expert_outputs: [n_experts, batch_size, expert_hidden_dim]
            # weighted: [batch_size, expert_hidden_dim]
            weighted = torch.einsum('be,ebh->bh', gate_weights, expert_outputs)

            # Task-specific output layer
            task_output = self.output_layers[task_idx](weighted)  # [batch_size, 1]
            task_outputs.append(task_output)

        # Stack task outputs: [batch_size, n_tasks]
        predictions = torch.cat(task_outputs, dim=1)

        # Store gate entropies for regularization in training loop
        if self.entropy_reg > 0 and self.training:
            avg_entropy = torch.stack(gate_entropies).mean()
            # Negative entropy (we want to MINIMIZE entropy to encourage specialization)
            # Or positive to MAXIMIZE entropy for diverse usage - depends on goal
            # For MMoE, typically we want diverse usage, so add entropy loss
            self._gate_entropy_loss = -self.entropy_reg * avg_entropy

        return predictions

    def get_gate_weights(self, h: Tensor) -> dict[int, Tensor]:
        """
        Get gate weight distributions for each task (for analysis/visualization).

        Args:
            h: Encoded representation [batch_size, input_dim]

        Returns:
            Dictionary mapping task_idx -> gate_weights [batch_size, n_experts]
        """
        gate_weights = {}
        for task_idx in range(self.n_tasks):
            gate_logits = self.gates[task_idx](h)
            gate_weights[task_idx] = F.softmax(gate_logits, dim=1)
        return gate_weights
```

#### Integration with Training Loop

**File**: `src/admet/model/chemprop/model.py` (or Lightning module)

```python
class ChempropLightningModule(pl.LightningModule):

    def training_step(self, batch, batch_idx):
        # ... existing code ...

        loss = self.criterion(predictions, targets, mask)

        # Add entropy regularization if using MMoE
        if hasattr(self.predictor, '_gate_entropy_loss'):
            loss += self.predictor._gate_entropy_loss

        return loss
```

### Phase 2: Grouped Multi-Head Implementation

**File**: `src/admet/model/chemprop/ffn.py`

#### Class Structure

```python
@PredictorRegistry.register("regression-grouped")
class GroupedMultiHeadFFN(Predictor, HyperparametersMixin):
    """
    Grouped multi-head architecture with task affinity-based clustering.

    Tasks are grouped based on gradient-based affinity, and each group
    gets its own decoder network.
    """

    n_targets = 1
    _T_default_criterion = MSE
    _T_default_metric = MSE

    def __init__(
        self,
        n_tasks: int,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        # Grouping parameters
        auto_group: bool = True,
        n_groups: int = 4,
        task_groups: list[list[int]] | None = None,
        group_clustering_method: str = "agglomerative",
        affinity_matrix: torch.Tensor | None = None,  # Pre-computed affinity
        # Shared trunk
        use_shared_trunk: bool = False,
        trunk_n_layers: int = 1,
        trunk_hidden_dim: int = 600,
        trunk_dropout: float = 0.1,
        # Group decoders
        decoder_n_layers: int = 2,
        decoder_hidden_dim: int | list[int] = 600,
        decoder_dropout: float = 0.1,
        # Other
        criterion: ChempropMetric | None = None,
        output_transform: UnscaleTransform | None = None,
    ):
        """
        Initialize grouped multi-head predictor.

        Args:
            n_tasks: Number of tasks
            input_dim: Input dimension from MPNN
            auto_group: Automatically compute groups from affinity
            n_groups: Number of groups (for auto_group=True)
            task_groups: Manual group specification (for auto_group=False)
                         e.g., [[0,1], [2,3], [4,5,6,7,8]]
            group_clustering_method: Clustering method (agglomerative, spectral, kmeans)
            affinity_matrix: Pre-computed task affinity matrix [n_tasks, n_tasks]
            use_shared_trunk: Add shared layers before group decoders
            trunk_n_layers: Layers in shared trunk
            trunk_hidden_dim: Hidden dim for trunk
            trunk_dropout: Trunk dropout
            decoder_n_layers: Layers per group decoder
            decoder_hidden_dim: Hidden dim per group (int) or per-group list
            decoder_dropout: Decoder dropout
            criterion: Loss criterion
            output_transform: Output transform
        """
        super().__init__(criterion, output_transform)

        self.n_tasks = n_tasks
        self.auto_group = auto_group
        self.n_groups = n_groups

        # Determine task groupings
        if auto_group:
            if affinity_matrix is None:
                raise ValueError("Must provide affinity_matrix for auto_group=True")
            self.task_groups = self._compute_groups(
                affinity_matrix,
                n_groups,
                group_clustering_method
            )
        else:
            if task_groups is None:
                raise ValueError("Must provide task_groups for auto_group=False")
            self.task_groups = task_groups
            self.n_groups = len(task_groups)

        # Validate groupings
        all_tasks = [t for group in self.task_groups for t in group]
        assert len(all_tasks) == n_tasks, "All tasks must be assigned to a group"
        assert len(set(all_tasks)) == n_tasks, "Tasks cannot be in multiple groups"

        # Build shared trunk (optional)
        current_dim = input_dim
        if use_shared_trunk:
            self.trunk = self._build_trunk(
                input_dim, trunk_hidden_dim, trunk_n_layers, trunk_dropout
            )
            current_dim = trunk_hidden_dim
        else:
            self.trunk = None

        # Build group-specific decoders
        if isinstance(decoder_hidden_dim, int):
            decoder_hidden_dims = [decoder_hidden_dim] * self.n_groups
        else:
            decoder_hidden_dims = decoder_hidden_dim
            assert len(decoder_hidden_dims) == self.n_groups

        self.group_decoders = nn.ModuleList([
            self._build_decoder(
                current_dim,
                decoder_hidden_dims[i],
                decoder_n_layers,
                len(self.task_groups[i]),  # Number of tasks in this group
                decoder_dropout
            )
            for i in range(self.n_groups)
        ])

        # Create task-to-group mapping for fast lookup
        self.task_to_group = {}
        self.task_to_group_idx = {}
        for group_idx, task_list in enumerate(self.task_groups):
            for task_idx_in_group, task_idx in enumerate(task_list):
                self.task_to_group[task_idx] = group_idx
                self.task_to_group_idx[task_idx] = task_idx_in_group

    def _compute_groups(
        self,
        affinity_matrix: torch.Tensor,
        n_groups: int,
        method: str
    ) -> list[list[int]]:
        """
        Compute task groupings from affinity matrix using clustering.

        Args:
            affinity_matrix: [n_tasks, n_tasks] similarity matrix
            n_groups: Number of groups to create
            method: Clustering method (agglomerative, spectral, kmeans)

        Returns:
            List of task groups, e.g., [[0,1], [2,3,4], [5,6,7,8]]
        """
        from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans

        # Convert to numpy
        affinity = affinity_matrix.cpu().numpy()
        n_tasks = affinity.shape[0]

        # Convert affinity to distance (for methods that need it)
        # distance = 1 - affinity (assumes affinity in [0, 1])
        distance = 1.0 - affinity

        if method == "agglomerative":
            clustering = AgglomerativeClustering(
                n_clusters=n_groups,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(distance)

        elif method == "spectral":
            # Spectral clustering uses affinity directly
            clustering = SpectralClustering(
                n_clusters=n_groups,
                affinity='precomputed',
                assign_labels='kmeans'
            )
            labels = clustering.fit_predict(affinity)

        elif method == "kmeans":
            # KMeans on affinity matrix rows as features
            clustering = KMeans(n_clusters=n_groups, random_state=42)
            labels = clustering.fit_predict(affinity)

        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Convert labels to task groups
        task_groups = [[] for _ in range(n_groups)]
        for task_idx, group_idx in enumerate(labels):
            task_groups[group_idx].append(task_idx)

        # Remove empty groups (shouldn't happen but safety check)
        task_groups = [g for g in task_groups if len(g) > 0]

        return task_groups

    def _build_trunk(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float
    ) -> nn.Module:
        """Build shared trunk."""
        layers = []
        current_dim = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ])
            current_dim = hidden_dim
        return nn.Sequential(*layers)

    def _build_decoder(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_tasks_in_group: int,
        dropout: float
    ) -> nn.Module:
        """Build decoder for one task group."""
        layers = []
        current_dim = input_dim

        for _ in range(n_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ])
            current_dim = hidden_dim

        # Final output layer for all tasks in group
        layers.append(nn.Linear(current_dim, n_tasks_in_group))

        return nn.Sequential(*layers)

    def forward(self, h: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Forward pass through grouped architecture.

        Args:
            h: Input from MPNN [batch_size, input_dim]
            mask: Task mask [batch_size, n_tasks]

        Returns:
            predictions: [batch_size, n_tasks]
        """
        # Shared trunk (if enabled)
        if self.trunk is not None:
            h = self.trunk(h)

        # Process each group
        # Build output tensor with correct task ordering
        batch_size = h.size(0)
        predictions = torch.zeros(batch_size, self.n_tasks, device=h.device)

        for group_idx, task_list in enumerate(self.task_groups):
            # Group decoder output: [batch_size, n_tasks_in_group]
            group_output = self.group_decoders[group_idx](h)

            # Assign to correct task indices
            for task_idx_in_group, task_idx in enumerate(task_list):
                predictions[:, task_idx] = group_output[:, task_idx_in_group]

        return predictions

    def get_task_groups(self) -> list[list[int]]:
        """Return task groupings for analysis."""
        return self.task_groups
```

#### Integration with Task Affinity Module

**File**: `src/admet/model/task_affinity.py` (or wherever it currently lives)

```python
from admet.model.chemprop.ffn import GroupedMultiHeadFFN

def create_grouped_ffn_from_affinity(
    model: ChempropModel,  # Existing model to compute affinity
    train_loader: DataLoader,
    n_groups: int,
    affinity_epochs: int = 2,
    clustering_method: str = "agglomerative",
    **ffn_kwargs
) -> GroupedMultiHeadFFN:
    """
    Create GroupedMultiHeadFFN with automatically computed task affinity.

    Args:
        model: Existing Chemprop model (for computing gradients)
        train_loader: Training data
        n_groups: Number of task groups
        affinity_epochs: Epochs to compute gradients for affinity
        clustering_method: Clustering method
        **ffn_kwargs: Additional kwargs for GroupedMultiHeadFFN

    Returns:
        Initialized GroupedMultiHeadFFN with computed groupings
    """
    # Compute affinity matrix (existing implementation)
    affinity_module = TaskAffinityModule(model)
    affinity_matrix = affinity_module.compute_task_affinity(
        train_loader,
        epochs=affinity_epochs
    )

    # Create FFN with affinity-based groupings
    grouped_ffn = GroupedMultiHeadFFN(
        n_tasks=model.n_tasks,
        auto_group=True,
        n_groups=n_groups,
        affinity_matrix=affinity_matrix,
        group_clustering_method=clustering_method,
        **ffn_kwargs
    )

    return grouped_ffn
```

### Phase 3: MMoE-Grouped Implementation

**File**: `src/admet/model/chemprop/ffn.py`

```python
@PredictorRegistry.register("regression-mmoe-grouped")
class MMoEGroupedFFN(Predictor, HyperparametersMixin):
    """
    Hierarchical MMoE with task grouping.

    Combines task affinity grouping with per-task gating within groups.
    Each group has its own expert pool, and tasks within a group have
    independent gating networks.
    """

    n_targets = 1
    _T_default_criterion = MSE
    _T_default_metric = MSE

    def __init__(
        self,
        n_tasks: int,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        # Task grouping
        auto_group: bool = True,
        n_groups: int = 4,
        task_groups: list[list[int]] | None = None,
        group_clustering_method: str = "agglomerative",
        affinity_matrix: torch.Tensor | None = None,
        # Experts per group
        experts_per_group: int = 4,
        expert_hidden_dim: int = 600,
        expert_n_layers: int = 2,
        expert_dropout: float = 0.1,
        # Gates per task
        gate_hidden_dim: int = 300,
        gate_n_layers: int = 1,
        gate_dropout: float = 0.0,
        # Cross-group sharing
        allow_cross_group_experts: bool = False,
        cross_group_penalty: float = 0.01,
        # Optional shared trunk
        use_shared_trunk: bool = False,
        trunk_n_layers: int = 1,
        trunk_hidden_dim: int = 600,
        trunk_dropout: float = 0.1,
        # Other
        criterion: ChempropMetric | None = None,
        output_transform: UnscaleTransform | None = None,
    ):
        """
        Initialize hierarchical MMoE with task grouping.

        Key difference from GroupedMultiHeadFFN: Instead of one decoder per group,
        each group has multiple experts + per-task gates.

        Key difference from MMoERegressionFFN: Experts are partitioned by group,
        so Group 1 tasks only access Group 1 experts (unless cross_group enabled).
        """
        super().__init__(criterion, output_transform)

        self.n_tasks = n_tasks
        self.experts_per_group = experts_per_group
        self.allow_cross_group = allow_cross_group
        self.cross_group_penalty = cross_group_penalty

        # Determine task groupings (same as GroupedMultiHeadFFN)
        if auto_group:
            if affinity_matrix is None:
                raise ValueError("Must provide affinity_matrix for auto_group=True")
            self.task_groups = self._compute_groups(
                affinity_matrix, n_groups, group_clustering_method
            )
        else:
            if task_groups is None:
                raise ValueError("Must provide task_groups for auto_group=False")
            self.task_groups = task_groups
            self.n_groups = len(task_groups)

        # Validate groupings
        all_tasks = [t for group in self.task_groups for t in group]
        assert len(all_tasks) == n_tasks
        assert len(set(all_tasks)) == n_tasks

        # Create task mappings
        self.task_to_group = {}
        for group_idx, task_list in enumerate(self.task_groups):
            for task_idx in task_list:
                self.task_to_group[task_idx] = group_idx

        # Shared trunk (optional)
        current_dim = input_dim
        if use_shared_trunk:
            self.trunk = self._build_trunk(
                input_dim, trunk_hidden_dim, trunk_n_layers, trunk_dropout
            )
            current_dim = trunk_hidden_dim
        else:
            self.trunk = None

        # Build expert pools per group
        # expert_pools: [n_groups] -> ModuleList of [experts_per_group] experts
        self.expert_pools = nn.ModuleList([
            nn.ModuleList([
                self._build_expert(
                    current_dim, expert_hidden_dim, expert_n_layers, expert_dropout
                )
                for _ in range(experts_per_group)
            ])
            for _ in range(self.n_groups)
        ])

        # Build per-task gating networks
        # gates: [n_tasks] -> gate network
        # Each gate outputs weights for its group's experts
        self.gates = nn.ModuleDict()
        for task_idx in range(n_tasks):
            group_idx = self.task_to_group[task_idx]
            n_experts_for_task = (
                experts_per_group if not allow_cross_group
                else experts_per_group * self.n_groups
            )
            self.gates[str(task_idx)] = self._build_gate(
                current_dim, gate_hidden_dim, gate_n_layers,
                n_experts_for_task, gate_dropout
            )

        # Task-specific output layers
        self.output_layers = nn.ModuleList([
            nn.Linear(expert_hidden_dim, 1) for _ in range(n_tasks)
        ])

    def _compute_groups(self, affinity_matrix, n_groups, method):
        """Same as GroupedMultiHeadFFN._compute_groups"""
        # ... (copy implementation from above)
        pass

    def _build_trunk(self, input_dim, hidden_dim, n_layers, dropout):
        """Same as GroupedMultiHeadFFN._build_trunk"""
        # ... (copy implementation)
        pass

    def _build_expert(self, input_dim, hidden_dim, n_layers, dropout):
        """Same as MMoERegressionFFN._build_expert"""
        # ... (copy implementation)
        pass

    def _build_gate(self, input_dim, hidden_dim, n_layers, n_experts, dropout):
        """Same as MMoERegressionFFN._build_gate (no activation param for simplicity)"""
        # ... (copy implementation with ReLU)
        pass

    def forward(self, h: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Hierarchical forward pass.

        For each task:
        1. Identify its group
        2. Compute gate weights over group's experts (or all experts if cross_group)
        3. Weighted combination
        4. Task-specific output layer
        """
        batch_size = h.size(0)

        # Shared trunk
        if self.trunk is not None:
            h = self.trunk(h)

        # Process each task
        task_outputs = []

        for task_idx in range(self.n_tasks):
            group_idx = self.task_to_group[task_idx]

            if self.allow_cross_group:
                # Gate over ALL experts across all groups
                all_expert_outputs = []
                for g_idx in range(self.n_groups):
                    for expert in self.expert_pools[g_idx]:
                        all_expert_outputs.append(expert(h))
                expert_outputs = torch.stack(all_expert_outputs, dim=0)
                # expert_outputs: [n_groups * experts_per_group, batch_size, hidden_dim]

            else:
                # Gate over only this group's experts
                group_expert_outputs = [
                    expert(h) for expert in self.expert_pools[group_idx]
                ]
                expert_outputs = torch.stack(group_expert_outputs, dim=0)
                # expert_outputs: [experts_per_group, batch_size, hidden_dim]

            # Compute gate weights
            gate_logits = self.gates[str(task_idx)](h)  # [batch_size, n_experts]
            gate_weights = F.softmax(gate_logits, dim=1)

            # Weighted combination
            weighted = torch.einsum('be,ebh->bh', gate_weights, expert_outputs)

            # Output layer
            task_output = self.output_layers[task_idx](weighted)
            task_outputs.append(task_output)

        predictions = torch.cat(task_outputs, dim=1)
        return predictions

    def get_gate_weights_per_group(self, h: Tensor) -> dict[int, dict[int, Tensor]]:
        """
        Get gate weights organized by group and task.

        Returns:
            {group_idx: {task_idx: gate_weights}}
        """
        result = {g: {} for g in range(self.n_groups)}

        for task_idx in range(self.n_tasks):
            group_idx = self.task_to_group[task_idx]
            gate_logits = self.gates[str(task_idx)](h)
            gate_weights = F.softmax(gate_logits, dim=1)
            result[group_idx][task_idx] = gate_weights

        return result
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/unit/test_ffn.py`

#### Phase 1: MMoE Unit Tests

```python
import pytest
import torch
from admet.model.chemprop.ffn import MMoERegressionFFN

class TestMMoEFFN:

    @pytest.fixture
    def mmoe_ffn(self):
        return MMoERegressionFFN(
            n_tasks=9,
            n_experts=4,
            input_dim=300,
            expert_hidden_dim=200,
            expert_n_layers=2,
            gate_hidden_dim=100,
            gate_n_layers=1,
        )

    def test_forward_shape(self, mmoe_ffn):
        """Test output shape is correct."""
        batch_size = 16
        h = torch.randn(batch_size, 300)

        output = mmoe_ffn(h)

        assert output.shape == (batch_size, 9)

    def test_gradient_flow(self, mmoe_ffn):
        """Test gradients flow to all experts and gates."""
        h = torch.randn(8, 300)
        output = mmoe_ffn(h)
        loss = output.sum()
        loss.backward()

        # Check all experts have gradients
        for expert in mmoe_ffn.experts:
            for param in expert.parameters():
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()

        # Check all gates have gradients
        for gate in mmoe_ffn.gates:
            for param in gate.parameters():
                assert param.grad is not None

    def test_gate_weights_sum_to_one(self, mmoe_ffn):
        """Test gate weights are valid probability distributions."""
        h = torch.randn(16, 300)
        gate_weights = mmoe_ffn.get_gate_weights(h)

        for task_idx, weights in gate_weights.items():
            # weights: [batch_size, n_experts]
            sums = weights.sum(dim=1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
            assert (weights >= 0).all()
            assert (weights <= 1).all()

    def test_different_gates_per_task(self, mmoe_ffn):
        """Test that different tasks learn different gate weights."""
        h = torch.randn(32, 300)
        gate_weights = mmoe_ffn.get_gate_weights(h)

        # After random init, gate weights should differ between tasks
        w_task0 = gate_weights[0].mean(dim=0)  # Average over batch
        w_task1 = gate_weights[1].mean(dim=0)

        # Not a strict requirement (could be same by chance), but very unlikely
        assert not torch.allclose(w_task0, w_task1, atol=0.1)

    def test_entropy_regularization(self):
        """Test entropy regularization term is computed."""
        mmoe = MMoERegressionFFN(
            n_tasks=9,
            n_experts=4,
            input_dim=300,
            entropy_regularization=0.01
        )
        mmoe.train()

        h = torch.randn(16, 300)
        output = mmoe(h)

        assert hasattr(mmoe, '_gate_entropy_loss')
        assert isinstance(mmoe._gate_entropy_loss, torch.Tensor)
```

#### Phase 2: Grouped Multi-Head Unit Tests

```python
from admet.model.chemprop.ffn import GroupedMultiHeadFFN

class TestGroupedMultiHeadFFN:

    @pytest.fixture
    def manual_grouped_ffn(self):
        """FFN with manual task grouping."""
        task_groups = [
            [0, 1],  # LogD, KSOL
            [2, 3],  # HLM, MLM
            [4, 5, 6, 7, 8]  # Rest
        ]
        return GroupedMultiHeadFFN(
            n_tasks=9,
            input_dim=300,
            auto_group=False,
            task_groups=task_groups,
            decoder_n_layers=2,
            decoder_hidden_dim=200,
        )

    @pytest.fixture
    def auto_grouped_ffn(self):
        """FFN with automatic affinity-based grouping."""
        # Create mock affinity matrix (high within groups, low across)
        affinity = torch.zeros(9, 9)
        # Group 1: tasks 0, 1
        affinity[0, 1] = affinity[1, 0] = 0.9
        # Group 2: tasks 2, 3, 4
        for i in [2, 3, 4]:
            for j in [2, 3, 4]:
                if i != j:
                    affinity[i, j] = 0.8
        # Group 3: tasks 5, 6, 7, 8
        for i in [5, 6, 7, 8]:
            for j in [5, 6, 7, 8]:
                if i != j:
                    affinity[i, j] = 0.7
        # Add self-affinity
        affinity.fill_diagonal_(1.0)

        return GroupedMultiHeadFFN(
            n_tasks=9,
            input_dim=300,
            auto_group=True,
            n_groups=3,
            affinity_matrix=affinity,
            group_clustering_method="agglomerative",
            decoder_n_layers=2,
            decoder_hidden_dim=200,
        )

    def test_manual_grouping(self, manual_grouped_ffn):
        """Test manual task groups are respected."""
        expected_groups = [[0, 1], [2, 3], [4, 5, 6, 7, 8]]
        assert manual_grouped_ffn.get_task_groups() == expected_groups

    def test_auto_grouping_produces_valid_groups(self, auto_grouped_ffn):
        """Test automatic grouping produces valid task assignments."""
        groups = auto_grouped_ffn.get_task_groups()

        # All tasks assigned
        all_tasks = [t for g in groups for t in g]
        assert sorted(all_tasks) == list(range(9))

        # No duplicates
        assert len(all_tasks) == len(set(all_tasks))

        # Correct number of groups
        assert len(groups) == 3

    def test_forward_shape(self, manual_grouped_ffn):
        """Test output shape."""
        h = torch.randn(16, 300)
        output = manual_grouped_ffn(h)
        assert output.shape == (16, 9)

    def test_gradient_flow_to_all_groups(self, manual_grouped_ffn):
        """Test all group decoders receive gradients."""
        h = torch.randn(8, 300)
        output = manual_grouped_ffn(h)
        loss = output.sum()
        loss.backward()

        for decoder in manual_grouped_ffn.group_decoders:
            for param in decoder.parameters():
                assert param.grad is not None

    def test_shared_trunk_optional(self):
        """Test model works with and without shared trunk."""
        task_groups = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]

        # Without trunk
        ffn_no_trunk = GroupedMultiHeadFFN(
            n_tasks=9,
            auto_group=False,
            task_groups=task_groups,
            use_shared_trunk=False,
        )
        assert ffn_no_trunk.trunk is None

        # With trunk
        ffn_with_trunk = GroupedMultiHeadFFN(
            n_tasks=9,
            auto_group=False,
            task_groups=task_groups,
            use_shared_trunk=True,
            trunk_n_layers=2,
            trunk_hidden_dim=400,
        )
        assert ffn_with_trunk.trunk is not None

        # Both should work
        h = torch.randn(16, 300)
        out1 = ffn_no_trunk(h)
        out2 = ffn_with_trunk(h)
        assert out1.shape == out2.shape == (16, 9)
```

#### Phase 3: MMoE-Grouped Unit Tests

```python
from admet.model.chemprop.ffn import MMoEGroupedFFN

class TestMMoEGroupedFFN:

    @pytest.fixture
    def mmoe_grouped_ffn(self):
        """Hierarchical MMoE with task grouping."""
        affinity = torch.eye(9)  # Simple affinity for testing
        affinity[0, 1] = affinity[1, 0] = 0.9
        affinity[2, 3] = affinity[3, 2] = 0.8

        return MMoEGroupedFFN(
            n_tasks=9,
            input_dim=300,
            auto_group=True,
            n_groups=3,
            affinity_matrix=affinity,
            experts_per_group=3,
            expert_hidden_dim=200,
            expert_n_layers=2,
            gate_hidden_dim=100,
            gate_n_layers=1,
        )

    def test_expert_pools_per_group(self, mmoe_grouped_ffn):
        """Test each group has its own expert pool."""
        assert len(mmoe_grouped_ffn.expert_pools) == 3
        for pool in mmoe_grouped_ffn.expert_pools:
            assert len(pool) == 3  # experts_per_group

    def test_forward_shape(self, mmoe_grouped_ffn):
        """Test output shape."""
        h = torch.randn(16, 300)
        output = mmoe_grouped_ffn(h)
        assert output.shape == (16, 9)

    def test_cross_group_isolation(self):
        """Test tasks only access their group's experts when cross_group=False."""
        affinity = torch.eye(9)
        affinity[:3, :3] = 0.9  # Group 1
        affinity[3:6, 3:6] = 0.8  # Group 2
        affinity[6:, 6:] = 0.7  # Group 3

        mmoe = MMoEGroupedFFN(
            n_tasks=9,
            input_dim=300,
            auto_group=True,
            n_groups=3,
            affinity_matrix=affinity,
            experts_per_group=4,
            allow_cross_group_experts=False,
        )

        h = torch.randn(16, 300)
        gate_weights = mmoe.get_gate_weights_per_group(h)

        # Each group's tasks should have gates over 4 experts (not 12)
        for group_idx, task_weights in gate_weights.items():
            for task_idx, weights in task_weights.items():
                assert weights.shape[1] == 4  # experts_per_group, not total

    def test_cross_group_sharing(self):
        """Test cross-group expert sharing when enabled."""
        affinity = torch.eye(9)

        mmoe = MMoEGroupedFFN(
            n_tasks=9,
            input_dim=300,
            auto_group=True,
            n_groups=3,
            affinity_matrix=affinity,
            experts_per_group=4,
            allow_cross_group_experts=True,
        )

        h = torch.randn(16, 300)

        # Forward pass should work (gates over 3 * 4 = 12 experts)
        output = mmoe(h)
        assert output.shape == (16, 9)
```

### Regression Tests

**File**: `tests/regression/test_architecture_performance.py`

```python
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from admet.model.chemprop.ffn import MMoERegressionFFN, GroupedMultiHeadFFN, MMoEGroupedFFN
from chemprop.nn.predictors import MLP

class TestArchitectureRegression:
    """
    Regression tests to ensure new architectures don't degrade performance
    on synthetic multi-task data.
    """

    @pytest.fixture
    def synthetic_dataset(self):
        """
        Create synthetic multi-task dataset with known task relationships.

        Tasks 0, 1: Highly correlated (share signal)
        Tasks 2, 3, 4: Moderately correlated
        Tasks 5, 6, 7, 8: Sparse, independent
        """
        torch.manual_seed(42)
        n_samples = 1000
        input_dim = 50

        X = torch.randn(n_samples, input_dim)

        # Shared signal for tasks 0, 1
        signal_01 = X[:, :10].sum(dim=1, keepdim=True) * 0.5
        y0 = signal_01 + torch.randn(n_samples, 1) * 0.1
        y1 = signal_01 + torch.randn(n_samples, 1) * 0.1

        # Shared signal for tasks 2, 3, 4
        signal_234 = X[:, 10:20].sum(dim=1, keepdim=True) * 0.3
        y2 = signal_234 + torch.randn(n_samples, 1) * 0.2
        y3 = signal_234 + torch.randn(n_samples, 1) * 0.2
        y4 = signal_234 + torch.randn(n_samples, 1) * 0.2

        # Independent tasks 5-8 (sparse: 50% missing)
        y5 = X[:, 20:30].sum(dim=1, keepdim=True) * 0.2
        y6 = X[:, 30:40].sum(dim=1, keepdim=True) * 0.2
        y7 = X[:, 40:50].sum(dim=1, keepdim=True) * 0.2
        y8 = torch.randn(n_samples, 1) * 0.5  # Pure noise

        # Add sparsity to tasks 5-8
        mask_5678 = torch.rand(n_samples, 4) > 0.5
        y5[~mask_5678[:, 0:1]] = float('nan')
        y6[~mask_5678[:, 1:2]] = float('nan')
        y7[~mask_5678[:, 2:3]] = float('nan')
        y8[~mask_5678[:, 3:4]] = float('nan')

        Y = torch.cat([y0, y1, y2, y3, y4, y5, y6, y7, y8], dim=1)

        # Split train/val
        train_X, train_Y = X[:800], Y[:800]
        val_X, val_Y = X[800:], Y[800:]

        train_dataset = TensorDataset(train_X, train_Y)
        val_dataset = TensorDataset(val_X, val_Y)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        return train_loader, val_loader

    def train_and_evaluate(self, model, train_loader, val_loader, epochs=50):
        """Simple training loop for regression testing."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss(reduction='none')

        model.train()
        for epoch in range(epochs):
            for X, Y in train_loader:
                optimizer.zero_grad()
                preds = model(X)

                # Masked MSE (ignore NaNs)
                mask = ~torch.isnan(Y)
                loss = criterion(preds, Y.nan_to_num(0.0))
                loss = (loss * mask).sum() / mask.sum()

                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        all_preds, all_targets, all_masks = [], [], []
        with torch.no_grad():
            for X, Y in val_loader:
                preds = model(X)
                all_preds.append(preds)
                all_targets.append(Y)
                all_masks.append(~torch.isnan(Y))

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        masks = torch.cat(all_masks, dim=0)

        # Compute per-task MSE
        mse_per_task = []
        for task_idx in range(9):
            mask_t = masks[:, task_idx]
            if mask_t.sum() == 0:
                mse_per_task.append(float('nan'))
                continue
            pred_t = preds[mask_t, task_idx]
            target_t = targets[mask_t, task_idx]
            mse = ((pred_t - target_t) ** 2).mean().item()
            mse_per_task.append(mse)

        return mse_per_task

    def test_mmoe_vs_baseline(self, synthetic_dataset):
        """Test MMoE performs comparably or better than baseline MLP."""
        train_loader, val_loader = synthetic_dataset

        # Baseline MLP
        baseline = MLP(
            n_tasks=9,
            input_dim=50,
            hidden_dim=100,
            n_layers=2,
            dropout=0.0,
        )
        baseline_mse = self.train_and_evaluate(baseline, train_loader, val_loader)

        # MMoE
        mmoe = MMoERegressionFFN(
            n_tasks=9,
            n_experts=4,
            input_dim=50,
            expert_hidden_dim=100,
            expert_n_layers=2,
            gate_hidden_dim=50,
            gate_n_layers=1,
        )
        mmoe_mse = self.train_and_evaluate(mmoe, train_loader, val_loader)

        # MMoE should not be significantly worse on any task
        for task_idx in range(9):
            if not (torch.isnan(torch.tensor(baseline_mse[task_idx])) or
                    torch.isnan(torch.tensor(mmoe_mse[task_idx]))):
                # Allow 20% degradation tolerance
                assert mmoe_mse[task_idx] < baseline_mse[task_idx] * 1.2, \
                    f"Task {task_idx}: MMoE MSE {mmoe_mse[task_idx]:.4f} >> Baseline {baseline_mse[task_idx]:.4f}"

        # MMoE should improve on at least one task
        improvements = [
            baseline_mse[i] - mmoe_mse[i] for i in range(9)
            if not (torch.isnan(torch.tensor(baseline_mse[i])) or torch.isnan(torch.tensor(mmoe_mse[i])))
        ]
        assert any(imp > 0 for imp in improvements), "MMoE should improve at least one task"

    def test_grouped_improves_sparse_tasks(self, synthetic_dataset):
        """Test grouped architecture improves sparse tasks (5-8)."""
        train_loader, val_loader = synthetic_dataset

        # Baseline
        baseline = MLP(n_tasks=9, input_dim=50, hidden_dim=100, n_layers=2)
        baseline_mse = self.train_and_evaluate(baseline, train_loader, val_loader)

        # Grouped (manual grouping based on known structure)
        task_groups = [[0, 1], [2, 3, 4], [5, 6, 7, 8]]
        grouped = GroupedMultiHeadFFN(
            n_tasks=9,
            input_dim=50,
            auto_group=False,
            task_groups=task_groups,
            decoder_n_layers=2,
            decoder_hidden_dim=100,
        )
        grouped_mse = self.train_and_evaluate(grouped, train_loader, val_loader)

        # Grouped should improve sparse tasks (5-8) on average
        sparse_baseline_avg = sum(baseline_mse[5:9]) / 4
        sparse_grouped_avg = sum(grouped_mse[5:9]) / 4

        assert sparse_grouped_avg < sparse_baseline_avg, \
            f"Grouped should improve sparse tasks: {sparse_grouped_avg:.4f} vs {sparse_baseline_avg:.4f}"
```

### Acceptance Tests

**File**: `tests/acceptance/test_expansionrx_subset.py`

```python
import pytest
import pandas as pd
from pathlib import Path
from admet.data import load_expansionrx_data
from admet.model.chemprop import ChempropModel
from admet.model.chemprop.ffn import MMoERegressionFFN, GroupedMultiHeadFFN, MMoEGroupedFFN

class TestExpansionRxAcceptance:
    """
    Acceptance tests on real ExpansionRx data subset.

    Requirements:
    - Test runtime < 2 minutes on RTX 3080
    - Use actual challenge data (subset)
    - Compare architectures on real performance metrics
    """

    @pytest.fixture(scope="class")
    def expansionrx_subset(self):
        """Load 1000-compound subset of ExpansionRx data."""
        # Assumes data is in standard location
        data_path = Path("data/expansionrx_challenge/train.csv")
        df = pd.read_csv(data_path)

        # Take stratified subset (1000 compounds, balanced across tasks)
        # Use temporal split logic (alphabetical on Molecule Name)
        df_sorted = df.sort_values('Molecule Name')
        subset = df_sorted.iloc[:1000]

        return subset

    @pytest.fixture
    def base_config(self):
        """Base Chemprop configuration for all tests."""
        return {
            'depth': 3,
            'message_hidden_dim': 500,
            'batch_size': 64,
            'max_epochs': 30,  # Reduced for fast testing
            'patience': 5,
            'num_workers': 4,
            'seed': 42,
        }

    def test_mmoe_trains_successfully(self, expansionrx_subset, base_config):
        """Test MMoE architecture trains without errors on real data."""
        config = base_config.copy()
        config.update({
            'ffn_type': 'mmoe',
            'n_experts': 4,
            'expert_hidden_dim': 300,
            'expert_n_layers': 2,
            'gate_hidden_dim': 150,
            'gate_n_layers': 1,
        })

        model = ChempropModel(config)

        # Train
        train_df = expansionrx_subset.iloc[:800]
        val_df = expansionrx_subset.iloc[800:]

        import time
        start = time.time()
        metrics = model.fit(train_df, val_df)
        elapsed = time.time() - start

        # Runtime check
        assert elapsed < 120, f"Training took {elapsed:.1f}s, should be < 120s"

        # Sanity checks
        assert 'val_mae' in metrics
        assert metrics['val_mae'] < 1.0, "Validation MAE should be reasonable"

    def test_grouped_trains_successfully(self, expansionrx_subset, base_config):
        """Test grouped architecture trains on real data."""
        config = base_config.copy()
        config.update({
            'ffn_type': 'grouped_multihead',
            'auto_group': False,
            'task_groups': [[0, 1], [2, 3], [4, 5], [6, 7, 8]],  # Manual ADMET grouping
            'decoder_n_layers': 2,
            'decoder_hidden_dim': 300,
        })

        model = ChempropModel(config)

        train_df = expansionrx_subset.iloc[:800]
        val_df = expansionrx_subset.iloc[800:]

        import time
        start = time.time()
        metrics = model.fit(train_df, val_df)
        elapsed = time.time() - start

        assert elapsed < 120
        assert metrics['val_mae'] < 1.0

    def test_mmoe_grouped_trains_successfully(self, expansionrx_subset, base_config):
        """Test MMoE-grouped architecture trains on real data."""
        config = base_config.copy()
        config.update({
            'ffn_type': 'mmoe_grouped',
            'auto_group': False,
            'task_groups': [[0, 1], [2, 3], [4, 5], [6, 7, 8]],
            'experts_per_group': 3,
            'expert_hidden_dim': 300,
            'expert_n_layers': 2,
            'gate_hidden_dim': 150,
            'gate_n_layers': 1,
        })

        model = ChempropModel(config)

        train_df = expansionrx_subset.iloc[:800]
        val_df = expansionrx_subset.iloc[800:]

        import time
        start = time.time()
        metrics = model.fit(train_df, val_df)
        elapsed = time.time() - start

        assert elapsed < 120
        assert metrics['val_mae'] < 1.0

    def test_architectures_improve_sparse_endpoints(self, expansionrx_subset, base_config):
        """
        Compare architectures on sparse endpoints (MBPB, MGMB).

        Expectation: MMoE and grouped architectures should improve
        R² on sparse tasks compared to baseline MLP.
        """
        sparse_task_indices = [7, 8]  # MBPB, MGMB

        train_df = expansionrx_subset.iloc[:800]
        val_df = expansionrx_subset.iloc[800:]

        # Baseline MLP
        baseline_config = base_config.copy()
        baseline_config.update({'ffn_type': 'regression'})
        baseline_model = ChempropModel(baseline_config)
        baseline_metrics = baseline_model.fit(train_df, val_df)
        baseline_r2_sparse = [
            baseline_metrics[f'val_r2_task_{i}'] for i in sparse_task_indices
        ]

        # MMoE
        mmoe_config = base_config.copy()
        mmoe_config.update({
            'ffn_type': 'mmoe',
            'n_experts': 4,
            'expert_hidden_dim': 300,
            'gate_hidden_dim': 150,
        })
        mmoe_model = ChempropModel(mmoe_config)
        mmoe_metrics = mmoe_model.fit(train_df, val_df)
        mmoe_r2_sparse = [
            mmoe_metrics[f'val_r2_task_{i}'] for i in sparse_task_indices
        ]

        # At least one sparse task should improve
        improvements = [mmoe_r2_sparse[i] - baseline_r2_sparse[i] for i in range(len(sparse_task_indices))]
        assert any(imp > 0.05 for imp in improvements), \
            f"MMoE should improve at least one sparse task by >0.05 R²"
```

---

## Experiment Design

### Ablation Study Structure

**Goal**: Systematically evaluate the contribution of each component.

**Baselines**:

1. **MLP (Current Best)**: Standard feed-forward network (from December 16 submission)
2. **Current MoE**: Existing single-gate MoE implementation

**New Architectures**:
3. **MMoE**: Multi-gate mixture-of-experts
4. **Grouped Multi-Head**: Task affinity-based grouping
5. **MMoE-Grouped**: Hierarchical combination

**Ablation Dimensions**:

- **Architecture**: MLP → MoE → MMoE → Grouped → MMoE-Grouped
- **Task Sampling**: Uniform → α-weighted oversampling
- **Uncertainty Weighting**: Fixed weights → Learnable task uncertainty
- **Pre-training**: Random init → CheMeleon encoder

### Experiment Configurations

#### Single-Fold Rapid Iteration

**Purpose**: Fast experimentation during development

**Configuration**:

- Single Butina split (split 0)
- Single CV fold (fold 0)
- HPO: 100 trials (ASHA, max 50 epochs)
- Metrics: Validation MAE, per-task R²
- Runtime: ~2-4 hours on RTX 3080

**Tracked Metrics**:

```python
{
    'val_mae_macro': float,  # Primary metric
    'val_r2_macro': float,
    'val_mae_per_task': List[float],  # 9 values
    'val_r2_per_task': List[float],
    'train_time_minutes': float,
    'convergence_epoch': int,
}
```

#### Full 5×5 Ensemble Evaluation

**Purpose**: Final performance assessment for leaderboard

**Configuration**:

- 5 Butina splits × 5 CV folds = 25 models
- HPO: 500 trials (ASHA, max 150 epochs)
- Ensemble averaging: Mean predictions across 25 models
- Metrics: MA-RAE, per-endpoint R²/MAE/RAE, Spearman, Kendall
- Runtime: ~3-5 days on RTX 3080

**Tracked Metrics**:

```python
{
    'test_ma_rae': float,  # Leaderboard metric
    'test_r2_macro': float,
    'test_per_endpoint': {
        'LogD': {'mae': float, 'rae': float, 'r2': float, 'spearman': float},
        'KSOL': {...},
        # ... all 9 endpoints
    },
    'ensemble_variance': float,  # Ensemble diversity metric
    'rank_improvement': int,  # vs. baseline
}
```

### HPO Strategy

**Search Algorithm**: Ray Tune ASHA (Asynchronous Successive Halving)

**Search Spaces** (see [Configuration Schema](#configuration-schema) for full details):

**MMoE**:

- `n_experts`: [4, 6, 8]  # Reduced from [4, 6, 8, 10, 12] for 10GB GPU budget
- `expert_hidden_dim`: [300, 450, 600]  # Reduced from [300, 600, 900, 1200] for 10GB GPU
- `expert_n_layers`: [1, 2, 3, 4]
- `gate_hidden_dim`: [100, 200, 300]  # Reduced for 10GB GPU
- `gate_n_layers`: [1, 2, 3]
- `entropy_regularization`: [0.0, 1e-4, 1e-3, 1e-2]
- `load_balance_weight`: [0.001, 0.01, 0.1]  # Industry best practice

**Grouped**:

- `n_groups`: [2, 3, 4, 5, 6]
- `group_clustering_method`: ['agglomerative', 'spectral']
- `use_shared_trunk`: [True, False]
- `trunk_n_layers`: [1, 2, 3] (if trunk enabled)
- `decoder_n_layers`: [1, 2, 3, 4]
- `decoder_hidden_dim`: [300, 450, 600]  # Reduced for 10GB GPU

**MMoE-Grouped**:

- Combination of above + `experts_per_group`: [2, 3, 4]  # Reduced for 10GB GPU

**Shared Hyperparameters** (also tuned):

- `depth`: [3, 4, 5, 6, 7]
- `message_hidden_dim`: [500, 700, 900, 1100]
- `dropout`: [0.0, 0.05, 0.1, 0.15, 0.2]
- `learning_rate`: [1e-5, 1e-4, 1e-3]

### Evaluation Protocol

#### Phase 4: Hyperparameter Optimization (Per Architecture)

**For each architecture (MLP, MMoE, Grouped, MMoE-Grouped)**:

1. **Run HPO** (500 trials, single fold)
   - Log all trials to MLflow
   - Track validation MA-RAE as primary metric
   - Early stop trials at 20/50/100/150 epochs (ASHA)

2. **Analyze Top-10 Configurations**
   - Select top-10 by validation MA-RAE
   - Check for consistent hyperparameter patterns
   - Identify any outliers or overfitting

3. **Select Best Configuration**
   - Choose config with best validation MA-RAE
   - Secondary criteria: per-endpoint R² balance, training stability

4. **Document Results**
   - Generate HPO report (Markdown)
   - Include hyperparameter importance analysis
   - Store in `results/hpo/{architecture}/`

#### Phase 5: Full Ensemble Evaluation

**For each architecture's best config**:

1. **Train 25-Model Ensemble**
   - 5 Butina splits × 5 CV folds
   - Use best HPO config
   - Log each model to MLflow with tags: `split={i}`, `fold={j}`, `architecture={name}`

2. **Generate Predictions**
   - Predict on held-out test set (12% temporal split from ExpansionRx)
   - Average predictions across 25 models
   - Compute ensemble variance as uncertainty metric

3. **Compute Metrics**
   - MA-RAE (primary leaderboard metric)
   - Per-endpoint: MAE, RAE, R², Spearman, Kendall
   - Overall: R² macro, Spearman macro

4. **Submit to Leaderboard**
   - Generate submission CSV
   - Upload to OpenADMET challenge page
   - Record timestamp and rank

5. **Ablation Analysis**
   - Compare architectures pairwise (statistical significance tests)
   - Identify which endpoints benefit most from each architecture
   - Generate comparison report

### Success Criteria

**Minimum Viable Success** (Phase 1-3 Complete):

- [ ] All three architectures (MMoE, Grouped, MMoE-Grouped) train without errors
- [ ] Unit tests pass (100% coverage on new FFN classes)
- [ ] Regression tests pass (no significant degradation vs. baseline on synthetic data)
- [ ] Acceptance tests pass (runtime < 2 min on 3080, reasonable validation metrics)

**Performance Success** (Phase 4-5):

- [ ] At least one architecture improves MA-RAE by ≥5% vs. baseline
- [ ] Sparse endpoints (MGMB, MBPB) show R² improvement of ≥0.10
- [ ] LogD rank improves to top 30 (from current rank 46)
- [ ] Overall leaderboard rank moves into top 10 (from current rank 17)

**Stretch Goals**:

- [ ] MA-RAE < 0.57 (would be top 10 as of December 16, 2025)
- [ ] All endpoints achieve R² > 0.50
- [ ] Publish findings as technical report or blog post

---

## Key Concepts (Plain English)

This section explains the core ideas behind each architecture in non-technical language.

### Multi-Gate Mixture-of-Experts (MMoE)

**The Core Idea**:
Imagine you're building a team to predict 9 different properties of molecules. Instead of having one generalist do all 9 tasks, you hire several specialists (experts) and give each task a manager (gate) who decides which specialists to consult for that specific task.

**Why It Helps**:

- **Different tasks need different knowledge**: Predicting lipophilicity (LogD) requires understanding fat-water partitioning, while predicting gut microbiome binding (MGMB) needs totally different chemistry knowledge
- **Prevents interference**: Without MMoE, the model tries to learn one shared representation for all tasks, which can hurt performance when tasks conflict
- **Selective sharing**: Related tasks (like human and mouse liver clearance) can still share experts, while unrelated tasks use different ones

**How It Works**:

1. Input molecule → 4-12 expert networks (each learns different patterns)
2. For each task (e.g., LogD), a task-specific gate network decides: "Expert 1 is 40% relevant, Expert 2 is 30%, Expert 3 is 20%, Expert 4 is 10%"
3. Task prediction = weighted combination of expert outputs based on gate weights
4. Different tasks learn different weightings → specialists emerge naturally

**Expected Impact on ADMET**:

- **Sparse tasks** (MGMB, MBPB) can use dedicated experts without competition from abundant tasks
- **Task conflicts reduced**: LogD won't be forced to share features with incompatible tasks
- **Better generalization**: Experts specialize in chemical patterns, improving robustness

### Grouped Multi-Head Architecture

**The Core Idea**:
Before training, analyze which tasks are similar (e.g., using gradient-based affinity), then group related tasks together and give each group its own predictor.

**Why It Helps**:

- **Pre-structured learning**: Instead of discovering task relationships from scratch, we give the model a head start
- **Reduced search space**: Fewer parameters to optimize, faster training
- **Interpretability**: Explicit groupings make it clear which tasks are related

**How It Works**:

1. **Compute task affinity**: Train a small model for 1-2 epochs, measure how similar the gradients are for each pair of tasks
   - High gradient similarity → tasks benefit from same features
   - Low similarity → tasks need different features
2. **Cluster tasks**: Use hierarchical clustering to group tasks (e.g., [LogD, KSOL], [HLM, MLM], [Caco-2 Papp, Efflux], [MPPB, MBPB, MGMB])
3. **Build group-specific decoders**: Each group gets its own multi-layer network
4. **Train**: Molecular encoder (MPNN) is shared, but decoders are group-specific

**Expected Impact on ADMET**:

- **Faster convergence**: Model doesn't waste time figuring out task relationships
- **Sparse task isolation**: Problematic tasks (MGMB) are isolated from abundant tasks, reducing negative transfer
- **Domain knowledge integration**: Can manually specify groups based on ADMET knowledge (e.g., all permeability tasks together)

### MMoE-Grouped (Hierarchical Hybrid)

**The Core Idea**:
Combine both approaches: First group tasks by affinity, then within each group use MMoE to dynamically route tasks to group-specific experts.

**Why It Helps**:

- **Best of both worlds**: Structure (grouping) + flexibility (gating)
- **Hierarchical specialization**:
  - Level 1: Group-level isolation (sparse tasks vs. abundant tasks)
  - Level 2: Within-group fine-tuning (task-specific gating)
- **Scalability**: Adding new endpoints easier (assign to existing group)

**How It Works**:

1. **Group tasks** using affinity (same as Grouped Multi-Head)
2. **Create expert pools per group**: Each group gets 3-5 expert networks
3. **Per-task gating within groups**: Tasks in same group share experts but have independent gates
4. **Cross-group isolation**: Group 1 tasks cannot access Group 2 experts (prevents interference)

**Visual Example**:

```
Group 1: [LogD, KSOL]
  - 4 experts for lipophilicity patterns
  - LogD's gate: [0.5, 0.3, 0.1, 0.1] → mostly uses Expert 1 & 2
  - KSOL's gate: [0.4, 0.4, 0.1, 0.1] → also uses Expert 1 & 2 (related task)

Group 2: [MBPB, MGMB]
  - 4 different experts for binding patterns
  - MBPB's gate: [0.6, 0.2, 0.1, 0.1] → uses Group 2's Expert 1
  - MGMB's gate: [0.1, 0.1, 0.7, 0.1] → uses Group 2's Expert 3 (different!)
```

**Expected Impact on ADMET**:

- **Maximum performance**: Combines advantages of both methods
- **Robustness**: Falls back to group-level sharing if task-specific gating fails
- **Efficiency**: Fewer total experts needed (4 per group × 4 groups = 16 total, vs. 12 global experts in standard MMoE)

### Task Affinity (Gradient-Based Similarity)

**The Core Idea**:
Two tasks are "related" if updating the model to improve one task also helps the other task.

**How We Measure It**:

1. Train model for 1-2 epochs on all tasks
2. For each task pair (e.g., LogD vs. KSOL), compute:
   - Gradient for LogD task (direction to improve LogD)
   - Gradient for KSOL task (direction to improve KSOL)
   - Cosine similarity between gradients → affinity score
3. High score → tasks want model to update in similar ways → they're related
4. Low/negative score → tasks conflict → they're unrelated

**Why This Works**:

- **Data-driven**: Discovers relationships automatically from actual training dynamics
- **Accounts for data coverage**: Tasks with overlapping compound sets will naturally have higher affinity
- **Chemical intuition validation**: Can check if discovered groups match domain knowledge (e.g., clearance tasks should cluster)

### Uncertainty-Weighted Loss (Future Extension)

**The Core Idea**:
Some tasks are inherently noisier than others. Instead of treating all tasks equally, learn a task-specific uncertainty weight that balances the loss contributions.

**Why It Helps**:

- **Automatic task balancing**: Model learns how much to trust each task's gradients
- **Noisy task robustness**: MGMB (sparse, high variance) gets downweighted automatically
- **High-quality task emphasis**: LogD (abundant, clean data) can dominate training appropriately

**How It Works**:
Each task gets a learnable uncertainty parameter σ_i. Loss becomes:

```
L = Σ (L_i / (2 * σ_i²) + log(σ_i))
```

- If task is easy/clean → model learns low σ_i → loss gets large weight
- If task is hard/noisy → model learns high σ_i → loss gets small weight
- The `log(σ_i)` term prevents all σ_i from going to infinity

**Expected Impact**:

- Better overall MA-RAE by automatically balancing task difficulties
- Improved sparse task performance by preventing them from destabilizing training

---

## Success Criteria

### Technical Milestones

**Phase 1 Complete** (Weeks 1-2):

- [ ] `MMoERegressionFFN` class implemented and tested
- [ ] YAML configuration schema defined
- [ ] Unit tests pass (tensor shapes, gradient flow, gate normalization)
- [ ] Regression tests pass (synthetic data performance comparable to baseline)
- [ ] Acceptance test on ExpansionRx subset completes in <2 min
- [ ] MLflow experiment tracking integrated

**Phase 2 Complete** (Weeks 3-4):

- [ ] `GroupedMultiHeadFFN` class implemented
- [ ] Task affinity integration functional
- [ ] Manual and automatic grouping modes working
- [ ] Unit tests pass (grouping correctness, decoder routing)
- [ ] Regression tests show sparse task improvement vs. ungrouped baseline

**Phase 3 Complete** (Weeks 5-6):

- [ ] `MMoEGroupedFFN` class implemented
- [ ] Hierarchical expert-gate structure functional
- [ ] Cross-group isolation verified
- [ ] All unit/regression/acceptance tests pass
- [ ] Documentation updated with architecture diagrams

**Phase 4 Complete** (Weeks 7-8):

- [ ] HPO completed for all architectures (~500 trials each)
- [ ] Top-10 configs identified per architecture
- [ ] Best config selected and documented
- [ ] HPO reports generated with hyperparameter importance analysis

**Phase 5 Complete** (Weeks 9-10):

- [ ] Full 25-model ensembles trained for all architectures
- [ ] Leaderboard submissions completed
- [ ] Ablation study results analyzed
- [ ] Final model selected and documented

### Performance Targets

**Minimum Success** (Any architecture):

- [ ] MA-RAE ≤ 0.60 (match current baseline)
- [ ] No endpoint degrades by >10% vs. baseline
- [ ] Overall R² ≥ 0.53 (maintain current level)

**Good Success** (At least one architecture):

- [ ] MA-RAE ≤ 0.57 (5% improvement → estimated rank 10-12)
- [ ] MGMB R² ≥ 0.35 (from 0.24, +46% improvement)
- [ ] MBPB R² ≥ 0.45 (from 0.35, +29% improvement)
- [ ] LogD rank ≤ 30 (from 46)

**Excellent Success** (Best architecture):

- [ ] MA-RAE ≤ 0.54 (10% improvement → estimated rank 5-8)
- [ ] All endpoints R² ≥ 0.50
- [ ] LogD rank ≤ 20
- [ ] Overall rank ≤ 10 (from 17)

**Stretch Goals**:

- [ ] MA-RAE ≤ 0.50 (would likely be top 3-5)
- [ ] MGMB R² ≥ 0.50 (more than doubling current performance)
- [ ] Overall rank ≤ 5

### Deliverables Checklist

**Code Artifacts**:

- [ ] Three new FFN predictor classes in `src/admet/model/chemprop/ffn.py`
- [ ] Configuration schemas in `configs/mmoe/`, `configs/grouped/`, `configs/mmoe_grouped/`
- [ ] Unit tests in `tests/unit/test_ffn.py`
- [ ] Regression tests in `tests/regression/test_architecture_performance.py`
- [ ] Acceptance tests in `tests/acceptance/test_expansionrx_subset.py`
- [ ] HPO scripts in `scripts/hpo/`
- [ ] Ensemble training scripts in `scripts/train/`

**Documentation**:

- [ ] Updated MODEL_CARD.md with new architectures
- [ ] Updated README.md with usage examples
- [ ] API documentation for new FFN classes
- [ ] HPO reports (per architecture)
- [ ] Ablation study report
- [ ] Final model selection justification document

**Experiment Tracking**:

- [ ] MLflow experiments for HPO (all architectures)
- [ ] MLflow experiments for full ensembles (all architectures)
- [ ] Comparison dashboard in MLflow UI
- [ ] Exported results CSVs in `results/`

**Leaderboard**:

- [ ] Submission CSV files in `submissions/`
- [ ] Leaderboard screenshot or scrape with timestamps
- [ ] Rank progression tracking document

---

## References

### Papers

1. **Multi-Gate Mixture-of-Experts (MMoE)**:
   - Ma, J., Zhao, Z., Yi, X., Chen, J., Hong, L., & Chi, E. H. (2018). "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts." In KDD 2018.
   - <https://dl.acm.org/doi/10.1145/3219819.3220007>

2. **Adaptive Mixture of Local Experts** (Original MoE):
   - Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). "Adaptive Mixtures of Local Experts." Neural Computation, 3(1), 79-87.

3. **Task Affinity in Multi-Task Learning**:
   - Standley, T., Zamir, A., Chen, D., Guibas, L., Malik, J., & Savarese, S. (2020). "Which Tasks Should Be Learned Together in Multi-task Learning?" In ICML 2020.

4. **Uncertainty Weighting for Multi-Task Learning**:
   - Kendall, A., Gal, Y., & Cipolla, R. (2018). "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics." In CVPR 2018.

### Codebases

- **Chemprop v2**: <https://github.com/chemprop/chemprop>
- **Ray Tune**: <https://docs.ray.io/en/latest/tune/index.html>
- **MLflow**: <https://mlflow.org/docs/latest/index.html>
- **Scikit-learn Clustering**: <https://scikit-learn.org/stable/modules/clustering.html>

### Challenge Resources

- **OpenADMET + ExpansionRx Challenge**: <https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge>
- **Challenge Blog Post**: <https://www.openadmet.com/blog> (link TBD)
- **Model Report Template**: <https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge> (template section)

---

## Appendix A: Expected Task Groupings

Based on domain knowledge, here are reasonable task groupings for ADMET endpoints:

**Option 1: By Property Type** (4 groups)

- Group 1 (Lipophilicity): [LogD, KSOL]
- Group 2 (Clearance): [HLM CLint, MLM CLint]
- Group 3 (Permeability): [Caco-2 Papp, Caco-2 Efflux]
- Group 4 (Binding): [MPPB, MBPB, MGMB]

**Option 2: By Data Coverage** (3 groups)

- Group 1 (High Coverage): [LogD, KSOL, HLM CLint, MLM CLint]
- Group 2 (Medium Coverage): [Caco-2 Papp, Caco-2 Efflux, MPPB]
- Group 3 (Sparse): [MBPB, MGMB]

**Option 3: By Chemical Mechanism** (5 groups)

- Group 1 (Passive Diffusion): [LogD, KSOL, Caco-2 Papp]
- Group 2 (Active Transport): [Caco-2 Efflux]
- Group 3 (Metabolic Clearance): [HLM CLint, MLM CLint]
- Group 4 (Plasma Binding): [MPPB, MBPB]
- Group 5 (Microbiome Binding): [MGMB]

**Recommendation**: Start with Option 1 for manual grouping experiments, then compare to auto-discovered groupings from task affinity.

---

## Appendix B: Debugging Tips

### Common Issues and Solutions

**Issue**: Gate weights collapse to uniform distribution (all experts used equally)

- **Cause**: Learning rate too high, insufficient training
- **Solution**: Lower gate LR, increase training epochs, add entropy regularization

**Issue**: One expert dominates all tasks (gate weights → [0.9, 0.05, 0.05, 0])

- **Cause**: Expert capacity mismatch, poor initialization
- **Solution**: Increase number of experts, use load balancing loss, try different random seeds

**Issue**: Grouped architecture performs worse than ungrouped

- **Cause**: Incorrect task groupings, groups too small/large
- **Solution**: Validate affinity matrix makes sense, try different clustering methods, increase `n_groups`

**Issue**: MMoE-Grouped trains very slowly

- **Cause**: Too many experts per group
- **Solution**: Reduce `experts_per_group` to 3-4, use shared trunk to reduce parameters

**Issue**: NaN losses during training

- **Cause**: Gradient explosion in gates, numerical instability in softmax
- **Solution**: Add gradient clipping (max_norm=1.0), reduce learning rate, increase gate dropout

### Validation Checklist

Before submitting results, verify:

- [ ] All gate weight distributions sum to 1.0 (within numerical precision)
- [ ] Task-to-group assignments are correct (no overlaps, all tasks assigned)
- [ ] Expert pools have correct dimensions (match task groups)
- [ ] Output predictions are in correct order (task index preserved)
- [ ] MLflow logging is working (experiments, metrics, artifacts)
- [ ] Ensemble predictions average correctly across 25 models
- [ ] Submission CSV format matches leaderboard requirements

---

**END OF DOCUMENT**

*This implementation plan is intended for Claude Opus 4.5 to execute systematically. All specifications are production-ready and aligned with current repository structure.*
