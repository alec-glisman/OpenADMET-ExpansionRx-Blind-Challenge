# OpenADMET + ExpansionRx Blind Challenge Submissions

* [Submission Link](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge)

---

## January 9, 2026 (Submission 0)

### Model

#### MLflow

* **Experiment ID**: `TODO`
* **Run ID**: `TODO`
* **Run Name**: `rank_001_task_weighted`

#### Architecture

**Model Type:** Chemprop MPNN with Task-Weighted Loss

**Key Change:** Task-specific loss weights derived from January 6 submission analysis to focus training on underperforming endpoints.

#### Hyperparameters

```yaml
# MPNN (unchanged from Jan-06 best model)
depth: 3
message_hidden_dim: 700
aggregation: norm
# FFN
ffn_type: regression
num_layers: 4
hidden_dim: 200
# Regularization
dropout: 0.15
batch_norm: true
# Training
batch_size: 128
criterion: MAE
# Learning Rate Schedule
init_lr: 0.00113
max_lr: 0.000227
final_lr: 0.000113
# Early Stopping
patience: 15
max_epochs: 150
# Sampling
task_oversampling_alpha: 0.02
# Reproducibility
seed: 42
```

#### Task Weights (NEW)

| Task | Weight | Formula | Justification |
|------|--------|---------|---------------|
| LogD | **1.5** | `1.0 × (1 + 0.286 × 1.75)` | Rank 63/289 (21.8%), 28.6% gap to leader. High improvement potential. |
| KSOL | **0.7** | `1.0 × 0.7` | Rank 17/289 (5.9%), 8.8% gap. Already excellent - prevent overfitting. |
| HLM CLint | **1.0** | `1.0` (baseline) | Rank 35/289 (12.1%), 10.0% gap. Decent performance - maintain. |
| MLM CLint | **1.1** | `1.0 × 1.1` | Rank 39/289 (13.5%), 11.1% gap. Slightly worse than HLM - small boost. |
| Caco-2 Papp A>B | **1.4** | `1.0 × (1 + 0.240 × 1.67)` | Rank 70/289 (24.2%), 24.0% gap. Significant improvement needed. |
| Caco-2 Efflux | **1.8** | `1.0 × (1 + 0.257 × 3.11)` | Rank 98/289 (33.9%), 25.7% gap. **WORST TASK** - highest priority. |
| MPPB | **1.3** | `1.0 × (1 + 0.263 × 1.14)` | Rank 47/289 (16.3%), 26.3% gap. Notable gap despite mid-rank. |
| MBPB | **1.4** | `1.0 × (1 + 0.312 × 1.28)` | Rank 65/289 (22.5%), 31.2% gap. Large gap to leader. |
| MGMB | **0.7** | `1.0 × 0.7` | Rank 18/289 (6.2%), 11.8% gap. Second best - reduce weight. |

**Weight Formula:**

```
weight = base_weight × rank_penalty × gap_multiplier

Where:
- base_weight = 1.0
- rank_penalty = 1.0 + (percentile_rank × scaling_factor) for tasks > 15th percentile
- gap_multiplier = 1.0 + (Δ_to_leader / 100) for tasks with gap > 20%
- scaling_factor adjusted per task to achieve target range [0.7, 1.8]
```

**Rationale:** MA-RAE is the mean across all 9 task RAEs. By up-weighting poorly-ranked tasks (Caco-2 Efflux, LogD, MBPB) and down-weighting already-excellent tasks (KSOL, MGMB), the model should allocate more learning capacity to closing the gap on weak endpoints.

### Statistics

#### Overall

| Rank | User | MA-RAE | Min MA-RAE | Δ MA-RAE to min (%)[^1] | R² | Spearman R | Kendall's τ | Submission Time | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| TODO | aglisman | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

#### By Task

| Rank | Task | MAE | Min MAE | Δ MAE to min (%)[^2] | R² | Spearman R | Kendall's τ | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| TODO | LogD | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | KSOL | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | MLM CLint | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | HLM CLint | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | Caco-2 Permeability Efflux | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | Caco-2 Permeability Papp A>B | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | MPPB | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | MBPB | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | MGMB | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

### Expected Outcomes

Based on the task weighting strategy:

| Task | Jan-06 Rank | Expected Change | Target Rank |
|------|-------------|-----------------|-------------|
| LogD | 63 | ↑ 15-25 ranks | 38-48 |
| KSOL | 17 | ↔ maintain | 15-20 |
| MLM CLint | 39 | ↑ 5-10 ranks | 29-34 |
| HLM CLint | 35 | ↔ maintain | 30-40 |
| Caco-2 Efflux | 98 | ↑ 25-40 ranks | 58-73 |
| Caco-2 Papp A>B | 70 | ↑ 15-25 ranks | 45-55 |
| MPPB | 47 | ↑ 5-15 ranks | 32-42 |
| MBPB | 65 | ↑ 10-20 ranks | 45-55 |
| MGMB | 18 | ↔ or slight ↓ | 18-25 |

**Expected MA-RAE:** 0.58-0.60 (improved from 0.61)
**Expected Rank:** Top 8-10% (improved from 10.7%)

### Conclusions

**Hypothesis:**

Task-weighted loss will improve MA-RAE by directing model capacity toward the worst-performing tasks (Caco-2 Efflux, LogD, MBPB) while accepting minor regression on already-strong tasks (KSOL, MGMB).

**What to watch:**

* Did Caco-2 Efflux improve significantly (target: rank < 70)?
* Did KSOL/MGMB regress unacceptably (threshold: rank > 30)?
* Did overall MA-RAE decrease?

**Next steps if successful:**

* Apply similar weighting to Chemeleon model
* Combine task-weighted Chemprop + Chemeleon in ensemble
* Explore more aggressive weights for worst tasks (2.0-2.5)

**Next steps if unsuccessful:**

* Weights may be too aggressive - try narrower range [0.8, 1.5]
* Consider task_oversampling_alpha increase (0.02 → 0.10) instead of loss weights
* Investigate if task correlations make some weight combinations counterproductive

---

## January 8, 2026

### Model

#### MLflow

* **Experiment ID**: `12`
* **Run ID**: `d7d51490fea9458e99e8e6677f425c37`
* **Run Name**: `rank_003`

#### Architecture

**Model Type:** Chemeleon (Pretrained Encoder)

#### Hyperparameters

```yaml
# Chemeleon Encoder
checkpoint_path: auto
freeze_encoder: false
unfreeze_encoder_lr_multiplier: 0.1
# FFN
ffn_type: mixture_of_experts
ffn_hidden_dim: 400
ffn_num_layers: 2
n_experts: 7
# Regularization
dropout: 0.248
batch_norm: true
weight_decay: 0.000259
# Training
batch_size: 128
criterion: MAE
use_mixed_precision: true
# Learning Rate Schedule
init_lr: 0.00162
max_lr: 0.00216
final_lr: 2.47e-06
# Early Stopping
patience: 15
max_epochs: 150
# Sampling
task_oversampling_alpha: 0.651
# Reproducibility
seed: 42
```

### Statistics

#### Overall

| Rank | User | MA-RAE | Min MA-RAE | Δ MA-RAE to min (%)[^1] | R² | Spearman R | Kendall's τ | Submission Time | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 55/302 | aglisman | 0.63 ± 0.03 | 0.51 | 19.0% | 0.48 ± 0.04 | 0.73 ± 0.02 | 0.56 ± 0.02 | 2026-01-08 21:58:41+00:00 | Top 18.2% |

#### By Task

| Rank | Task | MAE | Min MAE | Δ MAE to min (%)[^2] | R² | Spearman R | Kendall's τ | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 88 | LogD | 0.38 ± 0.01 | 0.25 | 34.2% | 0.68 ± 0.03 | 0.85 ± 0.01 | 0.70 ± 0.01 | Needs improvement |
| 69 | KSOL | 0.38 ± 0.01 | 0.31 | 18.4% | 0.54 ± 0.03 | 0.68 ± 0.02 | 0.50 ± 0.02 | Needs improvement |
| 146 | MLM CLint | 0.40 ± 0.01 | 0.33 | 17.5% | 0.28 ± 0.04 | 0.51 ± 0.03 | 0.36 ± 0.02 | Needs improvement |
| 74 | HLM CLint | 0.32 ± 0.01 | 0.26 | 18.8% | 0.32 ± 0.06 | 0.58 ± 0.04 | 0.42 ± 0.03 | Needs improvement |
| 116 | Caco-2 Permeability Efflux | 0.36 ± 0.01 | 0.25 | 30.6% | 0.16 ± 0.04 | 0.74 ± 0.02 | 0.54 ± 0.02 | Needs improvement |
| 126 | Caco-2 Permeability Papp A>B | 0.28 ± 0.01 | 0.19 | 32.1% | 0.21 ± 0.05 | 0.70 ± 0.02 | 0.51 ± 0.02 | Needs improvement |
| 24 | MPPB | 0.17 ± 0.01 | 0.14 | 17.6% | 0.69 ± 0.04 | 0.83 ± 0.02 | 0.64 ± 0.02 | Okay - Top 7.9% |
| 54 | MBPB | 0.15 ± 0.01 | 0.11 | 26.7% | 0.75 ± 0.03 | 0.86 ± 0.02 | 0.69 ± 0.02 | Top 17.9% |
| 5 | MGMB | 0.15 ± 0.01 | 0.15 | 0.0% | 0.70 ± 0.06 | 0.82 ± 0.03 | 0.66 ± 0.03 | 🏆 Excellent - Top 1.7% |

### Recommended Per-Task Training Weights

To improve MA-RAE, increase focus on underperforming tasks during training:

| Task | Current Rank | Δ to Min (%) | Suggested Weight | Priority |
|------|-------------|--------------|------------------|----------|
| LogD | 88 | 34.2% | **1.8** | 🔴 High |
| KSOL | 69 | 18.4% | **1.5** | 🔴 High |
| MLM CLint | 146 | 17.5% | **2.0** | 🔴 Critical |
| HLM CLint | 74 | 18.8% | **1.5** | 🔴 High |
| Caco-2 Efflux | 116 | 30.6% | **1.8** | 🔴 High |
| Caco-2 Papp A>B | 126 | 32.1% | **1.8** | 🔴 High |
| MPPB | 24 | 17.6% | 0.8 | 🟢 Low |
| MBPB | 54 | 26.7% | 1.0 | 🟡 Medium |
| MGMB | 5 | 0.0% | 0.5 | 🟢 Low |

**Normalized weights for `task_oversampling_alpha`:** `[1.8, 1.5, 2.0, 1.5, 1.8, 1.8, 0.8, 1.0, 0.5]`

### Model Improvement Recommendations

1. **Add Chemprop features to FFN input** - Concatenate Chemeleon embeddings with Chemprop molecular fingerprints for physicochemical tasks
2. **Reduce task_oversampling_alpha** - Current 0.65 may oversample binding tasks; try 0.3-0.4
3. **Increase FFN depth** - Current 2 layers may be insufficient for complex metabolism predictions
4. **Add auxiliary loss for LogD/KSOL** - Use additional supervision from related molecular properties
5. **Encoder fine-tuning schedule** - Gradually unfreeze more Chemeleon layers over training epochs
6. **Data augmentation** - SMILES enumeration and dropout-based augmentation for minority tasks

### Conclusions

**What worked:**

* Chemeleon pretrained encoder shows strong performance on plasma binding tasks (MPPB, MBPB, MGMB)
* Mixture of Experts FFN with 7 experts provides task specialization
* Higher task oversampling alpha (0.65) balances performance across tasks

**What didn't work:**

* Performance regressed on metabolism (MLM CLint) and permeability tasks compared to Chemprop
* LogD and KSOL performance dropped significantly
* Overall ranking worse than previous Chemprop submissions

**Next steps:**

* Consider ensemble of Chemprop + Chemeleon models
* Investigate task-specific model selection or weighting

---

## January 7, 2026

### Model

#### MLflow

* **Experiment ID**: `6`
* **Run ID**: `5ef1d4104f42489184188968ede410d6`
* **Run Name**: `rank_033`

#### Architecture

**Model Type:** Chemprop MPNN

#### Hyperparameters

```yaml
# MPNN
depth: 5
message_hidden_dim: 1100
aggregation: norm
# FFN
ffn_type: regression
num_layers: 2
hidden_dim: 900
# Regularization
dropout: 0.05
batch_norm: true
# Training
batch_size: 128
criterion: MAE
# Learning Rate Schedule
init_lr: 0.00149
max_lr: 0.000149
final_lr: 0.000149
# Early Stopping
patience: 15
max_epochs: 150
# Sampling
task_oversampling_alpha: 0.0
# Reproducibility
seed: 42
```

### Statistics

#### Overall

| Rank | User | MA-RAE | Min MA-RAE | Δ MA-RAE to min (%)[^1] | R² | Spearman R | Kendall's τ | Submission Time | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 38/295 | aglisman | 0.61 ± 0.03 | 0.50 | 18.0% | 0.53 ± 0.04 | 0.76 ± 0.02 | 0.59 ± 0.02 | 2026-01-07 21:12:55+00:00 | Top 12.9% |

#### By Task

| Rank | Task | MAE | Min MAE | Δ MAE to min (%)[^2] | R² | Spearman R | Kendall's τ | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 28 | LogD | 0.32 ± 0.01 | 0.25 | 21.9% | 0.75 ± 0.03 | 0.89 ± 0.01 | 0.75 ± 0.01 | Okay - Top 9.5% |
| 15 | KSOL | 0.34 ± 0.01 | 0.31 | 8.8% | 0.62 ± 0.02 | 0.74 ± 0.02 | 0.54 ± 0.01 | Good - Top 5.1% |
| 16 | MLM CLint | 0.35 ± 0.01 | 0.32 | 8.6% | 0.43 ± 0.03 | 0.61 ± 0.03 | 0.44 ± 0.02 | Good - Top 5.4% |
| 55 | HLM CLint | 0.31 ± 0.01 | 0.27 | 12.9% | 0.31 ± 0.05 | 0.56 ± 0.04 | 0.40 ± 0.03 | Top 18.6% |
| 53 | Caco-2 Permeability Efflux | 0.33 ± 0.01 | 0.26 | 21.2% | 0.30 ± 0.04 | 0.80 ± 0.01 | 0.60 ± 0.01 | Top 18.0% |
| 16 | Caco-2 Permeability Papp A>B | 0.22 ± 0.01 | 0.19 | 13.6% | 0.52 ± 0.03 | 0.78 ± 0.02 | 0.59 ± 0.01 | Good - Top 5.4% |
| 105 | MPPB | 0.22 ± 0.01 | 0.14 | 36.4% | 0.56 ± 0.04 | 0.81 ± 0.02 | 0.62 ± 0.02 | Needs improvement |
| 121 | MBPB | 0.18 ± 0.01 | 0.11 | 38.9% | 0.68 ± 0.03 | 0.87 ± 0.02 | 0.71 ± 0.02 | Needs improvement |
| 96 | MGMB | 0.20 ± 0.01 | 0.15 | 25.0% | 0.63 ± 0.04 | 0.82 ± 0.03 | 0.66 ± 0.03 | Needs improvement |

### Recommended Per-Task Training Weights

To improve MA-RAE, increase focus on underperforming tasks during training:

| Task | Current Rank | Δ to Min (%) | Suggested Weight | Priority |
|------|-------------|--------------|------------------|----------|
| LogD | 28 | 21.9% | 1.0 | 🟡 Medium |
| KSOL | 15 | 8.8% | 0.8 | 🟢 Low |
| MLM CLint | 16 | 8.6% | 0.8 | 🟢 Low |
| HLM CLint | 55 | 12.9% | 1.2 | 🟡 Medium |
| Caco-2 Efflux | 53 | 21.2% | 1.2 | 🟡 Medium |
| Caco-2 Papp A>B | 16 | 13.6% | 0.8 | 🟢 Low |
| MPPB | 105 | 36.4% | **2.0** | 🔴 Critical |
| MBPB | 121 | 38.9% | **2.0** | 🔴 Critical |
| MGMB | 96 | 25.0% | **1.8** | 🔴 High |

**Normalized weights for `task_oversampling_alpha`:** `[1.0, 0.8, 0.8, 1.2, 1.2, 0.8, 2.0, 2.0, 1.8]`

### Model Improvement Recommendations

1. **Increase task_oversampling_alpha** - Current 0.0 ignores task imbalance; try 0.3-0.5 to boost binding tasks
2. **Add plasma binding-specific features** - Incorporate descriptors like logP, PSA, rotatable bonds
3. **Increase dropout** - Current 0.05 is very low; try 0.15-0.20 to reduce overfitting on binding tasks
4. **Fix learning rate schedule** - init_lr ≈ max_lr ≈ final_lr is unusual; use proper warmup/decay
5. **Add regularization** - Weight decay (1e-4 to 1e-3) may help generalization on binding tasks
6. **Consider multi-task auxiliary heads** - Add intermediate predictions for related properties

### Conclusions

**What worked:**

* Very large MPNN (depth=5, hidden=1100) with large FFN (hidden=900)
* Strong performance on permeability tasks (Caco-2 Papp A>B)
* Good metabolism predictions (MLM CLint, KSOL)

**What didn't work:**

* Plasma binding tasks (MPPB, MBPB, MGMB) significantly underperformed
* Low dropout (0.05) may cause overfitting on some tasks
* Flat learning rate schedule (init ≈ max ≈ final) is unusual

**Next steps:**

* Test Chemeleon pretrained model for plasma binding tasks
* Investigate task-specific hyperparameter tuning

---

## January 6, 2026

### Model

#### MLflow

* **Experiment ID**: `6`
* **Run ID**: `ca2760b28f5945ee9b387915db9da875`
* **Run Name**: `rank_001`

#### Architecture

**Model Type:** Chemprop MPNN (Same as December 16 baseline)

#### Hyperparameters

```yaml
# MPNN
depth: 3
message_hidden_dim: 700
aggregation: norm
# FFN
ffn_type: regression
num_layers: 4
hidden_dim: 200
# Regularization
dropout: 0.15
batch_norm: true
# Training
batch_size: 128
criterion: MAE
# Learning Rate Schedule
init_lr: 0.00113
max_lr: 0.000227
final_lr: 0.000113
# Early Stopping
patience: 15
max_epochs: 150
# Sampling
task_oversampling_alpha: 0.02
# Reproducibility
seed: 42
```

### Statistics

#### Overall

| Rank | User | MA-RAE | Min MA-RAE | Δ MA-RAE to min (%)[^1] | R² | Spearman R | Kendall's τ | Submission Time | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 31/289 | aglisman | 0.61 ± 0.03 | 0.50 | 18.0% | 0.53 ± 0.04 | 0.76 ± 0.02 | 0.59 ± 0.02 | 2026-01-06 21:29:09+00:00 | Top 10.7% |

#### By Task

| Rank | Task | MAE | Min MAE | Δ MAE to min (%)[^2] | R² | Spearman R | Kendall's τ | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 63 | LogD | 0.35 ± 0.01 | 0.25 | 28.6% | 0.72 ± 0.03 | 0.88 ± 0.01 | 0.74 ± 0.01 | Needs improvement |
| 17 | KSOL | 0.34 ± 0.01 | 0.31 | 8.8% | 0.62 ± 0.02 | 0.72 ± 0.02 | 0.53 ± 0.01 | Good - Top 5.9% |
| 39 | MLM CLint | 0.36 ± 0.01 | 0.32 | 11.1% | 0.41 ± 0.03 | 0.59 ± 0.03 | 0.42 ± 0.02 | Okay - Top 13.5% |
| 35 | HLM CLint | 0.30 ± 0.01 | 0.27 | 10.0% | 0.37 ± 0.05 | 0.61 ± 0.04 | 0.45 ± 0.03 | Okay - Top 12.1% |
| 98 | Caco-2 Permeability Efflux | 0.35 ± 0.01 | 0.26 | 25.7% | 0.19 ± 0.04 | 0.79 ± 0.01 | 0.58 ± 0.01 | Needs improvement |
| 70 | Caco-2 Permeability Papp A>B | 0.25 ± 0.01 | 0.19 | 24.0% | 0.36 ± 0.04 | 0.74 ± 0.02 | 0.55 ± 0.02 | Needs improvement |
| 47 | MPPB | 0.19 ± 0.01 | 0.14 | 26.3% | 0.66 ± 0.03 | 0.84 ± 0.02 | 0.65 ± 0.02 | Top 16.3% |
| 65 | MBPB | 0.16 ± 0.01 | 0.11 | 31.2% | 0.74 ± 0.03 | 0.87 ± 0.02 | 0.70 ± 0.02 | Needs improvement |
| 18 | MGMB | 0.17 ± 0.01 | 0.15 | 11.8% | 0.69 ± 0.05 | 0.83 ± 0.03 | 0.67 ± 0.03 | Good - Top 6.2% |

### Recommended Per-Task Training Weights

To improve MA-RAE, increase focus on underperforming tasks during training:

| Task | Current Rank | Δ to Min (%) | Suggested Weight | Priority |
|------|-------------|--------------|------------------|----------|
| LogD | 63 | 28.6% | **1.6** | 🔴 High |
| KSOL | 17 | 8.8% | 0.8 | 🟢 Low |
| MLM CLint | 39 | 11.1% | 1.0 | 🟡 Medium |
| HLM CLint | 35 | 10.0% | 1.0 | 🟡 Medium |
| Caco-2 Efflux | 98 | 25.7% | **1.8** | 🔴 High |
| Caco-2 Papp A>B | 70 | 24.0% | **1.5** | 🔴 High |
| MPPB | 47 | 26.3% | 1.2 | 🟡 Medium |
| MBPB | 65 | 31.2% | **1.4** | 🟡 Medium |
| MGMB | 18 | 11.8% | 0.8 | 🟢 Low |

**Normalized weights for `task_oversampling_alpha`:** `[1.6, 0.8, 1.0, 1.0, 1.8, 1.5, 1.2, 1.4, 0.8]`

### Model Improvement Recommendations

1. **Increase task_oversampling_alpha** - Current 0.02 is very low; try 0.1-0.2 to boost Caco-2 tasks
2. **Increase MPNN depth** - Current depth=3 may miss long-range interactions; try depth=4-5
3. **Add message passing attention** - Use attention mechanism to focus on relevant substructures
4. **Incorporate 3D features** - Add conformer-based descriptors for permeability prediction
5. **Curriculum learning** - Start training on easier tasks (KSOL, MGMB) then add harder ones (Caco-2)
6. **Ensemble within architecture** - Train multiple seeds and average predictions

### Conclusions

**What worked:**

* Consistent performance with baseline hyperparameters
* Best overall ranking (Top 10.7%) among all submissions
* Strong KSOL and MGMB performance

**What didn't work:**

* LogD dropped from Dec 16 (rank 46 → 63) despite more competition
* Caco-2 tasks remain weak points
* No significant improvement over baseline configuration

**Next steps:**

* Experiment with alternative architectures (Chemeleon, MoE)
* Focus on improving Caco-2 and LogD predictions

---

## January 5, 2026

### Model

#### MLflow

* **Experiment ID**: `6`
* **Run ID**: `c781fb7efe4a4b70a6fb6263dd3dd8e9`
* **Run Name**: `rank_011`

#### Architecture

**Model Type:** Chemprop MPNN with Mixture of Experts FFN

#### Hyperparameters

```yaml
# MPNN
depth: 6
message_hidden_dim: 400
aggregation: norm
# FFN
ffn_type: mixture_of_experts
num_layers: 1
hidden_dim: 500
n_experts: 2
# Regularization
dropout: 0.2
batch_norm: true
# Training
batch_size: 64
criterion: MAE
# Learning Rate Schedule
init_lr: 0.000483
max_lr: 0.000967
final_lr: 1.93e-05
# Early Stopping
patience: 15
max_epochs: 150
# Sampling
task_oversampling_alpha: 0.0
# Reproducibility
seed: 42
```

### Statistics

#### Overall

| Rank | User | MA-RAE | Min MA-RAE | Δ MA-RAE to min (%)[^1] | R² | Spearman R | Kendall's τ | Submission Time | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 35/285 | aglisman | 0.61 ± 0.03 | 0.53 | 13.1% | 0.53 ± 0.04 | 0.77 ± 0.02 | 0.59 ± 0.02 | 2026-01-05 10:52:12+00:00 | Top 12.3% |

#### By Task

| Rank | Task | MAE | Min MAE | Δ MAE to min (%)[^2] | R² | Spearman R | Kendall's τ | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 15 | LogD | 0.31 ± 0.01 | 0.26 | 16.1% | 0.79 ± 0.02 | 0.91 ± 0.01 | 0.76 ± 0.01 | Good - Top 5.3% |
| 15 | KSOL | 0.34 ± 0.01 | 0.31 | 8.8% | 0.62 ± 0.02 | 0.73 ± 0.02 | 0.54 ± 0.01 | Good - Top 5.3% |
| 12 | MLM CLint | 0.34 ± 0.01 | 0.33 | 2.9% | 0.44 ± 0.04 | 0.62 ± 0.02 | 0.45 ± 0.02 | Good - Top 4.2% |
| 51 | HLM CLint | 0.32 ± 0.01 | 0.27 | 15.6% | 0.32 ± 0.06 | 0.58 ± 0.04 | 0.42 ± 0.03 | Top 17.9% |
| 94 | Caco-2 Permeability Efflux | 0.35 ± 0.01 | 0.25 | 28.6% | 0.20 ± 0.04 | 0.80 ± 0.01 | 0.60 ± 0.01 | Needs improvement |
| 71 | Caco-2 Permeability Papp A>B | 0.25 ± 0.01 | 0.19 | 24.0% | 0.37 ± 0.04 | 0.75 ± 0.02 | 0.55 ± 0.02 | Needs improvement |
| 100 | MPPB | 0.22 ± 0.01 | 0.14 | 36.4% | 0.58 ± 0.04 | 0.82 ± 0.02 | 0.62 ± 0.02 | Needs improvement |
| 51 | MBPB | 0.15 ± 0.01 | 0.12 | 20.0% | 0.75 ± 0.03 | 0.88 ± 0.02 | 0.72 ± 0.02 | Top 17.9% |
| 36 | MGMB | 0.17 ± 0.01 | 0.15 | 11.8% | 0.68 ± 0.05 | 0.82 ± 0.03 | 0.67 ± 0.03 | Okay - Top 12.6% |

### Recommended Per-Task Training Weights

To improve MA-RAE, increase focus on underperforming tasks during training:

| Task | Current Rank | Δ to Min (%) | Suggested Weight | Priority |
|------|-------------|--------------|------------------|----------|
| LogD | 15 | 16.1% | 0.8 | 🟢 Low |
| KSOL | 15 | 8.8% | 0.8 | 🟢 Low |
| MLM CLint | 12 | 2.9% | 0.6 | 🟢 Low |
| HLM CLint | 51 | 15.6% | 1.2 | 🟡 Medium |
| Caco-2 Efflux | 94 | 28.6% | **1.8** | 🔴 High |
| Caco-2 Papp A>B | 71 | 24.0% | **1.5** | 🔴 High |
| MPPB | 100 | 36.4% | **2.0** | 🔴 Critical |
| MBPB | 51 | 20.0% | 1.2 | 🟡 Medium |
| MGMB | 36 | 11.8% | 1.0 | 🟡 Medium |

**Normalized weights for `task_oversampling_alpha`:** `[0.8, 0.8, 0.6, 1.2, 1.8, 1.5, 2.0, 1.2, 1.0]`

### Model Improvement Recommendations

1. **Enable task_oversampling_alpha** - Current 0.0 ignores imbalance; try 0.2-0.4 to boost MPPB/Caco-2
2. **Increase number of experts** - Current 2 experts may be insufficient; try 4-8 for better task routing
3. **Add expert load balancing loss** - Prevent expert collapse where one expert handles all tasks
4. **Reduce MPNN depth** - Current depth=6 may cause oversmoothing; try depth=4-5
5. **Add residual connections** - Help gradient flow in deeper networks
6. **Task-specific gating** - Modify MoE to use task embeddings in gating function
7. **Plasma protein binding features** - Add logP, pKa, charge distribution for MPPB

### Conclusions

**What worked:**

* Mixture of Experts (MoE) FFN with 2 experts
* Excellent LogD performance (rank 15, Top 5.3%)
* Best MLM CLint performance to date (rank 12, Top 4.2%)
* Deeper MPNN (depth=6) captures more complex molecular features

**What didn't work:**

* MPPB dropped significantly (rank 100)
* Caco-2 tasks remain weak
* No task oversampling (alpha=0.0) may hurt minority tasks

**Next steps:**

* Increase number of experts in MoE
* Re-enable task oversampling with moderate alpha
* Focus on improving plasma binding predictions

---

## January 3, 2026 (Erroneous Submission)

### Model

#### MLflow

* **Experiment ID**: `5`
* **Run ID**: `041fca3d071e41e88a7091119dd6e66f`
* **Run Name**: `rank_001`

#### Architecture

**Model Type:** Chemprop MPNN

#### Hyperparameters

```yaml
# MPNN
depth: 3
message_hidden_dim: 700
aggregation: norm
# FFN
ffn_type: regression
num_layers: 4
hidden_dim: 200
# Regularization
dropout: 0.15
batch_norm: true
# Training
batch_size: 128
criterion: MAE
# Learning Rate Schedule
init_lr: 0.00113
max_lr: 0.000227
final_lr: 0.000113
# Early Stopping
patience: 15
max_epochs: 150
# Sampling
task_oversampling_alpha: 0.02
# Reproducibility
seed: 42
```

### Statistics

#### Overall

| Rank | User | MA-RAE | Min MA-RAE | Δ MA-RAE to min (%)[^1] | R² | Spearman R | Kendall's τ | Submission Time | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 60/280 | aglisman | 0.65 ± 0.03 | 0.53 | 18.5% | 0.48 ± 0.04 | 0.74 ± 0.02 | 0.56 ± 0.02 | 2026-01-04 10:43:03+00:00 | ⚠️ ERRONEOUS - Top 21.4% |

#### By Task

| Rank | Task | MAE | Min MAE | Δ MAE to min (%)[^2] | R² | Spearman R | Kendall's τ | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 114 | LogD | 0.41 ± 0.01 | 0.26 | 36.6% | 0.67 ± 0.03 | 0.84 ± 0.01 | 0.69 ± 0.01 | Needs improvement |
| 88 | KSOL | 0.40 ± 0.01 | 0.31 | 22.5% | 0.52 ± 0.03 | 0.68 ± 0.02 | 0.49 ± 0.02 | Needs improvement |
| 58 | MLM CLint | 0.37 ± 0.01 | 0.33 | 10.8% | 0.37 ± 0.03 | 0.56 ± 0.03 | 0.40 ± 0.02 | Top 20.7% |
| 33 | HLM CLint | 0.31 ± 0.01 | 0.28 | 9.7% | 0.34 ± 0.05 | 0.60 ± 0.04 | 0.43 ± 0.03 | Okay - Top 11.8% |
| 139 | Caco-2 Permeability Efflux | 0.38 ± 0.01 | 0.25 | 34.2% | 0.08 ± 0.04 | 0.79 ± 0.01 | 0.59 ± 0.01 | Needs improvement |
| 97 | Caco-2 Permeability Papp A>B | 0.26 ± 0.01 | 0.19 | 26.9% | 0.31 ± 0.04 | 0.74 ± 0.02 | 0.54 ± 0.02 | Needs improvement |
| 67 | MPPB | 0.20 ± 0.01 | 0.14 | 30.0% | 0.64 ± 0.04 | 0.83 ± 0.02 | 0.64 ± 0.02 | Needs improvement |
| 113 | MBPB | 0.18 ± 0.01 | 0.12 | 33.3% | 0.67 ± 0.03 | 0.83 ± 0.02 | 0.65 ± 0.02 | Needs improvement |
| 28 | MGMB | 0.17 ± 0.01 | 0.15 | 11.8% | 0.68 ± 0.06 | 0.82 ± 0.03 | 0.66 ± 0.03 | Okay - Top 10.0% |

### Conclusions

**⚠️ IMPORTANT:** This submission used ensemble predictions from local test splits instead of the final model trained on all data. This resulted in suboptimal performance and should be disregarded.

**Lessons learned:**

* Always verify submission pipeline uses production models trained on full data
* Test file configuration was incorrectly set to `local_test.csv`
* Data directory pointed to `split_train_val/` instead of `split_train_val_local_test/`

---

## December 16, 2025 (Baseline)

### Model

#### MLflow

* **Server URI**: `http://127.0.0.1:8084/#/experiments/4/runs/ce7470a8810148c39beba8a1a7089f80`
* **Backend Path**: `/media/aglisman/Data/models/mlflow-postgres`
* **Artifact Path**: `/media/aglisman/Data/models/mlflow-artifacts`
* **Experiment ID**: `4`
* **Run ID**: `ce7470a8810148c39beba8a1a7089f80`

#### Architecture

**Model Type:** Chemprop MPNN

#### Hyperparameters

```yaml
# MPNN
depth: 3
message_hidden_dim: 700
# FFN
ffn_type: regression
num_layers: 4
hidden_dim: 200
# Regularization
dropout: 0.15
batch_norm: true
# Training
batch_size: 128
criterion: MAE
# Learning Rate Schedule
final_lr: 0.000113
init_lr: 0.00113
max_lr: 0.000227
# Early Stopping
patience: 15
max_epochs: 150
# Sampling
task_sampling_alpha: 0.02
# Reproducibility
seed: 12345
```

### Statistics

#### Overall

| Rank | User | MA-RAE | Min MA-RAE | Δ MA-RAE to min (%)[^1] | R² | Spearman R | Kendall's τ | Submission Time | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 17/234 | aglisman | 0.60 ± 0.03 | 0.54 | 10.0% | 0.53 ± 0.04 | 0.77 ± 0.02 | 0.59 ± 0.02 | 2025-12-16 12:45:54+00:00 | 🏆 Top 7.3% overall |

#### By Task

| Rank | Task | MAE | Min MAE | Δ MAE to min (%)[^2] | R² | Spearman R | Kendall's τ | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 46 | LogD | 0.35 ± 0.01 | 0.27 | 22.9% | 0.73 ± 0.03 | 0.88 ± 0.01 | 0.74 ± 0.01 | Needs improvement |
| 10 | KSOL | 0.34 ± 0.01 | 0.31 | 8.8% | 0.62 ± 0.02 | 0.72 ± 0.02 | 0.53 ± 0.01 | 🏆 Excellent - Top 4.3% |
| 16 | MLM CLint | 0.35 ± 0.01 | 0.33 | 5.7% | 0.42 ± 0.03 | 0.60 ± 0.03 | 0.43 ± 0.02 | Good - Top 6.8% |
| 26 | HLM CLint | 0.31 ± 0.01 | 0.28 | 9.7% | 0.35 ± 0.06 | 0.62 ± 0.04 | 0.45 ± 0.03 | Okay - Top 11.1% |
| 86 | Caco-2 Permeability Efflux | 0.35 ± 0.01 | 0.25 | 28.6% | 0.19 ± 0.04 | 0.80 ± 0.01 | 0.59 ± 0.01 | Needs improvement |
| 69 | Caco-2 Permeability Papp A>B | 0.26 ± 0.01 | 0.19 | 26.9% | 0.32 ± 0.04 | 0.76 ± 0.02 | 0.56 ± 0.02 | Needs improvement |
| 36 | MPPB | 0.18 ± 0.01 | 0.14 | 22.2% | 0.67 ± 0.03 | 0.83 ± 0.02 | 0.64 ± 0.02 | Okay - Top 15.4% |
| 23 | MBPB | 0.14 ± 0.01 | 0.13 | 7.1% | 0.77 ± 0.03 | 0.87 ± 0.02 | 0.70 ± 0.02 | Okay - Top 9.8% |
| 2 | MGMB | 0.15 ± 0.01 | 0.15 | 0.0% | 0.71 ± 0.06 | 0.83 ± 0.03 | 0.68 ± 0.03 | 🏆 Excellent - Top 0.9% |

### Recommended Per-Task Training Weights

To improve MA-RAE, increase focus on underperforming tasks during training:

| Task | Current Rank | Δ to Min (%) | Suggested Weight | Priority |
|------|-------------|--------------|------------------|----------|
| LogD | 46 | 22.9% | **1.4** | 🟡 Medium |
| KSOL | 10 | 8.8% | 0.6 | 🟢 Low |
| MLM CLint | 16 | 5.7% | 0.7 | 🟢 Low |
| HLM CLint | 26 | 9.7% | 0.9 | 🟢 Low |
| Caco-2 Efflux | 86 | 28.6% | **1.8** | 🔴 High |
| Caco-2 Papp A>B | 69 | 26.9% | **1.6** | 🔴 High |
| MPPB | 36 | 22.2% | 1.2 | 🟡 Medium |
| MBPB | 23 | 7.1% | 0.8 | 🟢 Low |
| MGMB | 2 | 0.0% | 0.5 | 🟢 Low |

**Normalized weights for `task_oversampling_alpha`:** `[1.4, 0.6, 0.7, 0.9, 1.8, 1.6, 1.2, 0.8, 0.5]`

### Model Improvement Recommendations (⚠️ Weights Lost)

**Note:** These recommendations are for reference only as model weights were lost.

1. **Increase task_oversampling_alpha** - Current 0.02 is minimal; 0.15-0.25 would better balance Caco-2 performance
2. **Increase MPNN depth** - Depth=3 is conservative; depth=4-5 may capture more complex patterns
3. **Add attention mechanism** - Self-attention over message passing for better feature extraction
4. **Incorporate permeability-specific features** - TPSA, HBD, HBA are critical for Caco-2 prediction
5. **Use weighted loss** - Apply inverse-rank weights to loss function for task balancing
6. **Ensemble multiple seeds** - Average 3-5 models trained with different seeds

### Conclusions

**Baseline model performance:**

* Strong overall ranking (Top 7.3%) with a relatively simple Chemprop architecture
* Excellent performance on MGMB (rank 2) and KSOL (rank 10)
* Good metabolism predictions (MLM CLint, HLM CLint)
* Weak on Caco-2 permeability tasks (ranks 69-86)

**Key insights:**

* Moderate dropout (0.15) and task oversampling (alpha=0.02) provide good regularization
* 4-layer FFN with small hidden dim (200) is effective for multi-task learning
* Caco-2 tasks represent the biggest improvement opportunity

### Visual Highlights

* ![Overall rank distribution](assets/submissions/2025-12-16/figures/png/01_overall_rank_hist_ecdf.png) [Overall rank distribution](assets/submissions/2025-12-16/figures/01_overall_rank_hist_ecdf.png): Histogram + ECDF contextualizing leaderboard spread.
* ![Task-specific rankings](assets/submissions/2025-12-16/figures/png/02_task_rankings_bar.png) [Task-specific rankings](assets/submissions/2025-12-16/figures/02_task_rankings_bar.png): Horizontal bar chart with shaded performance zones.
* ![MAE comparison](assets/submissions/2025-12-16/figures/png/04_mae_comparison_bar.png) [MAE comparison](assets/submissions/2025-12-16/figures/04_mae_comparison_bar.png): User vs. top-performer MAE with uncertainty bars.
* ![Metrics heatmap](assets/submissions/2025-12-16/figures/png/06_metrics_heatmap_multi.png) [Metrics heatmap](assets/submissions/2025-12-16/figures/06_metrics_heatmap_multi.png): Per-task heatmaps for $R^2$, Spearman R, Kendall's $\tau$, and MAE.

### Actionable Insights for Next Round

* ![Priority Matrix](assets/submissions/2025-12-16/figures/png/20_priority_matrix.png) [Priority Matrix](assets/submissions/2025-12-16/figures/20_priority_matrix.png): Identifies quick wins (high impact, low effort) vs. major projects.
* ![Gap to Leader](assets/submissions/2025-12-16/figures/png/16_gap_to_leader_waterfall.png) [Gap to Leader](assets/submissions/2025-12-16/figures/16_gap_to_leader_waterfall.png): Absolute MAE improvement needed per task to match top performer.
* ![Percentile Rankings](assets/submissions/2025-12-16/figures/png/15_percentile_ranking.png) [Percentile Rankings](assets/submissions/2025-12-16/figures/15_percentile_ranking.png): Shows where you stand relative to all submissions per task.
* ![Task Difficulty vs Performance](assets/submissions/2025-12-16/figures/png/19_task_difficulty_vs_performance.png) [Task Difficulty vs Performance](assets/submissions/2025-12-16/figures/19_task_difficulty_vs_performance.png): Reveals if hard tasks are dragging down overall rank.
* ![Rank Improvement Potential](assets/submissions/2025-12-16/figures/png/18_rank_improvement_potential.png) [Rank Improvement Potential](assets/submissions/2025-12-16/figures/18_rank_improvement_potential.png): Visualizes how much rank could improve if MAE matched leader.
* ![Multi-Metric Radar](assets/submissions/2025-12-16/figures/png/14_radar_task_profile.png) [Multi-Metric Radar](assets/submissions/2025-12-16/figures/14_radar_task_profile.png): Spider chart showing balanced performance across metrics per task.

---

## Summary Table

| Date | Model | Rank | MA-RAE | Δ to Leader | Best Task | Worst Task | Notes |
|---|---|---:|---:|---:|---|---|---|
| 2025-12-16 | Chemprop | 17/234 (7.3%) | 0.60 | 10.0% | MGMB (#2) | Caco-2 Efflux (#86) | 🏆 Best overall |
| 2026-01-03 | Chemprop | 60/280 (21.4%) | 0.65 | 18.5% | MGMB (#28) | Caco-2 Efflux (#139) | ⚠️ Erroneous |
| 2026-01-05 | Chemprop+MoE | 35/285 (12.3%) | 0.61 | 13.1% | MLM CLint (#12) | MPPB (#100) | Best MLM CLint |
| 2026-01-06 | Chemprop | 31/289 (10.7%) | 0.61 | 18.0% | KSOL (#17) | Caco-2 Efflux (#98) | Stable baseline |
| 2026-01-07 | Chemprop (large) | 38/295 (12.9%) | 0.61 | 18.0% | KSOL (#15) | MBPB (#121) | Best Caco-2 Papp |
| 2026-01-08 | Chemeleon+MoE | 55/302 (18.2%) | 0.63 | 19.0% | MGMB (#5) | MLM CLint (#146) | Best binding |

---

## Detailed Analysis

### Task-Specific Model Selection Matrix

The following matrix shows the best-performing model for each task across all valid submissions (excluding the erroneous Jan 3 submission and Dec 16 whose weights were lost). This informs optimal task-specific model selection for ensemble strategies.

| Task | Best Model | Best Rank | MAE | 2nd Best Model | 2nd Rank | Δ Rank |
|------|------------|-----------|-----|----------------|----------|--------|
| LogD | Jan-05 MoE | #15 | 0.31 | Jan-07 Large | #28 | +13 |
| KSOL | Jan-07 Large | #15 | 0.34 | Jan-05 MoE | #15 | +0 |
| MLM CLint | Jan-05 MoE | #12 | 0.34 | Jan-07 Large | #16 | +4 |
| HLM CLint | Jan-06 Baseline | #35 | 0.30 | Jan-07 Large | #55 | +20 |
| Caco-2 Efflux | Jan-07 Large | #53 | 0.33 | Jan-05 MoE | #94 | +41 |
| Caco-2 Papp A>B | Jan-07 Large | #16 | 0.22 | Jan-05 MoE | #71 | +55 |
| MPPB | Jan-08 Chemeleon | #24 | 0.17 | Jan-06 Baseline | #47 | +23 |
| MBPB | Jan-05 MoE | #51 | 0.15 | Jan-08 Chemeleon | #54 | +3 |
| MGMB | Jan-08 Chemeleon | #5 | 0.15 | Jan-06 Baseline | #18 | +13 |

### Model Architecture Specialization

Different architectures excel at different task categories:

| Architecture | Strengths | Weaknesses | Available |
|--------------|-----------|------------|:---------:|
| **Chemprop Baseline** (depth=3, FFN=4×200) | KSOL, HLM CLint, MBPB, MGMB | Caco-2 tasks | ⚠️ Dec-16 weights lost |
| **Chemprop+MoE** (depth=6, 2 experts) | LogD, MLM CLint, MBPB | MPPB | ✓ Jan-05 |
| **Chemprop Large** (depth=5, FFN=2×900) | Caco-2 Papp A>B, Caco-2 Efflux, KSOL | MPPB, MBPB, MGMB | ✓ Jan-07 |
| **Chemeleon+MoE** (7 experts, α=0.65) | MPPB, MGMB | MLM CLint, LogD, KSOL | ✓ Jan-08 |
| **Chemprop Baseline** (Jan-06 retrain) | HLM CLint | Most tasks | ✓ Jan-06 |

### Competition Dynamics

| Metric | Dec 16 | Jan 8 | Change |
|--------|--------|-------|--------|
| Total Submissions | 234 | 302 | +29% |
| Min MA-RAE | 0.54 | 0.51 | -5.6% |
| Our Best Rank | 17 (7.3%) | 31 (10.7%) | -6.0% |
| Leader Gap | 10.0% | 18.0% | +8.0% |

**Key Insight:** The competition is intensifying rapidly. The gap to the leader has nearly doubled, suggesting top competitors are using more sophisticated approaches (likely ensembles, transfer learning, or external data).

### Recommended Task-Weighted Ensemble Strategy

Based on the analysis (excluding Dec-16 whose model weights were lost), an optimal ensemble would select predictions from:

```
LogD            → Jan-05 MoE         (rank 15, MAE 0.31)
KSOL            → Jan-07 Large       (rank 15, MAE 0.34)
MLM CLint       → Jan-05 MoE         (rank 12, MAE 0.34)
HLM CLint       → Jan-06 Baseline    (rank 35, MAE 0.30)
Caco-2 Efflux   → Jan-07 Large       (rank 53, MAE 0.33)
Caco-2 Papp A>B → Jan-07 Large       (rank 16, MAE 0.22)
MPPB            → Jan-08 Chemeleon   (rank 24, MAE 0.17)
MBPB            → Jan-05 MoE         (rank 51, MAE 0.15)
MGMB            → Jan-08 Chemeleon   (rank 5,  MAE 0.15)
```

**Note:** Dec-16 baseline achieved best ranks for KSOL (#10), MBPB (#23), and MGMB (#2), but model weights are no longer available.

**Estimated Improvement:** If this task-weighted selection achieves optimal per-task performance, the theoretical best rank would be significantly improved. The ensemble config is available at `configs/3-hpo-ensemble-production/2_task_weighted_ensemble/task_weighted_ensemble.yaml`.

### Hyperparameter Insights

| Parameter | Best for Physicochemical | Best for Metabolism | Best for Permeability | Best for Binding |
|-----------|--------------------------|---------------------|----------------------|------------------|
| MPNN Depth | 6 (MoE) | 3-6 | 5 (Large) | 3 (Baseline) |
| FFN Hidden | 500 (MoE) | 200 (Baseline) | 900 (Large) | 400 (Chemeleon) |
| Dropout | 0.2 | 0.15 | 0.05 | 0.25 |
| Task α | 0.0 | 0.02 | 0.0 | 0.65 |

---

[^1]: Δ MA-RAE to min (%) = ((mean MA-RAE - minimum MA-RAE) / mean MA-RAE) × 100%, rounded to 1 decimal place.
[^2]: Δ MAE to min (%) = ((mean MAE - minimum MAE) / mean MAE) × 100%, rounded to 1 decimal place.
