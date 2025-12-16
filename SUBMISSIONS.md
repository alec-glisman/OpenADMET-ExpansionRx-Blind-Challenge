# OpenADMET + ExpansionRx Blind Challenge Submissions

* [Submission Link](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge)

## December 16, 2025

### Model

#### MLflow

* **Server URI**: `http://127.0.0.1:8084/#/experiments/4/runs/ce7470a8810148c39beba8a1a7089f80`
* **Backend Path**: `/media/aglisman/Data/models/mlflow-postgres`
* **Artifact Path**: `/media/aglisman/Data/models/mlflow-artifacts`
* **Experiment ID**: `4`
* **Run ID:**: `ce7470a8810148c39beba8a1a7089f80`

#### Hyperparameters

```yaml
# MPNN
depth: 3
message_hidden_dim: 700
# FFN
ffn_type: regression
num_layers: 4
hidden_dim: 200
# Training
dropout: 0.15
batch_size: 128
batch_norm: true
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

| Rank | User | MA-RAE | Min MA-RAE | $\Delta$ MA-RAE to min (\%)[^1] | R2 | Spearman R | Kendall's Tau | Submission Time | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 17/234 | aglisman | 0.60 +/- 0.03 | 0.54 | 10.0% | 0.53 +/- 0.04 | 0.77 +/- 0.02 | 0.59 +/- 0.02 | 2025-12-16 12:45:54+00:00 | Top 7.3% overall |

#### By Task

| Rank | Task | User | MAE | Min MAE | $\Delta$ MAE to min (\%)[^2] | R2 | Spearman R | Kendall's Tau | Submission Time | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 46 | LogD | aglisman | 0.35 +/- 0.01 | 0.27 | 22.9% | 0.73 +/- 0.03 | 0.88 +/- 0.01 | 0.74 +/- 0.01 | 2025-12-16 12:45:54+00:00 | Poor performance |
| 10 | KSOL | aglisman | 0.34 +/- 0.01 | 0.31 | 8.8% | 0.62 +/- 0.02 | 0.72 +/- 0.02 | 0.53 +/- 0.01 | 2025-12-16 12:45:54+00:00 | Excellent performance |
| 16 | MLM CLint | aglisman | 0.35 +/- 0.01 | 0.33 | 5.7% | 0.42 +/- 0.03 | 0.60 +/- 0.03 | 0.43 +/- 0.02 | 2025-12-16 12:45:54+00:00 | Good performance |
| 26 | HLM CLint | aglisman | 0.31 +/- 0.01 | 0.28 | 9.7% | 0.35 +/- 0.06 | 0.62 +/- 0.04 | 0.45 +/- 0.03 | 2025-12-16 12:45:54+00:00 | Okay performance |
| 86 | Caco-2 Permeability Efflux | aglisman | 0.35 +/- 0.01 | 0.25 | 28.6% | 0.19 +/- 0.04 | 0.80 +/- 0.01 | 0.59 +/- 0.01 | 2025-12-16 12:45:54+00:00 | Terrible performance |
| 69 | Caco-2 Permeability Papp A>B | aglisman | 0.26 +/- 0.01 | 0.19 | 26.9% | 0.32 +/- 0.04 | 0.76 +/- 0.02 | 0.56 +/- 0.02 | 2025-12-16 12:45:54+00:00 | Terrible performance |
| 36 | MPPB | aglisman | 0.18 +/- 0.01 | 0.14 | 22.2% | 0.67 +/- 0.03 | 0.83 +/- 0.02 | 0.64 +/- 0.02 | 2025-12-16 12:45:54+00:00 | Okay performance |
| 23 | MBPB | aglisman | 0.14 +/- 0.01 | 0.13 | 7.1% | 0.77 +/- 0.03 | 0.87 +/- 0.02 | 0.70 +/- 0.02 | 2025-12-16 12:45:54+00:00 | Okay performance |
| 2 | MGMB | aglisman | 0.15 +/- 0.01 | 0.15 | 0.0% | 0.71 +/- 0.06 | 0.83 +/- 0.03 | 0.68 +/- 0.03 | 2025-12-16 12:45:54+00:00 | Excellent performance |

[^1]: $\Delta$ MA-RAE to min (\%) = ((mean MA-RAE - minimum MA-RAE) / mean MA-RAE) $\times$ 100\%, rounded to 1 decimal place.
[^2]: $\Delta$ MAE to min (\%) = ((mean MAE - minimum MAE) / mean MAE) $\times$ 100\%, rounded to 1 decimal place.
