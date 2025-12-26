# Model Card: OpenADMET ExpansionRx Blind Challenge

## Table of Contents

- [Methodology Summary](#methodology-summary)
- [Model Overview](#model-overview)
- [Model Architecture](#model-architecture)
- [Training Performance](#training-performance-observations)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Ensemble Strategy](#ensemble-strategy)
- [Evaluation](#evaluation)
- [Data Sources](#data-sources)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)
- [Reproducibility](#reproducibility)
- [References](#references)

---

## Methodology Summary

> **Challenge Requirements:** Model description, additional training steps, and performance observations.

### Model Description

Multi-task MPNN via [Chemprop v2](https://github.com/chemprop/chemprop) predicting 9 ADMET endpoints from SMILES. Ensemble of 25 models (5 Butina splits × 5 CV folds) with averaged predictions.

```mermaid
flowchart LR
    A[SMILES] --> B[Molecular Graph]
    B --> C[Message Passing<br/>5-7 layers]
    C --> D[Aggregation<br/>norm sum]
    D --> E[FFN<br/>1-4 layers]
    E --> F[9 Endpoints]
```

### Additional Training Steps

| Step | Description | Status |
|------|-------------|--------|
| HPO | Ray Tune ASHA ~2,000 trials | ✅ Used |
| Task Sampling | α-weighted oversampling of sparse endpoints | ✅ Used |
| FFN Variants | MLP, MoE, Branched architectures explored | ✅ Evaluated |
| Supplementary Data | KERMT, PharmaBench integration | 🔮 Future |
| Curriculum Learning | Quality-aware phased training | ✅ Implemented |

### Training Performance

- **Convergence:** 60–120 epochs with early stopping on validation MAE
- **Best validation MAE:** 0.4–0.6 (macro-averaged across endpoints)
- **R² range:** 0.40–0.85 (LogD best, Caco-2 and MGMB worst)

---

## Model Overview

- **Model Name:** OpenADMET Multi-Endpoint ADMET Predictor
- **Version:** 1.1

> **Note:** We implemented hyperparameter optimization (HPO) for single models and for ensembles, and added α-weighted oversampling to improve learning on sparser tasks in our multi-task models.

- **Task:** Multi-output regression for 9 ADMET endpoints
- **Architecture:** Chemprop MPNN ensemble
- **Training Framework:** PyTorch Lightning + Ray
- **Experiment Tracking:** MLflow

### Intended Use

Predict ADMET properties for small molecule drug candidates. Primary uses:

- Multi-endpoint ADMET prediction from SMILES
- Compound ranking by predicted profiles
- Identifying potential pharmacokinetic liabilities

**Out of scope:** Clinical decisions without validation, large biologics/polymers, regulatory submissions.

---

## Model Architecture

### Chemprop MPNN

```mermaid
flowchart TB
    subgraph Input
        S[SMILES String]
    end
    subgraph Encoder["Message Passing Network"]
        G[Molecular Graph] --> MP[Bond-based Messages<br/>depth: 3-7]
        MP --> AGG[Normalized Sum<br/>Aggregation]
    end
    subgraph Head["Prediction Head"]
        AGG --> FFN[Feed-Forward Network<br/>1-4 layers]
        FFN --> OUT[9 Endpoint Predictions]
    end
    S --> G
```

| Component | Configuration |
|-----------|---------------|
| Message Passing Depth | 3–7 (HPO-tuned) |
| Message Hidden Dim | 700–1100 |
| FFN Layers | 1–4 |
| FFN Hidden Dim | 200–1200 |
| Dropout | 0.0–0.2 |
| Aggregation | Normalized sum |

### FFN Architectures Explored

Three FFN types were evaluated during HPO:

| Type | Description | Performance |
|------|-------------|-------------|
| **MLP** | Standard multi-layer perceptron | Best overall; used in final ensemble |
| **MoE** | Mixture of Experts with gating network | Competitive (rank 3, 7 in top-10) |
| **Branched** | Shared trunk + task-specific branches | Competitive (rank 5, 6, 10 in top-10) |

The MoE and Branched architectures are implemented in [`src/admet/model/chemprop/ffn.py`](src/admet/model/chemprop/ffn.py).

### Multi-Task Learning

- **Output:** 9 endpoints predicted simultaneously
- **Loss:** MSE with NaN masking (missing endpoints ignored in gradient)
- **Task Weights:** Uniform (1.0 for all endpoints)
- **Task Sampling:** α-weighted to compensate for equal loss weighting and balance sparse endpoints

---

## Data Sources

### Primary Dataset: ExpansionRx Challenge Data

- **Source:** OpenADMET ExpansionRx Blind Challenge
- **Quality Designation:** **High**
- **Description:** Curated ADMET data from the challenge organizers with standardized assay protocols
- **Size:** Primary training set with all 9 endpoints
- **Local Test Set:** 12% of ExpansionRx data withheld via temporal split on `Molecule Name` column (sorted alphabetically, last 12% held out). Used only during HPO and architecture evaluation; final models trained on full 5×5 CV without separate time split.

### Supplementary Dataset: KERMT Public

- **Source:** KERMT benchmark (public subset)
- **Quality Designation:** **Medium**
- **Endpoints Available:**
  - LogD (pH 7.4)
  - Kinetic Solubility (log Saq → converted to μM)
  - HLM/MLM Clearance
  - Caco-2 Permeability (Papp)
  - Rat Plasma Protein Binding
  - P-gp Efflux (human) → mapped to Caco-2 Efflux (low confidence)

### Supplementary Dataset: KERMT Biogen

- **Source:** KERMT benchmark (Biogen proprietary subset, publicly released)
- **Quality Designation:** **Low to Medium** (endpoint-dependent)
- **Endpoints Available:**
  - Solubility at pH 6.8 (different assay conditions than challenge)
  - MDR1-MDCK Efflux Ratio (MDCK cells, not Caco-2)
  - HLM CLint
  - Rat Plasma Protein Binding
- **Caveats:**
  - Solubility measured at pH 6.8 vs pH 7.0-7.4 in challenge data
  - Efflux measured in MDCK cells vs Caco-2 cells

### Supplementary Dataset: PharmaBench

- **Source:** PharmaBench benchmark collection
- **Quality Designation:** **Low to Medium** (endpoint-dependent)
- **Endpoints Available:**
  - LogD
  - Water Solubility
  - HLM/MLM Clearance
  - Plasma Protein Binding
- **Filtering Applied:**
  - MW > 1000 Da: excluded
  - LogP < -2 or LogP > 8: excluded
  - Rotatable bonds > 20: excluded

---

## Data Harmonization and Transformations

### Unit Harmonization

All datasets were transformed to common units before merging:

| Endpoint | Final Units | Storage Format |
|----------|-------------|----------------|
| LogD | dimensionless | Linear (untransformed) |
| KSOL (Kinetic Solubility) | μM | log₁₀(μM) |
| HLM CLint | mL/min/kg | log₁₀(mL/min/kg) |
| MLM CLint | mL/min/kg | log₁₀(mL/min/kg) |
| Caco-2 Papp A→B | 10⁻⁶ cm/s | log₁₀(10⁻⁶ cm/s) |
| Caco-2 Efflux Ratio | dimensionless | log₁₀(ratio) |
| MPPB (Mouse PPB) | % unbound | log₁₀(% unbound) |
| MBPB (Mouse Brain) | % unbound | log₁₀(% unbound) |
| MGMB (Mouse Gut Microbiome) | % unbound | log₁₀(% unbound) |

### Source-Specific Transformations

#### KERMT Public

| Source Column | Transformation | Target Column |
|---------------|----------------|---------------|
| `LogD_pH_7.4` | Identity | LogD |
| `kinetic_logSaq` (log M) | 10^(x+6) | KSOL (μM) |
| `CL_microsome_human` (log mL/min/kg) | 10^x × 10⁻³ | HLM CLint |
| `CL_microsome_mouse` (log mL/min/kg) | 10^x × 10⁻³ | MLM CLint |
| `Papp_Caco2` (log 10⁻⁶ cm/s) | 10^x | Caco-2 Papp |
| `Rat_fraction_unbound_plasma` (log %) | 10^x | MPPB |
| `Pgp_human` (log ratio) | 10^x | Caco-2 Efflux* |

*Low confidence mapping: P-gp efflux ≠ Caco-2 efflux ratio

#### KERMT Biogen

| Source Column | Transformation | Target Column |
|---------------|----------------|---------------|
| `SOLY_6.8` (μg/mL) | (x / MW) × 10³ | KSOL (μM)* |
| `MDR1-MDCK_ER` | Identity | Caco-2 Efflux* |
| `HLM_CLint` (log) | 10^x | HLM CLint |
| `Rat_fraction_unbound_plasma` (log %) | 10^x | MPPB |

*Caveats: Different assay conditions (pH 6.8 vs 7.0-7.4; MDCK vs Caco-2)

#### PharmaBench

| Source Column | Transformation | Target Column |
|---------------|----------------|---------------|
| `logd_reg` | Identity | LogD |
| `water_sol_reg` (log nM) | 10^x / 1000 | KSOL (μM) |
| `hum_mic_cl_reg` (log mL/min/g) | 10^x | HLM CLint |
| `mou_mic_cl_reg` (log mL/min/g) | 10^x | MLM CLint |
| `ppb_reg` (%) | Identity | MPPB |

### Post-Transformation Processing

1. **Log₁₀ Transformation:** All endpoints except LogD are stored as log₁₀ values
2. **Outlier Handling:** For binding/clearance endpoints (MPPB, MBPB, MGMB, HLM CLint, MLM CLint), values below -3.0 (log scale) are set to NaN as likely measurement artifacts
3. **SMILES Canonicalization:** All SMILES canonicalized with RDKit (isomeric=True, canonical=True) using [`src/admet/data/smiles.py`](src/admet/data/smiles.py):
   - **Salt Removal:** RDKit SaltRemover strips counter-ions while preserving the parent molecule
   - **Validation:** Invalid SMILES that fail parsing return None and are excluded
   - **Empty Check:** Molecules with zero atoms after salt removal are excluded
   - **Parallel Processing:** Thread pool (max 32 workers) for batch canonicalization
4. **Duplicate Handling:** Duplicate canonical SMILES within datasets averaged (excluding NaN)

---

## Quality Categorization

### Quality Tiers

Data quality is assigned per-endpoint based on:

1. **Assay protocol alignment** with challenge data
2. **Measurement conditions** (pH, cell type, species)
3. **Distribution overlap** with ExpansionRx data

| Dataset | LogD | KSOL | HLM CLint | MLM CLint | Caco-2 Papp | Caco-2 Efflux | MPPB | MBPB | MGMB |
|---------|------|------|-----------|-----------|-------------|---------------|------|------|------|
| ExpansionRx | High | High | High | High | High | High | High | High | High |
| KERMT Public | Medium | Medium | Medium | Medium | Medium | Medium | Low | — | — |
| KERMT Biogen | — | Low | Low | — | — | Medium | Medium | — | — |
| PharmaBench | Medium | Low | Medium | Medium | — | — | Low | — | — |

### Quality Rationale

- **High:** ExpansionRx challenge data with standardized protocols
- **Medium:** Comparable assay conditions, similar value distributions
- **Low:** Different assay conditions, protocol mismatches, or distribution shifts
- **—:** Endpoint not available in dataset

---

## Training Performance Observations

### Convergence Behavior

- **Typical convergence:** 60–120 epochs with early stopping on validation MAE
- **Early stopping patience:** 15 epochs without improvement
- **Learning rate:** OneCycleLR with warmup critical for stability

### Per-Endpoint Performance

| Endpoint Category | Typical R² | Notes |
|-------------------|------------|-------|
| LogD | >0.85 | Well-represented, easiest |
| KSOL, HLM/MLM CLint | 0.65–0.75 | Good coverage |
| Caco-2 Papp/Efflux, MPPB | 0.55–0.70 | Moderate |
| MBPB, MGMB | 0.40–0.60 | Sparse, ExpansionRx only |

---

## Hyperparameter Optimization

### Framework

- **Tool:** Ray Tune with ASHA scheduler
- **Trials:** ~500
- **Objective:** Validation MAE (minimize)
- **Early stopping:** Grace period 10 epochs, reduction factor 2
- **Evaluation Split:** Local test set (12% temporal holdout) used during HPO to prevent overfitting to validation folds

### Search Space Summary

| Parameter | Range |
|-----------|-------|
| `learning_rate` | [1×10⁻⁴, 3×10⁻²] log-uniform |
| `depth` | [2, 8] |
| `message_hidden_dim` | {200, ..., 1200} |
| `ffn_type` | {mlp, moe, branched} |
| `ffn_num_layers` | [0, 5] |
| `dropout` | {0.0, ..., 0.3} |
| `task_sampling_alpha` | {0.0, ..., 1.0} |

### Top 10 Configurations

| Rank | FFN | Depth | Msg Dim | FFN Layers | FFN Dim | Dropout | α | LR | Val MAE |
|------|-----|-------|---------|------------|---------|---------|---|-----|---------|
| 1 | MLP | 3 | 700 | 4 | 200 | 0.15 | 0.02 | 2.27e-4 | 0.459 |
| 2 | MLP | 4 | 1100 | 1 | 1200 | 0.10 | 0.20 | 1.19e-3 | 0.460 |
| 3 | MoE | 6 | 1100 | 2 | 1100 | 0.00 | 0.00 | 2.94e-3 | 0.461 |
| 4 | MLP | 6 | 1000 | 3 | 700 | 0.20 | 0.20 | 1.77e-3 | 0.462 |
| 5 | Branched | 4 | 900 | 3 | 400 | 0.10 | 0.00 | 3.04e-4 | 0.463 |
| 6 | Branched | 5 | 800 | 2 | 800 | 0.05 | 0.05 | 2.74e-4 | 0.464 |
| 7 | MoE | 6 | 700 | 4 | 300 | 0.05 | 0.02 | 5.24e-4 | 0.464 |
| 8 | MLP | 3 | 1000 | 4 | 400 | 0.10 | 0.10 | 2.60e-4 | 0.465 |
| 9 | MLP | 4 | 800 | 2 | 1100 | 0.20 | 0.20 | 2.12e-3 | 0.467 |
| 10 | Branched | 7 | 700 | 4 | 800 | 0.15 | 0.02 | 2.41e-4 | 0.468 |

**Key insights:**

- MLP architectures dominate top ranks but MoE/Branched are competitive
- Task sampling α = 0.02–0.2 appears optimal
- Moderate dropout (0.05–0.15) preferred
- Deeper message passing (4–7) with moderate dimensions (700–1100)

### Task Sampling Alpha

**Purpose:** Compensates for uniform task weights in the loss function by oversampling sparse endpoints.

Since all endpoints have equal weight (1.0) in the MSE loss, task sampling alpha rebalances training to give more gradient updates to endpoints with fewer labeled examples.

Oversamples sparse endpoints using inverse-power weighting:

$$p_i \propto n_i^{-\alpha}$$

| α Value | Effect |
|---------|--------|
| 0.0 | Uniform task sampling (no rebalancing) |
| 0.02–0.1 | Mild rebalancing (recommended; used in top configs) |
| 0.5–1.0 | Aggressive; may hurt overall MAE |

---

## Ensemble Strategy

### Configuration

- **Splits:** 5 Butina clustering splits
- **Folds:** 5-fold CV per split
- **Total models:** 25
- **Clustering:** Taylor-Butina on molecular fingerprints
- **Training Strategy:** HPO and architecture evaluation used 88% of data (12% temporal holdout); final ensemble trained on full dataset with 5×5 CV

```mermaid
flowchart LR
    subgraph Ensemble["25 Model Ensemble"]
        direction TB
        S1[Split 1] --> F1[5 Folds]
        S2[Split 2] --> F2[5 Folds]
        S3[Split 3] --> F3[5 Folds]
        S4[Split 4] --> F4[5 Folds]
        S5[Split 5] --> F5[5 Folds]
    end
    F1 & F2 & F3 & F4 & F5 --> AGG[Mean Aggregation]
    AGG --> PRED[Final Predictions]
```

### Prediction Aggregation

```text
Y_mean = mean(Y_pred across 25 models)
Y_std = std(Y_pred across 25 models)
```

---

## Evaluation

### Primary Metric

**MAE (Mean Absolute Error)** — Validation MAE used for early stopping and HPO objective.

### Additional Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **RMSE** | Root Mean Squared Error | Penalizes large errors more than MAE |
| **R²** | Coefficient of determination | Proportion of variance explained |
| **Pearson r** | Linear correlation coefficient | Measures linear relationship strength |
| **Spearman ρ** | Rank-based correlation | Robust to outliers, captures monotonic relationships |
| **Kendall τ** | Rank concordance metric | More robust than Spearman for small samples |

**Correlation Metrics:**

- **Pearson r:** Assumes linear relationship; sensitive to outliers and distribution shape
- **Spearman ρ:** Non-parametric; evaluates monotonic (not necessarily linear) relationships by comparing ranks
- **Kendall τ:** Non-parametric; measures ordinal association; more computationally expensive but better for small datasets with ties

All three correlation metrics range from -1 (perfect negative correlation) to +1 (perfect positive correlation), with 0 indicating no correlation.

### Evaluation Spaces

- **Log Space:** Direct model outputs
- **Linear Space:** Back-transformed via 10^x (except LogD)

---

## Known Limitations

### Data Harmonization Caveats (Future Work)

> **Note:** Supplementary data integration is planned but not used in the final submission.

1. **Assay Condition Mismatches:**
   - KERMT Biogen solubility at pH 6.8 vs challenge pH ~7.0-7.4
   - KERMT Biogen efflux from MDCK cells vs Caco-2 cells
   - P-gp efflux (KERMT Public) ≠ Caco-2 efflux ratio

2. **Species Differences:** MPPB uses rat in some datasets, mouse in others

### Model Limitations

- **Domain:** Trained on drug-like small molecules; may not generalize to natural products, peptides, PROTACs
- **Sparse Endpoints:** MBPB/MGMB rely solely on ExpansionRx data
- **No Calibrated Uncertainty:** Ensemble stderr is spread, not confidence intervals
- **Uniform Task Weights:** Loss function uses equal weighting (1.0) for all endpoints; task sampling alpha compensates but per-endpoint loss weighting could improve performance

---

## Future Work

| Feature | Description | Status |
|---------|-------------|--------|
| Supplementary Data | KERMT, PharmaBench integration with harmonization | 🔮 Planned |
| Curriculum Learning | Quality-aware count-normalized sampling | ✅ Implemented, needs validation |
| Alternative Models | XGBoost, LightGBM, CheMeleon ensemble | 🔮 Planned |
| Uncertainty | Conformal prediction, Bayesian methods | 🔮 Planned |
| Task Weights | HPO-optimized per-endpoint loss weights (currently uniform) | 🔮 Planned |

---

## Reproducibility

### Random Seeds

- Base seed: 12345
- Per-fold seeds: base_seed + fold_index

### Environment

| Package | Version |
|---------|---------|
| Python | 3.11 |
| PyTorch | ≥2.1 |
| Chemprop | ≥2.0 |
| Ray | ≥2.4 |
| MLflow | ≥2.0 |

### Artifacts Logged

- Model checkpoints (best validation loss)
- Training metrics (per epoch)
- Predictions (train/val/test in log and linear space)
- Configuration YAML
- Run metadata (git hash, library versions)

---

## References

- [Chemprop](https://github.com/chemprop/chemprop): Message-passing neural networks for molecular property prediction
- [KERMT Benchmark](https://github.com/ncats/kermt): ADMET benchmark datasets
- [PharmaBench](https://github.com/mims-harvard/TDC): Therapeutics Data Commons benchmarks
- [Ray Tune](https://docs.ray.io/en/latest/tune/): Distributed hyperparameter tuning
- [BitBirch](https://github.com/jwildenhain/bitbirch): Scalable hierarchical molecular clustering

---

## Contact

**Repository:** [OpenADMET-ExpansionRx-Blind-Challenge](https://github.com/alec-glisman/OpenADMET-ExpansionRx-Blind-Challenge)
**Author:** Alec Glisman
