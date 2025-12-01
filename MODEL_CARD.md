# Model Card: OpenADMET ExpansionRx Blind Challenge

## Model Overview

**Model Name:** OpenADMET Multi-Endpoint ADMET Predictor
**Version:** 1.0
**Task:** Multi-output regression for 9 ADMET endpoints
**Architecture:** Chemprop Message-Passing Neural Network (MPNN) ensemble
**Training Framework:** PyTorch Lightning + Ray (parallel training)
**Experiment Tracking:** MLflow

## Intended Use

This model predicts ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) properties for small molecule drug candidates as part of the OpenADMET ExpansionRx Blind Challenge. The model is designed for early-stage drug discovery to prioritize compounds with favorable pharmacokinetic profiles.

### Primary Use Cases

- Predicting multiple ADMET endpoints simultaneously from molecular SMILES
- Ranking compounds by predicted ADMET profiles
- Identifying potential liabilities in drug candidates

### Out-of-Scope Uses

- Clinical decision-making without experimental validation
- Predictions for molecules outside the training domain (e.g., large biologics, polymers)
- Regulatory submissions without additional validation

---

## Data Sources

### Primary Dataset: ExpansionRx Challenge Data

- **Source:** OpenADMET ExpansionRx Blind Challenge
- **Quality Designation:** **High**
- **Description:** Curated ADMET data from the challenge organizers with standardized assay protocols
- **Size:** Primary training set with all 9 endpoints
- **Local Test Set:** 12% of ExpansionRx data withheld via temporal split on `Molecule Name` column (sorted alphabetically, last 12% held out before cross-validation)

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
3. **SMILES Canonicalization:** All SMILES canonicalized with RDKit (isomeric=True)
4. **Duplicate Handling:** Duplicate SMILES within datasets averaged (excluding NaN)

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

## Model Architecture

### Chemprop MPNN

- **Message Passing Depth:** 5 iterations
- **Message Hidden Dimension:** 600
- **FFN Layers:** 2
- **FFN Hidden Dimension:** 600
- **Dropout:** 0.1
- **Batch Normalization:** Enabled
- **FFN Type:** Standard regression (options: branched, mixture-of-experts)

### Multi-Task Learning

- **Output:** 9 endpoints predicted simultaneously
- **Loss Function:** MSE (Mean Squared Error)
- **Missing Value Handling:** NaN-masked loss (missing endpoints ignored in gradient computation)

---

## Training Configuration

### Data Splitting Strategy

1. **Local Test Set:** 12% temporal holdout from ExpansionRx (last entries by sorted `Molecule Name`)
2. **Cross-Validation:** 5 splits × 5 folds = 25 models per ensemble
3. **Clustering Methods Available:**
   - BitBirch (default): Hierarchical clustering on RDKit fingerprints
   - Scaffold-based: Bemis-Murcko scaffolds
   - K-means: Fingerprint-based clustering
   - Butina: Taylor-Butina clustering
   - Random: Stratified random splits
4. **Stratification:** Multi-label stratified K-fold on endpoint presence and quality labels

### Optimization

| Parameter | Value |
|-----------|-------|
| Initial LR | 1×10⁻⁴ |
| Max LR | 1×10⁻³ |
| Final LR | 1×10⁻⁴ |
| Warmup Epochs | 5 |
| Max Epochs | 150 |
| Early Stopping Patience | 15 |
| Batch Size | 32 |
| Optimizer | Adam (via Chemprop) |
| LR Schedule | OneCycleLR |

### Target Weights (Heuristic)

Per-endpoint loss weights to balance difficulty and importance:

| Endpoint | Weight | Rationale |
|----------|--------|-----------|
| LogD | 0.5 | Easier endpoint, well-represented |
| Log KSOL | 2.0 | Moderate difficulty |
| Log HLM CLint | 3.0 | Important, challenging |
| Log MLM CLint | 3.0 | Important, challenging |
| Log Caco-2 Papp | 3.0 | Important, challenging |
| Log Caco-2 Efflux | 2.0 | Moderate difficulty |
| Log MPPB | 2.0 | Moderate difficulty |
| Log MBPB | 3.0 | Challenging, sparse data |
| Log MGMB | 4.0 | Most challenging, sparsest data |

**Note:** Target weights are currently heuristic and should be optimized via hyperparameter search.

---

## Hyperparameter Selection

### Current Approach

Model hyperparameters are set based on literature defaults and preliminary experiments. No systematic hyperparameter optimization (HPO) has been performed yet.

### Planned HPO Strategy

- **Framework:** Ray Tune with ASHA scheduler
- **Objective:** Validation loss (MSE)
- **Search Space:**
  - Learning rates: log-uniform [10⁻⁵, 10⁻²]
  - Batch size: {16, 32, 64, 128}
  - Message passing depth: {2, 3, 4, 5, 6}
  - Hidden dimensions: {256, 300, 512, 600, 768}
  - Dropout: uniform [0.0, 0.4]
  - FFN type: {regression, branched, mixture_of_experts}
- **Resources:** 4 trials per GPU (0.25 GPU fraction)
- **Scheduler:** ASHA with grace period 10 epochs, max 150 epochs

---

## Curriculum Learning

### Implementation Status

Curriculum learning is **implemented but not currently active** in training.

### Curriculum Strategy (When Enabled)

The `CurriculumCallback` implements quality-aware training phases:

| Phase | Description | Weight Distribution (High/Med/Low) |
|-------|-------------|-----------------------------------|
| Warmup | Focus on high-quality data | 90% / 10% / 0% |
| Expand | Include medium-quality | 60% / 35% / 5% |
| Robust | Include low-quality | 40% / 40% / 20% |
| Polish | Re-focus on high-quality | 100% / 0% / 0% |

- **Phase Transition:** Triggered by lack of improvement on high-quality validation loss (patience-based)
- **Monitoring Metric:** `val_high_loss`

---

## Evaluation Metrics

### Primary Metric

**MAE (Mean Absolute Error)** — Used for model ranking and training

### Additional Metrics

- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of determination

### Evaluation Spaces

All metrics computed in both spaces:

1. **Log Space:** Direct model outputs (stored format)
2. **Linear Space:** Back-transformed via 10^x (except LogD)

### Aggregation

- **Per-Endpoint:** Individual metrics for each of 9 endpoints
- **Macro-Average:** Unweighted mean across all endpoints

---

## Ensemble Strategy

### Training

- **Configuration:** 5 splits × 5 folds = 25 base models
- **Parallelization:** Ray-based, configurable max parallel jobs

### Prediction Aggregation

For each molecule and endpoint:

```
Y_mean = mean(Y_pred across all models)
Y_std = std(Y_pred across all models)
Y_stderr = Y_std / sqrt(N_models)
```

### Output Format

Predictions saved in two formats:

1. **Log Space:** Direct ensemble outputs with mean, std, stderr
2. **Linear Space:** Back-transformed (10^x for non-LogD endpoints)

**Note:** Standard error is currently used only for visualization/diagnostics; not included in challenge submissions.

---

## Handling Missing Data

### Training

- **Masking Strategy:** Missing endpoint values (NaN) are masked in loss computation
- **No Imputation:** Missing values are not imputed; model learns from available data only

### Prediction

- Model predicts all 9 endpoints regardless of training data sparsity
- Endpoints with sparse training data may have higher uncertainty

### Endpoint Sparsity Notes

| Endpoint | Data Availability |
|----------|-------------------|
| LogD | Well-covered across datasets |
| KSOL | Well-covered |
| HLM/MLM CLint | Good coverage |
| Caco-2 Papp | Limited to ExpansionRx + KERMT Public |
| Caco-2 Efflux | Moderate coverage |
| MPPB | Good coverage |
| MBPB | **ExpansionRx only** |
| MGMB | **ExpansionRx only** |

---

## Model Selection (Planned Extensions)

### Current

- **Chemprop MPNN:** Primary model, SMILES-based

### Planned Additions

| Model | Input Type | Status |
|-------|------------|--------|
| XGBoost | Fingerprints | Planned |
| LightGBM | Fingerprints | Planned |
| CheMeleon | SMILES (pretrained) | Planned |

---

## Known Limitations and Caveats

### Data Harmonization Caveats

1. **Assay Condition Mismatches:**
   - KERMT Biogen solubility at pH 6.8 vs challenge pH ~7.0-7.4
   - KERMT Biogen efflux from MDCK cells vs Caco-2 cells
   - P-gp efflux (KERMT Public) ≠ Caco-2 efflux ratio

2. **Species Differences:**
   - MPPB: Some datasets use rat, others mouse plasma

3. **Low-Confidence Mappings:**
   - `Pgp_human` → `Caco-2 Efflux` (biological relevance uncertain)

### Model Limitations

1. **Domain Applicability:**
   - Trained on small molecule drug-like compounds
   - May not generalize to: natural products, peptides, PROTACs, or molecules outside Lipinski space

2. **Endpoint Sparsity:**
   - MBPB and MGMB predictions rely solely on ExpansionRx data
   - Higher uncertainty expected for sparse endpoints

3. **No Uncertainty Quantification:**
   - Ensemble stderr provides spread, not calibrated confidence intervals
   - No conformal prediction or Bayesian uncertainty

### Training Limitations

1. **Target Weights:** Heuristic, not optimized
2. **Curriculum Learning:** Implemented but inactive
3. **HPO:** Not yet performed; using literature defaults

---

## Reproducibility

### Random Seeds

- Base seed: 12345
- Per-fold seeds: base_seed + fold_index

### Environment

- Python: 3.11
- PyTorch: ≥2.1
- Chemprop: ≥2.0
- Ray: ≥2.4
- MLflow: ≥2.0

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
