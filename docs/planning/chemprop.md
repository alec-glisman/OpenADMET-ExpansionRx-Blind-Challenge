````markdown
# Quality-Aware Curriculum ML for Molecular Property Prediction

This repository is a **unified framework** for training and evaluating:

- **Classical models**: XGBoost, LightGBM  
- **Deep learning models**: Chemprop-like GNNs via PyTorch Lightning  

…with:

- **Data quality–aware training** (sample weights + curriculum)  
- **Chemically meaningful cross-validation** (BitBirch cluster–aware folds)  
- **Multi-task learning with task weighting**  
- **Ray Tune** for scalable hyperparameter optimization  
- **MLflow** (with nested runs) for experiment tracking  
- **OmegaConf** for configuration management  
- A growing **unit test suite** (pytest) to keep everything sane.

The goal: a *single repo* you can use to systematically compare classical and deep models on molecular datasets where **data quality varies** and **chemical diversity and leakage control matter**.

---

```mermaid

flowchart TD

  subgraph D[Data Preparation]
    A[Raw CSV: smiles + y_i + assay_quality + dataset_quality]
    B[augment_quality(df)\n- map assay_quality\n- map dataset_quality\n- combine to quality_score\n- assign quality_bucket]
    A --> B
  end

  subgraph CV[Cross-Validation Setup]
    C[make_folds(df, K)] 
  end

  D --> C

  subgraph R[Ray Tune]
    R1[Define search space\nlr, wd, depth, hidden_size, dropout,\ncurr_patience, batch_size]
    R2[Ray Tuner\n(num_samples = N)]
  end

  C --> R1
  R1 --> R2

  subgraph T[Per Trial (config)]
    direction TB
    T0[Start MLflow parent run\n(trial_id)]
    subgraph F[For each fold k in 1..K]
      direction TB
      F1[Start nested MLflow run\n(trial_id, fold_k)]
      F2[Create CurriculumState\n(patience = curr_patience)]
      F3[Create MolDataModule\n(train_idx_k, val_idx_k,\nquality-aware sampler)]
      F4[Init ChempropLightning\n(Chemprop model + MPNN)]
      F5[Train with PyTorch Lightning\n+ CurriculumCallback]
      F6[Compute val_high, val_med, val_low,\nval_combined for fold k]
      F7[Log fold metrics to nested MLflow run]
    end
    T1[Aggregate fold metrics\n(mean val_combined across folds)]
    T2[Log CV metrics to parent run]
    T3[Report val_combined to Ray Tune]
    T0 --> F --> T1 --> T2 --> T3
  end

  R2 --> T

  subgraph SEL[Selection & Final Model]
    S1[Ray Tune selects best config]
    S2[Train final model on full data\nwith best config + curriculum]
    S3[Log final training as MLflow run]
  end

  T --> S1 --> S2 --> S3

```


## 1. High-Level Architecture

**Single source dataset** (on disk):

- `data/molecules.csv`  
  - `smiles`  
  - `quality` (e.g. `high/medium/low`)  
  - multi-task targets: `y1`, `y2`, …  
  - optional extra feature / meta columns  

**Derived views and logic (in code):**

1. **Quality augmentation**
   - `quality_bucket` (categorical)
   - `sample_weight` (numeric, from config: `high → 1.0`, `medium → 0.5`, `low → 0.2`, etc.)

2. **BitBirch clustering**
   - RDKit fingerprints → BitBirch → `cluster_id` per molecule

3. **Stratified, cluster-aware K-fold splitting**
   - Combine **binned task label(s)** + `quality_bucket` into a joint `strata` label  
   - Use `StratifiedGroupKFold(y=strata, groups=cluster_id)` to get folds

4. **Model-specific datasets**
   - Classical: `X, y, w` matrices per fold  
   - Chemprop: `MolDataset` + DataLoaders per fold with `(x, y, quality_bucket)`

5. **Model training & evaluation**
   - Classical: XGBoost / LightGBM (multi-output) with `sample_weight`  
   - Chemprop: Lightning module with:
     - multi-task loss
     - task weighting
     - curriculum learning based on quality buckets

6. **HPO & tracking**
   - Ray Tune trials → parent MLflow run  
   - Each fold → nested MLflow run  
   - Metrics logged per fold & aggregated (CV mean)

---

## 2. Repository Layout

Planned structure (some files already implemented, others to be added incrementally):

```text
quality_curriculum_ml/
├── pyproject.toml           # packaging + dependencies
├── requirements.txt         # Python dependencies
├── README.md                # this file
├── data/
│   └── molecules.csv        # main dataset (user-provided)
├── config/
│   ├── base.yaml            # data, CV, quality, MLflow, Ray
│   ├── xgboost.yaml         # XGBoost model defaults
│   ├── lightgbm.yaml        # LightGBM model defaults
│   └── chemprop.yaml        # Chemprop model defaults
├── src/
│   └── quality_curriculum_ml/
│       ├── __init__.py
│       ├── classical/
│       │   ├── __init__.py
│       │   ├── data.py          # load/augment/encode data for classical models
│       │   ├── models.py        # XGB/LGBM wrappers
│       │   ├── metrics.py       # per-task & aggregated metrics
│       │   └── tune_trainable.py# Ray trainable for classical models
│       └── chemprop/
│           ├── __init__.py
│           ├── data.py          # SMILES featurization + datasets/loaders
│           ├── curriculum.py    # curriculum state + callbacks
│           ├── model.py         # LightningModule wrapping Chemprop GNN
│           └── tune_trainable.py# Ray trainable for Chemprop + Lightning
├── scripts/
│   ├── run_tune_xgboost.py      # Ray entrypoint for XGBoost
│   ├── run_tune_lightgbm.py     # Ray entrypoint for LightGBM
│   ├── run_tune_chemprop.py     # Ray entrypoint for Chemprop
│   └── run_mlflow_ui.sh         # convenience script for MLflow UI
└── tests/
    ├── __init__.py
    ├── test_data.py             # unit tests for data loading/splitting
    ├── test_classical_models.py # unit tests for XGB/LGBM pipeline
    ├── test_chemprop_model.py   # unit tests for Chemprop Lightning module
    └── test_integration.py      # light integration tests for end-to-end flow
````

---

## 3. Features & Implementation Plan

We’ll implement and test features in **phases**, walking from basics → sophistication. Each phase should introduce **code plus unit tests**.

### Phase 0 – Environment & Dependencies

**Goal**: Make the repo installable and runnable.

* [ ] Add `pyproject.toml` / `requirements.txt` with:

  * `pandas`, `numpy`, `scikit-learn`
  * `xgboost`, `lightgbm`
  * `rdkit` (or ensure it’s available via conda)
  * `bitbirch` (or your chosen BitBirch implementation)
  * `torch`, `pytorch-lightning`
  * `chemprop`
  * `ray[tune]`
  * `mlflow`
  * `omegaconf`
  * `pytest`
* [ ] Add `src/quality_curriculum_ml/__init__.py`
* [ ] Add `tests/` boilerplate and ensure `pytest` runs.

**Unit tests**

* `tests/test_environment.py` (optional)

  * Confirm imports work (e.g. `import xgboost`, `import lightgbm`, `import mlflow`).
  * Basic smoke test.

---

### Phase 1 – Core Data Handling (Classical Path)

**Goal**: Load `molecules.csv`, augment with quality, and produce `X, y, w` for classical models.

**Files**

* `config/base.yaml`:

  * `data.path`
  * `data.target_cols`
  * `data.quality_col`
  * `training.task_weights`
  * `quality_weights`
  * `cv.*`
  * `mlflow.*`
  * `ray.*`

* `src/quality_curriculum_ml/classical/data.py`:

  * `load_data(path)`
  * `augment_quality(df, quality_col, quality_weights)`
  * **temporary** `make_folds(df, ...)` using simple KFold (to be replaced with BitBirch-stratified later)
  * `get_xyw(df, target_cols)` → `X, y, sample_weight`

**Unit tests**

* `tests/test_data.py`:

  * Create small synthetic DataFrame:

    * columns: `f1, f2, quality, y1, y2`
  * Test `augment_quality`:

    * correct `quality_bucket` assignment
    * correct `sample_weight` mapping
  * Test `get_xyw`:

    * correct feature/target separation
    * correct shape and dtype
  * Test `make_folds`:

    * correct number of folds and non-overlapping indices.

---

### Phase 2 – Classical Models + Metrics + MLflow + Ray Tune

**Goal**: Full classical path with CV, HPO, and tracking.

**Files**

* `src/quality_curriculum_ml/classical/models.py`:

  * `build_model(model_type, params)` → `MultiOutputRegressor(XGBRegressor/LGBMRegressor)`
  * `fit_model(model, X, y, sample_weight)`
  * `predict_model(model, X)`

* `src/quality_curriculum_ml/classical/metrics.py`:

  * `per_task_rmse(y_true, y_pred)`
  * `aggregate_metric(per_task, task_weights)`
  * `build_metric_dict(y_true, y_pred, task_weights, prefix="val")`

* `src/quality_curriculum_ml/classical/tune_trainable.py`:

  * `tune_trainable(config, df, target_cols, task_weights, model_type, n_splits, experiment_name)`
  * Handles:

    * augment df with quality & weights
    * K-fold split
    * per-fold training and evaluation
    * MLflow parent + nested runs
    * reports aggregated `val_rmse_weighted` to Ray Tune

* `scripts/run_tune_xgboost.py`

* `scripts/run_tune_lightgbm.py`

**Unit tests**

* `tests/test_classical_models.py`:

  * Simple synthetic regression (e.g. 2D features, 2 tasks) where XGB/LGBM can learn easily.
  * Test `build_model` and `fit_model`:

    * training runs without errors
    * predictions shape matches targets
  * Test `metrics`:

    * `per_task_rmse` and `aggregate_metric` with small arrays.
* `tests/test_integration.py` (partial):

  * Smoke test: run `tune_trainable` for 1 trial, 2 folds, small synthetic df, `num_samples=1` with dummy MLflow tracking URI (e.g. local directory).

---

### Phase 3 – Chemprop + Lightning Curriculum

**Goal**: Add deep model path with curriculum, still using simple K-fold (no BitBirch yet).

**Files**

* `config/chemprop.yaml`:

  * chemprop model parameters
  * optimization hyperparams (lr, weight_decay, batch_size, max_epochs, curr_patience)

* `src/quality_curriculum_ml/chemprop/data.py`:

  * `augment_quality(df, quality_col, quality_weights)` (can reuse or share logic with classical)
  * `make_folds(df, ...)` (temporary)
  * `MolDataset(df, target_cols)`
  * `collate_mol_batch(batch)`
  * `MolDataModule` (or simple class with `setup()`, `train_dataloader()`, `val_dataloader()`)

* `src/quality_curriculum_ml/chemprop/curriculum.py`:

  * `CurriculumState`:

    * tracking `phase`, weights for `high/medium/low`
    * `update_from_val_high`, `maybe_advance_phase`, `set_polish`
  * `CurriculumCallback` (Lightning callback reading `val_high_loss` and updating state)

* `src/quality_curriculum_ml/chemprop/model.py`:

  * `ChempropLightning` (LightningModule):

    * wraps `MoleculeModel`
    * uses `task_weights` to compute multi-task loss
    * logs:

      * `val_high_loss`, `val_medium_loss`, `val_low_loss`, `val_combined`

* `src/quality_curriculum_ml/chemprop/tune_trainable.py`:

  * `tune_trainable_chemprop(config, df, target_cols, task_weights, n_splits, experiment_name)`
  * very similar structure to classical `tune_trainable`, but builds and trains Lightning model

* `scripts/run_tune_chemprop.py`

**Unit tests**

* `tests/test_chemprop_model.py`:

  * Use tiny synthetic MolDataset:

    * maybe use trivial “features” (e.g. random tensors) instead of real SMILES to start
  * Ensure:

    * forward pass works
    * training_step and validation_step don’t crash
    * `validation_epoch_end` returns proper keys
* `tests/test_integration.py`:

  * Add a small test that runs `tune_trainable_chemprop` on a tiny df with 1–2 epochs and 2 folds, `num_samples=1`.

---

### Phase 4 – BitBirch + StratifiedGroupKFold

**Goal**: Replace simple KFold with **BitBirch + task + quality**–aware CV.

**Files**

* Update **or add**:

  * `src/quality_curriculum_ml/classical/data.py` and `src/quality_curriculum_ml/chemprop/data.py`:

    * `compute_fingerprints(df)` → RDKit Morgan fingerprints
    * `run_bitbirch(fps)` → `cluster_id` (BitBirch)
    * `build_strata(df, primary_task, quality_col)`:

      * bin primary task into quantiles
      * combine bins + quality to `strata` labels
    * `make_bitbirch_stratified_folds(df, primary_task, quality_col, n_splits, random_state)`:

      * `StratifiedGroupKFold(y=strata, groups=cluster_id)`

* Update Ray trainables to call `make_bitbirch_stratified_folds` instead of simple `make_folds`.

**Unit tests**

* `tests/test_data.py` additions:

  * Use small synthetic df with:

    * dummy fingerprints / clusters (you can skip real RDKit in unit tests and feed fake cluster_ids)
    * known quality levels, task bins
  * Test:

    * `build_strata` produces expected label combinations
    * `make_bitbirch_stratified_folds`:

      * each cluster_id appears in only one fold
      * per-fold distribution of `strata` is reasonably balanced (rough check).

---

### Phase 5 – Polishing, Docs, and Additional Tests

**Goal**: Make the repo pleasant to use and robust.

**Tasks**

* [ ] Add **full examples** to README:

  * Example commands: `run_tune_xgboost.py`, `run_tune_chemprop.py`
  * Example of configuring model and quality weights via YAML
* [ ] Add more **integration tests**:

  * e.g., one end-to-end run for XGBoost and Chemprop on toy data.
* [ ] Add **continuous integration** (GitHub Actions or similar) to run pytest and maybe a minimal Ray Tune trial.
* [ ] Optional: add **Makefile** / `scripts/run_all.sh` to orchestrate:

  ```bash
  make tune-xgb
  make tune-lightgbm
  make tune-chemprop
  make mlflow-ui
  ```

---

## 4. How to Use (Once Implemented)

### 4.1. Prepare your dataset

Create `data/molecules.csv` with at least:

* `smiles`
* `quality` (values: `high`, `medium`, `low`)
* target columns, e.g. `y1`, `y2`

Update:

* `config/base.yaml`

  * `data.path: "data/molecules.csv"`
  * `data.target_cols: ["y1", "y2"]`
  * `data.quality_col: "quality"`
  * `training.task_weights: [0.5, 0.5]` (one per target)
  * `quality_weights: {high: 1.0, medium: 0.5, low: 0.2}`

### 4.2. Install & run

```bash
# create env and install dependencies
pip install -r requirements.txt

# Classical models
PYTHONPATH=src python scripts/run_tune_xgboost.py
PYTHONPATH=src python scripts/run_tune_lightgbm.py

# Chemprop deep model
PYTHONPATH=src python scripts/run_tune_chemprop.py
```

Start MLflow UI:

```bash
./scripts/run_mlflow_ui.sh
# open http://localhost:5000 in your browser
```

Run tests:

```bash
PYTHONPATH=src pytest -q
```

---

## 5. Extension Ideas

Once the basic plan is implemented and tested, you can:

* Add **ensembles** that combine classical and Chemprop predictions.
* Add **more sophisticated curricula** for Chemprop:

  * actual bucket-aware sampling schedules tied to `CurriculumState`.
* Add **more models**:

  * Random Forest, k-NN, SVM, etc., leveraging the same quality-weighted data.
* Add **custom evaluation suites**:

  * per-target calibration plots, parity plots, per-quality bucket performance.

---

This README is your **roadmap**: it tells you what files exist (or should be created), how they connect, what each phase adds, and where unit tests live. You can implement it phase by phase, running the tests as you go, and end up with a robust, quality-aware, multi-model molecular ML framework.

```
::contentReference[oaicite:0]{index=0}
```
