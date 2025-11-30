# Planned Model Training Steps

## High Level Overview

1. **Load pre-split datasets**: Load the datasets that have been split using various clustering methods (random, scaffold-based, k-means, Butina).
   - Input: Pre-split datasets from previous steps. Datasets are stored in a structured directory tree: `assets/dataset/splits/v2/{quality}_quality/{split_method}/`. Non-temporal splits also include N-split and K-fold information with subdirectories `split_{n}/fold_{k}/`. All datasets are in Hugging Face `Dataset` format.
   - Blinded Test Data: Blinded test data can be found at `assets/dataset/test/expansion_data_test_blinded.csv`.
   - User should specify through Typer CLI:
     - Quality level of the dataset (e.g., high, medium, low).
     - Splitting method used (e.g., random_cluster, scaffold_cluster, kmeans_cluster, butina_cluster, temporal_split).
   -
2. **Feature Extraction**: Extract metadata, SMILES, endpoints (predictors), and fingerprints from the datasets.
    - Metadata: `Molecule Name,Dataset`
    - SMILES: `SMILES`
    - Endpoints: `LogD,KSOL,HLM CLint,MLM CLint,Caco-2 Permeability Efflux,MPPB,Caco-2 Permeability Papp A>B,MBPB,MGMB`
    - Fingerprints: `Morgan_FP_[0-2047]`
    - Note that the blinded test data only contains `Molecule Name,SMILES`
3. **Model Training**: Train machine learning models using the extracted features.
    - Model options:
      - Classical ML models: Support Vector Machine, k-Nearest Neighbors, Random Forest
      - Gradient Boosting models: XGBoost, LightGBM
      - MPNN models: Chemprop
      - Pre-trained MPNN models: CheMeleon
      - Pre-trained Transformer models: ChemBERTA
    - Multi-output regression: All models should support multi-output regression to predict multiple endpoints simultaneously. If a model does not natively support multi-output regression, train separate models for each endpoint and aggregate the results. Notify the user with a warning message if this is the case.
    - API: Models should inherit from a base model class to ensure consistency. Classical ML models, gradient boosting models have the fingerprints input format, while MPNN models and transformer models use SMILES input format.
    - Ensemble training: They should run N-split, K-fold cross-validation as per the dataset splits. Each model should also have a random seed and can be trained in parallel. Hyperparameter configuration should be supported for all models. Expose all default hyperparameters for each model and allow user overrides via Typer CLI.
    - Weighted loss: Allow for weighted loss functions to handle class (`Dataset` metadata) imbalance if needed. This should be an optional parameter that dictates the weighting scheme for each class in the loss function. Unspecified classes should default to a weight of 1.
    - Early stopping: Implement early stopping based on validation loss to prevent overfitting. Allow user to specify patience and minimum delta for early stopping via Typer CLI.
    - Archive trained models: Save trained models in a structured directory tree similar to the input datasets for easy retrieval and comparison.
    - Ensemble policy: Final reported predictions for any dataset (including the blinded test set) are produced as ensembles of models trained across the specified splits and folds, with endpoint-wise means and standard errors aggregated across models.
4. **Model Evaluation**: Evaluate the trained models on the test sets and record performance metrics.
    - Metrics to compute:
      - For each endpoint, compute RMSE, MAE, R², and others as needed. MAE is the primary metric for ranking models and should be used for training models.
      - To aggregate multi-output regression metrics, compute the macro-average (unweighted mean) of the metrics across all endpoints.
    - Store evaluation results in a structured format for easy comparison across models and splits. This should be dumped to a CSV file and plotted using visualizations (e.g., box plots, bar charts).
    - Models should predict all quantities on the train/val/test dataset. They should be saved in 2 forms as CSV files:
      - 1. Direct outputs
      - 2. Transformed outputs. For all non `LogD` columns transform by `10^x` for easier comparison with experimental values.
      - On both predictions, plot the histograms (Seaborn with KDE) of the predicted values for each endpoint on a single figure with subplots. Each dataset (train, validation, test) should have its own figure.
      - For each endpoint, plot predicted vs experimental values with a parity line for each dataset (train, validation, test). Each dataset should have its own figure with subplots for each endpoint.
      - Plot the correlation heatmap (Seaborn) of predicted vs experimental values for each endpoint on the datasets.
      - Plot the correlation heatmap (Seaborn) of predicted values between each endpoint on the datasets.

## Scope

- No hyperparameter optimization or tuning is included in this plan; models will be trained with default or user-specified hyperparameters.
- No uncertainty quantification or conformal prediction is included in this plan.

## Detailed Implementation Plan

### Dataset contract and split API

- Use Hugging Face `Dataset` objects with a consistent schema for all pre-split datasets.
- Required columns for all train/val/test datasets:
  - `Molecule Name` (string)
  - `SMILES` (string)
  - `Dataset` (string or categorical, used for sample weights)
  - Endpoints (float): `LogD`, `KSOL`, `HLM CLint`, `MLM CLint`, `Caco-2 Permeability Efflux`, `MPPB`, `Caco-2 Permeability Papp A>B`, `MBPB`, `MGMB`
  - Fingerprints (float or int): `Morgan_FP_0` .. `Morgan_FP_2047`
- Targets are in log10 space for all endpoints except `LogD`, which is in linear space. Assume all variables are normalized appropriately at dataset creation time; do not apply further preprocessing in the training library.
- Missing endpoints for a given molecule are represented as `NaN` and are handled with masking in the loss function.
- Blinded test dataset:
  - Separate loader pointing to `assets/dataset/test/expansion_data_test_blinded.csv`.
  - Required columns: `Molecule Name`, `SMILES`.
  - No endpoints or fingerprints are stored; fingerprints are regenerated from SMILES using the same RDKit/Morgan parameters as during training.

Define a small dataset API in the library code (for example in `admet.data`):

- `load_dataset(quality, split_method, subset=None, split_level=None, split_id=None, fold_id=None)`
  - `quality`: `"high" | "medium" | "low"`.
  - `split_method`: `"random_cluster" | "scaffold_cluster" | "kmeans_cluster" | "butina_cluster" | "temporal_split"`.
  - `subset`: `"train" | "val" | "test" | None`.
  - `split_level`: `"nsplit" | "kfold" | "temporal" | None`.
  - `split_id`: integer index for N-split (optional for temporal).
  - `fold_id`: integer index for K-fold (optional; only used when `split_level="kfold"`).
  - Returns `Dataset` if `subset` is specified, or `DatasetDict` with `{"train", "val", "test"}` if `subset=None`.
- `load_blinded_dataset()`
  - Returns a `Dataset` with `Molecule Name` and `SMILES` columns for the blinded test set.

Split directory conventions:

- Non-temporal splits live under `assets/dataset/splits/v2/{quality}_quality/{split_method}/split_{split_id}/fold_{fold_id}/`.
- Temporal splits can omit `split_{n}/fold_{k}/` and use a temporal naming scheme. `split_level="temporal"` indicates this convention.

Data schema validation on loading (`validate_dataset_schema` helper):

- Check that all required columns are present with expected dtypes.
- Verify that fingerprint columns cover the full range `Morgan_FP_0` .. `Morgan_FP_2047`.
- Verify that endpoint columns are float-like and within reasonable numeric ranges for log10 and linear scales.
- Log warnings if extra unexpected columns exist but do not fail.
- Fail fast with a clear error message if any required columns are missing or dtypes are incompatible.

Provide a helper to iterate splits:

- `iter_splits(quality, split_method, split_level, split_ids, fold_ids=None)`
  - Yields `(split_id, fold_id)` pairs used to drive cross-validation and ensembling.

### Model training, multi-output behavior, and serialization

- Base model interface (for example in `admet.model.base`):
  - `fit(X_train, y_train, X_val=None, y_val=None, sample_weight=None, target_mask=None)`.
  - `predict(X)` → predictions for all endpoints in a fixed order.
  - `save(path)` and `@classmethod load(path)`.
  - `get_config()` → model and training hyperparameters.
  - `get_metadata()` → endpoints, dataset metadata, seeds, and training timestamps.
  - Attributes:
    - `input_type`: `"fingerprint"` or `"smiles"`.
    - `endpoints`: ordered list of endpoint names.

- Multi-output regression and missing endpoints:
  - Train on log10 targets for all endpoints except `LogD`, which is in linear space.
  - Use a target mask `M` with shape `(N, D)` where `M[i, d] = 1` if endpoint `d` is present for molecule `i`, else `0`.
  - For native multi-output models, compute the loss per-sample and per-endpoint only where `M[i, d] = 1` and normalize by the sum of mask entries.
  - Always predict all endpoints in a fixed order, even if some endpoints are sparse in the training data.
  - If a model does not support native multi-output regression, train separate models per endpoint and aggregate them in a wrapper that exposes the same multi-output interface.

- Sample weights:
  - When `Weighted loss` is enabled, compute a per-row `sample_weight` from the `Dataset` column using a user-specified mapping (for example via YAML config):
    - For each row, look up `weights[row["Dataset"]]` with a `default` fallback of `1.0`.
  - For models that support `sample_weight` (Random Forest, XGBoost, LightGBM, some neural network trainers), pass the computed weights into `fit`.
  - For models that do not support sample weights, log a warning and proceed without weighting.

- Serialization layout:
  - For each `(quality, split_method, split_id, fold_id, model_family, model_name)` combination, store model artifacts under:
    - `assets/models/{quality}/{split_method}/split_{split_id}/fold_{fold_id}/{model_family}/{model_name}/`.
  - Within each model directory, store:
    - `model.bin` (serialized model state, e.g. PyTorch checkpoint or pickle).
    - `config.yaml` (full experiment configuration, including model, training, and data settings).
    - `metrics.json` (per-endpoint and aggregated metrics on train/val/test, both in log10 and linear space).
    - `preprocessor.pkl` (if any runtime preprocessing is needed).
    - `train_predictions_log.csv`, `val_predictions_log.csv`, `test_predictions_log.csv`.
    - `train_predictions_linear.csv`, `val_predictions_linear.csv`, `test_predictions_linear.csv` (with `10^x` transform for non-`LogD` endpoints).
    - `run_info.json` (seeds, git hash, library versions, device, runtime, and CLI args).
  - For per-endpoint models, nest each endpoint under an `endpoints/` directory but still expose a multi-output wrapper that follows the base model interface.
  - Implement a helper `load_model(model_dir: str) -> BaseModel` that reads `config.yaml` to instantiate the correct subclass and loads `model.bin`.

### Hyperparameter configuration (single YAML file)

- Use a single YAML configuration file to control data, model, training, and ensemble behavior.
- Example structure:

```yaml
experiment_name: "openadmet_baseline_v1"

seed:
  python: 123
  numpy: 123
  torch: 123
  xgboost: 123
  lightgbm: 123

data:
  quality: "high"
  split_method: "random_cluster"
  split_level: "kfold"   # "nsplit" | "kfold" | "temporal"
  split_ids: [0, 1, 2]
  fold_ids: [0, 1, 2, 3, 4]
  endpoints:
    - LogD
    - KSOL
    - HLM CLint
    - MLM CLint
    - Caco-2 Permeability Efflux
    - MPPB
    - Caco-2 Permeability Papp A>B
    - MBPB
    - MGMB

training:
  ensemble:
    strategy: "mean"    # mean | median | custom
    include:
      # Optional explicit list of (split_id, fold_id) pairs used in ensembles
      - { split_id: 0, fold_id: 0 }
      - { split_id: 0, fold_id: 1 }
      - { split_id: 1, fold_id: 0 }
  parallel:
    backend: "ray"         # ray | joblib | multiprocessing
    max_parallel_jobs: 4

  sample_weights:
    enabled: true
    dataset_column: "Dataset"
    weights:
      dataset_a: 1.0
      dataset_b: 2.0
      default: 1.0

  early_stopping:
    enabled: true
    patience: 20
    min_delta: 0.0001

  device:
    preferred: "cuda"       # cpu | cuda | auto
    gpu_id: 0

models:
  random_forest:
    enabled: true
    input_type: "fingerprint"
    model_class: "RandomForestRegressor"
    model_params:
      n_estimators: 500
      max_depth: null
      max_features: "sqrt"
      min_samples_split: 2
      min_samples_leaf: 1
      n_jobs: -1
      bootstrap: true
      random_state: 123
    training_params:
      use_multi_output_wrapper: true

  xgboost:
    enabled: true
    input_type: "fingerprint"
    model_class: "XGBRegressor"
    model_params:
      n_estimators: 1000
      learning_rate: 0.05
      max_depth: 6
      subsample: 0.8
      colsample_bytree: 0.8
      reg_alpha: 0.0
      reg_lambda: 1.0
      tree_method: "hist"
      n_jobs: 8
      random_state: 123
    training_params:
      early_stopping_rounds: 50
      eval_metric: "mae"

  lightgbm:
    enabled: true
    input_type: "fingerprint"
    model_class: "LGBMRegressor"
    model_params:
      n_estimators: 2000
      learning_rate: 0.02
      num_leaves: 31
      max_depth: -1
      subsample: 0.8
      colsample_bytree: 0.8
      reg_alpha: 0.0
      reg_lambda: 0.0
      random_state: 123
      n_jobs: 8
    training_params:
      early_stopping_rounds: 100
      eval_metric: "mae"

  chemprop:
    enabled: true
    input_type: "smiles"
    model_class: "ChempropMPNN"
    model_params:
      hidden_size: 300
      depth: 5
      dropout: 0.1
      aggregation: "mean"
    training_params:
      batch_size: 64
      epochs: 100
      optimizer: "adam"
      learning_rate: 0.001
      weight_decay: 0.0
      early_stopping_patience: 20
      early_stopping_delta: 0.0001

  chemeleon:
    enabled: true
    input_type: "smiles"
    model_class: "CheMeleonMPNN"
    model_params:
      pretrained_checkpoint: "path/to/chemeleon.pt"
      freeze_backbone: false
      hidden_size: 300
    training_params:
      batch_size: 64
      epochs: 50
      learning_rate: 0.0005
      early_stopping_patience: 10
      early_stopping_delta: 0.0001

  chemberta:
    enabled: true
    input_type: "smiles"
    model_class: "ChemBERTAModel"
    model_params:
      pretrained_model_name: "seyonec/PubChem10M_SMILES_BPE_450k"
      freeze_encoder_layers: 0
      max_length: 256
    training_params:
      batch_size: 32
      epochs: 20
      learning_rate: 3e-5
      warmup_steps: 500
      weight_decay: 0.01
      early_stopping_patience: 5
      early_stopping_delta: 0.0005
```

### Parallelization with Ray and ensemble predictions

- Use Ray as the primary parallelization backend (add `ray` as a required dependency in `pyproject.toml`).
- Parallelize over `(split_id, fold_id, model_name)` tasks:
  - For each tuple, launch a Ray remote function that:
    - Loads the appropriate train/val/test splits.
    - Builds features (SMILES or fingerprints).
    - Computes sample weights if enabled.
    - Trains the model with early stopping and masked loss.
    - Saves artifacts and predictions to the model directory.
- Control concurrency with a global `max_parallel_jobs` parameter:
  - Expose `--max-parallel-jobs` in the CLI to override the YAML setting.
  - Use Ray resources (e.g. `num_cpus`, `num_gpus`) based on the model and `device` configuration.

Final predictions and ensembles:

- For any dataset (train, val, test, or blinded), ensemble predictions across all selected models:
  - For each model in the ensemble set, compute `Y_pred^(j)` with shape `(N, D)`.
  - Aggregate with the mean across models for each molecule and endpoint:
    - `Y_mean = mean_j Y_pred^(j)`.
  - Compute the standard deviation across models:
    - `Y_std = std_j Y_pred^(j)`.
  - Compute standard error across models:
    - `Y_se = Y_std / sqrt(J)`, where `J` is the number of models.
- Store ensemble predictions in separate CSV files:
  - `ensemble_predictions_log.csv` with columns:
    - `Molecule Name`, `SMILES`, and for each endpoint:
      - `{endpoint}_mean_log`, `{endpoint}_std_log`, `{endpoint}_se_log`.
  - `ensemble_predictions_linear.csv` with the same structure but endpoints converted to linear space:
    - `{endpoint}_mean_linear`, `{endpoint}_std_linear`, `{endpoint}_se_linear`, where non-`LogD` values are back-transformed with `10^x`.
- Blinded test handling:
  - Treat the blinded test as an independent dataset loaded via `load_blinded_dataset()`.
  - Generate fingerprints and any SMILES-based features.
  - Run inference through all trained models in the ensemble set.
  - Save ensemble predictions in `ensemble_predictions_log.csv` and `ensemble_predictions_linear.csv` for the blinded dataset.

### Evaluation metrics and visualization

- For each trained model and for the ensemble:
  - Compute metrics separately for train, val, and test splits.
  - For each endpoint and split:
    - In log space (dataset space): RMSE_log, MAE_log, R²_log.
    - In linear space: convert predictions and targets to linear units (identity for `LogD`, `10^x` for other endpoints) and compute RMSE_linear, MAE_linear, R²_linear.
  - Compute macro-average metrics across endpoints with valid data (non-empty masks):
    - `macro_log` and `macro_linear` for each metric and split.
- Visualizations:
  - Save plots under `assets/figures/model_eval/{quality}/{split_method}/split_{split_id}/fold_{fold_id}/{model_family}/{model_name}/`.
  - For each split (train, val, test) and space (log, linear):
    - Histograms with KDE of predicted values per endpoint (one figure per dataset with subplots).
    - Parity plots (predicted vs experimental) per endpoint with a parity line (one figure per dataset with subplots).
    - Correlation heatmap of predicted vs experimental values across endpoints.
    - Correlation heatmap of predicted values between endpoints.
  - For ensemble evaluation, produce analogous plots at the ensemble level.
  - Include metadata (model name, split id, fold id, endpoints, log/linear) in figure titles and optionally annotate macro MAE.

### CLI entrypoints and workflows

Implement Typer-based CLI commands in `admet.cli`:

- `admet train`:
  - Arguments:
    - `--config PATH` (required): YAML configuration file.
    - `--model-name TEXT` (optional): train only the specified model key from `models`.
    - `--max-parallel-jobs INT` (optional): override config.training.parallel.max_parallel_jobs.
    - `--device TEXT` (optional): override config.training.device.preferred (`cpu`, `cuda`, `auto`, `cuda:0`, etc.).
    - `--seed INT` (optional): override the seed values in the config.
    - `--log-level TEXT` (optional): `INFO`, `DEBUG`, etc.
    - `--dry-run` (flag): validate config, dataset loading, and shapes without training.
  - Workflow:
    - Load and validate the YAML config.
    - Apply CLI overrides (CLI values take precedence over YAML).
    - Seed all libraries (Python, NumPy, PyTorch, XGBoost, LightGBM, Ray).
    - Resolve device (CPU or GPU) based on config and CLI input.
    - Build the list of `(split_id, fold_id, model_name)` tasks from `data.split_ids`, `data.fold_ids`, and enabled models.
    - Launch training tasks in parallel with Ray respecting `max_parallel_jobs`.
    - Save all artifacts, predictions, and metrics to their respective directories.

- `admet evaluate`:
  - Arguments:
    - `--model-root PATH` (required): root directory containing one or more model runs.
    - `--config PATH` (optional): YAML to resolve data and endpoints when re-loading datasets.
    - `--dataset-split [train|val|test|all]` (default: `all`).
    - `--ensemble` (flag): compute ensemble-level metrics and plots across models under `model-root`.
    - `--log-level TEXT` (optional).
  - Workflow:
    - Discover model directories under `model-root`.
    - For each model, load predictions if available or recompute them from the saved model and dataset loader.
    - Compute metrics in log and linear space for the requested splits.
    - If `--ensemble` is set, form ensembles as specified in the config and compute ensemble metrics and plots.
    - Write/overwrite `metrics.json` and save updated figures.

- `admet predict`:
  - Arguments:
    - `--model-root PATH` (required): root directory containing trained models to include in the ensemble.
    - `--input PATH` (optional): custom CSV or dataset path for inference; if omitted, use the blinded test dataset loader.
    - `--output PATH` (required): directory where prediction CSVs will be written.
    - `--ensemble-strategy [mean|median]` (optional): override config.training.ensemble.strategy.
    - `--device TEXT` (optional).
    - `--log-level TEXT` (optional).
  - Workflow:
    - Load the target dataset (custom or blinded).
    - Generate fingerprints and SMILES-based features as needed.
    - Discover models under `model-root` that match the current data configuration.
    - Run inference through each model, collect predictions, and aggregate with the chosen ensemble strategy.
    - Save `ensemble_predictions_log.csv` and `ensemble_predictions_linear.csv` with mean, std, and std err for each endpoint.

- `admet list-models` (optional helper):
  - Arguments:
    - `--root PATH` (required): root of the model directory tree.
    - Filter options: `--quality`, `--split-method`, `--split-id`, `--fold-id`, `--model-family`, `--model-name`.
  - Output:
    - A table of available models with their paths, key config fields, macro metrics, seeds, and device information.

### Logging, seeding, and device management

- Configure a shared `logging` logger (for example `admet`) used throughout the library and CLI.
- CLI options set the global log level and optional log file location.
- Each run writes `run.log` in the corresponding model directory capturing:
  - CLI arguments.
  - Resolved config.
  - Seeds and device.
  - Dataset sizes and endpoint coverage.
  - Start/end times for training per model.
  - Key metric summaries.

- Seeding policy:
  - Accept seed values from the YAML config and allow CLI override.
  - Seed Python `random`, NumPy, PyTorch (CPU and CUDA), and any model-specific RNGs (XGBoost, LightGBM).
  - Log the effective seeds to `run_info.json` and to `run.log`.

- Device management:
  - Implement a `resolve_device(preferred, gpu_id)` helper:
    - If `preferred` is `"cuda"` or `"auto"` and a GPU is available, use `cuda:{gpu_id}`.
    - Otherwise fall back to CPU with a logged warning.
  - Pass the device to Chemprop, CheMeleon, and ChemBERTA models for both training and inference.

### Key milestones

1. Implement dataset loading and validation:
   - Implement `load_dataset`, `load_blinded_dataset`, `iter_splits`, and `validate_dataset_schema`.
   - Confirm that datasets load correctly for a sample `(quality, split_method, split_id, fold_id)`.

2. Implement the base model interface and a first baseline model:
   - Implement `BaseModel` and a fingerprint-based baseline (Random Forest or LightGBM).
   - Support multi-output regression and masking for missing endpoints.

3. Implement training orchestration with Ray:
   - Implement the `train_single_model` and `train_all_models` orchestrators.
   - Integrate Ray for parallel training across folds and splits, honoring `max_parallel_jobs`.

4. Implement evaluation and plotting utilities:
   - Implement metric computation in log and linear space.
   - Implement plotting utilities and save outputs to the defined directory structure.

5. Implement ensemble prediction and blinded inference:
   - Implement ensemble aggregation (mean, std, std err) across models.
   - Implement `predict` CLI command for arbitrary inputs and the blinded test dataset.

6. Integrate CLI commands with Typer:
   - Implement `admet train`, `admet evaluate`, `admet predict`, and optionally `admet list-models`.
   - Test CLI workflows end-to-end on a small subset of data.

### Pytest unit tests to write

- Dataset and schema tests:
  - `test_load_dataset_schema_valid`:
    - Load a sample dataset with `load_dataset`.
    - Assert that all required columns exist and have expected dtypes.
  - `test_validate_dataset_schema_missing_column`:
    - Create a toy `Dataset` missing an endpoint column and assert that `validate_dataset_schema` raises a clear error.

- Masking and multi-output tests:
  - `test_target_mask_drops_missing_endpoints`:
    - Construct a small toy dataset with some missing endpoints.
    - Verify that the loss only accumulates over non-missing entries.
  - `test_multi_output_prediction_shape`:
    - Fit a baseline multi-output model and assert that predictions have shape `(N, D)` for all endpoints.

- Serialization tests:
  - `test_model_save_load_equivalence`:
    - Train a small model, save it, reload it, and assert that predictions on a fixed input match within a tolerance.

- Ensemble tests:
  - `test_ensemble_mean_and_std_err`:
    - Simulate predictions from three models on a small dataset.
    - Compute ensemble mean and std err, and assert they match manually computed values.

- Evaluation tests:
  - `test_metrics_log_and_linear_space`:
    - For a toy dataset with known values, compute metrics in log and linear space and assert expected values.

- CLI and workflow tests (using `pytest` and `typer.testing`):
  - `test_admet_train_dry_run`:
    - Run `admet train --config config_example.yaml --dry-run` on a minimal config and assert successful exit and basic log output.
  - `test_admet_predict_blinded`:
    - Run `admet predict` on a toy blinded dataset using a small baseline model and assert that output CSVs contain the expected columns and shapes.

This planning section serves as a detailed implementation roadmap for the training, evaluation, and inference library and CLI framework.
