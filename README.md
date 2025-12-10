# OpenADMET + ExpansionRx Blind Challenge

**Authors**: Alec Glisman, PhD
**Date**: October 2025

## Getting Started

This repository contains code and documentation for participating in the OpenADMET + ExpansionRx Blind Challenge. The goal of this challenge is to develop machine learning models to predict various ADMET properties of small molecules using the provided dataset.

To get started, please follow the installation instructions in [INSTALLATION.md](./INSTALLATION.md) to set up your development environment.
You can find contribution guidelines in [CONTRIBUTING.md](./CONTRIBUTING.md) if you wish to contribute to this project.

### Training Models

#### Single Model Training

Train a single Chemprop model using a YAML configuration file:

```bash
# Train from command line
python -m admet.model.chemprop.model --config configs/example_chemprop.yaml

# With debug logging
python -m admet.model.chemprop.model -c configs/example_chemprop.yaml --log-level DEBUG
```

Or programmatically in Python:

```python
from omegaconf import OmegaConf
from admet.model.chemprop import ChempropConfig, ChempropModel

# Load configuration from YAML
config = OmegaConf.merge(
    OmegaConf.structured(ChempropConfig),
    OmegaConf.load("configs/example_chemprop.yaml")
)

# Create and train model
model = ChempropModel.from_config(config)
model.fit()

# Generate predictions
predictions = model.predict(test_df, generate_plots=True, split_name="test")

# Clean up MLflow run
model.close()
```

#### Ensemble Training

Train an ensemble of models across multiple splits and folds with Ray-based parallelization:

```bash
# Train ensemble from command line
python -m admet.model.chemprop.ensemble --config configs/ensemble_chemprop.yaml

# Limit parallel models to prevent OOM
python -m admet.model.chemprop.ensemble -c configs/ensemble_chemprop.yaml --max-parallel 2
```

Or programmatically:

```python
from omegaconf import OmegaConf
from admet.model.chemprop import EnsembleConfig, ChempropEnsemble

# Load ensemble configuration
config = OmegaConf.merge(
    OmegaConf.structured(EnsembleConfig),
    OmegaConf.load("configs/ensemble_chemprop.yaml")
)

# Create ensemble trainer
ensemble = ChempropEnsemble.from_config(config)

# Discover available splits/folds
splits_folds = ensemble.discover_splits_folds()
print(f"Found {len(splits_folds)} split/fold combinations")

# Train all models (uses Ray for parallelization)
ensemble.train_all(max_parallel=2)

# Generate ensemble predictions with uncertainty estimates
test_predictions = ensemble.predict_ensemble(test_df, split_name="test")
blind_predictions = ensemble.predict_ensemble(blind_df, split_name="blind")

# Access mean predictions and standard errors
print(test_predictions[["SMILES", "LogD_mean", "LogD_stderr"]])

# Generate ensemble plots with error bars
ensemble.generate_ensemble_plots(test_predictions, split_name="test")

# Clean up
ensemble.close()
```

The ensemble configuration extends the single-model config with additional options:

```yaml
# configs/ensemble_chemprop.yaml
ensemble:
  # Root directory containing split_*/fold_*/ subdirectories
  data_dir: "assets/dataset/splits/data"
  # Optional: filter specific splits/folds (null = use all)
  splits: null
  folds: null

# Rest of config same as single model...
data:
  test_file: "assets/dataset/set/local_test.csv"
  blind_file: "assets/dataset/set/blind_test.csv"
  # ...
```

#### Task Affinity Grouping

Use Task Affinity Grouping (TAG) to automatically discover which tasks benefit from joint training:

```bash
# Train with task affinity enabled
python -m admet.model.chemprop.model --config configs/chemprop_task_affinity.yaml

# Override number of task groups
python -m admet.model.chemprop.model -c configs/chemprop.yaml \
    --task-affinity.enabled true \
    --task-affinity.n-groups 3

# Compute affinity matrix only
python -m admet.cli.compute_task_affinity \
    --data-path data/admet_train.csv \
    --smiles-column SMILES \
    --target-columns LogD KSOL PAMPA hERG CLint \
    --save-path results/task_affinity.npz \
    --plot-heatmap results/affinity_heatmap.png
```

Configure task affinity in your YAML:

```yaml
# configs/chemprop_task_affinity.yaml
task_affinity:
  enabled: true
  n_groups: 3                    # Number of task groups
  affinity_epochs: 1             # Epochs for affinity computation
  affinity_batch_size: 64
  clustering_method: "agglomerative"

data:
  target_columns:
    - "LogD"
    - "KSOL"
    - "PAMPA"
    - "hERG"
    - "CLint"
```

Task affinity automatically groups related tasks (e.g., solubility, permeability, metabolism) to improve multi-task learning. See [docs/guide/task_affinity.rst](./docs/guide/task_affinity.rst) for detailed usage.

#### Hyperparameter Optimization

Run hyperparameter optimization (HPO) for Chemprop models using Ray Tune with ASHA scheduler:

```bash
# Run HPO from command line
python -m admet.model.chemprop.hpo --config configs/hpo_chemprop.yaml --num-samples 50

# With custom resource allocation
python -m admet.model.chemprop.hpo -c configs/hpo_chemprop.yaml \
    --gpus-per-trial 0.5 --cpus-per-trial 4 --output-dir hpo_results/
```

Or use the convenience bash script:

```bash
# Run HPO with default settings
./scripts/train_chemprop_hpo.sh configs/hpo_chemprop.yaml
```

Or programmatically in Python:

```python
from omegaconf import OmegaConf
from admet.model.chemprop import ChempropHPO, HPOConfig

# Load HPO configuration
config = OmegaConf.merge(
    OmegaConf.structured(HPOConfig),
    OmegaConf.load("configs/hpo_chemprop.yaml")
)

# Create HPO runner
hpo = ChempropHPO(config)

# Run optimization (returns Ray Tune ResultGrid)
result = hpo.run()

# Get top-k configurations as list of dicts
top_configs = hpo.get_top_k_configs(result, k=5)

# Results are also saved to:
# - hpo_results/top_k_configs.json (best hyperparameters)
# - hpo_results/ray_results/ (full Ray Tune artifacts)
```

Key HPO features:

- **ASHA Scheduler**: Early stopping of underperforming trials using Asynchronous Successive Halving
- **Conditional Search Spaces**: Automatic handling of architecture-dependent parameters (MoE experts, branched network trunk settings)
- **Transfer Learning**: Optional warm-start from pretrained CheMeleon checkpoints
- **MLflow Integration**: All trials logged as nested runs under a parent HPO experiment
- **Resource Management**: Fine-grained GPU/CPU allocation per trial for efficient parallelization

### Model Card

Comprehensive model documentation following ML transparency best practices is available in [`docs/model_card.md`](./docs/model_card.md). The model card includes:

- Intended use cases and limitations
- Training data descriptions and preprocessing
- Evaluation metrics and benchmark results
- Ethical considerations and bias analysis

### Ensemble Metrics

The ensemble training module reports comprehensive performance metrics:

- **MAE** (Mean Absolute Error): Primary regression metric
- **RMSE** (Root Mean Square Error): Penalizes large errors
- **RAE** (Relative Absolute Error): Scale-independent error relative to baseline
- **$R^2$**: Coefficient of determination
- **Pearson $r^2$**: Squared linear correlation coefficient
- **Spearman $\rho^2$**: Squared rank correlation (robust to outliers)
- **Kendall $\tau$**: Ordinal association measure

These metrics are computed per-target and aggregated with uncertainty estimates (mean ± stderr) across ensemble members.

### Data Splitting

Generate train/validation splits using cluster-based cross-validation with multiple stratification strategies:

```bash
# Run data splitting from command line
python -m admet.data.split --input data.csv --output outputs/ \
    --cluster-method bitbirch --split-method multilabel_stratified_kfold

# Or use the batch script for all configurations
./scripts/run_data_splits.sh --input data.csv --output-dir assets/dataset/splits/
```

Or programmatically in Python:

```python
from admet.data.split import pipeline
import pandas as pd

df = pd.read_csv("data.csv")

# Run clustering and cross-validation splitting
df_with_assignments = pipeline(
    df,
    cluster_method="bitbirch",      # Options: random, scaffold, kmeans, umap, butina, bitbirch
    split_method="multilabel_stratified_kfold",  # Options: group_kfold, stratified_kfold, multilabel_stratified_kfold
    n_splits=5,
    n_folds=5,
    smiles_col="SMILES",
    quality_col="Quality",
    fig_dir="outputs/figures",
)

# Access fold assignments
print(df_with_assignments[["SMILES", "cluster", "split_0_fold_0"]])
```

Key splitting features:

- **BitBirch Clustering**: Scalable hierarchical clustering using RDKit fingerprints (recommended)
- **Stratification Options**: Balance endpoint coverage and quality distributions across folds
- **Multi-label Support**: Handle sparse multi-task datasets with `MultilabelStratifiedKFold`
- **Diagnostic Plots**: Automatic visualization of cluster distributions and fold statistics

### Tech Stack Highlights

- `Python 3.11`: modern baseline with `Typer`/`Rich` powering the CLI.
- `RDKit` + `useful-rdkit-utils`: molecular featurization and cheminformatics helpers.
- `Chemprop`: graph neural networks tailored to molecular property prediction.
- `Transformers` (`ChemBERTa`): SMILES sequence modeling and transfer learning.
- `XGBoost` + `LightGBM`: strong tabular baselines for speed and interpretability.
- `PyTorch` + `TorchMetrics`: deep learning backbone with consistent metric logging.
- `Ray Tune`: distributed hyperparameter search (ASHA scheduler) and parallel training orchestration.
- `MLflow`: experiment tracking, artifacts, and configuration capture.
- `Matplotlib`/`Seaborn`: visualization stack for EDA and reporting.
- `Polaris`/`TDC`/`Hugging Face Datasets`: curated ADMET data access and augmentation.

### Pre-commit Quality Gates

- `pre-commit-hooks`: whitespace/EOF/merge-conflict/mixed-line-ending guards, secret and size checks, JSON/YAML/TOML validation.
- `Prettier`: consistent formatting for TOML and YAML configs.
- `nbstripout`: strips notebook outputs while keeping metadata tidy.
- `beautysh`: enforces clean formatting for shell scripts.
- `Black`: opinionated Python formatting to keep diffs small.
- `isort`: import ordering aligned with Black style.
- `flake8`: fast linting for common Python style and correctness issues.
- `pylint`: deeper Python linting to catch code smells.
- `mypy`: optional static type checks on the `src` package.
- `pytest -q`: quick test suite smoke run on commit.
- `docs` rebuild hook: ensures Sphinx docs compile cleanly before commits.
- `Commitizen`: commit message linting to enforce changelog-friendly messages.

### Documentation Build

Project documentation (Sphinx) lives under `docs/`.

- Initial HTML build:

```bash
sphinx-build -b html docs docs/_build/html
```

- Clean rebuild (remove previous output then build):

```bash
rm -rf docs/_build
sphinx-build -b html docs docs/_build/html
```

- Open locally: point your browser at `docs/_build/html/index.html`.
- Optional live autoreload

```bash
sphinx-autobuild docs docs/_build/html
```

  This serves docs at <http://127.0.0.1:8000> with automatic refresh on changes.

#### Using the Makefile

A convenience `Makefile` is provided under the `docs/` directory with
common Sphinx targets. From the repository root you can run:

```bash
make -C docs html
```

To remove previous build artifacts:

```bash
make -C docs clean
```

These targets call `sphinx-build` under the hood and produce output in `docs/_build/html`.

## Goals

### Models

We plan to benchmark the following models:

- XGBoost
- LightGBM
- Chemprop Multitask (trained from scratch)
- Chemprop CheMeleon (finetuned)
- GROVER (finetuned)
- KERMT (finetuned)
- ChemBERTa-3 (finetuned)

These models were selected to represent a range of architectures and training paradigms, including tree-based methods, graph neural networks, and transformer-based models. We will explore both training from scratch and transfer learning approaches.

Each of the models will be trained as ensemble models with 5-fold cross-validation to improve robustness and performance, as recommended in the literature (Practically Significant Method Comparison Protocols for Machine Learning in Small Molecule Drug Discovery). Each fold should have multiple random seeds to further enhance ensemble diversity. By default, we will try five random seeds per fold, resulting in a total of 25 models per endpoint per architecture.

We will explore various hyperparameter optimization strategies, including grid search, random search, and Bayesian optimization, to identify the best model configurations for each architecture.

We will also explore super-ensembling techniques to combine the embeddings and/or predictions from multiple models to further enhance predictive performance.

To begin, we will not use KERMT and ChemBERTa-3 due to local hardware constraints, but we plan to include them in future iterations.

### Endpoints

We are predicting the following ADMET endpoints:

| Column                       | Unit        | Type      | Description                                   |
|:---------------------------- |:----------: |:--------: |:----------------------------------------------|
| Molecule Name                |             |    str    | Identifier for the molecule |
| Smiles                       |             |    str    | Text representation of the 2D molecular structure |
| LogD                         |             |   float   | LogD calculation |
| KSol                         |    uM       |   float   | Kinetic Solubility |
| MLM CLint                    | mL/min/kg   |   float   | Mouse Liver Microsomal |
| HLM CLint                    | mL/min/kg   |   float   | Human Liver Microsomal |
| Caco-2 Permeability Efflux   |             |   float   | Caco-2 Permeability Efflux |
| Caco-2 Permeability Papp A>B | 10^-6 cm/s  |   float   | Caco-2 Permeability Papp A>B |
| MPPB                         | % Unbound   |   float   | Mouse Plasma Protein Binding |
| MBPB                         | % Unbound   |   float   | Mouse Brain Protein Binding |
| MGMB.                        | % Unbound   |   float   | Mouse Gastrocnemius Muscle Binding |

The challenge will be judged based on the following criteria:

- We welcome submissions of any kind, including machine learning and physics-based approaches. You can also employ pre-training approaches as you see fit, as well as incorporate data from external sources into your models and submissions.
- In the spirit of open science and open source we would love to see code showing how you created your submission if possible, in the form of a Github Repository. If not possible due to IP or other constraints you must at a minimum provide a short report written methodology based on the template here. Make sure your lat submission before the deadline includes a link to a report or to a Github repository.
- Each participant can submit as many times as they like, up to a limit of once per day. Only your latest submission will be considered for the final leaderboard.
- The endpoints will be judged individually by mean absolute error (MAE), while an overall leaderboard will be judged by the macro-averaged relative absolute error (MA-RAE).
- For endpoints that are not already on a log scale (e.g LogD) they will be transformed to log scale to minimize the impact of outliers on evaluation.
- We will estimate errors on the metrics using bootstrapping and use the statistical testing workflow outlined in this paper to determine if model performance is statistically distinct.

### Datasets

We will attempt to augment the provided training dataset with additional publicly available ADMET datasets to improve model performance. Potential sources for augmentation are listed in the Links section below.

## Links

### Challenge Information

- [Challenge Hugging Face Page](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge)
- [Teaser Dataset on Hugging Face](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-teaser)

    ```python
    # Hugging Face Datasets library
    from datasets import load_dataset
    ds = load_dataset("openadmet/openadmet-expansionrx-challenge-teaser")
    ```

- [Full Dataset on Hugging Face (not live)](https://huggingface.co/datasets/openadmet/openadmet-challenge-train-data)

### Reference Models

- [XGBoost Baseline](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html)
- [Chemprop Multitask](https://chemprop.readthedocs.io/en/latest/multi_task.html)
- [Chemprop Pretrained](https://chemprop.readthedocs.io/en/latest/chemeleon_foundation_finetuning.html)
- [ChemBERTa Foundation](https://deepchem.io/tutorials/transfer-learning-with-chemberta-transformers/)
- [KERMT Pretrained](https://github.com/NVIDIA-Digital-Bio/KERMT)

### External Datasets

- [x] [KERMT](https://figshare.com/articles/dataset/Datasets_for_Multitask_finetuning_and_acceleration_of_chemical_pretrained_models_for_small_molecule_drug_property_prediction_/30350548/2)
- [x] [Polaris Antiviral](https://polarishub.io/datasets/asap-discovery/antiviral-admet-2025-unblinded)
- [x] [Polaris ADME Fang](https://polarishub.io/datasets/biogen/adme-fang-v1)
- [x] [TDC](https://tdcommons.ai/benchmark/admet_group/overview/)
- [x] [PharmaBench](https://github.com/mindrank-ai/PharmaBench)
- [x] [NCATS](https://opendata.ncats.nih.gov/adme/data)
- [ ] [admetSAR 3.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC11223829/#:~:text=Data%20collection,are%20available%20in%20Text%20S2.)
  - NOTE: Appears to be proprietary data
- [x] [admetica](https://github.com/datagrok-ai/admetica)
- [x] [ChEMBL ADMET](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest)

### Papers and Blogs

- [Dataset Splitting](https://practicalcheminformatics.blogspot.com/2024/11/some-thoughts-on-splitting-chemical.html)
- [Benchmarking](https://practicalcheminformatics.blogspot.com/2023/08/we-need-better-benchmarks-for-machine.html)
- [Comparisons](https://practicalcheminformatics.blogspot.com/2025/03/even-more-thoughts-on-ml-method.html)
- [Practically Significant Method Comparison Protocols for Machine Learning in Small Molecule Drug Discovery](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5c01609)

### Examples

- [DataBricks Chemprop Training](https://community.databricks.com/t5/technical-blog/ai-drug-discovery-made-easy-your-complete-guide-to-chemprop-on/ba-p/111750#h_324287967181751055572426)
- [Chemprop Data Splitting](https://chemprop.readthedocs.io/en/latest/tutorial/python/data/splitting.html)
- [Chemprop Multitask Model](https://chemprop.readthedocs.io/en/latest/multi_task.html)
- [CheMeleon Foundation Finetuning](https://chemprop.readthedocs.io/en/latest/chemeleon_foundation_finetuning.html)

### Coding Assistants

- [Copilot Prompts](https://github.com/github/awesome-copilot/tree/main?tab=readme-ov-file)
