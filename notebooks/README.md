# Notebooks

Jupyter notebooks for exploratory data analysis, dataset preparation, and submission analysis for the OpenADMET ExpansionRx Blind Challenge.

## Contents

| Notebook                                     | Description                                                                                                        |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `0_exploratory_data_analysis.ipynb`          | Initial EDA of the challenge dataset including property distributions, molecular descriptors, and data quality checks |
| `0_exploratory_data_analysis_v2.ipynb`       | Updated EDA with additional visualizations and refined analysis methodology                                        |
| `1_datset_splits.ipynb`                      | Analysis of train/validation split strategies using BitBirch clustering                                           |
| `submission_comparison_analysis.ipynb`       | Comparison of model submissions against competition leaderboard                                                    |
| `submission_comparison_analysis_playground.ipynb` | Experimental analysis and visualization playground                                                            |

## Supporting Files

| File                                   | Description                                                   |
| -------------------------------------- | ------------------------------------------------------------- |
| `expansionrx_transformed_describe.csv` | Summary statistics for transformed dataset features           |
| `marimo/`                              | Marimo reactive notebook experiments (alternative to Jupyter) |

## Usage

### Prerequisites

Ensure the virtual environment is activated and Jupyter dependencies are installed:

```bash
source .venv/bin/activate
uv pip install jupyter jupyterlab
```

### Running Notebooks

```bash
# JupyterLab (recommended)
jupyter lab notebooks/

# Classic Jupyter Notebook
jupyter notebook notebooks/

# VS Code
# Open any .ipynb file directly in VS Code
```

### Environment

Notebooks use the project's Python environment with all dependencies installed. Key packages:

- **pandas**, **numpy**: Data manipulation
- **matplotlib**, **seaborn**, **plotly**: Visualization
- **rdkit**: Molecular structure handling
- **scikit-learn**: ML utilities and metrics

## Dataset Files

Notebooks reference data from `assets/dataset/`:

- `assets/dataset/raw/`: Original challenge data
- `assets/dataset/set/`: Processed train/test splits
- `assets/dataset/eda/`: EDA outputs and intermediate files

## Notes

- Notebooks are for exploration and analysis; production code lives in `src/admet/`
- Large outputs may be cleared before commits to reduce repository size
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for notebook contribution guidelines
