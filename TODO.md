# Future Work

## Exploratory Data Analysis (EDA)

- Download project datasets from Hugging Face.
- Perform EDA to understand data distributions, missing values, and correlations.
- Visualize key features using libraries like Matplotlib and Seaborn.
- Generate summary statistics and visualizations to inform model selection. Output to `assets/eda/project` with PNG images and MD report.
  
## Data Augmentation

- Investigate external datasets (KERMT, Polaris Antiviral, TDC, etc.) for potential augmentation.
- Implement data augmentation techniques to enhance training data diversity.
- Evaluate the impact of augmented data on dataset size, quality, and endpoint distributions.
- Document augmentation strategies and results in `assets/augmentation` with PNG images and MD report.
- Update EDA to reflect changes post-augmentation in `assets/eda/augmented`.
- Create a summary report comparing original and augmented datasets.

## Data Preprocessing

- Standardize molecular representations (e.g., SMILES canonicalization).
- Handle missing values and outliers.
- Normalize or scale features as needed.
- Split data into training, validation, and test sets. Allow for ensembe and cross validation. Split on Taylor-Butina clusters to avoid data leakage.
- Document preprocessing steps and rationale.
- Output preprocessed datasets and a preprocessing report in `assets/preprocessing`.
- Generate visualizations of feature distributions before and after preprocessing.
- Create a summary report detailing preprocessing steps and their impact on data quality.

## Model Training

- Implement and train baseline models (Chemprop Multitask, CheMeleon, ChemBERTa, KERMT).
- Fine-tune models using pre-trained weights where applicable.
- Use cross-validation to assess model robustness.
- Perform hyperparameter tuning using grid search or Bayesian optimization with `ray[tune]` to optimize model performance.
- Document model architectures, training procedures, and hyperparameter choices.
- Save trained models and training logs in `assets/models`.
- Generate training curves and performance metrics visualizations using `mlflow`.
- Create a summary report comparing model architectures and training strategies.
- Implement ensemble methods to combine predictions from multiple models for improved performance.
- Evaluate ensemble performance against individual models and document findings.
