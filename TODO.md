# Future Work

## Dataset Acquisition

- [x] Download the full training dataset from Hugging Face.
- [x] Add any additional external datasets identified for augmentation.
- [x] Check endpoint consistency and compatibility across datasets.

## Exploratory Data Analysis (EDA)

- [x] Download project datasets from Hugging Face.
- [x] Perform EDA to understand data distributions, missing values, and correlations.
- [x] Visualize key features using libraries like Matplotlib and Seaborn.
- [ ] Generate summary statistics and visualizations to inform model selection. Output to `assets/eda/project` with PNG images and MD report.

Look at Pat Walters's blog posts for further inspiration on EDA: <https://patwalters.github.io/OpenADMET-ExpansionRx-Data-Analysis/>

## Data Augmentation

- [x] Investigate external datasets (KERMT, Polaris Antiviral, TDC, etc.) for potential augmentation.
- [x] Implement data augmentation techniques to enhance training data diversity.
- [ ] Evaluate the impact of augmented data on dataset size, quality, and endpoint distributions.
- [ ] Document augmentation strategies and results in `assets/augmentation` with PNG images and MD report.
- [ ] Update EDA to reflect changes post-augmentation in `assets/eda/augmented`.
- [ ] Create a summary report comparing original and augmented datasets.

## Data Preprocessing

- [ ] Standardize molecular representations (e.g., SMILES canonicalization).
- [ ] Handle missing values and outliers.
- [ ] Normalize or scale features as needed.
- [ ] Split data into training, validation, and test sets. Allow for ensembe and cross validation. Split on Taylor-Butina clusters to avoid data leakage.
- [ ] Document preprocessing steps and rationale.
- [ ] Output preprocessed datasets and a preprocessing report in `assets/preprocessing`.
- [ ] Generate visualizations of feature distributions before and after preprocessing.
- [ ] Create a summary report detailing preprocessing steps and their impact on data quality.

## Model Training

- [ ] Try overweighting ExpansionRX data during training to improve performance on the target dataset.
- [ ] Explore training objectives that prioritize leaderboard metrics (e.g., MAE on individual tasks and MA-RAE across all tasks).
- [ ] Implement and train baseline models (Random Forest, XGBoost, LightGBM, Chemprop Multitask, CheMeleon, KERMT, ChemBERTa)
- [ ] Fine-tune models using pre-trained weights where applicable (CheMeleon, KERMT, ChemBERTa).
- [ ] Use cross-validation to assess model robustness (5x5 using useful_rdkit_utils with Taylor-Butina clustering, time-based splits on ExpansionRX dataset).
- [ ] Perform hyperparameter tuning using grid search or Bayesian optimization with `ray[tune]` to optimize model performance.
- [ ] Document model architectures, training procedures, and hyperparameter choices.
- [ ] Save trained models and training logs in `assets/models`.
- [ ] Generate training curves and performance metrics visualizations using `mlflow`.
- [ ] Create a summary report comparing model architectures and training strategies.
- [ ] Implement ensemble methods to combine predictions from multiple models for improved performance.
- [ ] Evaluate ensemble performance against individual models and document findings.
