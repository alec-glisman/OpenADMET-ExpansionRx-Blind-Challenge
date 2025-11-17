# Planned Model Training Steps

## High Level Overview

1. **Load pre-split datasets**: Load the datasets that have been split using various clustering methods (random, scaffold-based, k-means, Butina).
   - Input: Pre-split datasets from previous steps. Datasets are stored in a structured directory tree: `assets/dataset/splits/{quality}_quality/{split_method}/`. Non-temporal splits also include N-split and K-fold information with subdirectories `split_{n}/fold_{k}/`. All datasets are in Hugging Face `Dataset` format.
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
