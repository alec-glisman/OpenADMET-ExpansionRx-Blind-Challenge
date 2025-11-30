"""
Chemprop model wrapper for ADMET property prediction
=====================================================

This module provides a high-level wrapper around the Chemprop library for
training and inference of molecular property prediction models using
message-passing neural networks (MPNNs).

Key components
--------------
- ``ChempropHyperparams``: Dataclass holding all model hyperparameters including
  optimization settings, message passing configuration, and FFN architecture choices.
- ``ChempropModel``: Main class that handles data preparation, model construction,
  training via PyTorch Lightning, and prediction.

Supported FFN architectures
---------------------------
- ``regression``: Standard multi-task regression FFN (default)
- ``mixture_of_experts``: Mixture of experts regression FFN
- ``branched``: Branched FFN with shared trunk and task-specific heads

Example usage
-------------
>>> model = ChempropModel(
...     df_train=train_df,
...     df_validation=val_df,
...     smiles_col="smiles",
...     target_cols=["logD", "solubility"],
... )
>>> model.fit()
>>> predictions = model.predict(test_df)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, models, nn
from chemprop.nn import RegressionFFN
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from admet.model.chemprop.curriculum import CurriculumCallback, CurriculumState
from admet.model.chemprop.ffn import BranchedFFN, MixtureOfExpertsRegressionFFN

# TODO: implement sampling-based curriculum in addition to weighting
# TODO: implement different FFN types (e.g. MixtureOfExpertsRegressionFFN, BranchedFFN)
# TODO: hyperparameter optimize ChempropHyperparams
# TODO: think about incorporating task weights: _tasks = [1.018, 1.000, 1.364, 1.134, 2.377, 2.373, 3.939, 5.259, 23.099]


class CriterionName(str, Enum):
    MSE = "MSE"
    MAE = "MAE"
    RMSE = "RMSE"
    SID = "SID"
    BCE = "BCE"
    R2 = "R2Score"
    CROSS_ENTROPY = "CrossEntropy"
    DIRICHLET = "Dirichlet"
    EVIDENTIAL = "Evidential"
    MVE = "MVE"
    QUANTILE = "Quantile"

    @classmethod
    def resolve(cls, name: str) -> Any:
        try:
            key = cls(name)
        except ValueError as exc:
            raise ValueError(f"Unsupported criterion: {name}") from exc
        return _criterion_from_enum(key)


def _criterion_from_enum(criterion: CriterionName) -> Any:
    attr = getattr(nn.metrics, criterion.value, None)
    if attr is None:
        raise ValueError(f"Criterion class not found for '{criterion.value}'")
    if callable(attr):
        try:
            return attr()
        except TypeError as exc:
            raise TypeError(f"Unable to instantiate criterion '{criterion.value}'") from exc
    return attr


@dataclass
class ChempropHyperparams:
    """
    Hyperparameters for Chemprop MPNN model.

    This dataclass encapsulates all configurable hyperparameters for training
    a Chemprop message-passing neural network, including optimization settings,
    message passing architecture, and feed-forward network configuration.

    Attributes
    ----------
    init_lr : float, default=0.0001
        Initial learning rate for the optimizer.
    max_lr : float, default=0.001
        Maximum learning rate during warmup.
    final_lr : float, default=0.0001
        Final learning rate after decay.
    warmup_epochs : int, default=3
        Number of epochs for learning rate warmup.
    patience : int, default=15
        Number of epochs without improvement before early stopping.
    max_epochs : int, default=80
        Maximum number of training epochs.
    batch_size : int, default=16
        Batch size for training.
    num_workers : int, default=0
        Number of data loading workers.
    seed : int, default=12345
        Random seed for reproducibility.
    depth : int, default=3
        Number of message passing iterations.
    message_hidden_dim : int, default=300
        Hidden dimension for message passing layers.
    num_layers : int, default=2
        Number of layers in the feed-forward network.
    hidden_dim : int, default=300
        Hidden dimension for FFN layers.
    dropout : float, default=0.0
        Dropout probability.
    criterion : str, default='MSE'
        Loss criterion. Options: 'MSE', 'SID', 'BCE', 'CrossEntropy',
        'Dirichlet', 'Evidential', 'MVE', 'Quantile'.
    ffn_type : str, default='regression'
        Type of feed-forward network. Options: 'regression',
        'mixture_of_experts', 'branched'.
    trunk_n_layers : int, default=1
        Number of trunk layers for branched FFN.
    trunk_hidden_dim : int, default=300
        Hidden dimension for trunk layers in branched FFN.
    n_experts : int, default=3
        Number of experts for mixture of experts FFN.
    batch_norm : bool, default=True
        Whether to use batch normalization in MPNN.
    """

    # Optimization
    init_lr: float = 0.0001
    max_lr: float = 0.001
    final_lr: float = 0.0001
    warmup_epochs: int = 3
    patience: int = 15
    max_epochs: int = 80
    batch_size: int = 16
    num_workers: int = 0
    seed: int = 12345

    # Message passing
    depth: int = 3
    message_hidden_dim: int = 300

    # Feed forward
    num_layers: int = 2
    hidden_dim: int = 300
    dropout: float = 0.0
    criterion: str = "MSE"  # options: 'MSE', 'SID', 'BCE', 'CrossEntropy', 'Dirichlet', 'Evidential', 'MVE', 'Quantile'
    ffn_type: str = "regression"  # options: 'regression', 'mixture_of_experts', 'branched'

    # Branched FFN
    trunk_n_layers: int = 1
    trunk_hidden_dim: int = 300

    # Mixture of Experts FFN
    n_experts: int = 3

    # MPNN
    batch_norm: bool = True


class ChempropModel:
    """
    High-level wrapper for Chemprop MPNN training and inference.

    This class handles the full workflow of preparing data, building the model,
    training with PyTorch Lightning, and generating predictions. It supports
    multiple FFN architectures and automatic target normalization.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training dataframe containing SMILES and target columns.
    df_validation : pandas.DataFrame or None, optional
        Validation dataframe for early stopping and model selection.
    df_test : pandas.DataFrame or None, optional
        Test dataframe (not used during training).
    smiles_col : str, default='smiles'
        Column name containing SMILES strings.
    target_cols : list[str], default=[]
        List of target column names for multi-task prediction.
    target_weights : list[float], default=[]
        Per-task weights for the loss function. If empty, all tasks
        are weighted equally.
    output_dir : pathlib.Path or None, optional
        Directory to save model checkpoints.
    progress_bar : bool, default=False
        Whether to show training progress bar.
    hyperparams : ChempropHyperparams or None, optional
        Model hyperparameters. If None, defaults are used.

    Attributes
    ----------
    featurizer : SimpleMoleculeMolGraphFeaturizer
        Molecular graph featurizer.
    mpnn : models.MPNN
        The underlying Chemprop MPNN model.
    trainer : pl.Trainer
        PyTorch Lightning trainer instance.
    scaler : StandardScaler
        Target normalization scaler fitted on training data.
    metrics : list
        Evaluation metrics (RMSE, MAE, R2).

    Examples
    --------
    >>> model = ChempropModel(
    ...     df_train=train_df,
    ...     df_validation=val_df,
    ...     smiles_col="smiles",
    ...     target_cols=["logD", "solubility"],
    ... )
    >>> model.fit()
    >>> predictions = model.predict(test_df)
    """

    metrics = [nn.metrics.MAE(), nn.metrics.RMSE(), nn.metrics.R2Score()]

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_validation: pd.DataFrame | None = None,
        df_test: pd.DataFrame | None = None,
        smiles_col: str = "smiles",
        target_cols: List[str] = [],
        target_weights: List[float] = [],
        output_dir: Path | None = None,
        progress_bar: bool = False,
        hyperparams: ChempropHyperparams | None = None,
    ) -> None:
        """
        Initialize ChempropModel with data and configuration.

        Parameters
        ----------
        df_train : pandas.DataFrame
            Training dataframe containing SMILES and target columns.
        df_validation : pandas.DataFrame or None, optional
            Validation dataframe for early stopping.
        df_test : pandas.DataFrame or None, optional
            Test dataframe (stored but not used during training).
        smiles_col : str, default='smiles'
            Column name containing SMILES strings.
        target_cols : list[str], default=[]
            List of target column names.
        target_weights : list[float], default=[]
            Per-task loss weights.
        output_dir : pathlib.Path or None, optional
            Directory for saving checkpoints.
        progress_bar : bool, default=False
            Whether to display progress bar during training.
        hyperparams : ChempropHyperparams or None, optional
            Model hyperparameters.
        """
        self.smiles_col: str = smiles_col
        self.target_cols: List[str] = target_cols
        self.target_weights: List[float] = target_weights
        self.output_dir: Path | None = output_dir
        self.progress_bar: bool = progress_bar
        self.hyperparams: ChempropHyperparams = hyperparams or ChempropHyperparams()

        if self.target_weights == []:
            self.target_weights = [1.0] * len(self.target_cols)

        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.agg = nn.NormAggregation()
        self.mp = nn.BondMessagePassing(
            d_h=self.hyperparams.message_hidden_dim,
            depth=self.hyperparams.depth,
            dropout=self.hyperparams.dropout,
        )
        self.ffn: Any = None
        self.mpnn: Any = None
        self.trainer: Any = None

        self.dataframes: Dict[str, pd.DataFrame | None] = {
            "train": df_train,
            "validation": df_validation,
            "test": df_test,
        }
        self.dataloaders: Dict[str, Any] = {
            "train": None,
            "validation": None,
            "test": None,
        }
        self.scaler: Any = None
        self.transform: Any = None

        self._prepare_dataloaders()
        self._prepare_model()
        self._prepare_trainer()

    def _prepare_dataloaders(self) -> None:
        """
        Prepare PyTorch dataloaders for train/validation/test splits.

        This method converts dataframes to Chemprop MoleculeDatasets,
        fits a target scaler on training data, and creates dataloaders
        for each split.
        """
        datapoints = {}
        datasets = {}
        for split in ["train", "validation", "test"]:
            df = self.dataframes[split]
            if df is None:
                continue
            smis = df.loc[:, self.smiles_col].values
            ys = df.loc[:, self.target_cols].values

            datapoints[split] = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
            datasets[split] = data.MoleculeDataset(datapoints[split], self.featurizer)

            if split == "train":
                self.scaler = datasets[split].normalize_targets()
                self.transform = nn.transforms.UnscaleTransform.from_standard_scaler(self.scaler)

            elif split == "validation":
                datasets[split].normalize_targets(self.scaler)

            self.dataloaders[split] = data.build_dataloader(
                datasets[split],
                batch_size=self.hyperparams.batch_size,
                num_workers=self.hyperparams.num_workers,
                shuffle=(split == "train"),
                seed=self.hyperparams.seed,
            )

    def _prepare_model(self) -> None:
        """
        Build the MPNN model based on hyperparameters.

        Constructs the feed-forward network (FFN) based on ``ffn_type``
        hyperparameter and assembles the full MPNN with message passing,
        aggregation, and prediction components.

        Raises
        ------
        ValueError
            If ``ffn_type`` is not one of 'regression', 'mixture_of_experts',
            or 'branched'.
        """
        criterion = CriterionName.resolve(self.hyperparams.criterion)
        if self.hyperparams.ffn_type == "mixture_of_experts":
            self.ffn = MixtureOfExpertsRegressionFFN(
                n_tasks=len(self.target_cols),
                n_experts=self.hyperparams.n_experts,
                input_dim=self.hyperparams.message_hidden_dim,
                hidden_dim=self.hyperparams.hidden_dim,
                n_layers=self.hyperparams.num_layers,
                dropout=self.hyperparams.dropout,
                criterion=criterion,
                task_weights=self.target_weights,
                output_transform=self.transform,
            )
        elif self.hyperparams.ffn_type == "branched":
            self.ffn = BranchedFFN(
                task_groups=[[i] for i in range(len(self.target_cols))],
                n_tasks=len(self.target_cols),
                input_dim=self.hyperparams.message_hidden_dim,
                hidden_dim=self.hyperparams.hidden_dim,
                trunk_n_layers=self.hyperparams.trunk_n_layers,
                trunk_hidden_dim=self.hyperparams.trunk_hidden_dim,
                trunk_dropout=self.hyperparams.dropout,
                criterion=criterion,
                task_weights=self.target_weights,
                output_transform=self.transform,
            )
        elif self.hyperparams.ffn_type == "regression":
            self.ffn = RegressionFFN(
                n_tasks=len(self.target_cols),
                input_dim=self.hyperparams.message_hidden_dim,
                hidden_dim=self.hyperparams.hidden_dim,
                n_layers=self.hyperparams.num_layers,
                dropout=self.hyperparams.dropout,
                criterion=criterion,
                task_weights=self.target_weights,
                output_transform=self.transform,
            )
        else:
            raise ValueError(f"Unsupported ffn_type: {self.hyperparams.ffn_type}")

        self.mpnn = models.MPNN(
            message_passing=self.mp,
            agg=self.agg,
            predictor=self.ffn,
            batch_norm=self.hyperparams.batch_norm,
            metrics=self.metrics,
            warmup_epochs=self.hyperparams.warmup_epochs,
            init_lr=self.hyperparams.init_lr,
            max_lr=self.hyperparams.max_lr,
            final_lr=self.hyperparams.final_lr,
        )

    def _prepare_trainer(self) -> None:
        """
        Configure PyTorch Lightning trainer with callbacks.

        Sets up model checkpointing (saves best and last checkpoints)
        and early stopping based on validation loss.
        """
        torch.set_float32_matmul_precision("medium")

        checkpointing = ModelCheckpoint(
            dirpath=self.output_dir,
            filename="best-{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_last=True,
        )
        earlystopping = EarlyStopping(
            monitor="val_loss",
            patience=self.hyperparams.patience,
            mode="min",
        )

        self.trainer = pl.Trainer(
            logger=True,
            enable_checkpointing=True,
            enable_progress_bar=self.progress_bar,
            accelerator="auto",
            devices=1,
            max_epochs=self.hyperparams.max_epochs,
            callbacks=[checkpointing, earlystopping],
        )

    def fit(self) -> bool:
        """
        Train the MPNN model.

        Runs training using the PyTorch Lightning trainer with the
        configured train and validation dataloaders. Handles keyboard
        interrupts gracefully, allowing the model to be used for
        prediction even if training is interrupted.

        Returns
        -------
        bool
            True if training completed normally, False if interrupted.
        """
        try:
            self.trainer.fit(
                self.mpnn,
                train_dataloaders=self.dataloaders["train"],
                val_dataloaders=self.dataloaders["validation"],
            )
            return True
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user. Model state preserved.")
            print("   You can still use model.predict() with the current weights.")
            return False

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for a dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe containing SMILES column. Target columns
            are optional (used if present for dataset creation).

        Returns
        -------
        pandas.DataFrame
            Predictions with columns named after target columns.
        """
        smis = df.loc[:, self.smiles_col].values

        # gracefully handle labelled/unlabelled data
        try:
            ys = df.loc[:, self.target_cols].values
            datapoints = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
        except KeyError:
            datapoints = [data.MoleculeDatapoint.from_smi(smi) for smi in smis]

        dataset = data.MoleculeDataset(datapoints, self.featurizer)
        dataloader = data.build_dataloader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
        )
        results = self.trainer.predict(self.mpnn, dataloaders=dataloader)

        # Collect predictions from batches
        # results is a list of prediction tensors from each batch
        all_preds = []
        for batch_preds in results:
            # batch_preds is a tensor of shape [batch_size, n_tasks]
            if hasattr(batch_preds, "cpu"):
                preds = batch_preds.cpu().numpy()
            elif isinstance(batch_preds, dict) and "preds" in batch_preds:
                preds = batch_preds["preds"].cpu().numpy()
            else:
                preds = batch_preds
            all_preds.append(preds)

        all_preds = np.vstack(all_preds)
        pred_df = pd.DataFrame(all_preds, columns=[f"{t}" for t in self.target_cols])
        return pred_df

    def run(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience method to generate predictions on test data.

        Parameters
        ----------
        df_test : pandas.DataFrame
            Test dataframe containing SMILES column.

        Returns
        -------
        pandas.DataFrame
            Predictions with columns named after target columns.
        """
        preds = self.predict(df_test)
        return preds


def example_usage():
    """
    Demonstrate example usage of ChempropModel.

    Loads train/val/test CSVs, creates a model, trains it,
    and generates predictions. This function serves as a
    template for using the ChempropModel class.
    """
    # Example usage of ChempropModel
    df_train = pd.read_csv(
        "assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data/split_0/fold_0/train.csv",
        low_memory=False,
    )
    df_validation = pd.read_csv(
        "assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data/split_0/fold_0/validation.csv",
        low_memory=False,
    )
    df_test = pd.read_csv(
        "assets/dataset/set/local_test.csv",
        low_memory=False,
    )

    smiles_col = "SMILES"
    target_cols = [
        "LogD",
        "Log KSOL",
        "Log HLM CLint",
        "Log MLM CLint",
        "Log Caco-2 Permeability Papp A>B",
        "Log Caco-2 Permeability Efflux",
        "Log MPPB",
        "Log MBPB",
        "Log MGMB",
    ]
    output_dir = Path("./temp")
    hyperparams = ChempropHyperparams()

    model = ChempropModel(
        df_train=df_train,
        df_validation=df_validation,
        df_test=df_test,
        smiles_col=smiles_col,
        target_cols=target_cols,
        output_dir=output_dir,
        progress_bar=True,
        hyperparams=hyperparams,
    )

    model.fit()
    predictions = model.predict(df_test)
    print(predictions)


if __name__ == "__main__":
    example_usage()
