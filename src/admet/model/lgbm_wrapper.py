"""LightGBM wrapper helpers for property and Morgan fingerprint models.

This module provides thin convenience classes for quickly training single-
endpoint LightGBM regressors using either RDKit descriptor sets or Morgan
count fingerprints. These helpers are primarily intended for exploratory
baselines and notebooks, trading flexibility for brevity.

Contents
--------
Classes
    LGBMPropWrapper        : LightGBM model on RDKit descriptor vectors.
    LGBMMorganCountWrapper : LightGBM model on Morgan count fingerprints.
"""

from __future__ import annotations

from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import pandas as pd
import useful_rdkit_utils as uru


class LGBMPropWrapper:
    """LightGBM single-endpoint regressor using RDKit descriptor features.

    Parameters
    ----------
    y_col : str
        Target column name in provided DataFrames.
    """

    def __init__(self, y_col: str) -> None:
        self.lgbm = LGBMRegressor(verbose=-1)
        self.y_col = y_col
        self.rdkit_desc = uru.RDKitDescriptors(hide_progress=True)
        self.desc_name = "desc"

    def fit(self, train: pd.DataFrame) -> None:
        """Fit model on training DataFrame.

        Computes RDKit descriptors from ``SMILES`` column and trains LightGBM.
        """
        train[self.desc_name] = self.rdkit_desc.pandas_smiles(train.SMILES).values.tolist()
        self.lgbm.fit(np.stack(train[self.desc_name]), train[self.y_col])

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        """Predict target for provided test DataFrame.

        Returns
        -------
        numpy.ndarray
            1D array of predictions ordered as rows in ``test``.
        """
        test[self.desc_name] = self.rdkit_desc.pandas_smiles(test.SMILES).values.tolist()
        pred = self.lgbm.predict(np.stack(np.stack(test[self.desc_name])))
        return pred

    def validate(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        """Convenience wrapper: fit then predict (simple hold-out evaluation)."""
        self.fit(train)
        return self.predict(test)


class LGBMMorganCountWrapper:
    """LightGBM single-endpoint regressor using Morgan count fingerprints.

    Parameters
    ----------
    y_col : str
        Target column name in provided DataFrames.
    radius : int, optional
        Morgan fingerprint radius (default 2).
    fp_size : int, optional
        Bit size of fingerprint (default 1024).
    """

    def __init__(self, y_col: str, radius: int = 2, fp_size: int = 1024) -> None:
        self.lgbm = LGBMRegressor(verbose=-1)
        self.y_col = y_col
        self.fp_name = "fp"
        self.fg = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    def fit(self, train: pd.DataFrame) -> None:
        """Fit model on training DataFrame generating Morgan fingerprints."""
        train["mol"] = train.SMILES.apply(Chem.MolFromSmiles)
        train[self.fp_name] = train.mol.apply(self.fg.GetCountFingerprintAsNumPy)
        self.lgbm.fit(np.stack(train[self.fp_name]), train[self.y_col])

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        """Predict target for provided test DataFrame using fingerprints."""
        test["mol"] = test.SMILES.apply(Chem.MolFromSmiles)
        test[self.fp_name] = test.mol.apply(self.fg.GetCountFingerprintAsNumPy)
        pred = self.lgbm.predict(np.stack(np.stack(test[self.fp_name])))
        return pred

    def validate(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        """Convenience wrapper: fit then predict."""
        self.fit(train)
        return self.predict(test)


__all__ = ["LGBMPropWrapper", "LGBMMorganCountWrapper"]
