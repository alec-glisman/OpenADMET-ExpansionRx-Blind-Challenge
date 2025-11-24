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

from typing import Sequence, Dict, Any
from pathlib import Path

from lightgbm import LGBMRegressor
from rdkit import Chem  # type: ignore[import-not-found]
from rdkit.Chem import rdFingerprintGenerator  # type: ignore[import-not-found]
import numpy as np
import pandas as pd  # type: ignore[import-not-found]
import useful_rdkit_utils as uru  # type: ignore[import-not-found]


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
        self.endpoints: Sequence[str] = [y_col]
        self.input_type: str = "descriptors"
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
        # Return 2D array to align with multi-endpoint expectations
        return np.atleast_2d(pred).T

    def validate(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        """Convenience wrapper: fit then predict (simple hold-out evaluation)."""
        self.fit(train)
        return self.predict(test)

    def save(self, path: str) -> None:
        """Persist model to path using LightGBM native saver.

        Parameters
        ----------
        path : str
            Destination to save model artifacts.
        """
        # Ensure parent directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lgbm.booster_.save_model(str(path))

    @classmethod
    def load(cls, path: str) -> "LGBMPropWrapper":
        raise NotImplementedError("Loading LGBMPropWrapper from disk is not implemented.")

    def get_config(self) -> Dict[str, Any]:
        return {"type": "lgbm_prop", "y_col": self.y_col}

    def get_metadata(self) -> Dict[str, Any]:
        return {"n_estimators": getattr(self.lgbm, "n_estimators", None)}


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
        self.endpoints: Sequence[str] = [y_col]
        self.input_type: str = "fingerprint"
        self.fp_name = "fp"
        self.fg = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    def fit(self, train: pd.DataFrame) -> None:
        """Fit model on training DataFrame generating Morgan fingerprints."""
        mol_fn = getattr(Chem, "MolFromSmiles")  # type: ignore[attr-defined]
        train["mol"] = train.SMILES.apply(mol_fn)  # type: ignore[call-arg]
        train[self.fp_name] = train.mol.apply(self.fg.GetCountFingerprintAsNumPy)
        self.lgbm.fit(np.stack(train[self.fp_name]), train[self.y_col])

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        """Predict target for provided test DataFrame using fingerprints."""
        mol_fn = getattr(Chem, "MolFromSmiles")  # type: ignore[attr-defined]
        test["mol"] = test.SMILES.apply(mol_fn)  # type: ignore[call-arg]
        test[self.fp_name] = test.mol.apply(self.fg.GetCountFingerprintAsNumPy)
        pred = self.lgbm.predict(np.stack(np.stack(test[self.fp_name])))
        return np.atleast_2d(pred).T

    def validate(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        """Convenience wrapper: fit then predict."""
        self.fit(train)
        return self.predict(test)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lgbm.booster_.save_model(str(path))

    @classmethod
    def load(cls, path: str) -> "LGBMMorganCountWrapper":
        raise NotImplementedError("Loading LGBMMorganCountWrapper from disk is not implemented.")

    def get_config(self) -> Dict[str, Any]:
        return {"type": "lgbm_morgan_count", "y_col": self.y_col}

    def get_metadata(self) -> Dict[str, Any]:
        return {"n_estimators": getattr(self.lgbm, "n_estimators", None)}


__all__ = ["LGBMPropWrapper", "LGBMMorganCountWrapper"]
