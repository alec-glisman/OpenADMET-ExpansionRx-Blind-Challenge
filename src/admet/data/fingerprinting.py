"""Molecular fingerprint generation and feature extraction utilities.

This module provides a high-level class for generating RDKit Morgan
fingerprints (optionally with chirality and count simulation) and expanding
them into per-bit feature columns suitable for model training.

Contents
--------
Classes
    MorganFingerprintGenerator : Generates and inserts fingerprint features.

Public API
----------
Only the class ``MorganFingerprintGenerator`` is exported; helper methods are
kept private to avoid namespace clutter.
"""

import logging

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

logger = logging.getLogger(__name__)


class MorganFingerprintGenerator:
    """
    Generate Morgan fingerprints for molecules and expand them into feature columns.

    This class encapsulates the process of converting SMILES strings to Morgan
    fingerprints and expanding them into individual numerical feature columns
    for use in machine learning pipelines.

    Parameters
    ----------
    radius : int, optional
        Radius of the Morgan fingerprint (default: 3).
    count_simulation : bool, optional
        Whether to use count simulation (default: False).
    include_chirality : bool, optional
        Whether to include chirality information (default: False).
    fp_size : int, optional
        Size of the fingerprint in bits (default: 2048).

    Attributes
    ----------
    _fpgen : rdkit.Chem.rdFingerprintGenerator.FingerprintGenerator
        The underlying RDKit fingerprint generator.
    """

    def __init__(
        self,
        radius: int = 3,
        count_simulation: bool = False,
        include_chirality: bool = False,
        fp_size: int = 2048,
    ) -> None:
        """
        Initialize the Morgan fingerprint generator with specified parameters.

        Parameters
        ----------
        radius : int, optional
            Radius of the Morgan fingerprint (default: 3).
        count_simulation : bool, optional
            Whether to use count simulation (default: False).
        include_chirality : bool, optional
            Whether to include chirality information (default: False).
        fp_size : int, optional
            Size of the fingerprint in bits (default: 2048).
        """
        self.radius = radius
        self.count_simulation = count_simulation
        self.include_chirality = include_chirality
        self.fp_size = fp_size

        self._fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            countSimulation=count_simulation,
            includeChirality=include_chirality,
            fpSize=fp_size,
        )
        logger.debug(
            f"Initialized Morgan fingerprint generator: "
            f"radius={radius}, count_simulation={count_simulation}, "
            f"include_chirality={include_chirality}, fp_size={fp_size}"
        )

    def _smiles_to_mol(self, smiles: str) -> Chem.Mol | None:
        """
        Convert SMILES string to RDKit molecule object.

        Parameters
        ----------
        smiles : str
            SMILES representation of a molecule.

        Returns
        -------
        rdkit.Chem.Mol or None
            RDKit molecule object, or None if conversion fails.
        """
        return Chem.MolFromSmiles(smiles)

    def _mol_to_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """
        Convert molecule to fingerprint array.

        Parameters
        ----------
        mol : rdkit.Chem.Mol
            RDKit molecule object.

        Returns
        -------
        np.ndarray
            Fingerprint as a numpy array.
        """
        if mol is None:
            return np.zeros(self.fp_size, dtype=np.uint8)
        return self._fpgen.GetCountFingerprintAsNumPy(mol)

    def calculate_fingerprints(self, smiles_series: pd.Series) -> pd.DataFrame:
        """
        Calculate Morgan fingerprints for a series of SMILES strings.

        Parameters
        ----------
        smiles_series : pd.Series
            Series containing SMILES strings.

        Returns
        -------
        pd.DataFrame
            DataFrame with fingerprint features as individual columns.
        """
        logger.debug(f"Calculating fingerprints for {len(smiles_series)} molecules")

        # Convert SMILES to molecules
        mols = smiles_series.apply(self._smiles_to_mol)

        # Generate fingerprints
        fingerprints = mols.apply(self._mol_to_fingerprint)

        # Stack fingerprints into array
        fp_array = np.vstack(fingerprints.values)

        # Convert to DataFrame with column names
        fp_columns = [f"Morgan_FP_{i}" for i in range(fp_array.shape[1])]
        fp_df = pd.DataFrame(fp_array, columns=fp_columns, dtype=np.uint8)

        logger.debug(f"Generated {fp_df.shape[1]} fingerprint features")
        return fp_df

    def add_fingerprints_to_dataframe(
        self, df: pd.DataFrame, smiles_column: str = "SMILES", insertion_index: int = 3
    ) -> pd.DataFrame:
        """
        Add fingerprint columns to a dataframe.

        Fingerprints are calculated from the SMILES column and inserted
        at the specified position while maintaining other column order.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with SMILES column.
        smiles_column : str, optional
            Name of the SMILES column (default: "SMILES").
        insertion_index : int, optional
            Column index where fingerprints should be inserted (default: 3).

        Returns
        -------
        pd.DataFrame
            Dataframe with fingerprint columns added and reordered.
        """
        if smiles_column not in df.columns:
            raise ValueError(f"SMILES column '{smiles_column}' not found in dataframe")

        logger.debug(f"Adding fingerprints to dataframe with {len(df)} rows")

        # Calculate fingerprints
        fp_df = self.calculate_fingerprints(df[smiles_column])

        # Concatenate with original dataframe
        result_df = pd.concat([df.reset_index(drop=True), fp_df.reset_index(drop=True)], axis=1)

        # Reorder columns to place fingerprints after specified index
        cols = result_df.columns.tolist()
        # Move Morgan_FP columns to the insertion_index position
        fp_cols = [col for col in cols if col.startswith("Morgan_FP_")]
        non_fp_cols = [col for col in cols if not col.startswith("Morgan_FP_")]

        # Insert fingerprint columns at the specified position
        reordered_cols = non_fp_cols[:insertion_index] + fp_cols + non_fp_cols[insertion_index:]

        logger.debug(f"Reordered columns: {len(reordered_cols)} total columns")
        return result_df[reordered_cols]

    __all__ = ["MorganFingerprintGenerator"]
