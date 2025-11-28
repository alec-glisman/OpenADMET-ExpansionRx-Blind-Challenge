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
from dataclasses import dataclass
from typing import Any, Mapping, Optional, cast

import numpy as np
import pandas as pd
from rdkit import Chem  # type: ignore[import-not-found]
from rdkit.Chem import rdFingerprintGenerator  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FingerprintConfig:
    """Configuration for Morgan fingerprint generation."""

    radius: int = 2
    n_bits: int = 1024
    use_counts: bool = True
    include_chirality: bool = False

    @classmethod
    def from_mapping(
        cls,
        cfg: Optional[Mapping[str, object]],
        default: Optional["FingerprintConfig"] = None,
    ) -> "FingerprintConfig":
        """Construct a config from a mapping, falling back to defaults."""
        base = default or cls()
        if cfg is None:
            return base
        return cls(
            radius=int(cast(Any, cfg.get("radius", base.radius))),
            n_bits=int(cast(Any, cfg.get("n_bits", base.n_bits))),
            use_counts=bool(cast(Any, cfg.get("use_counts", base.use_counts))),
            include_chirality=bool(cast(Any, cfg.get("include_chirality", base.include_chirality))),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "radius": self.radius,
            "n_bits": self.n_bits,
            "use_counts": self.use_counts,
            "include_chirality": self.include_chirality,
        }


DEFAULT_FINGERPRINT_CONFIG = FingerprintConfig()


class MorganFingerprintGenerator:
    """
    Generate Morgan fingerprints for molecules and expand them into feature columns.

    This class encapsulates the process of converting SMILES strings to Morgan
    fingerprints and expanding them into individual numerical feature columns
    for use in machine learning pipelines.

    Parameters
    ----------
    radius : int, optional
        Radius of the Morgan fingerprint (default: 2).
    count_simulation : bool, optional
        Whether to use count simulation (default: True).
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
        radius: int = 2,
        count_simulation: bool = False,
        include_chirality: bool = False,
        fp_size: int = 1024,
        *,
        config: Optional[FingerprintConfig] = None,
    ) -> None:
        """
        Initialize the Morgan fingerprint generator with specified parameters.

        Parameters
        ----------
        radius : int, optional
            Radius of the Morgan fingerprint (default: 2).
        count_simulation : bool, optional
            Whether to use count simulation (default: True).
        include_chirality : bool, optional
            Whether to include chirality information (default: False).
        fp_size : int, optional
            Size of the fingerprint in bits (default: 1024).
        """
        cfg = config or FingerprintConfig(
            radius=radius,
            n_bits=fp_size,
            use_counts=count_simulation,
            include_chirality=include_chirality,
        )
        self.radius = cfg.radius
        self.count_simulation = cfg.use_counts
        self.include_chirality = cfg.include_chirality
        self.fp_size = cfg.n_bits
        self.config = cfg

        self._fpgen: Any = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            countSimulation=self.count_simulation,
            includeChirality=self.include_chirality,
            fpSize=self.fp_size,
        )
        logger.debug(
            "Initialized Morgan fingerprint generator: radius=%d, count_simulation=%s, "
            "include_chirality=%s, fp_size=%d",
            self.radius,
            self.count_simulation,
            self.include_chirality,
            self.fp_size,
        )

    def _smiles_to_mol(self, smiles: str) -> Optional[Any]:
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
        return Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]

    def _mol_to_fingerprint(self, mol: Optional[Any]) -> np.ndarray:
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
        return self._fpgen.GetCountFingerprintAsNumPy(mol)  # type: ignore[attr-defined]

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
        logger.debug("Calculating fingerprints for %d molecules", len(smiles_series))

        # Convert SMILES to molecules
        mols = smiles_series.apply(self._smiles_to_mol)

        # Generate fingerprints
        fingerprints = mols.apply(self._mol_to_fingerprint)

        # Stack fingerprints into array
        fp_array = np.vstack(fingerprints.values)

        # Convert to DataFrame with column names
        fp_columns = [f"Morgan_FP_{i}" for i in range(fp_array.shape[1])]
        fp_df = pd.DataFrame(fp_array, columns=fp_columns, dtype=np.uint8)

        logger.debug("Generated %d fingerprint features", fp_df.shape[1])
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

        logger.debug("Adding fingerprints to dataframe with %d rows", len(df))

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

        logger.debug("Reordered columns: %d total columns", len(reordered_cols))
        return result_df[reordered_cols]


__all__ = ["MorganFingerprintGenerator", "FingerprintConfig", "DEFAULT_FINGERPRINT_CONFIG"]
