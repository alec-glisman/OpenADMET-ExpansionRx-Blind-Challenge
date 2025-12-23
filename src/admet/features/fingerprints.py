"""Fingerprint and descriptor generation for molecules.

This module provides a unified interface for generating molecular fingerprints
and descriptors using RDKit and mordred-community.

Supported fingerprint types:
- Morgan: Circular fingerprints (default: radius=2, n_bits=2048)
- RDKit: Path-based fingerprints
- MACCS: Fixed 167-bit structural keys
- Mordred: Molecular descriptors (~1800 descriptors)

Example:
    >>> from admet.features.fingerprints import FingerprintGenerator
    >>> from admet.model.config import FingerprintConfig
    >>> config = FingerprintConfig(type="morgan")
    >>> generator = FingerprintGenerator(config)
    >>> features = generator.generate(["CCO", "CCCO", "c1ccccc1"])
    >>> features.shape
    (3, 2048)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors

if TYPE_CHECKING:
    from rdkit.Chem import Mol

    from admet.model.config import FingerprintConfig

logger = logging.getLogger(__name__)


class FingerprintGenerator:
    """Unified fingerprint and descriptor generator for molecules.

    Generates molecular fingerprints or descriptors from SMILES strings.
    Supports multiple fingerprint types with configurable parameters.

    Parameters
    ----------
    config : FingerprintConfig
        Configuration specifying fingerprint type and parameters.

    Attributes
    ----------
    config : FingerprintConfig
        The fingerprint configuration.
    fingerprint_dim : int
        Dimensionality of the generated fingerprint/descriptor vectors.

    Examples
    --------
    Generate Morgan fingerprints:

    >>> from admet.model.config import FingerprintConfig
    >>> config = FingerprintConfig(type="morgan")
    >>> gen = FingerprintGenerator(config)
    >>> fps = gen.generate(["CCO", "CCCO"])
    >>> fps.shape
    (2, 2048)

    Generate MACCS keys:

    >>> config = FingerprintConfig(type="maccs")
    >>> gen = FingerprintGenerator(config)
    >>> fps = gen.generate(["CCO"])
    >>> fps.shape
    (1, 167)
    """

    def __init__(self, config: FingerprintConfig) -> None:
        """Initialize fingerprint generator with configuration.

        Parameters
        ----------
        config : FingerprintConfig
            Configuration specifying fingerprint type and parameters.
        """
        self.config = config
        self._fp_func = self._get_fingerprint_function()
        self._fingerprint_dim: int | None = None

    @property
    def fingerprint_dim(self) -> int:
        """Get the dimensionality of generated fingerprints.

        Returns
        -------
        int
            Number of features per fingerprint.

        Raises
        ------
        ValueError
            If fingerprint type is unknown.
        """
        if self._fingerprint_dim is not None:
            return self._fingerprint_dim

        fp_type = self.config.type
        if fp_type == "morgan":
            self._fingerprint_dim = self.config.morgan.n_bits
        elif fp_type == "rdkit":
            self._fingerprint_dim = self.config.rdkit.n_bits
        elif fp_type == "maccs":
            self._fingerprint_dim = 167
        elif fp_type == "mordred":
            self._fingerprint_dim = self._get_mordred_dim()
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")

        return self._fingerprint_dim

    def _get_mordred_dim(self) -> int:
        """Get mordred descriptor count.

        Returns
        -------
        int
            Number of mordred descriptors.
        """
        try:
            from mordred import Calculator, descriptors

            calc = Calculator(descriptors, ignore_3D=self.config.mordred.ignore_3d)
            return len(calc.descriptors)
        except ImportError:
            logger.warning("mordred-community not installed. " "Install with: pip install mordred-community")
            return 0

    def _get_fingerprint_function(self):
        """Get appropriate fingerprint function based on config.

        Returns
        -------
        callable
            Function that generates fingerprint from RDKit Mol.

        Raises
        ------
        ValueError
            If fingerprint type is unknown.
        """
        fp_type = self.config.type
        if fp_type == "morgan":
            return self._morgan_fp
        elif fp_type == "rdkit":
            return self._rdkit_fp
        elif fp_type == "maccs":
            return self._maccs_fp
        elif fp_type == "mordred":
            return self._mordred_fp
        raise ValueError(f"Unknown fingerprint type: {fp_type}")

    def generate(self, smiles: list[str]) -> np.ndarray:
        """Generate fingerprints for list of SMILES.

        Parameters
        ----------
        smiles : list[str]
            List of SMILES strings.

        Returns
        -------
        np.ndarray
            Feature array of shape (n_molecules, fingerprint_dim).
            Invalid SMILES result in zero vectors.
        """
        fps: list[np.ndarray] = []
        n_invalid = 0

        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smi}")
                fps.append(self._get_null_fp())
                n_invalid += 1
            else:
                fps.append(self._fp_func(mol))

        if n_invalid > 0:
            logger.info(f"Generated fingerprints with {n_invalid}/{len(smiles)} invalid SMILES")

        return np.vstack(fps) if fps else np.array([]).reshape(0, self.fingerprint_dim)

    def generate_single(self, smiles: str) -> np.ndarray:
        """Generate fingerprint for a single SMILES.

        Parameters
        ----------
        smiles : str
            SMILES string.

        Returns
        -------
        np.ndarray
            Fingerprint vector of shape (fingerprint_dim,).
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return self._get_null_fp()
        return self._fp_func(mol)

    def _morgan_fp(self, mol: Mol) -> np.ndarray:
        """Generate Morgan (circular) fingerprint.

        Parameters
        ----------
        mol : Mol
            RDKit molecule object.

        Returns
        -------
        np.ndarray
            Morgan fingerprint as numpy array.
        """
        cfg = self.config.morgan
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=cfg.radius,
            nBits=cfg.n_bits,
            useChirality=cfg.use_chirality,
            useBondTypes=cfg.use_bond_types,
            useFeatures=cfg.use_features,
        )
        return np.array(fp)

    def _rdkit_fp(self, mol: Mol) -> np.ndarray:
        """Generate RDKit (path-based) fingerprint.

        Parameters
        ----------
        mol : Mol
            RDKit molecule object.

        Returns
        -------
        np.ndarray
            RDKit fingerprint as numpy array.
        """
        cfg = self.config.rdkit
        fp = Chem.RDKFingerprint(
            mol,
            minPath=cfg.min_path,
            maxPath=cfg.max_path,
            fpSize=cfg.n_bits,
            branchedPaths=cfg.branched_paths,
        )
        return np.array(fp)

    def _maccs_fp(self, mol: Mol) -> np.ndarray:
        """Generate MACCS keys fingerprint.

        Parameters
        ----------
        mol : Mol
            RDKit molecule object.

        Returns
        -------
        np.ndarray
            MACCS keys as numpy array (167 bits).
        """
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp)

    def _mordred_fp(self, mol: Mol) -> np.ndarray:
        """Generate Mordred molecular descriptors.

        Parameters
        ----------
        mol : Mol
            RDKit molecule object.

        Returns
        -------
        np.ndarray
            Mordred descriptors as numpy array.

        Raises
        ------
        ImportError
            If mordred-community is not installed.
        """
        try:
            from mordred import Calculator, descriptors
            from mordred.error import Error as MordredError
        except ImportError as e:
            raise ImportError(
                "mordred-community is required for Mordred descriptors. " "Install with: pip install mordred-community"
            ) from e

        calc = Calculator(descriptors, ignore_3D=self.config.mordred.ignore_3d)
        result = calc(mol)

        arr = np.array([float(x) if not isinstance(x, MordredError) else np.nan for x in result.values()])

        if self.config.mordred.normalize:
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        return arr

    def _get_null_fp(self) -> np.ndarray:
        """Return null fingerprint for invalid molecules.

        Returns
        -------
        np.ndarray
            Zero vector of appropriate dimension.
        """
        return np.zeros(self.fingerprint_dim)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"FingerprintGenerator(type={self.config.type!r}, " f"dim={self.fingerprint_dim})"
