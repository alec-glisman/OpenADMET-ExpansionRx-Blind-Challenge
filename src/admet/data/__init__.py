"""Data subpackage: dataset constants and chemistry utilities."""

from .constants import (
    DATASETS,
    DEFAULT_DATASET_DIR,
    DatasetInfo,
    COLS_WITH_UNITS,
    TRANSFORMATIONS,
    cols_with_units,
    transformations,
)
from .chem import (
    canonicalize_smiles,
    compute_molecular_properties,
    parallel_canonicalize_smiles,
)

__all__ = [
    # datasets
    "DatasetInfo",
    "DEFAULT_DATASET_DIR",
    "DATASETS",
    # constants
    "COLS_WITH_UNITS",
    "TRANSFORMATIONS",
    "cols_with_units",
    "transformations",
    # chem
    "canonicalize_smiles",
    "parallel_canonicalize_smiles",
    "compute_molecular_properties",
]
