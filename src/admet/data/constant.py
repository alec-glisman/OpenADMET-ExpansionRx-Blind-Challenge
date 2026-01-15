"""ADMET dataset registry and lightweight numeric transform helpers.

This module centralizes dataset configuration metadata (source type, URI and
canonical output filename) and exposes small numeric transformation helpers
used during dataset harmonization and exploratory analysis.

Contents
--------
Classes
    DatasetInfo : Typed dict structure for per-dataset metadata.

Constants
    DEFAULT_DATASET_DIR : Root path for downloaded raw datasets.
    DATASETS            : Mapping of lowercase dataset name -> DatasetInfo.
    COLS_WITH_UNITS     : Mapping of column name -> display units string.
    TRANSFORMATIONS     : Mapping of transformation label -> callable.
    CANONICAL_TARGET_COLUMNS : List of canonical target column names.
    COLUMN_ALIASES      : Mapping of alternative column names -> canonical names.

Notes
-----
The dataset list is partially populated dynamically from TDC benchmark names.
All dynamically retrieved names are converted to lowercase for uniform CLI
and API usage.
"""

from typing import Any, Callable, Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# EDA constants and lightweight numeric transformations
# ---------------------------------------------------------------------------

#: Column names with units for ExpansionRX and related datasets.
COLS_WITH_UNITS: Dict[str, str] = {
    "Molecule Name": "(None)",
    "LogD": "(None)",
    "KSOL": "(uM)",
    "HLM CLint": "(mL/min/kg)",
    "MLM CLint": "(mL/min/kg)",
    "Caco-2 Permeability Papp A>B": "(10^-6 cm/s)",
    "Caco-2 Permeability Efflux": "(None)",
    "MPPB": "(% unbound)",
    "MBPB": "(% unbound)",
    "MGMB": "(% unbound)",
}

# ---------------------------------------------------------------------------
# Canonical column names and aliases for multi-format CSV support
# ---------------------------------------------------------------------------

#: Canonical target column names used throughout the pipeline.
CANONICAL_TARGET_COLUMNS: List[str] = [
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

#: Mapping of alternative column names (from different CSV formats) to canonical names.
#: Handles LaTeX notation ($>$, $^{-6}$), unit annotations, and various header styles.
COLUMN_ALIASES: Dict[str, str] = {
    # Format 1: LaTeX-escaped headers (Log Caco-2 Permeability Papp A$>$B)
    "Log Caco-2 Permeability Papp A$>$B": "Log Caco-2 Permeability Papp A>B",
    # Format 2: Raw value headers with units (challenge test set format)
    # These map to canonical Log-prefixed names (data is assumed pre-transformed)
    "LogD (None)": "LogD",
    "KSOL (uM)": "Log KSOL",
    "HLM CLint (mL/min/kg)": "Log HLM CLint",
    "MLM CLint (mL/min/kg)": "Log MLM CLint",
    "Caco-2 Permeability Papp A$>$B (10$^{-6}$ cm/s)": "Log Caco-2 Permeability Papp A>B",
    "Caco-2 Permeability Efflux (None)": "Log Caco-2 Permeability Efflux",
    "MPPB ( pct unbound)": "Log MPPB",
    "MBPB ( pct unbound)": "Log MBPB",
    "MGMB ( pct unbound)": "Log MGMB",
    # Format 2 variant: without parenthetical units
    "KSOL": "Log KSOL",
    "HLM CLint": "Log HLM CLint",
    "MLM CLint": "Log MLM CLint",
    "Caco-2 Permeability Papp A$>$B": "Log Caco-2 Permeability Papp A>B",
    "Caco-2 Permeability Efflux": "Log Caco-2 Permeability Efflux",
    "MPPB": "Log MPPB",
    "MBPB": "Log MBPB",
    "MGMB": "Log MGMB",
    # Additional SMILES column alias
    "Molecule Name (None)": "Molecule Name",
}

#: Simple numeric transformations used during dataset harmonization.
TRANSFORMATIONS: Dict[str, Callable[..., Any]] = {
    "None": lambda x: x,
    "log10(x)": lambda x: np.log10(x) if x > 0 else np.log10(1e-6),
    # For CLint and permeability: replace 0s with half of smallest non-zero value
    "log10_clint_perm": lambda series: series.apply(
        lambda x: np.log10(x) if x > 0 else np.log10(series[series > 0].min() / 2)
    ),
    # For PPB endpoints: set 0s to 10^(-6) before log10
    "log10_ppb": lambda series: series.apply(lambda x: np.log10(x) if x > 0 else np.log10(1e-6)),
    "e^(x)": lambda x: np.exp(x),
    "10^(x+6)": lambda x: np.power(10.0, x + 6.0),
    "10^(x)": lambda x: np.power(10.0, x),
    "10^(x+2)": lambda x: np.power(10.0, x) * 1.0e2,
    "10^(x-2)": lambda x: np.power(10.0, x) * 1.0e-2,
    "10^(x); 1/g to 1/kg": lambda x: np.power(0.0, x) * 1.0e3,
    "10^(x); 1/kg to 1/g": lambda x: np.power(10.0, x) * 1.0e-3,
    # requires MW in g/mol; converts ug/mL to uM
    "ug/mL to uM": lambda x, mw: (x / mw) * 1.0e3 if (mw is not None and mw > 0) else float("nan"),
    # unit conversions
    "nM to uM": lambda x: x / 1000.0,
    "g to kg": lambda x: x / 1000.0,
    "kg to g": lambda x: x * 1000.0,
}


__all__ = [
    "CANONICAL_TARGET_COLUMNS",
    "COLS_WITH_UNITS",
    "COLUMN_ALIASES",
    "TRANSFORMATIONS",
]
