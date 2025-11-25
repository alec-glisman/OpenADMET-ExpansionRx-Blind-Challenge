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

Notes
-----
The dataset list is partially populated dynamically from TDC benchmark names.
All dynamically retrieved names are converted to lowercase for uniform CLI
and API usage.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, TypedDict

import numpy as np
from tdc import utils


class DatasetInfo(TypedDict):
    """Type definition for dataset information."""

    type: str
    uri: str
    output_file: str


#: Default output directory for downloaded datasets.
DEFAULT_DATASET_DIR = Path(__file__).parents[3] / "assets/dataset/raw"

#: Registry of available datasets keyed by lowercase name.
DATASETS: Mapping[str, DatasetInfo] = {}

# Dynamically add all TDC ADMET_Group datasets
for name in utils.retrieve_benchmark_names("ADMET_Group"):
    DATASETS[name.lower()] = {
        "type": "tdc",
        "uri": name,  # TDC benchmark name
        "output_file": f"tdc_{name.lower()}.csv",
    }

DATASETS["expansion_teaser"] = {
    "type": "huggingface",
    "uri": ("hf://datasets/openadmet/" "openadmet-expansionrx-challenge-teaser/" "expansion_data_teaser.csv"),
    "output_file": "expansion_teaser.csv",
}
DATASETS["expansion"] = {
    "type": "huggingface",
    "uri": ("hf://datasets/openadmet/openadmet-expansionrx-challenge-train-data/expansion_data_train.csv"),
    "output_file": "expansion.csv",
}
DATASETS["antiviral_admet_2025"] = {
    "type": "polaris",
    "uri": "asap-discovery/antiviral-admet-2025-unblinded",
    "output_file": "antiviral_admet_2025_unblinded.csv",
}
DATASETS["biogen_admet"] = {
    "type": "polaris",
    "uri": "biogen/adme-fang-v1",
    "output_file": "biogen_admet.csv",
}

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

#: Simple numeric transformations used during dataset harmonization.
TRANSFORMATIONS: Dict[str, Callable[..., Any]] = {
    "None": lambda x: x,
    "log10(x)": lambda x: np.log10(x + 1e-6),
    "e^(x)": lambda x: np.exp(x),
    "10^(x+6)": lambda x: np.power(10.0, x + 6.0),
    "10^(x)": lambda x: np.power(10.0, x),
    "10^(x); 1/g to 1/kg": lambda x: np.power(10.0, x) * 1.0e3,
    "10^(x); 1/kg to 1/g": lambda x: np.power(10.0, x) * 1.0e-3,
    # requires MW in g/mol; converts ug/mL to uM
    "ug/mL to uM": lambda x, mw: (x / mw) * 1.0e3 if (mw is not None and mw > 0) else float("nan"),
    # unit conversions
    "nM to uM": lambda x: x / 1000.0,
    "g to kg": lambda x: x / 1000.0,
    "kg to g": lambda x: x * 1000.0,
}


__all__ = [
    "DatasetInfo",
    "DEFAULT_DATASET_DIR",
    "DATASETS",
    "COLS_WITH_UNITS",
    "TRANSFORMATIONS",
]
