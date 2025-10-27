from typing import Mapping, TypedDict, Any, Callable, Dict
from pathlib import Path
from tdc import utils


class DatasetInfo(TypedDict):
    """Type definition for dataset information."""

    type: str
    uri: str
    output_file: str


# Default output directory for downloaded datasets
DEFAULT_DATASET_DIR = Path(__file__).parents[3] / "assets/dataset/raw"

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
    "uri": (
        "hf://datasets/openadmet/" "openadmet-expansionrx-challenge-teaser/" "expansion_data_teaser.csv"
    ),
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

# Column names with units for ExpansionRX and related datasets
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

# Simple numeric transformations used during dataset harmonization
TRANSFORMATIONS: Dict[str, Callable[..., Any]] = {
    "None": lambda x: x,
    "10^(x+6)": lambda x: 10.0 ** (x + 6.0),
    "10^(x)": lambda x: 10.0 ** (x),
    "10^(x); /mg to /kg": lambda x: (10.0**x) / 1.0e-6,
    # requires MW in g/mol; converts ug/mL to uM
    "ug/mL to uM": lambda x, mw: (x * 1000) / mw if (mw is not None and mw > 0) else float("nan"),
}

# Backwards-compatible lowercase aliases matching earlier module
cols_with_units = COLS_WITH_UNITS
transformations = TRANSFORMATIONS

__all__ = [
    "DatasetInfo",
    "DEFAULT_DATASET_DIR",
    "DATASETS",
    "COLS_WITH_UNITS",
    "TRANSFORMATIONS",
    "cols_with_units",
    "transformations",
]
