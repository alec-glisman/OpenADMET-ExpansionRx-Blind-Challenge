from typing import Mapping, TypedDict
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
