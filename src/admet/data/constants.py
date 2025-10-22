from typing import Mapping, TypedDict
from pathlib import Path


class DatasetInfo(TypedDict):
    """Type definition for dataset information."""

    type: str
    uri: str
    output_file: str


# Default output directory for downloaded datasets
DEFAULT_DATASET_DIR = Path(__file__).parents[3] / "assets/dataset/raw"

DATASETS: Mapping[str, DatasetInfo] = {
    "expansion_teaser": {
        "type": "huggingface",
        "uri": (
            "hf://datasets/openadmet/" "openadmet-expansionrx-challenge-teaser/" "expansion_data_teaser.csv"
        ),
        "output_file": "expansion_teaser.csv",
    }
}
