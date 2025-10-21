from typing import Mapping, Union, Iterable

DATASETS: Mapping[str, dict[str, Union[str, Iterable[str]]]] = {
    "expansion_teaser": {
        "type": "pandas",
        "uri": "hf://datasets/openadmet/openadmet-expansionrx-challenge-teaser/expansion_data_teaser.csv",
        "output_file": "expansion_teaser.csv",
    }
}
