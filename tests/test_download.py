import os
import csv
from typer.testing import CliRunner
import pytest

from admet.data import download as dl_mod
from admet import cli

runner = CliRunner()


def test_download_dataset_to_csv_single_split(tmp_path, monkeypatch):
    # Mock datasets.load_dataset to return a simple list of dicts
    sample = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    import os
    import csv
    from typer.testing import CliRunner
    import pytest

    from admet.data import download as dl_mod
    from admet import cli

    runner = CliRunner()


    def test_download_dataset_to_csv_single_split(tmp_path, monkeypatch):
        # Mock datasets.load_dataset to return a simple list of dicts
        sample = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

        import os
        import csv
        from typer.testing import CliRunner
        import pytest

        from admet.data import download as dl_mod
        from admet import cli

        runner = CliRunner()


        def test_download_dataset_to_csv_single_split(tmp_path, monkeypatch):
            # Mock datasets.load_dataset to return a simple list of dicts
            sample = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

            def fake_load(dataset_id):
                assert dataset_id == "openadmet/openadmet-expansionrx-challenge-teaser"
                return sample

            monkeypatch.setattr(dl_mod, "load_dataset", fake_load)

            out = tmp_path / "out"
            out.mkdir()

            written = dl_mod.download_dataset_to_csv(
                "openadmet/openadmet-expansionrx-challenge-teaser",
                split="train",
                output_dir=str(out),
            )

            assert len(written) == 1
            out_path = written[0]
            assert os.path.exists(out_path)

            with open(out_path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["a"] == "1"
            assert rows[1]["b"] == "y"


        def test_cli_data_download_huggingface(monkeypatch, tmp_path):
            # Patch constants to point to a small sample mapping and patch downloader
            from admet.data import constants

            constants.DATASETS = {
                "openadmet-teaser": {
                    "type": "huggingface",
                    "dataset_id": "openadmet/openadmet-expansionrx-challenge-teaser",
                    "output_file": "openadmet_teaser.csv",
                }
            }

            calls = {}

            def fake_save(dataset_id, output_file=None):
                calls["dataset_id"] = dataset_id
                calls["output_file"] = output_file
                # create a dummy file in tmp_path
                p = tmp_path / (output_file or "out.csv")
                p.write_text("a,b\n1,x\n")
                return [str(p)]

            # make Downloader use tmp_path as destination
            monkeypatch.setattr(dl_mod, "Downloader", lambda dest_dir: dl_mod.Downloader(dest_dir=str(tmp_path)))
            # monkeypatch the save_huggingface method on the Downloader class
            monkeypatch.setattr(
                dl_mod.Downloader,
                "save_huggingface",
                lambda self, did, output_file=None: fake_save(did, output_file),
            )

            result = runner.invoke(cli.app, ["data", "download", "openadmet-teaser"])

            assert result.exit_code == 0
            assert calls["dataset_id"] == "openadmet/openadmet-expansionrx-challenge-teaser"

    with open(out_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["a"] == "1"
    assert rows[1]["b"] == "y"


def test_cli_data_download_huggingface(monkeypatch, tmp_path):
    # Patch constants to point to a small sample mapping and patch downloader
    from admet.data import constants

    constants.DATASETS = {
        "openadmet-teaser": {
            "type": "huggingface",
            "dataset_id": "openadmet/openadmet-expansionrx-challenge-teaser",
            "output_file": "openadmet_teaser.csv",
        }
    }

    calls = {}

    def fake_save(dataset_id, output_file=None):
        calls["dataset_id"] = dataset_id
        calls["output_file"] = output_file
        # create a dummy file in tmp_path
        p = tmp_path / (output_file or "out.csv")
        p.write_text("a,b\n1,x\n")
        return [str(p)]

    monkeypatch.setattr(dl_mod, "Downloader", lambda dest_dir: dl_mod.Downloader(dest_dir=str(tmp_path)))
    # monkeypatch the save_huggingface method
    monkeypatch.setattr(dl_mod.Downloader, "save_huggingface", lambda self, did, output_file=None: fake_save(did, output_file))

    result = runner.invoke(cli.app, ["data", "download", "openadmet-teaser"])

    assert result.exit_code == 0
    assert calls["dataset_id"] == "openadmet/openadmet-expansionrx-challenge-teaser"
*** End File