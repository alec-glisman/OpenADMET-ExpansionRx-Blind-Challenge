import json
import pytest
from pathlib import Path
import logging

# Explicitly import pytest and use its utilities where appropriate

from admet.logging import configure_logging, get_logging_config


def test_configure_logging_sets_level_and_file(tmp_path: Path):
    logfile = tmp_path / "test.log"
    configure_logging(level="DEBUG", file=str(logfile), structured=False)
    cfg = get_logging_config()
    assert cfg["level"] in ("DEBUG", "10") or cfg["level"] == "DEBUG"
    assert cfg["file"] == str(logfile)
    assert cfg["structured"] is False
    # write a log entry
    logger = logging.getLogger(__name__)
    logger.debug("debug message for testing")
    if not logfile.exists():
        pytest.fail("Expected logfile to exist after logging, but it does not")
    text = logfile.read_text()
    if "debug message for testing" not in text:
        pytest.fail("Expected log message not found in logfile")


def test_configure_logging_structured_json(tmp_path: Path):
    logfile = tmp_path / "test_json.log"
    configure_logging(level="INFO", file=str(logfile), structured=True)
    cfg = get_logging_config()
    assert cfg["structured"] is True
    assert cfg["file"] == str(logfile)
    logger = logging.getLogger(__name__)
    logger.info("structured test")
    if not logfile.exists():
        pytest.fail("Expected structured logfile to exist after logging, but it does not")
    # Check file contains at least one parseable JSON line
    with logfile.open("r") as fh:
        for line in fh:
            if line.strip():
                json.loads(line)
                break


def test_get_logging_config_default_no_file():
    configure_logging(level="WARNING", file=None, structured=False)
    cfg = get_logging_config()
    if cfg["file"] is not None:
        pytest.fail("Expected cfg['file'] to be None by default")
    if cfg["structured"] is not False:
        pytest.fail("Expected cfg['structured'] to be False by default")
