"""Centralized logging helpers for the ADMET application.

This module provides a `configure_logging` function to set a global logging
handler and optionally enable structured JSON logs. It's intentionally
lightweight and avoids hard dependencies on structured logging libraries, but
supports using the `json` module for very simple structured logs.
"""

from __future__ import annotations

import logging
import json
from typing import Optional
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - simple serializer
        payload = {
            "time": self.formatTime(record, self.datefmt),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(
    level: str = "INFO",
    fmt: Optional[str] = None,
    file: Optional[str] = None,
    structured: bool = False,
) -> None:
    """Configure application-wide logging.

    Parameters
    ----------
    level:
        String level name - DEBUG, INFO, WARNING, ERROR, CRITICAL
    fmt:
        Optional log format string - ignored when `structured=True` as JSON format used.
    file:
        Optional path to a file to write logs into in addition to stderr.
    structured:
        If True, emit logs as JSON objects using the builtin JsonFormatter.
    """
    try:
        lvl = getattr(logging, str(level).upper())
    except AttributeError:
        lvl = logging.INFO

    root = logging.getLogger()
    # Remove pre-existing handlers
    for h in list(root.handlers):
        root.removeHandler(h)

    # Use JSON formatter when structured logging requested
    if structured:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
    else:
        handler = logging.StreamHandler()
        fmt_str = fmt or "%(asctime)s %(name)-20s %(levelname)-8s %(message)s"
        handler.setFormatter(logging.Formatter(fmt_str, datefmt="%Y-%m-%d %H:%M:%S"))
    handler.setLevel(lvl)
    root.addHandler(handler)
    root.setLevel(lvl)

    # Optional file logging using same formatter
    if file:
        fpath = Path(file)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(fpath)
        if structured:
            fh.setFormatter(JsonFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
        else:
            fmt_str = fmt or "%(asctime)s %(name)-20s %(levelname)-8s %(message)s"
            fh.setFormatter(logging.Formatter(fmt_str, datefmt="%Y-%m-%d %H:%M:%S"))
        fh.setLevel(lvl)
        root.addHandler(fh)


__all__ = ["configure_logging", "JsonFormatter"]


def get_logging_config() -> dict:
    """Return a dict describing the current root logging configuration."""
    root = logging.getLogger()
    level = logging.getLevelName(root.level)
    file_path = None
    structured = False
    for h in root.handlers:
        # JsonFormatter is defined earlier in this module
        JsonFormatterLocal = JsonFormatter
        # File handler
        if isinstance(h, logging.FileHandler):
            try:
                file_path = getattr(h, "baseFilename", None)
            except AttributeError:
                file_path = None
        # Structured if using JsonFormatter
        if (
            hasattr(h, "formatter")
            and JsonFormatterLocal is not None
            and isinstance(h.formatter, JsonFormatterLocal)
        ):
            structured = True
    return {"level": level, "file": file_path, "structured": structured}
