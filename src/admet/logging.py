"""Logging Utilities
====================

Centralized logging helpers for the ADMET application.

This module provides utilities to configure root logging for both the CLI
and Ray workers. It intentionally avoids external structured logging
dependencies while supporting a minimal JSON formatter.

Contents
--------
Classes
^^^^^^^
* :class:`JsonFormatter` – Lightweight JSON log formatter.

Functions
^^^^^^^^^
* :func:`configure_logging` – Configure root logger (optionally structured).
* :func:`get_logging_config` – Introspect current root logging configuration.
"""

from __future__ import annotations

import logging
import json
from typing import Optional
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter.

    Serializes standard log record fields plus an optional exception stack.

    Notes
    -----
    This formatter is intentionally small; if richer structured logging is
    desired a third‑party library can be integrated later.
    """

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - simple serializer
        """Format a log record as a JSON string.

        Parameters
        ----------
        record : logging.LogRecord
            The record emitted by the logger.

        Returns
        -------
        str
            JSON encoded representation of the log record.
        """
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

    Sets the root logger level, replaces existing handlers, and optionally
    enables structured JSON logging.

    Parameters
    ----------
    level : str, default="INFO"
        Logging level name (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``).
    fmt : str, optional
        Log format string. Ignored when ``structured=True``.
    file : str, optional
        Path to a file to append logs to (same formatter as stderr).
    structured : bool, default=False
        If ``True`` use :class:`JsonFormatter` for structured logs.

    Returns
    -------
    None
        This function configures global state and returns nothing.
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
    """Return current root logging configuration.

    Inspects the root logger to determine level, file handler (if present),
    and whether structured logging is active.

    Returns
    -------
    dict
        Mapping with keys: ``level`` (str), ``file`` (str|None),
        ``structured`` (bool).
    """
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
