"""Parser utilities for leaderboard data."""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def to_dataframe(gradio_value: Any) -> pd.DataFrame:
    """Convert common Gradio table payload formats into a pandas DataFrame.

    Handles multiple Gradio output formats:
    - {"headers": [...], "data": [[...], ...]}
    - {"columns": [...], "data": [[...], ...]}
    - {"data": [[...], ...]} (headers inferred as col0, col1, ...)
    - [[...], ...] (list of rows, headers inferred)
    - pandas DataFrame (returned as-is)

    Parameters
    ----------
    gradio_value : Any
        Gradio table output value

    Returns
    -------
    pd.DataFrame
        Parsed DataFrame. Returns empty DataFrame if parsing fails.

    Examples
    --------
    >>> data = {"headers": ["rank", "user"], "data": [[1, "alice"], [2, "bob"]]}
    >>> df = to_dataframe(data)
    >>> df.columns.tolist()
    ['rank', 'user']
    """
    if gradio_value is None:
        return pd.DataFrame()

    if isinstance(gradio_value, pd.DataFrame):
        return gradio_value

    if isinstance(gradio_value, dict):
        headers = None
        data = gradio_value.get("data")

        if "headers" in gradio_value:
            headers = gradio_value["headers"]
        elif "columns" in gradio_value:
            headers = gradio_value["columns"]

        if data is None:
            logger.warning("Gradio dict missing 'data' key")
            return pd.DataFrame()

        if not isinstance(data, list):
            logger.warning("Gradio 'data' is not a list")
            return pd.DataFrame()

        if not data:
            return pd.DataFrame(columns=headers if headers else [])

        if headers:
            return pd.DataFrame(data, columns=headers)
        else:
            df = pd.DataFrame(data)
            df.columns = [f"col{i}" for i in range(len(df.columns))]
            return df

    if isinstance(gradio_value, list):
        if not gradio_value:
            return pd.DataFrame()

        if isinstance(gradio_value[0], dict):
            return pd.DataFrame(gradio_value)
        else:
            df = pd.DataFrame(gradio_value)
            df.columns = [f"col{i}" for i in range(len(df.columns))]
            return df

    logger.warning("Unknown Gradio value format: %s", type(gradio_value))
    return pd.DataFrame()


def normalize_user_cell(x: Any) -> str:
    """Normalize user cell text for case-insensitive matching.

    Converts to lowercase and strips whitespace.

    Parameters
    ----------
    x : Any
        User cell value (typically str or None)

    Returns
    -------
    str
        Normalized lowercase string, empty if None

    Examples
    --------
    >>> normalize_user_cell("  Alice  ")
    'alice'
    >>> normalize_user_cell(None)
    ''
    """
    s = "" if x is None else str(x)
    return s.strip().lower()


def extract_value_uncertainty(val: Any) -> Tuple[Optional[float], Optional[float]]:
    """Parse strings like '0.35 +/- 0.01' or '0.35 ± 0.01'.

    Extracts mean value and uncertainty from formatted strings.

    Parameters
    ----------
    val : Any
        Value to parse (typically str)

    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        (value, uncertainty). Returns (None, None) if parsing fails.

    Examples
    --------
    >>> extract_value_uncertainty("0.35 +/- 0.01")
    (0.35, 0.01)
    >>> extract_value_uncertainty("0.42 ± 0.02")
    (0.42, 0.02)
    >>> extract_value_uncertainty("invalid")
    (None, None)
    """
    if val is None:
        return (None, None)

    s = str(val).strip()
    try:
        # Match patterns: "value +/- uncertainty" or "value ± uncertainty"
        match = re.search(r"([-+]?\d+\.?\d*)\s*[±+]/?-?\s*([-+]?\d+\.?\d*)", s)
        if match:
            value = float(match.group(1))
            uncertainty = float(match.group(2))
            return (value, uncertainty)

        # Try parsing as plain float
        return (float(s), None)
    except (ValueError, AttributeError):
        return (None, None)


def find_user_rank(df: pd.DataFrame, target_user: str) -> Optional[int]:
    """Return 1-based row rank for target_user based on displayed row order.

    Matches both plain usernames (e.g., 'aglisman') and markdown links
    (e.g., '[aglisman](https://...)'). Matching is case-insensitive.

    Parameters
    ----------
    df : pd.DataFrame
        Leaderboard DataFrame with a 'user' column
    target_user : str
        Username to find (case-insensitive)

    Returns
    -------
    Optional[int]
        1-based rank (row index + 1), or None if not found

    Examples
    --------
    >>> df = pd.DataFrame({"user": ["alice", "bob", "charlie"]})
    >>> find_user_rank(df, "bob")
    2
    >>> find_user_rank(df, "BOB")
    2
    >>> find_user_rank(df, "dave")
    None
    """
    if df.empty:
        return None

    # Identify user column
    user_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower == "user" or "user" in col_lower:
            user_col = col
            break

    if user_col is None:
        logger.warning("Could not find 'user' column in DataFrame")
        return None

    # Normalize target for matching
    target_norm = target_user.strip().lower()

    # Search for user in normalized column
    for idx, cell_val in enumerate(df[user_col]):
        cell_norm = normalize_user_cell(cell_val)

        # Match plain username or markdown link containing username
        if target_norm in cell_norm:
            return idx + 1

    logger.debug("User '%s' not found in leaderboard", target_user)
    return None
