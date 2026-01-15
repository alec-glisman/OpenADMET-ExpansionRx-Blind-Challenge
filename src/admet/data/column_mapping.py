"""Column name normalization for multi-format CSV support.

This module provides utilities to automatically detect and normalize column names
from different CSV header formats to canonical column names used throughout the
ADMET pipeline.

Supported input formats
-----------------------
Format 1 (Training data):
    SMILES, Dataset, Quality, Molecule Name, LogD, Log KSOL, Log HLM CLint,
    Log MLM CLint, Log Caco-2 Permeability Papp A$>$B, Log Caco-2 Permeability Efflux,
    Log MPPB, Log MBPB, Log MGMB

Format 2 (Challenge test set):
    Molecule Name (None), SMILES, LogD (None), KSOL (uM), HLM CLint (mL/min/kg),
    MLM CLint (mL/min/kg), Caco-2 Permeability Papp A$>$B (10$^{-6}$ cm/s),
    Caco-2 Permeability Efflux (None), MPPB ( pct unbound), MBPB ( pct unbound),
    MGMB ( pct unbound)

Output format (Canonical):
    LogD, Log KSOL, Log HLM CLint, Log MLM CLint, Log Caco-2 Permeability Papp A>B,
    Log Caco-2 Permeability Efflux, Log MPPB, Log MBPB, Log MGMB

Data is assumed to already be log-transformed regardless of input header naming.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import pandas as pd

from admet.data.constant import CANONICAL_TARGET_COLUMNS, COLUMN_ALIASES

logger = logging.getLogger("admet.data.column_mapping")


class ColumnMappingError(ValueError):
    """Raised when column mapping fails due to unresolvable columns."""

    pass


def normalize_column_name(column: str) -> str:
    """Normalize a single column name to its canonical form.

    Applies alias mapping and LaTeX normalization to convert alternative
    column names to their canonical equivalents.

    Parameters
    ----------
    column : str
        Input column name (may contain LaTeX notation or unit annotations).

    Returns
    -------
    str
        Canonical column name, or original if no mapping exists.

    Examples
    --------
    >>> normalize_column_name("Log Caco-2 Permeability Papp A$>$B")
    'Log Caco-2 Permeability Papp A>B'
    >>> normalize_column_name("KSOL (uM)")
    'Log KSOL'
    >>> normalize_column_name("LogD")
    'LogD'
    """
    if column in COLUMN_ALIASES:
        return COLUMN_ALIASES[column]

    # LaTeX normalization: $>$ -> >, $^{...}$ -> removed
    normalized = column.replace("$>$", ">").replace("$<$", "<")

    # Remove any remaining LaTeX $ markers
    while "$" in normalized:
        start = normalized.find("$")
        end = normalized.find("$", start + 1)
        if end == -1:
            break
        normalized = normalized[:start] + normalized[end + 1 :]

    if normalized != column and normalized in COLUMN_ALIASES:
        return COLUMN_ALIASES[normalized]

    return normalized


def normalize_dataframe_columns(
    df: pd.DataFrame,
    target_cols: Optional[List[str]] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Normalize DataFrame column names to canonical format.

    Auto-detects the input format and applies appropriate column name mappings.
    Raises an error if required target columns cannot be resolved.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with potentially non-canonical column names.
    target_cols : list[str], optional
        List of canonical target column names that must be present after
        normalization. If None, uses CANONICAL_TARGET_COLUMNS.
    inplace : bool, default=False
        If True, modifies the DataFrame in place. Otherwise returns a copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized column names.

    Raises
    ------
    ColumnMappingError
        If required target columns cannot be resolved after normalization.

    Examples
    --------
    >>> df = pd.read_csv("test_data.csv")
    >>> df_normalized = normalize_dataframe_columns(df)
    >>> assert "Log Caco-2 Permeability Papp A>B" in df_normalized.columns
    """
    if not inplace:
        df = df.copy()

    original_columns = list(df.columns)
    column_mapping: Dict[str, str] = {}

    for col in original_columns:
        normalized = normalize_column_name(col)
        if normalized != col:
            column_mapping[col] = normalized
            logger.debug("Column mapping: '%s' -> '%s'", col, normalized)

    if column_mapping:
        df.rename(columns=column_mapping, inplace=True)
        logger.info(
            "Normalized %d column names: %s",
            len(column_mapping),
            {k: v for k, v in list(column_mapping.items())[:5]},
        )

    # Validate required columns if target_cols specified
    if target_cols is not None:
        validate_target_columns(df, target_cols)

    return df


def validate_target_columns(
    df: pd.DataFrame,
    target_cols: List[str],
    strict: bool = True,
) -> List[str]:
    """Validate that required target columns are present in DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    target_cols : list[str]
        List of canonical target column names that must be present.
    strict : bool, default=True
        If True, raises an error if any target column is missing.
        If False, returns only the columns that are present.

    Returns
    -------
    list[str]
        List of validated target columns that are present in the DataFrame.

    Raises
    ------
    ColumnMappingError
        If strict=True and any target column is missing.
    """
    present_cols = [col for col in target_cols if col in df.columns]
    missing_cols = [col for col in target_cols if col not in df.columns]

    if missing_cols:
        # Try to find potential matches for missing columns
        suggestions = _suggest_column_matches(df.columns.tolist(), missing_cols)

        error_msg = (
            f"Missing {len(missing_cols)} target column(s) after normalization: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )

        if suggestions:
            error_msg += f"\nPossible matches: {suggestions}"
            error_msg += "\nConsider adding these mappings to COLUMN_ALIASES in src/admet/data/constant.py"

        if strict:
            raise ColumnMappingError(error_msg)
        else:
            logger.warning(error_msg)

    return present_cols


def _suggest_column_matches(
    available_cols: List[str],
    missing_cols: List[str],
) -> Dict[str, List[str]]:
    """Suggest potential column matches for missing columns.

    Uses simple substring matching to suggest which available columns
    might correspond to missing target columns.

    Parameters
    ----------
    available_cols : list[str]
        List of available column names in the DataFrame.
    missing_cols : list[str]
        List of missing target column names.

    Returns
    -------
    dict[str, list[str]]
        Mapping of missing column names to lists of potential matches.
    """
    suggestions: Dict[str, List[str]] = {}

    for missing in missing_cols:
        # Extract key terms from the missing column name
        key_terms = _extract_key_terms(missing)
        matches = []

        for available in available_cols:
            available_lower = available.lower()
            # Check if any key term appears in the available column
            if any(term in available_lower for term in key_terms):
                matches.append(available)

        if matches:
            suggestions[missing] = matches

    return suggestions


def _extract_key_terms(column_name: str) -> Set[str]:
    """Extract key terms from a column name for fuzzy matching."""
    # Remove common prefixes and convert to lowercase
    name = column_name.lower()
    name = name.replace("log ", "")

    # Split on common delimiters and filter short terms
    terms = set()
    for delimiter in [" ", "-", "_"]:
        for part in name.split(delimiter):
            if len(part) >= 3:
                terms.add(part)

    return terms


def detect_csv_format(df: pd.DataFrame) -> str:
    """Detect the CSV format based on column names.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze.

    Returns
    -------
    str
        Format identifier: 'canonical', 'format1_latex', 'format2_units', or 'unknown'.

    Examples
    --------
    >>> df = pd.read_csv("training_data.csv")
    >>> format_type = detect_csv_format(df)
    >>> print(format_type)
    'format1_latex'
    """
    columns = set(df.columns)

    # Check for canonical format
    canonical_present = sum(1 for col in CANONICAL_TARGET_COLUMNS if col in columns)
    if canonical_present >= 7:
        return "canonical"

    # Check for Format 1 (LaTeX-escaped, Log-prefixed)
    latex_markers = ["$>$", "$<$", "$^"]
    has_latex = any(any(marker in col for marker in latex_markers) for col in columns)
    has_log_prefix = any(col.startswith("Log ") for col in columns)

    if has_latex and has_log_prefix:
        return "format1_latex"

    # Check for Format 2 (unit annotations, no Log prefix)
    unit_markers = ["(uM)", "(mL/min/kg)", "( pct unbound)", "(None)", "(10$^"]
    has_units = any(any(marker in col for marker in unit_markers) for col in columns)

    if has_units:
        return "format2_units"

    # Check for Format 2 variant (raw names without Log prefix)
    raw_names = {"KSOL", "HLM CLint", "MLM CLint", "MPPB", "MBPB", "MGMB"}
    raw_present = sum(1 for col in columns if col in raw_names)
    if raw_present >= 3:
        return "format2_raw"

    return "unknown"


def load_csv_with_normalized_columns(
    filepath: str,
    target_cols: Optional[List[str]] = None,
    validate: bool = True,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load a CSV file and normalize column names to canonical format.

    Convenience function that combines pd.read_csv with column normalization.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    target_cols : list[str], optional
        Target columns to validate. If None, no validation is performed.
    validate : bool, default=True
        Whether to validate target columns are present.
    **read_csv_kwargs
        Additional arguments passed to pd.read_csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized column names.

    Raises
    ------
    ColumnMappingError
        If validation fails and required columns are missing.
    """
    df = pd.read_csv(filepath, **read_csv_kwargs)

    detected_format = detect_csv_format(df)
    logger.info("Detected CSV format: %s for file: %s", detected_format, filepath)

    validation_cols = target_cols if validate else None
    df = normalize_dataframe_columns(df, target_cols=validation_cols)

    return df


__all__ = [
    "ColumnMappingError",
    "detect_csv_format",
    "load_csv_with_normalized_columns",
    "normalize_column_name",
    "normalize_dataframe_columns",
    "validate_target_columns",
]
