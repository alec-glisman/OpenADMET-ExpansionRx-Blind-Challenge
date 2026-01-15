"""Data transformation utilities for ADMET endpoints.

This module provides functions for applying appropriate transformations to
different types of ADMET endpoints based on their physical and analytical
constraints.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def sanitize_column_names_for_latex(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """Sanitize column names to be LaTeX-compatible.

    Replaces Unicode and ASCII characters that LaTeX cannot process with
    LaTeX-compatible equivalents:
    - Percent (%) becomes \\%
    - ASCII caret notation (10^-6) becomes 10$^{-6}$

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with potentially Unicode column names.
    inplace : bool, default False
        If True, modify the dataframe in place.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with sanitized column names, or None if inplace=True.

    Examples
    --------
    >>> df = pd.DataFrame(columns=["Caco-2 Papp A→B (10⁻⁶ cm/s)"])
    >>> sanitized = sanitize_column_names_for_latex(df)
    >>> sanitized.columns[0]
    'Caco-2 Papp A$\\rightarrow$B (10$^{-6}$ cm/s)'

    >>> df2 = pd.DataFrame(columns=["Caco-2 Papp A>B (10^-6 cm/s)"])
    >>> sanitized2 = sanitize_column_names_for_latex(df2)
    >>> sanitized2.columns[0]
    'Caco-2 Papp A$>$B (10$^{-6}$ cm/s)'
    """
    if not inplace:
        df = df.copy()

    # Unicode to LaTeX mapping - handle superscripts as a group
    new_columns = []
    for col in df.columns:
        new_col = str(col)

        # Check if already sanitized (makes function idempotent)
        if "$^{" in new_col or "$\\rightarrow$" in new_col or "\\%" in new_col or " pct " in new_col:
            new_columns.append(new_col)
            continue

        # Replace percent with "pct" to avoid LaTeX line-breaking issues
        new_col = new_col.replace("% unbound", " pct unbound")
        new_col = new_col.replace("%", " pct")

        # Replace arrow symbols (both Unicode and ASCII)
        new_col = new_col.replace("→", r"$\rightarrow$")
        new_col = new_col.replace(">", r"$>$")  # ASCII replacement for arrow

        # Handle superscript sequences (e.g., "⁻⁶" -> "$^{-6}$")
        # Map each superscript to its regular form
        superscript_map = {
            "⁰": "0",
            "¹": "1",
            "²": "2",
            "³": "3",
            "⁴": "4",
            "⁵": "5",
            "⁶": "6",
            "⁷": "7",
            "⁸": "8",
            "⁹": "9",
            "⁻": "-",
            "⁺": "+",
        }

        # Find and replace sequences of superscripts
        i = 0
        result = []
        while i < len(new_col):
            if new_col[i] in superscript_map:
                # Start of superscript sequence
                superscript_text = ""
                j = i
                while j < len(new_col) and new_col[j] in superscript_map:
                    superscript_text += superscript_map[new_col[j]]
                    j += 1
                result.append(f"$^{{{superscript_text}}}$")
                i = j
            else:
                result.append(new_col[i])
                i += 1

        new_col = "".join(result)

        # Handle ASCII caret notation (e.g., "10^-6" -> "10$^{-6}$")
        # This pattern matches number^number or number^-number
        new_col = re.sub(r"(\d+)\^(-?\d+)", r"\1$^{\2}$", new_col)

        new_columns.append(new_col)

    df.columns = new_columns

    if not inplace:
        return df
    return None


def apply_endpoint_transformations(
    df: pd.DataFrame,
    exclude_columns: Optional[list[str]] = None,
    clint_endpoints: Optional[list[str]] = None,
    permeability_endpoints: Optional[list[str]] = None,
    ppb_endpoints: Optional[list[str]] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Apply endpoint-specific log10 transformations to ADMET data.

    Different ADMET endpoints require different handling of zero values before
    log transformation:

    - **CLint and permeability**: Zeros replaced with half of smallest non-zero
      value before log10 transformation
    - **PPB (Plasma Protein Binding)**: Zeros set to 10^(-6) before log10,
      considering typical PPB detection limits
    - **Other endpoints**: Standard log10 transformation with zeros set to 10^(-6)
    - **Excluded columns**: No transformation applied (e.g., LogD)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing ADMET endpoint columns.
    exclude_columns : list[str], optional
        Column names to exclude from transformation (e.g., SMILES, LogD, identifiers).
        Default: ["SMILES", "Molecule Name (None)", "LogD (None)"]
    clint_endpoints : list[str], optional
        CLint endpoint column names. Default includes HLM and MLM CLint.
    permeability_endpoints : list[str], optional
        Permeability endpoint column names. Default includes Caco-2 permeability.
    ppb_endpoints : list[str], optional
        Plasma protein binding endpoint column names. Default includes MPPB, MBPB, MGMB.
    inplace : bool, default False
        If True, modify the dataframe in place and return None.
        If False, return a new dataframe with transformations applied.

    Returns
    -------
    pd.DataFrame or None
        Transformed dataframe, or None if inplace=True.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "SMILES": ["CCO", "CCN"],
    ...     "Log HLM CLint (ul/min/mg)": [10.5, 0.0],
    ...     "Log MPPB (%)": [95.0, 0.0],
    ... })
    >>> df_transformed = apply_endpoint_transformations(df)

    Notes
    -----
    The function logs the transformation strategy used for each column.
    Columns not in any endpoint category receive standard log10 transformation.
    """
    if exclude_columns is None:
        exclude_columns = ["SMILES", "Molecule Name (None)", "LogD (None)"]

    if clint_endpoints is None:
        clint_endpoints = [
            "Log HLM CLint (ul/min/mg)",
            "Log MLM CLint (ul/min/mg)",
            "HLM CLint (mL/min/kg)",
            "MLM CLint (mL/min/kg)",
        ]

    if permeability_endpoints is None:
        permeability_endpoints = [
            "Log Caco-2 Permeability Papp A>B (nm/s)",
            "Log Caco-2 Permeability Efflux (None)",
            "Caco-2 Permeability Papp A>B (10^-6 cm/s)",
            "Caco-2 Permeability Efflux (None)",
        ]

    if ppb_endpoints is None:
        ppb_endpoints = [
            "Log MPPB (%)",
            "Log MBPB (%)",
            "Log MGMB (%)",
            "MPPB ( pct unbound)",
            "MBPB ( pct unbound)",
            "MGMB ( pct unbound)",
        ]

    if not inplace:
        df = df.copy()

    # Sanitize column names for LaTeX compatibility
    sanitize_column_names_for_latex(df, inplace=True)

    # Also sanitize the endpoint lists to match sanitized column names
    def sanitize_name(name: str) -> str:
        """Sanitize endpoint names to match LaTeX-sanitized column names."""
        import re

        # Handle percent signs
        name = name.replace("% unbound", " pct unbound")
        name = name.replace("%", " pct")

        # Handle arrows (both Unicode and ASCII)
        name = name.replace("\u2192", "$\\rightarrow$")  # Unicode arrow
        name = name.replace("→", "$\\rightarrow$")
        name = name.replace(">", "$>$")
        name = name.replace("<", "$<$")

        # Handle Unicode superscripts
        superscript_map = {
            "\u2070": "0",
            "\u00b9": "1",
            "\u00b2": "2",
            "\u00b3": "3",
            "\u2074": "4",
            "\u2075": "5",
            "\u2076": "6",
            "\u2077": "7",
            "\u2078": "8",
            "\u2079": "9",
            "\u207b": "-",
            "\u207a": "+",
        }

        # Replace sequences of Unicode superscripts
        i = 0
        result = []
        while i < len(name):
            if name[i] in superscript_map:
                superscript_text = ""
                j = i
                while j < len(name) and name[j] in superscript_map:
                    superscript_text += superscript_map[name[j]]
                    j += 1
                result.append(f"$^{{{superscript_text}}}$")
                i = j
            else:
                result.append(name[i])
                i += 1
        name = "".join(result)

        # Handle ASCII caret notation (e.g., "10^-6" -> "10$^{-6}$")
        name = re.sub(r"(\d+)\^(-?\d+)", r"\1$^{\2}$", name)

        return name

    clint_endpoints = [sanitize_name(ep) for ep in clint_endpoints]
    permeability_endpoints = [sanitize_name(ep) for ep in permeability_endpoints]
    ppb_endpoints = [sanitize_name(ep) for ep in ppb_endpoints]
    exclude_columns = [sanitize_name(col) for col in exclude_columns]

    for col in df.columns:
        if col in exclude_columns:
            logger.warning(f"Skipping transformation for excluded column: {col}")
            continue

        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Skipping non-numeric column: {col}")
            continue

        if col in clint_endpoints or col in permeability_endpoints:
            # CLint and permeability: replace 0s with half of smallest non-zero
            logger.info(f"Transforming {col} (CLint/Permeability strategy)")
            min_nonzero = df[df[col] > 0][col].min()
            logger.info(msg=f"Minimum positive value: {min_nonzero}")
            if pd.isna(min_nonzero):
                logger.warning(f"No positive values in {col}, using detection limit 1e-6")
                min_nonzero = 1e-6
            df[col] = df[col].apply(lambda x: np.log10(x) if x != 0.0 else np.log10(min_nonzero / 2))
        elif col in ppb_endpoints:
            # PPB: set 0s to 10^(-6) (detection limit)
            logger.info(f"Transforming {col} (PPB strategy)")
            df[col] = df[col].apply(lambda x: np.log10(x) if x != 0.0 else np.log10(1e-6))
        else:
            # Other endpoints: standard log10
            logger.info(f"Transforming {col} (standard log10)")
            df[col] = df[col].apply(lambda x: np.log10(x))

    if not inplace:
        return df
    return None
