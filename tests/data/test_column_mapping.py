"""Tests for column name normalization and multi-format CSV support.

Tests cover:
- Column alias mapping
- LaTeX notation normalization
- Format detection
- Validation of target columns
- Error handling for missing columns
"""

from __future__ import annotations

import pandas as pd
import pytest

from admet.data.column_mapping import (
    ColumnMappingError,
    detect_csv_format,
    normalize_column_name,
    normalize_dataframe_columns,
    validate_target_columns,
)
from admet.data.constant import CANONICAL_TARGET_COLUMNS, COLUMN_ALIASES


class TestNormalizeColumnName:
    """Tests for normalize_column_name function."""

    def test_canonical_column_unchanged(self):
        """Canonical column names should remain unchanged."""
        for col in CANONICAL_TARGET_COLUMNS:
            assert normalize_column_name(col) == col

    def test_latex_greater_than_normalized(self):
        """LaTeX $>$ notation should be normalized to >."""
        input_col = "Log Caco-2 Permeability Papp A$>$B"
        expected = "Log Caco-2 Permeability Papp A>B"
        assert normalize_column_name(input_col) == expected

    def test_format2_ksol_with_units(self):
        """Format 2 column with units should map to canonical."""
        input_col = "KSOL (uM)"
        expected = "Log KSOL"
        assert normalize_column_name(input_col) == expected

    def test_format2_hlm_clint(self):
        """Format 2 HLM CLint should map to canonical."""
        input_col = "HLM CLint (mL/min/kg)"
        expected = "Log HLM CLint"
        assert normalize_column_name(input_col) == expected

    def test_format2_caco2_with_latex_and_units(self):
        """Format 2 Caco-2 with LaTeX and units should map correctly."""
        input_col = "Caco-2 Permeability Papp A$>$B (10$^{-6}$ cm/s)"
        expected = "Log Caco-2 Permeability Papp A>B"
        assert normalize_column_name(input_col) == expected

    def test_format2_ppb_columns(self):
        """Format 2 PPB columns should map to Log versions."""
        assert normalize_column_name("MPPB ( pct unbound)") == "Log MPPB"
        assert normalize_column_name("MBPB ( pct unbound)") == "Log MBPB"
        assert normalize_column_name("MGMB ( pct unbound)") == "Log MGMB"

    def test_format2_raw_names_without_units(self):
        """Format 2 raw names without units should map to Log versions."""
        assert normalize_column_name("KSOL") == "Log KSOL"
        assert normalize_column_name("HLM CLint") == "Log HLM CLint"
        assert normalize_column_name("MPPB") == "Log MPPB"

    def test_unknown_column_unchanged(self):
        """Unknown column names should be returned unchanged."""
        assert normalize_column_name("SMILES") == "SMILES"
        assert normalize_column_name("Dataset") == "Dataset"
        assert normalize_column_name("Custom Column") == "Custom Column"

    def test_molecule_name_alias(self):
        """Molecule Name alias should be normalized."""
        assert normalize_column_name("Molecule Name (None)") == "Molecule Name"


class TestNormalizeDataframeColumns:
    """Tests for normalize_dataframe_columns function."""

    def test_canonical_format_unchanged(self):
        """DataFrame with canonical columns should be mostly unchanged."""
        df = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCC"],
                "LogD": [1.0, 2.0],
                "Log KSOL": [0.5, 0.6],
            }
        )
        result = normalize_dataframe_columns(df)
        assert list(result.columns) == ["SMILES", "LogD", "Log KSOL"]

    def test_format1_latex_normalized(self):
        """Format 1 with LaTeX should be normalized."""
        df = pd.DataFrame(
            {
                "SMILES": ["CCO"],
                "Log Caco-2 Permeability Papp A$>$B": [1.0],
            }
        )
        result = normalize_dataframe_columns(df)
        assert "Log Caco-2 Permeability Papp A>B" in result.columns
        assert "Log Caco-2 Permeability Papp A$>$B" not in result.columns

    def test_format2_units_normalized(self):
        """Format 2 with unit annotations should be normalized."""
        df = pd.DataFrame(
            {
                "SMILES": ["CCO"],
                "KSOL (uM)": [100.0],
                "HLM CLint (mL/min/kg)": [10.0],
            }
        )
        result = normalize_dataframe_columns(df)
        assert "Log KSOL" in result.columns
        assert "Log HLM CLint" in result.columns

    def test_inplace_modification(self):
        """Inplace modification should modify original DataFrame."""
        df = pd.DataFrame(
            {
                "SMILES": ["CCO"],
                "KSOL (uM)": [100.0],
            }
        )
        result = normalize_dataframe_columns(df, inplace=True)
        assert result is df
        assert "Log KSOL" in df.columns

    def test_copy_by_default(self):
        """Should return a copy by default."""
        df = pd.DataFrame(
            {
                "SMILES": ["CCO"],
                "KSOL (uM)": [100.0],
            }
        )
        original_cols = list(df.columns)
        result = normalize_dataframe_columns(df)
        assert result is not df
        assert list(df.columns) == original_cols

    def test_validation_with_target_cols(self):
        """Should validate target columns when specified."""
        df = pd.DataFrame(
            {
                "SMILES": ["CCO"],
                "LogD": [1.0],
            }
        )
        # Should pass with LogD
        result = normalize_dataframe_columns(df, target_cols=["LogD"])
        assert "LogD" in result.columns

        # Should fail with missing column
        with pytest.raises(ColumnMappingError) as exc_info:
            normalize_dataframe_columns(df, target_cols=["LogD", "Log KSOL"])
        assert "Log KSOL" in str(exc_info.value)


class TestValidateTargetColumns:
    """Tests for validate_target_columns function."""

    def test_all_columns_present(self):
        """Should return all columns when present."""
        df = pd.DataFrame(
            {
                "LogD": [1.0],
                "Log KSOL": [2.0],
            }
        )
        result = validate_target_columns(df, ["LogD", "Log KSOL"])
        assert result == ["LogD", "Log KSOL"]

    def test_partial_columns_strict_raises(self):
        """Should raise error with strict=True for missing columns."""
        df = pd.DataFrame({"LogD": [1.0]})
        with pytest.raises(ColumnMappingError) as exc_info:
            validate_target_columns(df, ["LogD", "Log KSOL"], strict=True)
        assert "Log KSOL" in str(exc_info.value)
        assert "Missing" in str(exc_info.value)

    def test_partial_columns_non_strict_returns_present(self):
        """Should return present columns with strict=False."""
        df = pd.DataFrame({"LogD": [1.0]})
        result = validate_target_columns(df, ["LogD", "Log KSOL"], strict=False)
        assert result == ["LogD"]

    def test_suggestions_for_missing_columns(self):
        """Should suggest potential matches for missing columns."""
        df = pd.DataFrame(
            {
                "KSOL (uM)": [1.0],  # Should suggest for "Log KSOL"
            }
        )
        with pytest.raises(ColumnMappingError) as exc_info:
            validate_target_columns(df, ["Log KSOL"])
        error_msg = str(exc_info.value)
        # Should suggest the similar column
        assert "KSOL" in error_msg or "Possible matches" in error_msg


class TestDetectCSVFormat:
    """Tests for detect_csv_format function."""

    def test_canonical_format(self):
        """Should detect canonical format."""
        df = pd.DataFrame({col: [1.0] for col in CANONICAL_TARGET_COLUMNS})
        assert detect_csv_format(df) == "canonical"

    def test_format1_latex(self):
        """Should detect Format 1 with LaTeX."""
        df = pd.DataFrame(
            {
                "Log Caco-2 Permeability Papp A$>$B": [1.0],
                "Log KSOL": [2.0],
            }
        )
        assert detect_csv_format(df) == "format1_latex"

    def test_format2_units(self):
        """Should detect Format 2 with units."""
        df = pd.DataFrame(
            {
                "KSOL (uM)": [1.0],
                "HLM CLint (mL/min/kg)": [2.0],
            }
        )
        assert detect_csv_format(df) == "format2_units"

    def test_format2_raw(self):
        """Should detect Format 2 with raw names."""
        df = pd.DataFrame(
            {
                "LogD": [1.0],
                "KSOL": [2.0],
                "MPPB": [3.0],
                "MBPB": [4.0],
            }
        )
        assert detect_csv_format(df) == "format2_raw"

    def test_unknown_format(self):
        """Should return 'unknown' for unrecognized format."""
        df = pd.DataFrame(
            {
                "col1": [1.0],
                "col2": [2.0],
            }
        )
        assert detect_csv_format(df) == "unknown"


class TestEndToEndNormalization:
    """End-to-end tests for complete normalization workflows."""

    def test_format1_full_normalization(self):
        """Full normalization of Format 1 data."""
        # Simulate Format 1 CSV headers
        df = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCC"],
                "Dataset": ["train", "train"],
                "Quality": ["high", "medium"],
                "Molecule Name": ["ethanol", "propane"],
                "LogD": [0.5, 1.0],
                "Log KSOL": [2.0, 2.5],
                "Log HLM CLint": [1.0, 1.5],
                "Log MLM CLint": [1.2, 1.7],
                "Log Caco-2 Permeability Papp A$>$B": [-5.0, -4.5],
                "Log Caco-2 Permeability Efflux": [0.1, 0.2],
                "Log MPPB": [1.0, 1.2],
                "Log MBPB": [0.8, 0.9],
                "Log MGMB": [0.9, 1.0],
            }
        )

        result = normalize_dataframe_columns(df, target_cols=CANONICAL_TARGET_COLUMNS)

        # Check all canonical columns are present
        for col in CANONICAL_TARGET_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

        # Check LaTeX was normalized
        assert "Log Caco-2 Permeability Papp A>B" in result.columns
        assert "Log Caco-2 Permeability Papp A$>$B" not in result.columns

    def test_format2_full_normalization(self):
        """Full normalization of Format 2 data."""
        # Simulate Format 2 CSV headers (challenge test set format)
        df = pd.DataFrame(
            {
                "Molecule Name (None)": ["mol1", "mol2"],
                "SMILES": ["CCO", "CCC"],
                "LogD (None)": [0.5, 1.0],
                "KSOL (uM)": [2.0, 2.5],
                "HLM CLint (mL/min/kg)": [1.0, 1.5],
                "MLM CLint (mL/min/kg)": [1.2, 1.7],
                "Caco-2 Permeability Papp A$>$B (10$^{-6}$ cm/s)": [-5.0, -4.5],
                "Caco-2 Permeability Efflux (None)": [0.1, 0.2],
                "MPPB ( pct unbound)": [1.0, 1.2],
                "MBPB ( pct unbound)": [0.8, 0.9],
                "MGMB ( pct unbound)": [0.9, 1.0],
            }
        )

        result = normalize_dataframe_columns(df, target_cols=CANONICAL_TARGET_COLUMNS)

        # Check all canonical columns are present
        for col in CANONICAL_TARGET_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

        # Check data values are preserved
        assert result["LogD"].iloc[0] == 0.5
        assert result["Log KSOL"].iloc[0] == 2.0

    def test_data_preservation(self):
        """Data values should be preserved during normalization."""
        df = pd.DataFrame(
            {
                "KSOL (uM)": [100.0, 200.0, 300.0],
                "HLM CLint (mL/min/kg)": [10.0, 20.0, 30.0],
            }
        )

        result = normalize_dataframe_columns(df)

        # Values should be identical, just column names changed
        assert list(result["Log KSOL"]) == [100.0, 200.0, 300.0]
        assert list(result["Log HLM CLint"]) == [10.0, 20.0, 30.0]


class TestColumnAliasesCompleteness:
    """Tests to verify COLUMN_ALIASES covers expected formats."""

    def test_all_format2_unit_columns_mapped(self):
        """All Format 2 columns with units should have mappings."""
        format2_columns = [
            "KSOL (uM)",
            "HLM CLint (mL/min/kg)",
            "MLM CLint (mL/min/kg)",
            "Caco-2 Permeability Papp A$>$B (10$^{-6}$ cm/s)",
            "Caco-2 Permeability Efflux (None)",
            "MPPB ( pct unbound)",
            "MBPB ( pct unbound)",
            "MGMB ( pct unbound)",
        ]

        for col in format2_columns:
            assert col in COLUMN_ALIASES, f"Missing alias for: {col}"
            # Verify mapping is to a canonical column
            canonical = COLUMN_ALIASES[col]
            assert (
                canonical in CANONICAL_TARGET_COLUMNS or canonical == "LogD"
            ), f"Alias {col} maps to non-canonical: {canonical}"

    def test_latex_column_mapped(self):
        """LaTeX notation column should be mapped."""
        latex_col = "Log Caco-2 Permeability Papp A$>$B"
        assert latex_col in COLUMN_ALIASES
        assert COLUMN_ALIASES[latex_col] == "Log Caco-2 Permeability Papp A>B"
