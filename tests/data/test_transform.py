"""Tests for data transformation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from admet.data.transform import apply_endpoint_transformations


class TestApplyEndpointTransformations:
    """Test suite for apply_endpoint_transformations function."""

    def test_clint_transformation_with_zeros(self):
        """CLint endpoints: zeros replaced with half of smallest non-zero."""
        df = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCN", "CCC"],
                "Log HLM CLint (ul/min/mg)": [10.0, 0.0, 5.0],
            }
        )

        result = apply_endpoint_transformations(df)

        # Check that transformation was applied
        assert result["Log HLM CLint (ul/min/mg)"][0] == pytest.approx(1.0)  # log10(10)
        assert result["Log HLM CLint (ul/min/mg)"][2] == pytest.approx(np.log10(5.0))

        # Zero should be replaced with log10(min_nonzero / 2) = log10(5/2) = log10(2.5)
        expected_zero_value = np.log10(5.0 / 2)
        assert result["Log HLM CLint (ul/min/mg)"][1] == pytest.approx(expected_zero_value)

    def test_permeability_transformation_with_zeros(self):
        """Permeability endpoints: zeros replaced with half of smallest non-zero."""
        df = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCN"],
                "Log Caco-2 Permeability Papp A$>$B (nm/s)": [100.0, 0.0],
            }
        )

        result = apply_endpoint_transformations(df)

        # Zero should be replaced with log10(min_nonzero / 2) = log10(100/2) = log10(50)
        expected_zero_value = np.log10(100.0 / 2)
        # After sanitization, column name remains A$>$B (idempotent now)
        assert result["Log Caco-2 Permeability Papp A$>$B (nm/s)"][1] == pytest.approx(expected_zero_value)

    def test_ppb_transformation_with_zeros(self):
        """PPB endpoints: zeros set to 10^(-6) before log10."""
        df = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCN"],
                "Log MPPB ( pct)": [95.0, 0.0],
            }
        )

        result = apply_endpoint_transformations(df)

        # Zero should be replaced with log10(1e-6)
        assert result["Log MPPB ( pct)"][1] == pytest.approx(np.log10(1e-6))
        assert result["Log MPPB ( pct)"][0] == pytest.approx(np.log10(95.0))

    def test_standard_transformation(self):
        """Other endpoints: standard log10 with zeros handled."""
        df = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCN"],
                "Log KSOL (ug/ml)": [1000.0, 1e-6],
            }
        )

        result = apply_endpoint_transformations(df)

        # Should apply standard log10 transformation
        assert result["Log KSOL (ug/ml)"][1] == pytest.approx(np.log10(1e-6))
        assert result["Log KSOL (ug/ml)"][0] == pytest.approx(3.0)  # log10(1000)

    def test_excluded_columns_not_transformed(self):
        """Excluded columns should not be transformed."""
        df = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCN"],
                "LogD (None)": [2.5, 3.0],
                "Molecule Name (None)": ["mol1", "mol2"],
            }
        )

        result = apply_endpoint_transformations(df)

        # These columns should remain unchanged
        assert result["LogD (None)"].equals(df["LogD (None)"])
        assert result["Molecule Name (None)"].equals(df["Molecule Name (None)"])
        assert result["SMILES"].equals(df["SMILES"])

    def test_inplace_modification(self):
        """Test inplace=True modifies the original dataframe."""
        df = pd.DataFrame(
            {
                "SMILES": ["CCO"],
                "Log HLM CLint (ul/min/mg)": [10.0],
            }
        )

        result = apply_endpoint_transformations(df, inplace=True)

        # Result should be None
        assert result is None
        # Original dataframe should be modified
        assert df["Log HLM CLint (ul/min/mg)"][0] == pytest.approx(1.0)

    def test_copy_not_inplace(self):
        """Test inplace=False returns a new dataframe."""
        df = pd.DataFrame(
            {
                "SMILES": ["CCO"],
                "Log HLM CLint (ul/min/mg)": [10.0],
            }
        )

        original_value = df["Log HLM CLint (ul/min/mg)"][0]
        result = apply_endpoint_transformations(df, inplace=False)

        # Original should be unchanged
        assert df["Log HLM CLint (ul/min/mg)"][0] == original_value
        # Result should be different
        assert result["Log HLM CLint (ul/min/mg)"][0] == pytest.approx(1.0)

    def test_custom_endpoint_lists(self):
        """Test with custom endpoint lists."""
        df = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCN"],
                "Custom Endpoint": [0.0, 10.0],
            }
        )

        result = apply_endpoint_transformations(
            df,
            clint_endpoints=["Custom Endpoint"],
        )

        # Should use CLint strategy (half of min non-zero)
        expected_zero_value = np.log10(10.0 / 2)
        assert result["Custom Endpoint"][0] == pytest.approx(expected_zero_value)

    def test_all_zeros_in_column(self):
        """Test handling when all values in a column are zero."""
        df = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCN"],
                "Log HLM CLint (ul/min/mg)": [0.0, 0.0],
            }
        )

        result = apply_endpoint_transformations(df)

        # Should fall back to detection limit (1e-6)
        expected_value = np.log10(1e-6 / 2)
        assert result["Log HLM CLint (ul/min/mg)"][0] == pytest.approx(expected_value)
        assert result["Log HLM CLint (ul/min/mg)"][1] == pytest.approx(expected_value)

    def test_mixed_endpoint_types(self):
        """Test dataframe with multiple endpoint types."""
        df = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCN", "CCC"],
                "Log HLM CLint (ul/min/mg)": [10.0, 0.0, 5.0],
                "Log MPPB ( pct)": [95.0, 0.0, 80.0],
                "Log KSOL (ug/ml)": [1000.0, 1e-6, 500.0],
                "LogD (None)": [2.5, 3.0, 2.0],
            }
        )

        result = apply_endpoint_transformations(df)

        # CLint: zeros replaced with half of min non-zero
        assert result["Log HLM CLint (ul/min/mg)"][1] == pytest.approx(np.log10(5.0 / 2))

        # PPB: zeros set to 1e-6
        assert result["Log MPPB ( pct)"][1] == pytest.approx(np.log10(1e-6))

        # KSOL: standard transformation
        assert result["Log KSOL (ug/ml)"][1] == pytest.approx(np.log10(1e-6))

        # LogD: unchanged
        assert result["LogD (None)"].equals(df["LogD (None)"])

    def test_non_numeric_columns_skipped(self):
        """Non-numeric columns should be skipped."""
        df = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCN"],
                "Text Column": ["A", "B"],
                "Log HLM CLint (ul/min/mg)": [10.0, 5.0],
            }
        )

        result = apply_endpoint_transformations(df)

        # Text column should remain unchanged
        assert result["Text Column"].equals(df["Text Column"])
        # Numeric column should be transformed
        assert result["Log HLM CLint (ul/min/mg)"][0] == pytest.approx(1.0)
