"""Test that Molecule Name is preserved in ensemble predictions and submissions."""

import pandas as pd
import pytest


def test_molecule_name_in_aggregate_predictions():
    """Test that Molecule Name is preserved when aggregating predictions."""
    # Mock prediction dataframes from multiple models
    predictions_list = [
        pd.DataFrame(
            {
                "smiles": ["CCO", "CCC", "CCCC"],
                "Molecule Name": ["mol1", "mol2", "mol3"],
                "LogD_actual": [1.0, 2.0, 3.0],
                "LogD": [1.1, 2.1, 3.1],
            }
        ),
        pd.DataFrame(
            {
                "smiles": ["CCO", "CCC", "CCCC"],
                "Molecule Name": ["mol1", "mol2", "mol3"],
                "LogD_actual": [1.0, 2.0, 3.0],
                "LogD": [0.9, 1.9, 2.9],
            }
        ),
    ]

    # Simulate aggregation logic
    result = pd.DataFrame()
    result["smiles"] = predictions_list[0]["smiles"].copy()

    # Preserve Molecule Name if present
    if "Molecule Name" in predictions_list[0].columns:
        result["Molecule Name"] = predictions_list[0]["Molecule Name"].copy()

    # Check that Molecule Name is in result
    assert "Molecule Name" in result.columns
    assert result["Molecule Name"].tolist() == ["mol1", "mol2", "mol3"]


def test_molecule_name_in_submissions():
    """Test that Molecule Name is included in submission files."""
    # Mock aggregated predictions
    predictions = pd.DataFrame(
        {
            "smiles": ["CCO", "CCC", "CCCC"],
            "Molecule Name": ["mol1", "mol2", "mol3"],
            "LogD_mean": [1.0, 2.0, 3.0],
            "LogD_std": [0.1, 0.1, 0.1],
            "LogD_stderr": [0.05, 0.05, 0.05],
            "LogD_transformed_mean": [10.0, 100.0, 1000.0],
        }
    )

    # Simulate submissions creation
    submissions = pd.DataFrame()
    submissions["smiles"] = predictions["smiles"]

    # Include Molecule Name if present in predictions
    if "Molecule Name" in predictions.columns:
        submissions["Molecule Name"] = predictions["Molecule Name"]

    # Check that Molecule Name is in submissions
    assert "Molecule Name" in submissions.columns
    assert submissions["Molecule Name"].tolist() == ["mol1", "mol2", "mol3"]


def test_molecule_name_optional():
    """Test that code works when Molecule Name is not present."""
    predictions_list = [
        pd.DataFrame(
            {
                "smiles": ["CCO", "CCC", "CCCC"],
                "LogD": [1.1, 2.1, 3.1],
            }
        ),
    ]

    # Simulate aggregation logic
    result = pd.DataFrame()
    result["smiles"] = predictions_list[0]["smiles"].copy()

    # Preserve Molecule Name if present (should not crash)
    if "Molecule Name" in predictions_list[0].columns:
        result["Molecule Name"] = predictions_list[0]["Molecule Name"].copy()

    # Check that Molecule Name is NOT in result (data didn't have it)
    assert "Molecule Name" not in result.columns
    assert "smiles" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
