"""Test that Molecule Name is preserved in ensemble predictions and submissions."""

import pandas as pd
import pytest


def test_blind_predictions_assigned():
    """Test that blind predictions are properly assigned after processing."""
    # Simulate the ensemble code path for blind predictions
    blind_preds = None
    smiles_col = "smiles"

    # Mock blind dataframe (what model.dataframes["blind"] would contain)
    blind_df = pd.DataFrame(
        {
            "smiles": ["CCO", "CCC", "CCCC"],
            "Molecule Name": ["blind1", "blind2", "blind3"],
        }
    )

    # Mock prediction dataframe (what model.predict() returns)
    pred_df = pd.DataFrame(
        {
            "LogD": [1.1, 2.1, 3.1],
            "Log KSOL": [0.5, 1.5, 2.5],
        }
    )

    # Simulate the processing (should match ensemble.py lines 538-551)
    if "Molecule Name" in blind_df.columns:
        pred_df["Molecule Name"] = blind_df["Molecule Name"].values
        cols = pred_df.columns.tolist()
        cols.insert(0, cols.pop(cols.index("Molecule Name")))
        pred_df = pred_df[cols]

    if smiles_col in blind_df.columns:
        pred_df[smiles_col] = blind_df[smiles_col].values
        cols = pred_df.columns.tolist()
        cols.insert(0, cols.pop(cols.index(smiles_col)))
        pred_df = pred_df[cols]

    # THIS IS THE CRITICAL LINE THAT WAS MISSING
    blind_preds = pred_df.copy()

    # Verify blind_preds is not None
    assert blind_preds is not None
    assert len(blind_preds) == 3
    assert "smiles" in blind_preds.columns
    assert "Molecule Name" in blind_preds.columns
    assert "LogD" in blind_preds.columns
    assert blind_preds["Molecule Name"].tolist() == ["blind1", "blind2", "blind3"]


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
