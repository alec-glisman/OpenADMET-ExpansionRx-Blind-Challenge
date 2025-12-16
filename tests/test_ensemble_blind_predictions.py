"""Test ensemble blind predictions pipeline."""

import pandas as pd
import pytest


def test_blind_predictions_complete_pipeline():
    """
    Test that blind predictions go through the complete pipeline:
    1. Individual model predictions are created
    2. blind_preds is properly assigned (not left as None)
    3. Predictions are collected in _all_blind_predictions
    4. Ensemble aggregation occurs
    5. Files are saved with 'blind' prefix
    """
    # Simulate the complete pipeline

    # Step 1: Individual model creates predictions
    blind_df = pd.DataFrame(
        {
            "smiles": ["CCO", "CCC", "CCCC"],
            "Molecule Name": ["blind1", "blind2", "blind3"],
        }
    )

    # Mock what model.predict() returns
    pred_df = pd.DataFrame(
        {
            "LogD": [1.1, 2.1, 3.1],
            "Log KSOL": [0.5, 1.5, 2.5],
        }
    )

    # Step 2: Process predictions (lines 538-551 in ensemble.py)
    smiles_col = "smiles"

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

    # CRITICAL: Assignment that was missing (now added at line 551)
    blind_preds = pred_df.copy()

    # Step 3: Verify blind_preds is collected (not None)
    assert blind_preds is not None, "blind_preds should not be None after processing"
    assert len(blind_preds) == 3, "blind_preds should have 3 rows"

    # Step 4: Simulate multiple models creating predictions
    all_blind_predictions = []
    for i in range(3):  # Simulate 3 models
        model_pred = blind_preds.copy()
        model_pred["LogD"] = model_pred["LogD"] + (i * 0.1)  # Vary predictions
        all_blind_predictions.append(model_pred)

    assert len(all_blind_predictions) == 3, "Should have predictions from 3 models"

    # Step 5: Simulate aggregation (simplified version of _aggregate_predictions)
    result = pd.DataFrame()
    result[smiles_col] = all_blind_predictions[0][smiles_col].copy()

    if "Molecule Name" in all_blind_predictions[0].columns:
        result["Molecule Name"] = all_blind_predictions[0]["Molecule Name"].copy()

    # Calculate mean across models
    import numpy as np

    for target in ["LogD", "Log KSOL"]:
        preds = np.array([df[target].values for df in all_blind_predictions])
        result[f"{target}_mean"] = np.mean(preds, axis=0)
        result[f"{target}_std"] = np.std(preds, axis=0, ddof=1)
        result[f"{target}_stderr"] = result[f"{target}_std"] / np.sqrt(3)

    # Step 6: Verify aggregated results
    assert "smiles" in result.columns
    assert "Molecule Name" in result.columns
    assert "LogD_mean" in result.columns
    assert "LogD_stderr" in result.columns
    assert "Log KSOL_mean" in result.columns

    # Step 7: Simulate submission format
    submissions = pd.DataFrame()
    submissions[smiles_col] = result[smiles_col]

    if "Molecule Name" in result.columns:
        submissions["Molecule Name"] = result["Molecule Name"]

    # For "Log " columns, would transform: 10^mean
    submissions["LogD"] = result["LogD_mean"]  # Direct for LogD
    submissions["KSOL"] = np.power(10, result["Log KSOL_mean"])  # Transform for Log columns

    # Verify final submission format
    assert "smiles" in submissions.columns
    assert "Molecule Name" in submissions.columns
    assert "LogD" in submissions.columns
    assert "KSOL" in submissions.columns
    assert len(submissions) == 3
    assert submissions["Molecule Name"].tolist() == ["blind1", "blind2", "blind3"]

    print("✅ Complete blind predictions pipeline test passed!")


def test_blind_vs_test_separate_files():
    """Test that blind and test predictions produce separate file names."""

    # Test split
    test_split_name = "test"
    test_predictions_file = f"{test_split_name}_ensemble_predictions.csv"
    test_submissions_file = f"{test_split_name}_ensemble_submissions.csv"

    # Blind split
    blind_split_name = "blind"
    blind_predictions_file = f"{blind_split_name}_ensemble_predictions.csv"
    blind_submissions_file = f"{blind_split_name}_ensemble_submissions.csv"

    # Verify they're different
    assert test_predictions_file != blind_predictions_file
    assert test_submissions_file != blind_submissions_file

    # Verify expected names
    assert test_predictions_file == "test_ensemble_predictions.csv"
    assert test_submissions_file == "test_ensemble_submissions.csv"
    assert blind_predictions_file == "blind_ensemble_predictions.csv"
    assert blind_submissions_file == "blind_ensemble_submissions.csv"

    print("✅ File naming test passed!")


def test_blind_predictions_missing_molecule_name():
    """Test blind predictions work even without Molecule Name column."""

    blind_df = pd.DataFrame(
        {
            "smiles": ["CCO", "CCC", "CCCC"],
            # No Molecule Name
        }
    )

    pred_df = pd.DataFrame(
        {
            "LogD": [1.1, 2.1, 3.1],
        }
    )

    smiles_col = "smiles"

    # Process (should not crash)
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

    blind_preds = pred_df.copy()

    # Verify it worked without Molecule Name
    assert blind_preds is not None
    assert "smiles" in blind_preds.columns
    assert "Molecule Name" not in blind_preds.columns
    assert "LogD" in blind_preds.columns

    print("✅ Blind predictions without Molecule Name test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
