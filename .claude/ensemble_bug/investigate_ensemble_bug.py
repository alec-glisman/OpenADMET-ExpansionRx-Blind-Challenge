#!/usr/bin/env python3
"""
Investigation Script: Ensemble Regression Bug
==============================================

This script helps identify the bug between commits 0d41199f and 1f74f83.
Run this after checking out each commit to compare behavior.

Usage:
    # On good commit
    git checkout 0d41199f
    python investigate_ensemble_bug.py --label good --output /tmp/good_analysis.json

    # On bad commit
    git checkout 1f74f83
    python investigate_ensemble_bug.py --label bad --output /tmp/bad_analysis.json

    # Compare
    python investigate_ensemble_bug.py --compare /tmp/good_analysis.json /tmp/bad_analysis.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def get_git_info():
    """Get current git commit information."""
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

        commit_message = subprocess.check_output(["git", "log", "-1", "--pretty=%B"], text=True).strip()

        return {"commit_hash": commit_hash, "commit_message": commit_message}
    except subprocess.CalledProcessError as e:
        return {"error": str(e)}


def analyze_ensemble_module():
    """Analyze the ensemble module for potential issues."""
    ensemble_path = Path("src/admet/model/chemprop/ensemble.py")

    if not ensemble_path.exists():
        return {"error": "ensemble.py not found"}

    with open(ensemble_path, "r") as f:
        content = f.read()

    # Extract key functions
    analysis = {
        "file_size": len(content),
        "line_count": content.count("\n"),
        "has_aggregate_predictions": "_aggregate_predictions" in content,
        "has_save_predictions": "_save_ensemble_predictions" in content,
    }

    # Check for specific patterns that might indicate bugs
    patterns = {
        "np.mean(preds": "np.mean(preds" in content,
        "np.power(10": "np.power(10" in content,
        'startswith("Log ")': 'startswith("Log ")' in content,
        "target}_mean": "{target}_mean" in content or 'f"{target}_mean"' in content,
        "pred_col = target": "pred_col = target" in content,
    }
    analysis["patterns"] = patterns

    # Extract the _aggregate_predictions function
    agg_pred_start = content.find("def _aggregate_predictions(")
    if agg_pred_start != -1:
        # Find the end of the function (next def or class)
        agg_pred_end = content.find("\n    def ", agg_pred_start + 1)
        if agg_pred_end == -1:
            agg_pred_end = content.find("\nclass ", agg_pred_start + 1)

        if agg_pred_end != -1:
            func_content = content[agg_pred_start:agg_pred_end]
            analysis["aggregate_predictions_func"] = {
                "length": len(func_content),
                "lines": func_content.count("\n"),
                "content_hash": hash(func_content),  # Simple hash to detect changes
            }

            # Extract key lines
            lines = func_content.split("\n")
            key_lines = []
            for i, line in enumerate(lines):
                if any(keyword in line for keyword in ["np.mean", "np.power", "startswith", "result["]):
                    key_lines.append(f"Line {i}: {line.strip()}")

            analysis["aggregate_predictions_func"]["key_lines"] = key_lines

    return analysis


def analyze_config():
    """Analyze configuration structure."""
    config_path = Path("configs/2-hpo-ensemble/ensemble_chemprop_hpo_001.yaml")

    if not config_path.exists():
        return {"error": "config file not found"}

    with open(config_path, "r") as f:
        content = f.read()

    return {
        "file_size": len(content),
        "has_target_cols": "target_cols:" in content,
        "has_log_endpoints": "Log " in content,
    }


def analyze_tests():
    """Check test coverage for ensemble."""
    test_files = [
        "tests/test_ensemble_chemprop.py",
        "tests/test_ensemble_blind_predictions.py",
    ]

    results = {}
    for test_file in test_files:
        test_path = Path(test_file)
        if test_path.exists():
            with open(test_path, "r") as f:
                content = f.read()
            results[test_file] = {
                "exists": True,
                "size": len(content),
                "has_aggregate_test": "aggregate" in content.lower(),
            }
        else:
            results[test_file] = {"exists": False}

    return results


def run_analysis(label):
    """Run complete analysis of current commit."""
    print(f"Running analysis for: {label}")

    analysis = {
        "label": label,
        "git_info": get_git_info(),
        "ensemble_module": analyze_ensemble_module(),
        "config": analyze_config(),
        "tests": analyze_tests(),
    }

    return analysis


def compare_analyses(good_file, bad_file):
    """Compare two analysis files to identify differences."""
    with open(good_file, "r") as f:
        good = json.load(f)

    with open(bad_file, "r") as f:
        bad = json.load(f)

    print("\n" + "=" * 80)
    print("COMPARISON: GOOD vs BAD COMMITS")
    print("=" * 80)

    print(f"\nGood commit: {good['git_info'].get('commit_hash', 'unknown')}")
    print(f"Bad commit:  {bad['git_info'].get('commit_hash', 'unknown')}")

    # Compare ensemble module
    print("\n--- Ensemble Module Changes ---")

    good_ens = good["ensemble_module"]
    bad_ens = bad["ensemble_module"]

    if good_ens.get("file_size") != bad_ens.get("file_size"):
        print(f"✗ File size changed: {good_ens.get('file_size')} → {bad_ens.get('file_size')}")

    if good_ens.get("line_count") != bad_ens.get("line_count"):
        print(f"✗ Line count changed: {good_ens.get('line_count')} → {bad_ens.get('line_count')}")

    # Compare patterns
    print("\n--- Pattern Changes ---")
    good_patterns = good_ens.get("patterns", {})
    bad_patterns = bad_ens.get("patterns", {})

    for pattern, good_val in good_patterns.items():
        bad_val = bad_patterns.get(pattern)
        if good_val != bad_val:
            print(f"✗ Pattern '{pattern}': {good_val} → {bad_val}")

    # Compare aggregate_predictions function
    print("\n--- _aggregate_predictions Function ---")
    good_func = good_ens.get("aggregate_predictions_func", {})
    bad_func = bad_ens.get("aggregate_predictions_func", {})

    if good_func.get("content_hash") != bad_func.get("content_hash"):
        print("✗ Function content CHANGED!")
        print(f"  Good hash: {good_func.get('content_hash')}")
        print(f"  Bad hash:  {bad_func.get('content_hash')}")

        print("\n--- Key Lines Comparison ---")
        print("GOOD commit:")
        for line in good_func.get("key_lines", []):
            print(f"  {line}")

        print("\nBAD commit:")
        for line in bad_func.get("key_lines", []):
            print(f"  {line}")
    else:
        print("✓ Function content unchanged")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Investigate ensemble regression bug")
    parser.add_argument("--label", help='Label for this analysis (e.g., "good" or "bad")')
    parser.add_argument("--output", type=Path, help="Output JSON file")
    parser.add_argument("--compare", nargs=2, type=Path, help="Compare two analysis files (good bad)")

    args = parser.parse_args()

    if args.compare:
        compare_analyses(args.compare[0], args.compare[1])
    elif args.label and args.output:
        analysis = run_analysis(args.label)

        with open(args.output, "w") as f:
            json.dump(analysis, f, indent=2)

        print(f"\nAnalysis saved to: {args.output}")
        print(f"Commit: {analysis['git_info'].get('commit_hash', 'unknown')}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
