#!/bin/bash
#
# Quick Commit Diff Analyzer
# ==========================
#
# This script extracts and analyzes the exact changes between
# the good and bad commits.
#
# Usage: bash analyze_commit_diff.sh

set -e

GOOD_COMMIT="0d41199f07d930062b943681357a7029554961f6"
BAD_COMMIT="1f74f8396f7be56ad573256d4769ed0e96d9d69b"
OUTPUT_DIR="/tmp/ensemble_bug_analysis"

mkdir -p "$OUTPUT_DIR"

echo "================================"
echo "Ensemble Regression Bug Analysis"
echo "================================"
echo ""
echo "Good commit: $GOOD_COMMIT"
echo "Bad commit:  $BAD_COMMIT"
echo "Output dir:  $OUTPUT_DIR"
echo ""

# Step 1: Get list of changed files
echo "[Step 1] Getting list of changed files..."
git diff --name-only $GOOD_COMMIT $BAD_COMMIT > "$OUTPUT_DIR/changed_files.txt"
NUM_FILES=$(wc -l < "$OUTPUT_DIR/changed_files.txt")
echo "  → $NUM_FILES files changed"
cat "$OUTPUT_DIR/changed_files.txt"
echo ""

# Step 2: Get full diff
echo "[Step 2] Extracting full diff..."
git diff $GOOD_COMMIT $BAD_COMMIT > "$OUTPUT_DIR/full_diff.patch"
DIFF_SIZE=$(wc -l < "$OUTPUT_DIR/full_diff.patch")
echo "  → $DIFF_SIZE lines in diff"
echo ""

# Step 3: Extract ensemble.py specific changes
echo "[Step 3] Analyzing src/admet/model/chemprop/ensemble.py..."
if git diff $GOOD_COMMIT $BAD_COMMIT -- src/admet/model/chemprop/ensemble.py > "$OUTPUT_DIR/ensemble_diff.patch"; then
    ENSEMBLE_DIFF_SIZE=$(wc -l < "$OUTPUT_DIR/ensemble_diff.patch")
    if [ $ENSEMBLE_DIFF_SIZE -gt 0 ]; then
        echo "  → ✗ ensemble.py CHANGED ($ENSEMBLE_DIFF_SIZE lines)"
        echo ""
        echo "  Showing diff:"
        echo "  ------------"
        cat "$OUTPUT_DIR/ensemble_diff.patch"
        echo ""

        # Extract just the _aggregate_predictions changes
        echo "  [3a] Extracting _aggregate_predictions changes..."
        git show $GOOD_COMMIT:src/admet/model/chemprop/ensemble.py | \
            sed -n '/def _aggregate_predictions/,/^    def /p' | head -n -1 \
            > "$OUTPUT_DIR/aggregate_predictions_good.py"

        git show $BAD_COMMIT:src/admet/model/chemprop/ensemble.py | \
            sed -n '/def _aggregate_predictions/,/^    def /p' | head -n -1 \
            > "$OUTPUT_DIR/aggregate_predictions_bad.py"

        echo "  → Saved to aggregate_predictions_good.py and aggregate_predictions_bad.py"
        echo ""
        echo "  Diff of _aggregate_predictions:"
        diff -u "$OUTPUT_DIR/aggregate_predictions_good.py" "$OUTPUT_DIR/aggregate_predictions_bad.py" || true
    else
        echo "  → ✓ ensemble.py unchanged"
    fi
else
    echo "  → ensemble.py not found in one of the commits"
fi
echo ""

# Step 4: Check other critical files
echo "[Step 4] Checking other critical files..."
for file in \
    "src/admet/model/chemprop/model.py" \
    "src/admet/model/base.py" \
    "src/admet/data/smiles.py" \
    "configs/2-hpo-ensemble/ensemble_chemprop_hpo_001.yaml"; do

    if git diff $GOOD_COMMIT $BAD_COMMIT -- "$file" > "$OUTPUT_DIR/$(basename $file).diff" 2>/dev/null; then
        DIFF_SIZE=$(wc -l < "$OUTPUT_DIR/$(basename $file).diff")
        if [ $DIFF_SIZE -gt 0 ]; then
            echo "  → ✗ $file CHANGED ($DIFF_SIZE lines)"
        else
            echo "  → ✓ $file unchanged"
        fi
    else
        echo "  → ⚠ $file not accessible"
    fi
done
echo ""

# Step 5: Extract commit message
echo "[Step 5] Extracting commit message for bad commit..."
git log -1 --pretty=format:"%H%n%an <%ae>%n%ad%n%n%B" $BAD_COMMIT > "$OUTPUT_DIR/bad_commit_message.txt"
echo "  Commit message:"
echo "  --------------"
cat "$OUTPUT_DIR/bad_commit_message.txt"
echo ""

# Step 6: Check for imports/dependencies changes
echo "[Step 6] Checking for import changes in ensemble.py..."
echo "  Good commit imports:"
git show $GOOD_COMMIT:src/admet/model/chemprop/ensemble.py | grep "^import\|^from" | head -20
echo ""
echo "  Bad commit imports:"
git show $BAD_COMMIT:src/admet/model/chemprop/ensemble.py | grep "^import\|^from" | head -20
echo ""

# Step 7: Summary
echo "================================"
echo "SUMMARY"
echo "================================"
echo ""
echo "Analysis files saved to: $OUTPUT_DIR"
echo ""
echo "Key files to review:"
echo "  1. full_diff.patch - Complete diff between commits"
echo "  2. ensemble_diff.patch - Changes to ensemble.py"
echo "  3. aggregate_predictions_*.py - The critical function comparison"
echo "  4. changed_files.txt - List of all changed files"
echo ""
echo "Next steps:"
echo "  1. Review ensemble_diff.patch for obvious bugs"
echo "  2. Compare aggregate_predictions_good.py vs aggregate_predictions_bad.py"
echo "  3. Check if column naming or array operations changed"
echo "  4. Look for changes in log transformation logic"
echo ""

# Step 8: Quick bug checks
echo "================================"
echo "AUTOMATED BUG CHECKS"
echo "================================"
echo ""

# Check for common bug patterns in the bad commit
BAD_ENSEMBLE="$OUTPUT_DIR/ensemble_bad_full.py"
git show $BAD_COMMIT:src/admet/model/chemprop/ensemble.py > "$BAD_ENSEMBLE"

echo "[Check 1] Looking for 'pred_col = target' usage..."
if grep -n "pred_col = target" "$BAD_ENSEMBLE"; then
    echo "  → Found. Verify this is correct column name."
else
    echo "  → Not found. May use different column naming."
fi
echo ""

echo "[Check 2] Looking for log transformation logic..."
if grep -A 5 "startswith.*Log " "$BAD_ENSEMBLE" | grep -n "np.power"; then
    echo "  → Found log transformation. Check order: mean THEN transform, not transform THEN mean."
else
    echo "  → Log transformation pattern not found as expected."
fi
echo ""

echo "[Check 3] Looking for aggregation of predictions..."
if grep -n "np.mean.*preds.*axis=0" "$BAD_ENSEMBLE"; then
    echo "  → Found np.mean with axis=0. Verify this averages across models correctly."
else
    echo "  → Standard mean pattern not found."
fi
echo ""

echo "[Check 4] Checking for array stacking..."
if grep -n "np.array.*\[df\[" "$BAD_ENSEMBLE"; then
    echo "  → Found list comprehension with df access. Verify column names are correct."
else
    echo "  → Array stacking pattern different."
fi
echo ""

echo "================================"
echo "Analysis complete!"
echo "================================"
