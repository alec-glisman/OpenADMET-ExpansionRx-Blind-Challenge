#!/usr/bin/env bash
# =============================================================================
# Train Chemprop HPO Ensemble Models
# =============================================================================
# This script trains ensemble models using the top 100 HPO configurations
# in order from best (rank 1) to worst (rank 100).
#
# Usage:
#   ./scripts/train_chemprop_hpo_ensembles.sh
#   ./scripts/train_chemprop_hpo_ensembles.sh --start 1 --end 10
#   ./scripts/train_chemprop_hpo_ensembles.sh --ranks 1,5,10
#
# Environment:
#   Assumes virtual environment is activated and all dependencies installed.
# =============================================================================

set -euo pipefail

# Default values
START_RANK=1
END_RANK=100
SPECIFIC_RANKS=""
MAX_PARALLEL=4
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --start)
            START_RANK="$2"
            shift 2
            ;;
        --end)
            END_RANK="$2"
            shift 2
            ;;
        --ranks)
            SPECIFIC_RANKS="$2"
            shift 2
            ;;
        --max-parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --start N          Start from rank N (default: 1)"
            echo "  --end N            End at rank N (default: 100)"
            echo "  --ranks N,M,K      Train specific ranks only (comma-separated)"
            echo "  --max-parallel N   Max parallel jobs (default: 5)"
            echo "  --dry-run          Print commands without executing"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to train a single ensemble
train_ensemble() {
    local rank=$1
    local config_file="configs/ensemble_chemprop_hpo_$(printf "%03d" "$rank").yaml"
    
    if [[ ! -f "$config_file" ]]; then
        echo "ERROR: Config file not found: $config_file"
        return 1
    fi
    
    echo "=========================================="
    echo "Training HPO Ensemble Rank $rank"
    echo "Config: $config_file"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would execute:"
        echo "python -m admet.model.chemprop.ensemble --config $config_file --max-parallel $MAX_PARALLEL"
        return 0
    fi
    
    # Run ensemble training
    if python -m admet.model.chemprop.ensemble \
        --config "$config_file" \
        --max-parallel "$MAX_PARALLEL"; then
        echo "✓ Successfully completed ensemble rank $rank"
        echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
        return 0
    else
        echo "✗ Failed to train ensemble rank $rank"
        echo "Failed at: $(date '+%Y-%m-%d %H:%M:%S')"
        return 1
    fi
}

# Main execution
echo "============================================="
echo "Chemprop HPO Ensemble Training Pipeline"
echo "============================================="
echo "Start rank: $START_RANK"
echo "End rank: $END_RANK"
echo "Max parallel: $MAX_PARALLEL"
echo "Dry run: $DRY_RUN"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
echo ""

# Track success/failure
SUCCESSFUL=()
FAILED=()

# Train specific ranks if provided
if [[ -n "$SPECIFIC_RANKS" ]]; then
    IFS=',' read -ra RANKS <<< "$SPECIFIC_RANKS"
    for rank in "${RANKS[@]}"; do
        if train_ensemble "$rank"; then
            SUCCESSFUL+=("$rank")
        else
            FAILED+=("$rank")
        fi
        echo ""
    done
else
    # Train range of ranks
    for rank in $(seq "$START_RANK" "$END_RANK"); do
        if train_ensemble "$rank"; then
            SUCCESSFUL+=("$rank")
        else
            FAILED+=("$rank")
        fi
        echo ""
    done
fi

# Summary
echo "============================================="
echo "Training Complete"
echo "============================================="
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Successful: ${#SUCCESSFUL[@]}"
if [[ ${#SUCCESSFUL[@]} -gt 0 ]]; then
    echo "  Ranks: ${SUCCESSFUL[*]}"
fi
echo ""
echo "Failed: ${#FAILED[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Ranks: ${FAILED[*]}"
fi
echo "============================================="

# Exit with error if any failed
if [[ ${#FAILED[@]} -gt 0 ]]; then
    exit 1
fi
