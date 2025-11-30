#!/usr/bin/env bash
# =============================================================================
# Chemprop Single Model Training Script
# =============================================================================
# This script trains individual Chemprop models for each split/fold combination.
# Unlike the ensemble script, this trains models one at a time without
# Ray parallelization.
#
# Usage:
#   ./scripts/train_chemprop_model.sh [--dry-run] [--log-level LEVEL] [--splits N] [--folds N]
#
# Options:
#   --dry-run     Print commands without executing
#   --log-level   Set logging level (DEBUG, INFO, WARNING, ERROR)
#   --splits      Number of splits to train (default: all)
#   --folds       Number of folds per split (default: all)
#
# =============================================================================

set -euo pipefail

# Source common library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

# Configuration
PROJECT_ROOT="$(get_project_root "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/configs/single_chemprop.yaml"

# Default options
DRY_RUN=false
LOG_LEVEL="INFO"
MAX_SPLITS=""
MAX_FOLDS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --splits)
            MAX_SPLITS="$2"
            shift 2
            ;;
        --folds)
            MAX_FOLDS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--log-level LEVEL] [--splits N] [--folds N]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Change to project root
cd "$PROJECT_ROOT"

# Check config file exists
check_config_exists "$CONFIG_FILE" || exit 1

# Get data directories
mapfile -t BASE_DATA_DIRS < <(get_data_dirs)

log_info "Starting single model training"
log_info "Config file: $CONFIG_FILE"
log_info "Base data directories: ${#BASE_DATA_DIRS[@]}"

# Track results
TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

# Iterate through each base data directory
for base_dir in "${BASE_DATA_DIRS[@]}"; do
    print_header "Processing base directory: $base_dir"

    # Check if directory exists
    if ! check_dir_exists "$base_dir"; then
        log_warn "Directory not found, skipping: $base_dir"
        continue
    fi

    # Find all split directories
    mapfile -t split_dirs < <(find_split_dirs "$base_dir" "$MAX_SPLITS")

    log_info "Found ${#split_dirs[@]} splits"

    # Iterate through splits
    for split_dir in "${split_dirs[@]}"; do
        split_name=$(basename "$split_dir")

        # Find all fold directories
        mapfile -t fold_dirs < <(find_fold_dirs "$split_dir" "$MAX_FOLDS")

        log_info "  $split_name: ${#fold_dirs[@]} folds"

        # Iterate through folds
        for fold_dir in "${fold_dirs[@]}"; do
            fold_name=$(basename "$fold_dir")
            ((TOTAL++))

            echo ""
            log_info "Training: $split_name/$fold_name"
            log_info "  Data dir: $fold_dir"

            # Check for required files
            if ! check_training_files "$fold_dir"; then
                log_warn "Missing train.csv or validation.csv, skipping"
                ((SKIPPED++))
                continue
            fi

            # Create temporary config with updated data_dir
            TEMP_CONFIG=$(create_temp_config "$CONFIG_FILE" "$fold_dir")

            # Build command
            CMD="python -m admet.model.chemprop.model --config $TEMP_CONFIG --log-level $LOG_LEVEL"

            # Execute command
            if execute_command "$CMD" "$DRY_RUN" "$split_name/$fold_name"; then
                if [[ "$DRY_RUN" != "true" ]]; then
                    log_success "Completed: $split_name/$fold_name"
                    ((SUCCESS++))
                fi
            else
                log_error "Failed: $split_name/$fold_name"
                ((FAILED++))
            fi

            # Clean up temp config
            cleanup_temp_config "$TEMP_CONFIG"
        done
    done
done

# Print summary and exit with appropriate code
print_summary "$TOTAL" "$SUCCESS" "$FAILED" "$SKIPPED" || exit 1
