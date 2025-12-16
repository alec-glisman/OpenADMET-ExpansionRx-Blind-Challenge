#!/usr/bin/env bash
# =============================================================================
# Production Ensemble Training Script
# =============================================================================
# This script trains all production ensemble models from configs/3-production/
# These are the top-performing configurations selected from HPO results.
#
# Usage:
#   ./scripts/training/train_production_ensembles.sh
#   ./scripts/training/train_production_ensembles.sh --dry-run
#   ./scripts/training/train_production_ensembles.sh --config ensemble_chemprop_hpo_001.yaml
#   ./scripts/training/train_production_ensembles.sh --max-parallel 2
#
# Options:
#   --dry-run         Print commands without executing
#   --max-parallel N  Override max parallel models (default: from config)
#   --log-level L     Set logging level (DEBUG, INFO, WARNING, ERROR)
#   --config FILE     Train specific config file only
#   --continue-from N Continue from config number N (useful after failures)
#
# =============================================================================

set -euo pipefail

# Source common library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

# Configuration
# Project root is two levels above this script (repo root)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PRODUCTION_CONFIG_DIR="${PROJECT_ROOT}/configs/3-production"

# Default options
DRY_RUN=false
MAX_PARALLEL="5"
LOG_LEVEL="INFO"
SPECIFIC_CONFIG=""
CONTINUE_FROM=""
DATA_DIR_OVERRIDE=""
SKIP_RANKS=""

# =============================================================================
# Argument Parsing
# =============================================================================

show_help() {
  cat <<EOF
Usage: $0 [OPTIONS]

Train all production ensemble models from configs/3-production/

Options:
  --dry-run         Print commands without executing
  --max-parallel N  Override max parallel models (default: from config)
  --log-level L     Set logging level (DEBUG, INFO, WARNING, ERROR)
  --config FILE     Train specific config file only (e.g., ensemble_chemprop_hpo_001.yaml)
  --continue-from N Continue from config number N (useful after failures)
  --data-dir DIR    Override `data.data_dir` in each config with this path
  --skip-ranks L    Comma-separated list of rank numbers to skip (e.g. 1 or 001,6)
  -h, --help        Show this help message

Examples:
  $0
  $0 --dry-run
  $0 --config ensemble_chemprop_hpo_001.yaml
  $0 --max-parallel 2
  $0 --continue-from 19
  $0 --data-dir assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data

  # Skip examples:
  $0 --skip-ranks 1

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
  --dry-run)
    DRY_RUN=true
    shift
    ;;
  --max-parallel)
    MAX_PARALLEL="$2"
    shift 2
    ;;
  --log-level)
    LOG_LEVEL="$2"
    shift 2
    ;;
  --config)
    SPECIFIC_CONFIG="$2"
    shift 2
    ;;
  --continue-from)
    CONTINUE_FROM="$2"
    shift 2
    ;;
  --data-dir)
    DATA_DIR_OVERRIDE="$2"
    shift 2
    ;;
  --skip-ranks)
    SKIP_RANKS="$2"
    shift 2
    ;;
  -h | --help)
    show_help
    exit 0
    ;;
  *)
    log_error "Unknown option: $1"
    show_help
    exit 1
    ;;
  esac
done

# Change to project root
cd "$PROJECT_ROOT"

# =============================================================================
# Validation
# =============================================================================

print_header "Production Ensemble Training"

# Check production config directory exists
if [[ ! -d "$PRODUCTION_CONFIG_DIR" ]]; then
  log_error "Production config directory not found: $PRODUCTION_CONFIG_DIR"
  exit 1
fi

log_info "Production config directory: $PRODUCTION_CONFIG_DIR"
log_info "Log level: $LOG_LEVEL"
log_info "Dry run: $DRY_RUN"

# =============================================================================
# Get Config Files
# =============================================================================

if [[ -n "$SPECIFIC_CONFIG" ]]; then
  # Train specific config only
  CONFIG_PATH="${PRODUCTION_CONFIG_DIR}/${SPECIFIC_CONFIG}"
  
  if [[ ! -f "$CONFIG_PATH" ]]; then
    log_error "Config file not found: $CONFIG_PATH"
    exit 1
  fi
  
  CONFIG_FILES=("$CONFIG_PATH")
  log_info "Training single config: $SPECIFIC_CONFIG"
else
  # Get all config files, sorted numerically
  mapfile -t CONFIG_FILES < <(find "$PRODUCTION_CONFIG_DIR" -maxdepth 1 -name "ensemble_chemprop_hpo_*.yaml" | sort -V)
  
  if [[ ${#CONFIG_FILES[@]} -eq 0 ]]; then
    log_error "No config files found in $PRODUCTION_CONFIG_DIR"
    exit 1
  fi
  
  log_info "Found ${#CONFIG_FILES[@]} production config files"
  
  # Filter by continue-from if specified
  if [[ -n "$CONTINUE_FROM" ]]; then
    FILTERED_FILES=()
    for config_file in "${CONFIG_FILES[@]}"; do
      # Extract config number (e.g., 001, 006, 010, etc.)
      config_num=$(basename "$config_file" | grep -oP '\d{3}')
      
      # Convert to integer for comparison
      config_num_int=$((10#$config_num))
      continue_from_int=$((10#$CONTINUE_FROM))
      
      if [[ $config_num_int -ge $continue_from_int ]]; then
        FILTERED_FILES+=("$config_file")
      fi
    done
    
    CONFIG_FILES=("${FILTERED_FILES[@]}")
    log_info "Continuing from config $CONTINUE_FROM: ${#CONFIG_FILES[@]} configs to train"
  fi

  # Filter out any skipped ranks if provided (e.g., --skip-ranks 1,6)
  if [[ -n "$SKIP_RANKS" ]]; then
    IFS=',' read -ra SKIP_ARR <<< "$SKIP_RANKS"
    FILTERED2=()
    for config_file in "${CONFIG_FILES[@]}"; do
      config_num=$(basename "$config_file" | grep -oP '\d{3}')
      config_num_int=$((10#$config_num))
      skip=false
      for s in "${SKIP_ARR[@]}"; do
        # normalize s to integer
        s_int=$((10#$s))
        if [[ $config_num_int -eq $s_int ]]; then
          skip=true
          break
        fi
      done
      if [[ "$skip" == "true" ]]; then
        log_info "Skipping config: $(basename "$config_file") (rank $config_num_int)"
        continue
      fi
      FILTERED2+=("$config_file")
    done
    CONFIG_FILES=("${FILTERED2[@]}")
    log_info "After skipping ranks: ${#CONFIG_FILES[@]} configs to train"
  fi
fi

# =============================================================================
# Training Execution
# =============================================================================

# Build command options
CMD_OPTS="--log-level $LOG_LEVEL"
if [[ -n "$MAX_PARALLEL" ]]; then
  CMD_OPTS="$CMD_OPTS --max-parallel $MAX_PARALLEL"
fi

# Track results
TOTAL=${#CONFIG_FILES[@]}
SUCCESS=0
FAILED=0
FAILED_CONFIGS=()

# Record start time
OVERALL_START=$(date +%s)

log_info "Starting training for $TOTAL configuration(s)"
echo ""

# Train each config
for i in "${!CONFIG_FILES[@]}"; do
  config_file="${CONFIG_FILES[$i]}"
  config_name=$(basename "$config_file")
  config_num=$((i + 1))
  
  print_header "Training Config $config_num/$TOTAL: $config_name"
  
  # Record config start time
  CONFIG_START=$(date +%s)
  
  # Build command (optionally override data_dir by creating a temporary config)
  TEMP_CONFIG=""
  CMD_CONFIG="$config_file"
  if [[ -n "$DATA_DIR_OVERRIDE" ]]; then
    # create temp config that replaces data_dir: ... in the YAML
    TEMP_CONFIG=$(create_temp_config "$config_file" "$DATA_DIR_OVERRIDE")
    CMD_CONFIG="$TEMP_CONFIG"
  fi

  CMD="python -m admet.model.chemprop.ensemble --config $CMD_CONFIG $CMD_OPTS"

  log_cmd "$CMD"

  if [[ "$DRY_RUN" == "true" ]]; then
    if [[ -n "$DATA_DIR_OVERRIDE" ]]; then
      log_info "[DRY RUN] Would execute ensemble training (with data_dir override: $DATA_DIR_OVERRIDE)"
    else
      log_info "[DRY RUN] Would execute ensemble training"
    fi
    # cleanup temp config if created (dry-run: just remove it if present)
    if [[ -n "$TEMP_CONFIG" && -f "$TEMP_CONFIG" ]]; then
      rm -f "$TEMP_CONFIG"
    fi
    continue
  fi

  # Execute command
  if eval "$CMD"; then
    CONFIG_END=$(date +%s)
    CONFIG_DURATION=$((CONFIG_END - CONFIG_START))
    
    log_success "✓ Completed: $config_name (${CONFIG_DURATION}s)"
    ((SUCCESS++))
  else
    CONFIG_END=$(date +%s)
    CONFIG_DURATION=$((CONFIG_END - CONFIG_START))
    
    log_error "✗ Failed: $config_name (${CONFIG_DURATION}s)"
    ((FAILED++))
    FAILED_CONFIGS+=("$config_name")
  fi
  # Clean up temporary config if used
  if [[ -n "$TEMP_CONFIG" && -f "$TEMP_CONFIG" ]]; then
    cleanup_temp_config "$TEMP_CONFIG"
  fi
  
  echo ""
done

# =============================================================================
# Summary
# =============================================================================

OVERALL_END=$(date +%s)
OVERALL_DURATION=$((OVERALL_END - OVERALL_START))

print_header "Training Complete"

log_info "Total configs: $TOTAL"
log_info "Successful: $SUCCESS"
log_info "Failed: $FAILED"
log_info "Total duration: ${OVERALL_DURATION}s ($(($OVERALL_DURATION / 60))m $(($OVERALL_DURATION % 60))s)"

if [[ "$DRY_RUN" == "true" ]]; then
  log_warn "Dry run mode - no commands were executed"
  exit 0
fi

# List failed configs if any
if [[ $FAILED -gt 0 ]]; then
  echo ""
  log_error "Failed configurations:"
  for failed_config in "${FAILED_CONFIGS[@]}"; do
    echo "  - $failed_config"
  done
  echo ""
  log_info "To retry failed configs, run individual configs with:"
  log_info "  $0 --config <CONFIG_FILE>"
  exit 1
fi

log_success "All production ensemble models trained successfully! ✓"
exit 0
