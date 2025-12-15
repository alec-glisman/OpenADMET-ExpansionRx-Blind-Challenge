#!/usr/bin/env bash
# =============================================================================
# Chemprop Ensemble Training Script
# =============================================================================
# This script trains Chemprop ensemble models across multiple data splits.
# It iterates through all data directories and trains an ensemble for each.
#
# Usage:
#   ./scripts/train_chemprop_ensembles.sh [--dry-run] [--max-parallel N] [--log-level LEVEL]
#
# Options:
#   --dry-run       Print commands without executing
#   --max-parallel  Override max parallel models (default: from config)
#   --log-level     Set logging level (DEBUG, INFO, WARNING, ERROR)
#
# =============================================================================

# Source common library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

# Configuration
PROJECT_ROOT="$(get_project_root "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/configs/ensemble_chemprop_tag.yaml"

# Default options
DRY_RUN=false
MAX_PARALLEL=""
LOG_LEVEL="INFO"
CONFIG_DIR=""

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
  --config-dir)
    CONFIG_DIR="$2"
    shift 2
    ;;
  -h | --help)
    echo "Usage: $0 [--dry-run] [--max-parallel N] [--log-level LEVEL] [--config-dir DIR]"
    exit 0
    ;;
  *)
    echo "Unknown option: $1"
    exit 1
    ;;
  esac
done

# Change to project root
cd "$PROJECT_ROOT" || exit 1

# Build command options
CMD_OPTS="--log-level $LOG_LEVEL"
if [[ -n "$MAX_PARALLEL" ]]; then
  CMD_OPTS="$CMD_OPTS --max-parallel $MAX_PARALLEL"
fi

if [[ -n "$CONFIG_DIR" ]]; then
  # Config Directory Mode
  if [[ ! -d "$CONFIG_DIR" ]]; then
    log_error "Config directory not found: $CONFIG_DIR"
    exit 1
  fi

  log_info "Starting ensemble training from config directory: $CONFIG_DIR"

  # Get sorted config files
  mapfile -t CONFIG_FILES < <(find "$CONFIG_DIR" -maxdepth 1 -name "*.yaml" -o -name "*.yml" | sort)

  if [[ ${#CONFIG_FILES[@]} -eq 0 ]]; then
    log_error "No config files found in $CONFIG_DIR"
    exit 1
  fi

  log_info "Found ${#CONFIG_FILES[@]} config files"

  TOTAL=${#CONFIG_FILES[@]}
  SUCCESS=0
  FAILED=0
  SKIPPED=0

  for config_file in "${CONFIG_FILES[@]}"; do
    print_header "Processing config: $(basename "$config_file")"

    CMD="python -m admet.model.chemprop.ensemble --config $config_file $CMD_OPTS"

    if execute_command "$CMD" "$DRY_RUN" "ensemble training for $(basename "$config_file")"; then
      if [[ "$DRY_RUN" != "true" ]]; then
        log_success "Completed: $(basename "$config_file")"
        ((SUCCESS++))
      fi
    else
      log_error "Failed: $(basename "$config_file")"
      ((FAILED++))
    fi
  done

else
  # Data Directory Mode (Original)

  # Check config file exists
  check_config_exists "$CONFIG_FILE" || exit 1

  # Get data directories
  mapfile -t DATA_DIRS < <(get_data_dirs)

  log_info "Starting ensemble training"
  log_info "Config file: $CONFIG_FILE"
  log_info "Data directories: ${#DATA_DIRS[@]}"

  # Track results
  TOTAL=${#DATA_DIRS[@]}
  SUCCESS=0
  FAILED=0
  SKIPPED=0

  # Train ensemble for each data directory
  for data_dir in "${DATA_DIRS[@]}"; do
    print_header "Processing: $data_dir"

    # Check if directory exists
    if ! check_dir_exists "$data_dir"; then
      log_warn "Directory not found, skipping: $data_dir"
      ((SKIPPED++))
      continue
    fi

    # Create temporary config with updated data_dir
    TEMP_CONFIG=$(create_temp_config "$CONFIG_FILE" "$data_dir")

    # Build command
    CMD="python -m admet.model.chemprop.ensemble --config $TEMP_CONFIG $CMD_OPTS"

    # Execute command
    if execute_command "$CMD" "$DRY_RUN" "ensemble training for $data_dir"; then
      if [[ "$DRY_RUN" != "true" ]]; then
        log_success "Completed: $data_dir"
        ((SUCCESS++))
      fi
    else
      log_error "Failed: $data_dir"
      ((FAILED++))
    fi

    # Clean up temp config
    cleanup_temp_config "$TEMP_CONFIG"
  done
fi

# Print summary and exit with appropriate code
print_summary "$TOTAL" "$SUCCESS" "$FAILED" "$SKIPPED" || exit 1
