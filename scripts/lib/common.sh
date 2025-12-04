#!/usr/bin/env bash
# =============================================================================
# Common Library for Training Scripts
# =============================================================================
# This file contains shared functions and variables used across training scripts.
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
#
# =============================================================================

# Prevent multiple sourcing
if [[ -n "${_COMMON_SH_LOADED:-}" ]]; then
  return 0
fi
_COMMON_SH_LOADED=1

# =============================================================================
# Directory Configuration
# =============================================================================

# Get the project root directory
get_project_root() {
  local script_dir="$1"
  dirname "$script_dir"
}

# =============================================================================
# Data Directory Definitions
# =============================================================================

# Base data directories containing split_*/fold_*/ subdirectories
# These are used by both ensemble and single model training scripts
get_data_dirs() {
  local dirs=(
    # High quality data
    "assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data"
    # "assets/dataset/split_train_val/v3/quality_high/bitbirch/stratified_kfold/data"
    # "assets/dataset/split_train_val/v3/quality_high/bitbirch/group_kfold/data"

    # # High + Medium quality data
    # "assets/dataset/split_train_val/v3/quality_high_medium/bitbirch/multilabel_stratified_kfold/data"
    # "assets/dataset/split_train_val/v3/quality_high_medium/bitbirch/stratified_kfold/data"
    # "assets/dataset/split_train_val/v3/quality_high_medium/bitbirch/group_kfold/data"

    # # High + Medium + Low quality data
    # "assets/dataset/split_train_val/v3/quality_high_medium_low/bitbirch/multilabel_stratified_kfold/data"
    # "assets/dataset/split_train_val/v3/quality_high_medium_low/bitbirch/stratified_kfold/data"
    # "assets/dataset/split_train_val/v3/quality_high_medium_low/bitbirch/group_kfold/data"
  )
  printf '%s\n' "${dirs[@]}"
}

# =============================================================================
# Color Definitions
# =============================================================================

# ANSI color codes for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# =============================================================================
# Logging Functions
# =============================================================================

log_info() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

log_cmd() {
  echo -e "${BLUE}[CMD]${NC} $1"
}

log_section() {
  echo -e "${CYAN}$1${NC}"
}

log_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_debug() {
  if [[ "${DEBUG:-false}" == "true" ]]; then
    echo -e "${MAGENTA}[DEBUG]${NC} $1"
  fi
}

# Print a horizontal separator
print_separator() {
  echo "=========================================="
}

# Print a section header
print_header() {
  local title="$1"
  echo ""
  log_section "=========================================="
  log_section "$title"
  log_section "=========================================="
}

# =============================================================================
# Configuration Helpers
# =============================================================================

# Create a temporary config file with updated data_dir
# Args:
#   $1 - Base config file path
#   $2 - New data_dir value
# Returns:
#   Path to temporary config file (caller must clean up)
create_temp_config() {
  local base_config="$1"
  local data_dir="$2"
  local temp_config

  temp_config=$(mktemp /tmp/training_config_XXXXXX.yaml)
  sed "s|data_dir:.*|data_dir: \"$data_dir\"|" "$base_config" >"$temp_config"
  echo "$temp_config"
}

# Clean up temporary config file
cleanup_temp_config() {
  local temp_config="$1"
  if [[ -f "$temp_config" ]]; then
    rm -f "$temp_config"
  fi
}

# =============================================================================
# Validation Helpers
# =============================================================================

# Check if a config file exists
# Args:
#   $1 - Config file path
# Returns:
#   0 if exists, 1 otherwise
check_config_exists() {
  local config_file="$1"
  if [[ ! -f "$config_file" ]]; then
    log_error "Config file not found: $config_file"
    return 1
  fi
  return 0
}

# Check if a directory exists
# Args:
#   $1 - Directory path
# Returns:
#   0 if exists, 1 otherwise
check_dir_exists() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    return 1
  fi
  return 0
}

# Check if required training files exist in a fold directory
# Args:
#   $1 - Fold directory path
# Returns:
#   0 if both train.csv and validation.csv exist, 1 otherwise
check_training_files() {
  local fold_dir="$1"
  if [[ ! -f "$fold_dir/train.csv" ]] || [[ ! -f "$fold_dir/validation.csv" ]]; then
    return 1
  fi
  return 0
}

# =============================================================================
# Directory Discovery
# =============================================================================

# Find all split directories in a base data directory
# Args:
#   $1 - Base data directory
#   $2 - Optional: max number of splits to return
# Outputs:
#   Sorted list of split directory paths (one per line)
find_split_dirs() {
  local base_dir="$1"
  local max_splits="${2:-}"
  local split_dirs=()

  for split_dir in "$base_dir"/split_*; do
    if [[ -d "$split_dir" ]]; then
      split_dirs+=("$split_dir")
    fi
  done

  # Sort directories
  IFS=$'\n' split_dirs=($(sort <<<"${split_dirs[*]}"))
  unset IFS

  # Limit if specified
  if [[ -n "$max_splits" ]] && [[ ${#split_dirs[@]} -gt 0 ]]; then
    split_dirs=("${split_dirs[@]:0:$max_splits}")
  fi

  printf '%s\n' "${split_dirs[@]}"
}

# Find all fold directories in a split directory
# Args:
#   $1 - Split directory
#   $2 - Optional: max number of folds to return
# Outputs:
#   Sorted list of fold directory paths (one per line)
find_fold_dirs() {
  local split_dir="$1"
  local max_folds="${2:-}"
  local fold_dirs=()

  for fold_dir in "$split_dir"/fold_*; do
    if [[ -d "$fold_dir" ]]; then
      fold_dirs+=("$fold_dir")
    fi
  done

  # Sort directories
  IFS=$'\n' fold_dirs=($(sort <<<"${fold_dirs[*]}"))
  unset IFS

  # Limit if specified
  if [[ -n "$max_folds" ]] && [[ ${#fold_dirs[@]} -gt 0 ]]; then
    fold_dirs=("${fold_dirs[@]:0:$max_folds}")
  fi

  printf '%s\n' "${fold_dirs[@]}"
}

# =============================================================================
# Summary Reporting
# =============================================================================

# Print a training summary
# Args:
#   $1 - Total count
#   $2 - Success count
#   $3 - Failed count
#   $4 - Skipped count
print_summary() {
  local total="$1"
  local success="$2"
  local failed="$3"
  local skipped="$4"

  echo ""
  log_section "=========================================="
  log_section "Training Summary"
  log_section "=========================================="
  log_info "Total: $total"
  log_info "Success: $success"
  log_info "Failed: $failed"
  log_info "Skipped: $skipped"

  if [[ $failed -gt 0 ]]; then
    return 1
  fi
  return 0
}

# =============================================================================
# Command Execution
# =============================================================================

# Execute a command with optional dry-run mode
# Args:
#   $1 - Command to execute
#   $2 - Dry run flag ("true" or "false")
#   $3 - Description for dry run message
# Returns:
#   Command exit code, or 0 for dry run
execute_command() {
  local cmd="$1"
  local dry_run="${2:-false}"
  local description="${3:-command}"
  local custom_log_file="${4:-}"

  # Determine log file:
  # Priority: custom argument > LOG_FILE env var > default to assets/logs/{datetime}_chemprop_ensemble.log
  local log_file
  if [[ -n "$custom_log_file" ]]; then
    log_file="$custom_log_file"
  elif [[ -n "${LOG_FILE:-}" ]]; then
    log_file="${LOG_FILE}"
  else
    local logs_dir="assets/logs"
    mkdir -p "$logs_dir"
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    log_file="${logs_dir}/${ts}_chemprop_ensemble.log"
  fi

  log_cmd "$cmd"
  log_info "Logging to: $log_file"

  # Ensure we can write to the log file
  if ! touch "$log_file" 2>/dev/null; then
    log_error "Cannot write to log file: $log_file"
    return 1
  fi

  if [[ "$dry_run" == "true" ]]; then
    log_info "[DRY RUN] Would execute: $description"
    printf '[%s] [DRY RUN] Would execute: %s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$description" | tee -a "$log_file" >/dev/null
    return 0
  fi

  # Header for command execution
  printf '[%s] === START: %s ===\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$description" | tee -a "$log_file"

  # Execute the command and pipe both stdout and stderr to the log file while still showing it on console.
  # Capture the exit code of the evaluated command via PIPESTATUS.
  eval "$cmd" 2>&1 | tee -a "$log_file"
  local rc="${PIPESTATUS[0]}"

  # Footer with exit code
  printf '[%s] === END: %s (exit %d) ===\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$description" "$rc" | tee -a "$log_file"

  return "$rc"
}
