#!/usr/bin/env bash
# =============================================================================
# Chemprop Hyperparameter Optimization Script
# =============================================================================
# This script runs hyperparameter optimization for Chemprop models using
# Ray Tune with ASHA scheduler.
#
# Usage:
#   ./scripts/run_chemprop_hpo.sh [CONFIG_FILE] [OPTIONS]
#
# Arguments:
#   CONFIG_FILE   Path to HPO configuration YAML file
#                 (default: configs/1-hpo-single/hpo_chemprop.yaml)
#
# Options:
#   --dry-run     Print commands without executing
#   --log-level   Set logging level (DEBUG, INFO, WARNING, ERROR)
#   --num-gpus    Number of GPUs to use (default: auto-detect)
#   --num-cpus    Number of CPUs to use (default: auto-detect)
#
# Example:
#   ./scripts/run_chemprop_hpo.sh configs/1-hpo-single/hpo_chemprop.yaml --num-gpus 2
#
# =============================================================================

set -euo pipefail

# Source common library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

# Configuration
PROJECT_ROOT="$(get_project_root "$SCRIPT_DIR")"
DEFAULT_CONFIG="${PROJECT_ROOT}/configs/1-hpo-single/hpo_chemprop.yaml"

# Default options
CONFIG_FILE=""
DRY_RUN=false
LOG_LEVEL="INFO"
NUM_GPUS=""
NUM_CPUS=""

# =============================================================================
# Argument Parsing
# =============================================================================

show_help() {
  cat <<EOF
Usage: $0 [CONFIG_FILE] [OPTIONS]

Run Chemprop hyperparameter optimization with Ray Tune.

Arguments:
  CONFIG_FILE   Path to HPO configuration YAML file
                (default: configs/hpo_chemprop.yaml)

Options:
  --dry-run     Print commands without executing
  --log-level   Set logging level (DEBUG, INFO, WARNING, ERROR)
  --num-gpus    Number of GPUs to use (default: auto-detect)
  --num-cpus    Number of CPUs to use (default: auto-detect)
  -h, --help    Show this help message

Examples:
  $0 configs/hpo_chemprop.yaml
  $0 configs/hpo_chemprop.yaml --num-gpus 2
  $0 --dry-run

EOF
}

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
  --num-gpus)
    NUM_GPUS="$2"
    shift 2
    ;;
  --num-cpus)
    NUM_CPUS="$2"
    shift 2
    ;;
  -h | --help)
    show_help
    exit 0
    ;;
  -*)
    log_error "Unknown option: $1"
    show_help
    exit 1
    ;;
  *)
    # Positional argument is config file
    if [[ -z "$CONFIG_FILE" ]]; then
      CONFIG_FILE="$1"
    else
      log_error "Unexpected argument: $1"
      exit 1
    fi
    shift
    ;;
  esac
done

# Use default config if not specified
if [[ -z "$CONFIG_FILE" ]]; then
  CONFIG_FILE="$DEFAULT_CONFIG"
fi

# Change to project root
cd "$PROJECT_ROOT"

# =============================================================================
# Validation
# =============================================================================

print_header "Chemprop Hyperparameter Optimization"

# Check config file exists
if ! check_config_exists "$CONFIG_FILE"; then
  log_error "Config file not found: $CONFIG_FILE"
  exit 1
fi

log_info "Configuration file: $CONFIG_FILE"
log_info "Log level: $LOG_LEVEL"
log_info "Dry run: $DRY_RUN"

# =============================================================================
# Environment Setup
# =============================================================================

# Check Python environment
if ! command -v python &>/dev/null; then
  log_error "Python not found. Please activate your virtual environment."
  exit 1
fi

log_info "Python: $(which python)"
log_info "Python version: $(python --version)"

# Check Ray installation
if ! python -c "import ray" &>/dev/null; then
  log_error "Ray not installed. Please run: pip install 'ray[tune]'"
  exit 1
fi

# Detect GPUs if not specified
if [[ -z "$NUM_GPUS" ]]; then
  if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    log_info "Auto-detected $NUM_GPUS GPU(s)"
  else
    NUM_GPUS=0
    log_warn "No NVIDIA GPUs detected. Running on CPU only."
  fi
else
  log_info "Using $NUM_GPUS GPU(s) as specified"
fi

# =============================================================================
# Setup Logging
# =============================================================================

# Create logs directory
LOGS_DIR="${PROJECT_ROOT}/assets/logs"
mkdir -p "$LOGS_DIR"

# Generate log filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOGS_DIR}/${TIMESTAMP}_chemprop_hpo.log"
export LOG_FILE

log_info "Log file: $LOG_FILE"

# =============================================================================
# Build Command
# =============================================================================

# Build the HPO command
HPO_CMD="python -m admet.model.chemprop.hpo --config '$CONFIG_FILE'"

# Add Ray resource hints via environment variables if specified
RAY_ENV=""
if [[ -n "$NUM_GPUS" ]]; then
  RAY_ENV="RAY_NUM_GPUS=$NUM_GPUS"
fi
if [[ -n "$NUM_CPUS" ]]; then
  if [[ -n "$RAY_ENV" ]]; then
    RAY_ENV="$RAY_ENV RAY_NUM_CPUS=$NUM_CPUS"
  else
    RAY_ENV="RAY_NUM_CPUS=$NUM_CPUS"
  fi
fi

# Prepend environment variables if any
if [[ -n "$RAY_ENV" ]]; then
  HPO_CMD="$RAY_ENV $HPO_CMD"
fi

# =============================================================================
# Execute HPO
# =============================================================================

print_header "Starting Hyperparameter Optimization"

log_info "Command: $HPO_CMD"

if [[ "$DRY_RUN" == "true" ]]; then
  log_info "[DRY RUN] Would execute HPO"
  exit 0
fi

# Execute the command
START_TIME=$(date +%s)

if execute_command "$HPO_CMD" "$DRY_RUN" "Chemprop HPO"; then
  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))

  print_header "HPO Completed Successfully"
  log_success "Duration: ${DURATION}s"
  log_info "Results logged to MLflow"
  log_info "Top configurations saved to output directory"
else
  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))

  print_header "HPO Failed"
  log_error "Duration: ${DURATION}s"
  log_error "Check log file for details: $LOG_FILE"
  exit 1
fi
