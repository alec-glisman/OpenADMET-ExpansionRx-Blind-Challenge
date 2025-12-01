#!/usr/bin/env bash
# =============================================================================
# Data Splitting Script
# =============================================================================
# Runs the admet.data.split module across multiple configurations of:
# - Split methods (group_kfold, stratified_kfold, multilabel_stratified_kfold)
# - Clustering methods (random, scaffold, kmeans, umap, butina, bitbirch)
# - Quality filters (high, medium, low combinations)
#
# Usage:
#   ./scripts/run_data_splits.sh --input data.csv --output-dir outputs/
#   ./scripts/run_data_splits.sh -i data.csv -o outputs/ --cluster-methods bitbirch scaffold
#   ./scripts/run_data_splits.sh -i data.csv -o outputs/ --qualities "high" "high,medium"
#
# =============================================================================

set -euo pipefail

# =============================================================================
# Default Configuration
# =============================================================================

# Script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
INPUT_FILE=""
OUTPUT_DIR="assets/dataset/split_train_val"
SMILES_COL="SMILES"
QUALITY_COL="Quality"
LOG_LEVEL="INFO"
DRY_RUN=false
MAX_PARALLEL=1

# Available options
ALL_SPLIT_METHODS=("group_kfold" "stratified_kfold" "multilabel_stratified_kfold")
ALL_CLUSTER_METHODS=("random" "kmeans" "umap" "bitbirch") # NOTE: butina excluded for speed/memory
ALL_QUALITY_COMBINATIONS=(
  "high"
  "high,medium"
  "high,medium,low"
)

# Selected options (empty = use all)
SPLIT_METHODS=()
CLUSTER_METHODS=()
QUALITY_COMBINATIONS=()
TARGET_COLS=()

# =============================================================================
# Logging Functions
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() {
  echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_cmd() {
  echo -e "${BLUE}[CMD]${NC} $*"
}

log_section() {
  echo ""
  echo -e "${CYAN}==============================================================================${NC}"
  echo -e "${CYAN}  $*${NC}"
  echo -e "${CYAN}==============================================================================${NC}"
}

# =============================================================================
# Usage and Help
# =============================================================================

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run data splitting across multiple configurations.

Required:
  -i, --input FILE          Input CSV file path

Options:
  -o, --output-dir DIR      Output directory (default: assets/dataset/split_train_val)
  --smiles-col COL          SMILES column name (default: SMILES)
  --quality-col COL         Quality column name (default: Quality)

Split Configuration:
  -s, --split-methods M...  Split methods to use (space-separated)
                            Options: ${ALL_SPLIT_METHODS[*]}
                            Default: all methods

  -c, --cluster-methods M...  Clustering methods to use (space-separated)
                              Options: ${ALL_CLUSTER_METHODS[*]}
                              Default: all methods

  -q, --qualities Q...      Quality filter combinations (space-separated, comma-delimited)
                            Example: "high" "high,medium" "high,medium,low"
                            Default: all combinations

  -t, --target-cols T...    Target columns for stratification (space-separated)
                            Default: auto-detect from data

Execution:
  --dry-run                 Print commands without executing
  --max-parallel N          Maximum parallel jobs (default: 1, sequential)
  --log-level LEVEL         Logging level (DEBUG, INFO, WARNING, ERROR)
                            Default: INFO

Help:
  -h, --help                Show this help message

Examples:
  # Run all configurations
  $(basename "$0") -i data.csv -o outputs/splits/

  # Run specific methods
  $(basename "$0") -i data.csv -s multilabel_stratified_kfold -c bitbirch scaffold

  # Run with specific quality filters
  $(basename "$0") -i data.csv -q "high" "high,medium"

  # Dry run to see commands
  $(basename "$0") -i data.csv --dry-run

EOF
}

# =============================================================================
# Argument Parsing
# =============================================================================

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
    -i | --input)
      INPUT_FILE="$2"
      shift 2
      ;;
    -o | --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --smiles-col)
      SMILES_COL="$2"
      shift 2
      ;;
    --quality-col)
      QUALITY_COL="$2"
      shift 2
      ;;
    -s | --split-methods)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        SPLIT_METHODS+=("$1")
        shift
      done
      ;;
    -c | --cluster-methods)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        CLUSTER_METHODS+=("$1")
        shift
      done
      ;;
    -q | --qualities)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        QUALITY_COMBINATIONS+=("$1")
        shift
      done
      ;;
    -t | --target-cols)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        TARGET_COLS+=("$1")
        shift
      done
      ;;
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
    -h | --help)
      usage
      exit 0
      ;;
    *)
      log_error "Unknown option: $1"
      usage
      exit 1
      ;;
    esac
  done
}

# =============================================================================
# Validation
# =============================================================================

validate_args() {
  # Check input file
  if [[ -z "$INPUT_FILE" ]]; then
    log_error "Input file is required (-i, --input)"
    usage
    exit 1
  fi

  if [[ ! -f "$INPUT_FILE" ]]; then
    log_error "Input file not found: $INPUT_FILE"
    exit 1
  fi

  # Use defaults if not specified
  if [[ ${#SPLIT_METHODS[@]} -eq 0 ]]; then
    SPLIT_METHODS=("${ALL_SPLIT_METHODS[@]}")
  fi

  if [[ ${#CLUSTER_METHODS[@]} -eq 0 ]]; then
    CLUSTER_METHODS=("${ALL_CLUSTER_METHODS[@]}")
  fi

  if [[ ${#QUALITY_COMBINATIONS[@]} -eq 0 ]]; then
    QUALITY_COMBINATIONS=("${ALL_QUALITY_COMBINATIONS[@]}")
  fi

  # Validate split methods
  for method in "${SPLIT_METHODS[@]}"; do
    if [[ ! " ${ALL_SPLIT_METHODS[*]} " =~ " ${method} " ]]; then
      log_error "Invalid split method: $method"
      log_error "Valid options: ${ALL_SPLIT_METHODS[*]}"
      exit 1
    fi
  done

  # Validate cluster methods
  for method in "${CLUSTER_METHODS[@]}"; do
    if [[ ! " ${ALL_CLUSTER_METHODS[*]} " =~ " ${method} " ]]; then
      log_error "Invalid cluster method: $method"
      log_error "Valid options: ${ALL_CLUSTER_METHODS[*]}"
      exit 1
    fi
  done
}

# =============================================================================
# Quality Label Helpers
# =============================================================================

# Convert comma-separated quality values to underscore-joined directory name
quality_to_dirname() {
  local quality_str="$1"
  # Replace commas with underscores, sort values
  echo "quality_${quality_str//,/_}"
}

# Convert comma-separated quality values to space-separated for CLI
quality_to_args() {
  local quality_str="$1"
  # Replace commas with spaces
  echo "${quality_str//,/ }"
}

# =============================================================================
# Run Split Command
# =============================================================================

run_split() {
  local cluster_method="$1"
  local split_method="$2"
  local quality_str="$3"

  # Build output directory path
  local quality_dir
  quality_dir=$(quality_to_dirname "$quality_str")
  local output_path="${OUTPUT_DIR}/${quality_dir}/${cluster_method}/${split_method}"

  # Build quality args
  local quality_args
  quality_args=$(quality_to_args "$quality_str")

  # Build command
  local cmd=(
    python -m admet.data.split
    --input "$INPUT_FILE"
    --output "$output_path"
    --cluster-method "$cluster_method"
    --split-method "$split_method"
    --quality-col "$QUALITY_COL"
    --quality-values $quality_args
    --smiles-col "$SMILES_COL"
    --log-level "$LOG_LEVEL"
  )

  # Add target columns if specified
  if [[ ${#TARGET_COLS[@]} -gt 0 ]]; then
    cmd+=(--target-cols "${TARGET_COLS[@]}")
  fi

  log_cmd "${cmd[*]}"

  if [[ "$DRY_RUN" == "true" ]]; then
    return 0
  fi

  # Run the command
  if "${cmd[@]}"; then
    log_info "✓ Completed: $quality_dir / $cluster_method / $split_method"
    return 0
  else
    log_error "✗ Failed: $quality_dir / $cluster_method / $split_method"
    return 1
  fi
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
  parse_args "$@"
  validate_args

  log_section "Data Splitting Configuration"
  log_info "Input file:       $INPUT_FILE"
  log_info "Output directory: $OUTPUT_DIR"
  log_info "SMILES column:    $SMILES_COL"
  log_info "Quality column:   $QUALITY_COL"
  log_info "Split methods:    ${SPLIT_METHODS[*]}"
  log_info "Cluster methods:  ${CLUSTER_METHODS[*]}"
  log_info "Quality combos:   ${QUALITY_COMBINATIONS[*]}"
  log_info "Target columns:   ${TARGET_COLS[*]:-auto}"
  log_info "Dry run:          $DRY_RUN"
  log_info "Log level:        $LOG_LEVEL"

  # Calculate total runs
  local n_splits=${#SPLIT_METHODS[@]}
  local n_clusters=${#CLUSTER_METHODS[@]}
  local n_qualities=${#QUALITY_COMBINATIONS[@]}
  local total_runs=$((n_splits * n_clusters * n_qualities))

  log_section "Starting $total_runs split configurations"

  # Track results
  local success_count=0
  local fail_count=0
  local run_count=0

  # Create output directory
  mkdir -p "$OUTPUT_DIR"

  # Run all combinations
  for quality_str in "${QUALITY_COMBINATIONS[@]}"; do
    for cluster_method in "${CLUSTER_METHODS[@]}"; do
      for split_method in "${SPLIT_METHODS[@]}"; do
        run_count=$((run_count + 1))
        log_section "Run $run_count / $total_runs"
        log_info "Quality:  $quality_str"
        log_info "Cluster:  $cluster_method"
        log_info "Split:    $split_method"

        if run_split "$cluster_method" "$split_method" "$quality_str"; then
          success_count=$((success_count + 1))
        else
          fail_count=$((fail_count + 1))
        fi
      done
    done
  done

  # Summary
  log_section "Summary"
  log_info "Total runs:  $total_runs"
  log_info "Successful:  $success_count"
  log_info "Failed:      $fail_count"

  if [[ "$DRY_RUN" == "true" ]]; then
    log_warn "Dry run mode - no commands were executed"
  fi

  if [[ $fail_count -gt 0 ]]; then
    log_error "Some runs failed!"
    return 1
  fi

  log_info "All splits completed successfully!"
  return 0
}

# Run main function
main "$@"
