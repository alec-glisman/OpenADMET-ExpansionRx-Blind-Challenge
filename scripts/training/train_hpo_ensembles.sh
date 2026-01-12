#!/usr/bin/env bash
# =============================================================================
# Train HPO Ensemble Models (Chemprop or Chemeleon)
# =============================================================================
# This script trains ensemble models using the top HPO configurations
# in order from best (rank 1) to worst (rank N).
#
# Usage:
#   ./scripts/training/train_hpo_ensembles.sh
#   ./scripts/training/train_hpo_ensembles.sh --start 1 --end 10
#   ./scripts/training/train_hpo_ensembles.sh --ranks 1,5,10
#   ./scripts/training/train_hpo_ensembles.sh --config-dir 3-production
#   ./scripts/training/train_hpo_ensembles.sh --model-type chemeleon --config-dir 2-hpo-ensemble/1_chemeleon_v1
#
# Environment:
#   Assumes virtual environment is activated and all dependencies installed.
# =============================================================================

set -euo pipefail

# Default values
START_RANK=1
END_RANK=100
SPECIFIC_RANKS=""
DRY_RUN=false
CONFIG_DIR="2-hpo-ensemble"
MODEL_TYPE="chemprop"
SKIP_CONFIRMATION=false

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
  --config-dir)
    CONFIG_DIR="$2"
    shift 2
    ;;
  --model-type)
    MODEL_TYPE="$2"
    shift 2
    ;;
  --dry-run)
    DRY_RUN=true
    shift
    ;;
  -y | --yes)
    SKIP_CONFIRMATION=true
    shift
    ;;
  -h | --help)
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --start N          Start from rank N (default: 1)"
    echo "  --end N            End at rank N (default: 100)"
    echo "  --ranks N,M,K      Train specific ranks only (comma-separated)"
    echo "  --config-dir DIR   Config directory name (default: 2-hpo-ensemble)"
    echo "                     Examples: 2-hpo-ensemble, 3-production, 2-hpo-ensemble/1_chemeleon_v1"
    echo "  --model-type TYPE  Model type: chemprop or chemeleon (default: chemprop)"
    echo "  --dry-run          Print commands without executing"
    echo "  -y, --yes          Skip confirmation prompt"
    echo "  -h, --help         Show this help message"
    exit 0
    ;;
  *)
    echo "Unknown option: $1"
    exit 1
    ;;
  esac
done

# Strip 'configs/' prefix from CONFIG_DIR if present
CONFIG_DIR="${CONFIG_DIR#configs/}"

# Function to discover available config files and extract ranks
discover_available_ranks() {
  local config_pattern="configs/${CONFIG_DIR}/ensemble_${MODEL_TYPE}_hpo_*.yaml"
  local -a available_ranks=()

  if ! compgen -G "$config_pattern" >/dev/null; then
    echo "ERROR: No config files found matching: $config_pattern"
    exit 1
  fi

  # Extract rank numbers from filenames
  for config_file in configs/"${CONFIG_DIR}"/ensemble_"${MODEL_TYPE}"_hpo_*.yaml; do
    if [[ -f "$config_file" ]]; then
      # Extract the number from ensemble_{model_type}_hpo_NNN.yaml
      local basename
      basename=$(basename "$config_file")
      if [[ $basename =~ ensemble_${MODEL_TYPE}_hpo_([0-9]+)\.yaml ]]; then
        local rank="${BASH_REMATCH[1]}"
        # Remove leading zeros
        rank=$((10#$rank))
        available_ranks+=("$rank")
      fi
    fi
  done

  # Sort ranks numerically
  IFS=$'\n' available_ranks=($(sort -n <<<"${available_ranks[*]}"))
  unset IFS

  echo "${available_ranks[@]}"
}

# Function to filter ranks based on user input
filter_ranks() {
  local -a all_ranks=("$@")
  local -a filtered_ranks=()

  if [[ -n "$SPECIFIC_RANKS" ]]; then
    # Use specific ranks if provided
    IFS=',' read -ra requested_ranks <<<"$SPECIFIC_RANKS"
    for requested in "${requested_ranks[@]}"; do
      for available in "${all_ranks[@]}"; do
        if [[ "$available" -eq "$requested" ]]; then
          filtered_ranks+=("$available")
          break
        fi
      done
    done
  else
    # Filter by range
    for rank in "${all_ranks[@]}"; do
      if [[ "$rank" -ge "$START_RANK" && "$rank" -le "$END_RANK" ]]; then
        filtered_ranks+=("$rank")
      fi
    done
  fi

  echo "${filtered_ranks[@]}"
}

# Function to extract gpu_ids from config and set CUDA_VISIBLE_DEVICES
extract_gpu_ids() {
  local config_file=$1
  # Extract gpu_ids from YAML config using grep and sed
  # Looking for pattern like: gpu_ids: [1] or gpu_ids: [0, 1, 2]
  local gpu_line
  gpu_line=$(grep -E '^\s*gpu_ids:\s*\[' "$config_file" 2>/dev/null || true)

  if [[ -n "$gpu_line" ]]; then
    # Extract the array contents between [ and ]
    local gpu_ids
    gpu_ids=$(echo "$gpu_line" | sed -E 's/.*\[([^]]*)\].*/\1/' | tr -d ' ')
    if [[ -n "$gpu_ids" && "$gpu_ids" != "null" ]]; then
      echo "$gpu_ids"
      return 0
    fi
  fi

  # No gpu_ids in config - detect all available GPUs from nvidia-smi
  if command -v nvidia-smi &>/dev/null; then
    local all_gpus
    all_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    if [[ -n "$all_gpus" ]]; then
      echo "$all_gpus"
    fi
  fi
}

# Function to train a single ensemble
train_ensemble() {
  local rank=$1
  local config_file="configs/${CONFIG_DIR}/ensemble_${MODEL_TYPE}_hpo_$(printf "%03d" "$rank").yaml"

  echo "=========================================="
  echo "Training ${MODEL_TYPE} HPO Ensemble Rank $rank"
  echo "Config: $config_file"
  echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=========================================="

  # Extract gpu_ids and set CUDA_VISIBLE_DEVICES before Python import
  local gpu_ids
  gpu_ids=$(extract_gpu_ids "$config_file")

  if [[ -n "$gpu_ids" ]]; then
    echo "Setting CUDA_VISIBLE_DEVICES=$gpu_ids"
    export CUDA_VISIBLE_DEVICES="$gpu_ids"
  else
    echo "Warning: Could not detect GPUs, CUDA_VISIBLE_DEVICES not set"
  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would execute:"
    echo "CUDA_VISIBLE_DEVICES=$gpu_ids python -m admet.model.${MODEL_TYPE}.ensemble --config $config_file"
    return 0
  fi

  # Run ensemble training
  if python -m "admet.model.${MODEL_TYPE}.ensemble" \
    --config "$config_file"; then
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
echo "HPO Ensemble Training Pipeline"
echo "============================================="
echo "Model type: $MODEL_TYPE"
echo "Config directory: configs/$CONFIG_DIR"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
echo ""

# Discover available config files
echo "Discovering available ensemble configs..."
read -ra AVAILABLE_RANKS <<<"$(discover_available_ranks)"

if [[ ${#AVAILABLE_RANKS[@]} -eq 0 ]]; then
  echo "ERROR: No config files found in configs/$CONFIG_DIR/"
  exit 1
fi

echo "Found ${#AVAILABLE_RANKS[@]} available configs: ${AVAILABLE_RANKS[*]}"
echo ""

# Filter ranks based on user input
echo "Filtering ranks based on input..."
if [[ -n "$SPECIFIC_RANKS" ]]; then
  echo "  Requested ranks: $SPECIFIC_RANKS"
else
  echo "  Rank range: $START_RANK to $END_RANK"
fi

read -ra RANKS_TO_RUN <<<"$(filter_ranks "${AVAILABLE_RANKS[@]}")"

if [[ ${#RANKS_TO_RUN[@]} -eq 0 ]]; then
  echo ""
  echo "ERROR: No matching config files found for the specified ranks."
  echo "Available ranks: ${AVAILABLE_RANKS[*]}"
  exit 1
fi

echo ""
echo "============================================="
echo "Will train ${#RANKS_TO_RUN[@]} ensemble(s):"
echo "  Ranks: ${RANKS_TO_RUN[*]}"
echo "  Dry run: $DRY_RUN"
echo "============================================="
echo ""

# Ask for confirmation if not in dry-run mode and confirmation not skipped
if [[ "$DRY_RUN" == "false" && "$SKIP_CONFIRMATION" == "false" ]]; then
  read -p "Proceed with training? (y/N): " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
  fi
  echo ""
fi

# Track success/failure
SUCCESSFUL=()
FAILED=()

# Train each rank
for rank in "${RANKS_TO_RUN[@]}"; do
  if train_ensemble "$rank"; then
    SUCCESSFUL+=("$rank")
  else
    FAILED+=("$rank")
  fi
  echo ""
done

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
