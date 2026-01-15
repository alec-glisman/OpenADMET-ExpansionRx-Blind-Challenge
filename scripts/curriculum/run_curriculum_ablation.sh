#!/bin/bash
# Curriculum Learning Ablation Study Runner
#
# This script runs all 5 ablation experiments for curriculum learning.
# Results are logged to MLflow experiment: curriculum_ablation_study
#
# Usage:
#   ./scripts/run_curriculum_ablation.sh           # Run all experiments
#   ./scripts/run_curriculum_ablation.sh 1         # Run only experiment 1
#   ./scripts/run_curriculum_ablation.sh 1 3       # Run experiments 1 and 3

set -e # Exit on error

# Get the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Activate virtual environment if it exists
if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  source "${PROJECT_ROOT}/.venv/bin/activate"
  echo "Virtual environment activated"
  echo ""
elif command -v admet &> /dev/null; then
  echo "Using admet from PATH: $(which admet)"
  echo ""
else
  echo "ERROR: Virtual environment not found at ${PROJECT_ROOT}/.venv and 'admet' not in PATH"
  exit 1
fi

# Configuration directory
CONFIG_DIR="configs/0-experiment/curriculum-learning/ablation"

# Define experiments
declare -A EXPERIMENTS=(
  ["01"]="01_baseline_curriculum.yaml"
  ["02"]="02_two_quality_only.yaml"
  ["03"]="03_selective_tasks.yaml"
  ["04"]="04_high_quality_focus.yaml"
  ["05"]="05_finetune_approach.yaml"
)

# Function to run a single experiment
run_experiment() {
  local exp_num="$1"
  local config="${EXPERIMENTS[$exp_num]}"

  if [[ -z "$config" ]]; then
    echo "ERROR: Unknown experiment number: $exp_num"
    echo "Valid options: ${!EXPERIMENTS[*]}"
    return 1
  fi

  echo "=============================================="
  echo "Running Experiment $exp_num: $config"
  echo "=============================================="
  echo ""

  admet model train -c "${CONFIG_DIR}/${config}"

  echo ""
  echo "Experiment $exp_num completed!"
  echo ""
}

# Main logic
main() {
  echo "Curriculum Learning Ablation Study"
  echo "==================================="
  echo ""
  echo "MLflow experiment: curriculum_ablation_study"
  echo ""

  # If specific experiments are requested, run only those
  if [[ $# -gt 0 ]]; then
    for exp_num in "$@"; do
      # Pad with leading zero if needed
      exp_num=$(printf "%02d" "$exp_num")
      run_experiment "$exp_num"
    done
  else
    # Run all experiments in order
    echo "Running all 5 experiments..."
    echo ""

    for exp_num in 01 02 03 04 05; do
      run_experiment "$exp_num"
    done
  fi

  echo "=============================================="
  echo "All requested experiments completed!"
  echo ""
  echo "View results in MLflow:"
  echo "  mlflow ui --backend-store-uri http://127.0.0.1:8084"
  echo ""
  echo "Or compare experiments with:"
  echo "  admet leaderboard compare --experiment curriculum_ablation_study"
  echo "=============================================="
}

# Run main function
main "$@"
