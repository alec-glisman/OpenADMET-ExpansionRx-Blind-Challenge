#!/bin/bash
# Multi-stage HPO orchestration script
# Runs 3-phase HPO strategy: Exploration -> Exploitation -> Refinement

set -e # Exit on error

MODEL_TYPE=${1:-"chemprop"} # chemprop or chemeleon
PHASE=${2:-"all"}           # 1, 2, 3, or all

PHASES_DIR="configs/1-hpo-single-fold/phases"
BASE_CMD="admet model hpo"

echo "====================================================================="
echo "          Multi-Stage HPO for $MODEL_TYPE"
echo "====================================================================="
echo "Phase to run: $PHASE"
echo "Started at: $(date)"
echo ""

run_phase() {
  local phase_num=$1
  local phase_name=$2
  local config_file="${PHASES_DIR}/phase${phase_num}_${phase_name}_${MODEL_TYPE}.yaml"

  echo ""
  echo "---------------------------------------------------------------------"
  echo "  Phase $phase_num: ${phase_name^^}"
  echo "---------------------------------------------------------------------"
  echo "Config: $config_file"
  echo "Started: $(date)"
  echo ""

  # Check if config exists
  if [ ! -f "$config_file" ]; then
    echo "ERROR: Config file not found: $config_file"
    exit 1
  fi

  # Run HPO
  $BASE_CMD -c "$config_file"

  echo ""
  echo "Phase $phase_num complete at: $(date)"
  echo ""
}

# Run requested phases
if [[ "$PHASE" == "all" || "$PHASE" == "1" ]]; then
  run_phase 1 "explore"
fi

if [[ "$PHASE" == "all" || "$PHASE" == "2" ]]; then
  run_phase 2 "exploit"
fi

if [[ "$PHASE" == "all" || "$PHASE" == "3" ]]; then
  # Warn user to update Phase 3 config with narrowed search space
  echo ""
  echo "====================================================================="
  echo "                  ⚠️  PHASE 3 PREPARATION REQUIRED  ⚠️"
  echo "====================================================================="
  echo ""
  echo "Before running Phase 3, you should:"
  echo "  1. Review Phase 2 parameter importance plot"
  echo "  2. Check Phase 2 top 10 trial configurations"
  echo "  3. Update Phase 3 config to narrow search space around best values"
  echo ""
  echo "Phase 3 config: ${PHASES_DIR}/phase3_refine_${MODEL_TYPE}.yaml"
  echo ""
  read -p "Press Enter to continue with Phase 3, or Ctrl+C to abort..."
  echo ""

  run_phase 3 "refine"
fi

echo ""
echo "====================================================================="
echo "          Multi-Stage HPO Complete!"
echo "====================================================================="
echo "Completed at: $(date)"
echo ""
echo "Results:"
echo "  - Phase 1 (Explore): /media/aglisman/Data/models/${MODEL_TYPE}-hpo/phase1/"
echo "  - Phase 2 (Exploit): /media/aglisman/Data/models/${MODEL_TYPE}-hpo/phase2/"
echo "  - Phase 3 (Refine):  /media/aglisman/Data/models/${MODEL_TYPE}-hpo/phase3/"
echo ""
echo "Optuna studies: /media/aglisman/Data/models/hpo_results/optuna_studies/"
echo "  - ${MODEL_TYPE}_phase1_explore"
echo "  - ${MODEL_TYPE}_phase2_exploit"
echo "  - ${MODEL_TYPE}_phase3_refine"
echo ""
echo "Next steps:"
echo "  1. Review MLflow UI: http://127.0.0.1:8084"
echo "  2. Examine Optuna visualizations in output directories"
echo "  3. Select best models for ensemble training"
echo ""
