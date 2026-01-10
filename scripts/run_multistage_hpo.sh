#!/bin/bash
# Multi-stage HPO orchestration script
# Runs 3-phase HPO strategy: Exploration -> Exploitation -> Refinement

set -e # Exit on error

MODEL_TYPE=${1:-"chemprop"} # chemprop or chemeleon
PHASE=${2:-"all"}           # 1, 2, 3, all, 2-3, 1-2, etc.

PHASES_DIR="configs/1-hpo-single-fold/phases"
BASE_CMD="admet model hpo"

# Parse phase range (e.g., "2-3" means start at 2, end at 3)
if [[ "$PHASE" == "all" ]]; then
  START_PHASE=1
  END_PHASE=3
elif [[ "$PHASE" =~ ^([1-3])-([1-3])$ ]]; then
  START_PHASE=${BASH_REMATCH[1]}
  END_PHASE=${BASH_REMATCH[2]}
  if [[ $START_PHASE -gt $END_PHASE ]]; then
    echo "ERROR: Invalid phase range: $PHASE (start must be <= end)"
    exit 1
  fi
elif [[ "$PHASE" =~ ^[1-3]$ ]]; then
  START_PHASE=$PHASE
  END_PHASE=$PHASE
else
  echo "ERROR: Invalid phase specification: $PHASE"
  echo "Usage: $0 [MODEL_TYPE] [PHASE]"
  echo "  MODEL_TYPE: chemprop (default) or chemeleon"
  echo "  PHASE: 1, 2, 3, all, or range like 2-3"
  exit 1
fi

echo "====================================================================="
echo "          Multi-Stage HPO for $MODEL_TYPE"
echo "====================================================================="
if [[ $START_PHASE -eq $END_PHASE ]]; then
  echo "Phase to run: $START_PHASE"
else
  echo "Phases to run: $START_PHASE through $END_PHASE"
fi
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
if [[ $START_PHASE -le 1 && $END_PHASE -ge 1 ]]; then
  run_phase 1 "explore"
fi

if [[ $START_PHASE -le 2 && $END_PHASE -ge 2 ]]; then
  run_phase 2 "exploit"
fi

if [[ $START_PHASE -le 3 && $END_PHASE -ge 3 ]]; then
  phase3_config="${PHASES_DIR}/phase3_refine_${MODEL_TYPE}.yaml"

  # Check if automatic refinement is configured
  refinement_enabled=$(grep -E "^\s*enabled:\s*true" "$phase3_config" 2>/dev/null | head -1)
  previous_phase_dir=$(grep -E "^\s*previous_phase_dir:" "$phase3_config" 2>/dev/null | sed 's/.*previous_phase_dir:\s*//' | tr -d ' ')

  if [[ -n "$refinement_enabled" && -n "$previous_phase_dir" ]]; then
    # Auto-refinement is configured - check if Phase 2 output exists
    top_k_file="${previous_phase_dir}/top_k_configs.json"

    if [[ -f "$top_k_file" ]]; then
      echo ""
      echo "====================================================================="
      echo "           ✅ AUTOMATIC SEARCH SPACE REFINEMENT ENABLED"
      echo "====================================================================="
      echo ""
      echo "Phase 3 will automatically narrow search ranges using:"
      echo "  Source: $top_k_file"
      config_count=$(grep -c '"' "$top_k_file" 2>/dev/null | head -1 || echo "unknown")
      echo "  Configs available: ~$((config_count / 10)) trials"
      echo ""
      echo "Refinement settings from config:"
      grep -E "^\s*(top_k|margin_factor|use_percentiles):" "$phase3_config" 2>/dev/null | sed 's/^/    /'
      echo ""
    else
      # Auto-refinement configured but Phase 2 output missing
      echo ""
      echo "====================================================================="
      echo "           ⚠️  PHASE 2 OUTPUT NOT FOUND"
      echo "====================================================================="
      echo ""
      echo "Auto-refinement is enabled but Phase 2 results are missing:"
      echo "  Expected: $top_k_file"
      echo "  Status:   NOT FOUND"
      echo ""
      echo "Phase 3 will fall back to base search space from config."
      echo "To enable auto-refinement, ensure Phase 2 completed successfully."
      echo ""
      read -p "Press Enter to continue with fallback ranges, or Ctrl+C to abort..."
      echo ""
    fi
  else
    # Auto-refinement not configured - show manual preparation warning
    echo ""
    echo "====================================================================="
    echo "                  ⚠️  PHASE 3 PREPARATION REQUIRED  ⚠️"
    echo "====================================================================="
    echo ""
    echo "Auto-refinement is NOT enabled in Phase 3 config."
    echo ""
    echo "Before running Phase 3, you should:"
    echo "  1. Review Phase 2 parameter importance plot"
    echo "  2. Check Phase 2 top 10 trial configurations"
    echo "  3. Update Phase 3 config to narrow search space around best values"
    echo ""
    echo "Alternatively, enable automatic refinement by adding to config:"
    echo "  refinement:"
    echo "    enabled: true"
    echo "    previous_phase_dir: /path/to/phase2/output"
    echo ""
    echo "Phase 3 config: $phase3_config"
    echo ""
    read -p "Press Enter to continue with Phase 3, or Ctrl+C to abort..."
    echo ""
  fi

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
