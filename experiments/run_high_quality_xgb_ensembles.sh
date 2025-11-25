#!/usr/bin/env bash
#
# run_high_quality_xgb_ensembles.sh
# ================================
# Purpose:
#   Batch-train and evaluate every ensemble defined under configs/high_quality.
#
# What it does:
#   For each high-quality split directory (butina, kmeans, random, scaffold, temporal, umap),
#   the script executes:
#     1. `admet --log-level INFO train xgb --config <split>/xgb_train_ensemble.yaml`
#     2. `admet --log-level INFO ensemble-eval --config <split>/xgb_predict_ensemble.yaml`
#
# Requirements:
#   - `admet` CLI must be on PATH with the proper environment activated.
#   - YAML configs in each split directory must follow the expected naming convention.
#
# Usage:
#   bash experiments/run_high_quality_xgb_ensembles.sh
#
# Notes:
#   - Missing config files are skipped with a warning so partial runs still succeed.
#   - Logs stream directly from the admet CLI; consider redirecting stdout/stderr if desired.

set -euo pipefail
echo "Starting high-quality XGB ensemble runs..."

# Directory setup
CONFIG_ROOT="configs/high_quality"
MODEL_NAME="xgb"

for split_dir in "${CONFIG_ROOT}"/*/; do
  [[ -d "${split_dir}" ]] || continue # Skip non-directory files
  
  split_name="$(basename "${split_dir}")" # Get the split name from the directory
  
  train_cfg="${split_dir}${MODEL_NAME}_train_ensemble.yaml"
  predict_cfg="${split_dir}${MODEL_NAME}_predict_ensemble.yaml"

  if [[ -f "${train_cfg}" ]]; then
    echo "[${split_name}] Training ensemble"
    admet --log-level INFO \
        train "${MODEL_NAME}" \
        --config "${train_cfg}"
  else
    echo "[${split_name}] Missing ${MODEL_NAME}_train_ensemble.yaml, skipping train" >&2
  fi

  if [[ -f "${predict_cfg}" ]]; then
    echo "[${split_name}] Evaluating ensemble"
    admet --log-level INFO \
        ensemble-eval \
        --config "${predict_cfg}"
  else
    echo "[${split_name}] Missing ${MODEL_NAME}_predict_ensemble.yaml, skipping eval" >&2
  fi

done
