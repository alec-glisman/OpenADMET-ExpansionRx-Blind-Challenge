#!/usr/bin/env bash
#
# run_xgb_ensembles.sh
# ====================
# Purpose:
#   Batch-train and evaluate every ensemble defined under configs/{high,medium,low}_quality.
#
# What it does:
#   For each quality tier and split directory (butina, kmeans, random, scaffold, temporal, umap),
#   the script executes:
#     1. `admet --log-level INFO train xgb --config <quality>/<split>/xgb_train_ensemble.yaml`
#     2. `admet --log-level INFO ensemble-eval --config <quality>/<split>/xgb_predict_ensemble.yaml`
#
# Requirements:
#   - `admet` CLI must be on PATH with the proper environment activated.
#   - YAML configs in each split directory must follow the expected naming convention.
#
# Usage:
#   bash experiments/run_xgb_ensembles.sh
#
# Notes:
#   - Missing config files are skipped with a warning so partial runs still succeed.
#   - Logs stream directly from the admet CLI; consider redirecting stdout/stderr if desired.

set -euo pipefail

MODEL_NAME="xgb"
QUALITIES=(
  #   "high_quality"
  "medium_quality"
  "low_quality"
)

echo "Starting ${MODEL_NAME} ensemble for ${#QUALITIES[@]}} quality tiers."
for quality in "${QUALITIES[@]}"; do
  CONFIG_ROOT="configs/${quality}"
  [[ -d "${CONFIG_ROOT}" ]] || {
    echo "[${quality}] Config root missing, skipping"
    continue
  }
  echo "[${quality}] Starting ensemble runs..."

  for split_dir in "${CONFIG_ROOT}"/*/; do
    [[ -d "${split_dir}" ]] || continue # Skip non-directory files

    split_name="$(basename "${split_dir}")" # Get the split name from the directory

    train_cfg="${split_dir}${MODEL_NAME}_train_ensemble.yaml"
    predict_cfg="${split_dir}${MODEL_NAME}_predict_ensemble.yaml"

    if [[ -f "${train_cfg}" ]]; then
      echo "[${quality}/${split_name}] Training ensemble"
      admet --log-level INFO \
        train "${MODEL_NAME}" \
        --config "${train_cfg}"
    else
      echo "[${quality}/${split_name}] Missing ${MODEL_NAME}_train_ensemble.yaml, skipping train" >&2
    fi

    if [[ -f "${predict_cfg}" ]]; then
      echo "[${quality}/${split_name}] Evaluating ensemble"
      admet --log-level INFO \
        ensemble-eval \
        --config "${predict_cfg}"
    else
      echo "[${quality}/${split_name}] Missing ${MODEL_NAME}_predict_ensemble.yaml, skipping eval" >&2
    fi

  done
done
