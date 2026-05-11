#!/usr/bin/env bash
# =============================================================================
# scGPT Probe — wandb sweep launcher
#
# Runs 4 datasets sequentially via wandb sweep (grid search over dataset_id).
# Each run logs to the "scgpt-probe" wandb project.
#
# Usage:
#   bash run_sweep.sh
# or in background:
#   nohup bash run_sweep.sh > run_sweep.log 2>&1 &
# =============================================================================
set -euo pipefail

WANDB="/lichaohan/miniconda3/envs/scvi/bin/wandb"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}"

# ─── Create sweep and capture the sweep ID ──────────────────────────────────
echo "Creating sweep from sweep.yaml ..."
SWEEP_OUT=$("${WANDB}" sweep sweep.yaml 2>&1)
echo "${SWEEP_OUT}"

# Extract "entity/project/sweepid" from the agent command line wandb prints
SWEEP_ID=$(echo "${SWEEP_OUT}" | grep 'wandb agent' | awk '{print $NF}')

if [[ -z "${SWEEP_ID}" ]]; then
    echo "ERROR: could not parse sweep ID from wandb output." >&2
    exit 1
fi

echo ""
echo "Sweep ID : ${SWEEP_ID}"
echo "Starting agent (datasets run sequentially) ..."
echo ""

# ─── Run agent — processes all 4 trials one by one ──────────────────────────
"${WANDB}" agent "${SWEEP_ID}"
