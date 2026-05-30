#!/usr/bin/env bash
# =============================================================================
# scGPT Probe — visualize embeddings
# Requires probe to have been run with SAVE_EMBEDDINGS="--save_embeddings".
# Usage:
#   bash run_visualize.sh
# or in background:
#   nohup bash run_visualize.sh > run_visualize.log 2>&1 &
# =============================================================================
set -euo pipefail

PYTHON="/lichaohan/miniconda3/envs/scvi/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Configuration ──────────────────────────────────────────────────────────
# Point RUN_DIR to a single run directory (e.g. outputs_probe/probe_GSE96583_hgnc)
# or to the outputs_probe/ root to visualize all runs at once.
RUN_DIR="/lichaohan/scGPT/prob/outputs_probe"

METHOD="both"      # umap | tsne | both
MAX_CELLS=20000    # subsample to this many cells before UMAP/t-SNE
SEED=42
N_JOBS=4           # runs processed in parallel (each run uses 1 core; set to -1 for all cores)

# Comma-separated list of splits to visualize: val, train, all
# Examples:
#   SPLITS="val"           → only validation embeddings
#   SPLITS="val,all"       → val and full-dataset embeddings
#   SPLITS="val,train,all" → all three splits
SPLITS="joint"

# Filter run directories by name substrings (comma-separated, ALL must match).
# Leave empty ("") to process every run under RUN_DIR.
# Examples:
#   RUN_NAME_FILTER="20260512"              → only runs from 2026-05-12
#   RUN_NAME_FILTER="20260512,hgnc"         → 2026-05-12 runs with hgnc gene space
RUN_NAME_FILTER="20260512"

# ─── Run ────────────────────────────────────────────────────────────────────
cd "${SCRIPT_DIR}"

FILTER_ARG=()
if [[ -n "${RUN_NAME_FILTER}" ]]; then
    FILTER_ARG=(--run_name_filter "${RUN_NAME_FILTER}")
fi

$PYTHON visualize.py \
    --run_dir   "${RUN_DIR}" \
    --method    "${METHOD}" \
    --max_cells "${MAX_CELLS}" \
    --seed      "${SEED}" \
    --n_jobs    "${N_JOBS}" \
    --splits    "${SPLITS}" \
    "${FILTER_ARG[@]}"
