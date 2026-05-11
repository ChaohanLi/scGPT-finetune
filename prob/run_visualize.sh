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
RUN_DIR="${SCRIPT_DIR}/outputs_probe"

METHOD="both"      # umap | tsne | both
MAX_CELLS=20000    # subsample to this many cells before UMAP/t-SNE
SEED=42

# ─── Run ────────────────────────────────────────────────────────────────────
cd "${SCRIPT_DIR}"

$PYTHON visualize.py \
    --run_dir   "${RUN_DIR}" \
    --method    "${METHOD}" \
    --max_cells "${MAX_CELLS}" \
    --seed      "${SEED}"
