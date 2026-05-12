"""
Paper-quality UMAP / t-SNE visualization of probe embeddings.

Requires embeddings saved with --save_embeddings when running probe.py.
Looks for:
    <run_dir>/embeddings_val.npy   — (N, D) float32 embedding matrix
    <run_dir>/labels_val.npy       — (N,)   int64  integer labels
    <run_dir>/class_names.json     — list[str] label names (index = integer label)
    <run_dir>/probe_metrics.json   — used for title / subtitle (optional)

Usage examples:
    # Single run
    python visualize.py --run_dir outputs_probe/probe_5w_symbol_ensembl

    # All runs under a parent directory
    python visualize.py --run_dir outputs_probe/

    # Only UMAP, subsample to 10k cells
    python visualize.py --run_dir outputs_probe/probe_5w_symbol_ensembl \\
                        --method umap --max_cells 10000
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Matplotlib paper style ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "font.family":      "DejaVu Sans",   # Arial substitute always available
    "font.size":        9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  7,
    "figure.figsize":   (8.5, 7),
    "pdf.fonttype":     42,             # embed fonts as Type 42 (editable in Illustrator)
    "ps.fonttype":      42,
})


# ── Colour helpers ────────────────────────────────────────────────────────

def _build_palette(n_classes: int):
    """
    Return a list of n_classes RGB colours.
      ≤ 8  → Set2   (colour-blind friendly)
      ≤ 20 → Tab20
      > 20 → Tab20 + Tab20b  (up to 40 distinct colours)
    """
    import matplotlib.cm as cm

    if n_classes <= 8:
        cmap = plt.get_cmap("Set2")
        return [cmap(i / 8) for i in range(n_classes)]

    elif n_classes <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(i / 20) for i in range(n_classes)]

    else:
        tab20  = [plt.get_cmap("tab20")(i / 20)  for i in range(20)]
        tab20b = [plt.get_cmap("tab20b")(i / 20) for i in range(20)]
        combined = tab20 + tab20b          # 40 distinct colours
        # cycle if > 40 classes (rare)
        return [combined[i % len(combined)] for i in range(n_classes)]


# ── Dimensionality reduction ──────────────────────────────────────────────

def _reduce_umap(embeddings: np.ndarray, seed: int) -> np.ndarray:
    import umap as umap_lib
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.3,
        spread=1.0,
        random_state=seed,
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return reducer.fit_transform(embeddings)


def _reduce_tsne(embeddings: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE
    n = embeddings.shape[0]
    perplexity = min(30, max(5, n // 100))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=1000,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return reducer.fit_transform(embeddings)


# ── Single plot ───────────────────────────────────────────────────────────

def _plot_embedding(
    coords_2d:   np.ndarray,      # (N, 2)
    labels:      np.ndarray,      # (N,) int
    class_names: list,
    method:      str,
    title:       str,
    subtitle:    str,
    save_path:   str,
    point_size:  float,
    alpha:       float,
):
    unique_labels = sorted(np.unique(labels))
    n_classes     = len(unique_labels)
    palette       = _build_palette(n_classes)

    # ── figure size: wider when legend is long ───────────────────────
    # Roughly 2.5 cm per legend row in the side panel
    n_legend_cols = 1 if n_classes <= 15 else 2
    fig_w = 8.5 + 3.0 * n_legend_cols
    fig, ax = plt.subplots(figsize=(fig_w, 7), dpi=300)

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = class_names[lbl] if lbl < len(class_names) else str(lbl)
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[palette[i]],
            s=point_size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
            label=name,
            zorder=2,
        )

    # ── axis cosmetics ───────────────────────────────────────────────
    axis_label = "UMAP" if method == "umap" else "t-SNE"
    ax.set_xlabel(f"{axis_label} 1")
    ax.set_ylabel(f"{axis_label} 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── title ────────────────────────────────────────────────────────
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    if subtitle:
        ax.text(
            0.5, 1.01, subtitle,
            transform=ax.transAxes,
            fontsize=7, ha="center", va="bottom", color="#555555",
        )

    # ── legend (outside right) ───────────────────────────────────────
    legend = ax.legend(
        title="Cell type",
        title_fontsize=8,
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        ncol=n_legend_cols,
        borderpad=0,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.8,
        markerscale=3.0,
    )

    plt.tight_layout()

    # ── save PDF + PNG ───────────────────────────────────────────────
    pdf_path = save_path + ".pdf"
    png_path = save_path + ".png"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {pdf_path}")
    print(f"    Saved: {png_path}")


# ── Visualize one run directory ───────────────────────────────────────────

def visualize_run(run_dir: str, methods: list, max_cells: int, seed: int):
    # ── check required files ─────────────────────────────────────────
    emb_path    = os.path.join(run_dir, "embeddings_val.npy")
    lbl_path    = os.path.join(run_dir, "labels_val.npy")
    names_path  = os.path.join(run_dir, "class_names.json")
    metrics_path = os.path.join(run_dir, "probe_metrics.json")

    for p in [emb_path, lbl_path, names_path]:
        if not os.path.exists(p):
            print(
                f"  [SKIP] {run_dir}\n"
                f"    Missing: {os.path.basename(p)}\n"
                f"    → Re-run probe.py with --save_embeddings"
            )
            return

    # ── load ─────────────────────────────────────────────────────────
    embeddings  = np.load(emb_path)      # (N, D)
    labels      = np.load(lbl_path)      # (N,)
    with open(names_path) as f:
        class_names = json.load(f)

    n_total = len(labels)
    print(f"  Loaded: {n_total} cells × {embeddings.shape[1]} dims, "
          f"{len(class_names)} classes")

    # ── build title / subtitle ────────────────────────────────────────
    run_name = os.path.basename(run_dir.rstrip("/"))
    subtitle = ""
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            mdata = json.load(f)
        m = mdata.get("metrics", {}).get("test", {})
        parts = []
        if "macro_f1" in m:
            parts.append(f"macro F1={m['macro_f1']:.3f}")
        if "balanced_accuracy" in m:
            parts.append(f"bal acc={m['balanced_accuracy']:.3f}")
        subtitle = "  ·  ".join(parts)

    # ── subsample ────────────────────────────────────────────────────
    if n_total > max_cells:
        rng   = np.random.default_rng(seed)
        idx   = rng.choice(n_total, max_cells, replace=False)
        embeddings = embeddings[idx]
        labels     = labels[idx]
        print(f"  Subsampled to {max_cells} cells for dimensionality reduction")

    # ── point size: smaller for more cells ───────────────────────────
    n = len(labels)
    point_size = max(0.5, min(3.0, 30000 / n))
    alpha      = max(0.4, min(0.7, 20000 / n))

    # ── run reductions ────────────────────────────────────────────────
    vis_dir = os.path.join(run_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    for method in methods:
        print(f"  Computing {method.upper()}...", flush=True)
        if method == "umap":
            coords = _reduce_umap(embeddings, seed)
        else:
            coords = _reduce_tsne(embeddings, seed)

        title = f"{run_name}  [{method.upper()}]"
        save_path = os.path.join(vis_dir, method)
        _plot_embedding(
            coords, labels, class_names,
            method=method,
            title=title,
            subtitle=subtitle,
            save_path=save_path,
            point_size=point_size,
            alpha=alpha,
        )

        # save 2D coords for reuse (avoid recomputing)
        np.save(os.path.join(vis_dir, f"{method}_coords.npy"), coords)
        print(f"    Coords cached: {vis_dir}/{method}_coords.npy")


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Paper-quality UMAP/t-SNE for probe embedding outputs"
    )
    p.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to a single probe run directory (containing embeddings_val.npy), "
             "or a parent directory to visualize all runs inside it.",
    )
    p.add_argument(
        "--method", type=str, default="both",
        choices=["umap", "tsne", "both"],
        help="Dimensionality reduction method (default: both)",
    )
    p.add_argument(
        "--max_cells", type=int, default=20000,
        help="Maximum cells to use for UMAP/t-SNE (subsampled if larger). "
             "Default 20000.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p.add_argument(
        "--n_jobs", type=int, default=1,
        help="Number of runs to process in parallel (default: 1). "
             "Set to -1 to use all available CPU cores.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    methods = ["umap", "tsne"] if args.method == "both" else [args.method]

    run_dir = os.path.abspath(args.run_dir)

    # Determine whether run_dir itself is a run or a parent of runs
    emb_in_dir = os.path.exists(os.path.join(run_dir, "embeddings_val.npy"))
    if emb_in_dir:
        run_dirs = [run_dir]
    else:
        # scan immediate children
        children = sorted([
            os.path.join(run_dir, d)
            for d in os.listdir(run_dir)
            if os.path.isdir(os.path.join(run_dir, d))
        ])
        run_dirs = children
        if not run_dirs:
            print(f"No subdirectories found in {run_dir}")
            sys.exit(1)

    print(f"Visualizing {len(run_dirs)} run(s) | methods={methods} | "
          f"max_cells={args.max_cells} | seed={args.seed} | "
          f"n_jobs={args.n_jobs}\n")

    def _do(rd):
        print(f"→ {os.path.basename(rd)}", flush=True)
        visualize_run(rd, methods, args.max_cells, args.seed)
        print()

    if args.n_jobs == 1 or len(run_dirs) == 1:
        for rd in run_dirs:
            _do(rd)
    else:
        from joblib import Parallel, delayed
        Parallel(n_jobs=args.n_jobs, backend="loky", verbose=0)(
            delayed(_do)(rd) for rd in run_dirs
        )


if __name__ == "__main__":
    main()
