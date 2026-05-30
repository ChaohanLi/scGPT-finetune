"""
Unified cell-type evaluation for scGPT embeddings.

This script keeps the scGPT encoder fixed, extracts one embedding per cell
(CLS-token, 512-dim), and evaluates cell-type separability with a LinearSVC
probe — mirroring the scFoundation probe protocol:

    validation embeddings -> 5-fold StratifiedKFold
    each fold: StandardScaler -> optional PCA(100) -> LinearSVC
    report mean CV train/test accuracy and macro-F1
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import wandb
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

_PROB_DIR = os.path.dirname(os.path.abspath(__file__))
# Reuse dataset.py from celltype — identical data loading logic
_CELLTYPE_DIR = os.path.join(_PROB_DIR, "..", "celltype")
sys.path.insert(0, _CELLTYPE_DIR)
from dataset import load_data  # noqa: E402

sys.path.insert(0, _PROB_DIR)
from model import build_model   # noqa: E402


# ---------------------------------------------------------------------------
# Dataset registry — maps dataset_id → (h5ad, n_class, preprocess)
# Sweep agents only need to pass --dataset_id; other dataset-specific fields
# are resolved automatically from this table.
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    "5w_symbol":     {
        "h5ad":       "/lichaohan/readData/5w_allcelltype_anno_symbol.h5ad",
        "n_class":    29,
        "preprocess": False,
    },
    "5w_GSE196830":  {
        "h5ad":       "/lichaohan/readData/5w_PBMC_GSE196830/5w_allcelltype.h5ad",
        "n_class":    29,
        "preprocess": True,
    },
    "GSE96583":      {
        "h5ad":       "/lichaohan/readData/GSE96583_PBMC/GSE96583_merged_dedup.h5ad",
        "n_class":    8,
        "preprocess": True,
    },
    "10w_GSE196830": {
        "h5ad":       "/lichaohan/readData/10w_PBMC_GSE196830/10w_allcelltype.h5ad",
        "n_class":    29,
        "preprocess": True,
    },
    "20w_GSE196830": {
        "h5ad":       "/lichaohan/readData/20w_PBMC_GSE196830/20w_allcelltype.h5ad",
        "n_class":    29,
        "preprocess": True,
    },
    "40w_GSE196830": {
        "h5ad":       "/lichaohan/readData/40w_PBMC_GSE196830/GSE196830_40w_subset.h5ad",
        "n_class":    29,
        "preprocess": True,
    },
    "80w_GSE196830": {
        "h5ad":       "/lichaohan/readData/80w_PBMC_GSE196830/GSE196830_80w_subset.h5ad",
        "n_class":    29,
        "preprocess": True,
    },
    "120w_GSE196830": {
        "h5ad":       "/lichaohan/readData/120w_PBMC_GSE196830/GSE196830_120w_subset.h5ad",
        "n_class":    29,
        "preprocess": True,
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract scGPT embeddings and evaluate a LinearSVC probe"
    )
    p.add_argument("--model_dir", type=str,
                   default="/lichaohan/scGPT/scGPT_human")
    p.add_argument("--h5ad", type=str,
                   default="/lichaohan/readData/5w_allcelltype_anno_symbol.h5ad")
    p.add_argument("--dataset_id", type=str, default="5w_symbol",
                   help="Short dataset tag appended to run_name (e.g. 5w_symbol, "
                        "5w_GSE196830, GSE96583, 10w_GSE196830, 20w_GSE196830, 40w_GSE196830)")
    p.add_argument("--n_class", type=int, default=29)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--train_size", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--max_samples", type=int, default=5000)
    p.add_argument("--pca_dim", type=int, default=100)
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument("--n_jobs", type=int, default=16,
                   help="Total CPU cores for parallel fold evaluation.")
    p.add_argument("--no_frozenmore", action="store_true",
                   help="Also unfreeze token/value embeddings before embedding extraction")
    p.add_argument("--n_hvg", type=int, default=3000,
                   help="Number of highly variable genes to subset before feeding to scGPT. "
                        "Matches scGPT paper Tutorial_Cell_Embedding protocol (1200–3000). "
                        "Set to 0 to disable HVG selection (feed full vocab-matched genes).")
    p.add_argument("--preprocess", action="store_true",
                   help="Apply normalize_total+log1p before processing (for raw count datasets)")
    p.add_argument("--symbol_map", type=str,
                   default="/lichaohan/readData/gene_id_to_symbol.tsv",
                   help="TSV (gene_id, gene_symbol) used when h5ad lacks a gene_symbol column. "
                        "Set to '' to disable.")
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(_PROB_DIR, "outputs_probe"))
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--save_embeddings", action="store_true")
    # ── Weights & Biases ───────────────────────────────────────────────
    p.add_argument("--wandb_project", type=str, default="scgpt-probe")
    p.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    return p.parse_args()


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_embeddings, all_labels = [], []

    for batch in loader:
        gene = batch["gene"].to(device, non_blocking=True)
        expr = batch["expr"].to(device, non_blocking=True)
        emb  = model.encode({"gene": gene, "expr": expr})
        all_embeddings.append(emb.cpu().numpy())
        all_labels.append(batch["targets"].cpu().numpy())

    return np.concatenate(all_embeddings), np.concatenate(all_labels)


def build_probe(train_embeddings, args):
    steps = [("scaler", StandardScaler())]
    if args.pca_dim is not None:
        pca_dim = min(
            int(args.pca_dim),
            train_embeddings.shape[0],
            train_embeddings.shape[1],
        )
        if pca_dim >= 1 and pca_dim < train_embeddings.shape[1]:
            steps.append(("pca", PCA(n_components=pca_dim,
                                     random_state=args.seed)))
    steps.append(("svc", LinearSVC(
        random_state=args.seed,
        dual=False,
        max_iter=args.max_iter,
    )))
    return Pipeline(steps)


def compute_metrics(labels, preds):
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted",
                                      zero_division=0)),
        "n_samples": int(len(labels)),
    }


def run_svc_cv(embeddings, labels, args):
    unique, counts = np.unique(labels, return_counts=True)
    keep_classes   = unique[counts >= args.cv_folds]
    dropped_classes = unique[counts < args.cv_folds]
    if len(keep_classes) < 2:
        raise ValueError(
            f"Need at least 2 classes with >= {args.cv_folds} samples for SVC CV; "
            f"got {len(keep_classes)}."
        )

    dropped_info = [
        {"class_id": int(cls_id), "count": int(cls_count)}
        for cls_id, cls_count in zip(unique, counts)
        if cls_count < args.cv_folds
    ]
    if len(keep_classes) != len(unique):
        mask       = np.isin(labels, keep_classes)
        embeddings = embeddings[mask]
        labels     = labels[mask]
        labels     = np.searchsorted(keep_classes, labels)

    splitter = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.seed,
    )

    n_fold_jobs = min(args.cv_folds, args.n_jobs)
    n_jobs_ovr  = max(1, args.n_jobs // n_fold_jobs)
    print(f"Parallelism: {n_fold_jobs} fold workers × {n_jobs_ovr} OvR cores "
          f"= {n_fold_jobs * n_jobs_ovr} cores used (of {args.n_jobs})",
          flush=True)

    splits = list(splitter.split(embeddings, labels))

    def _run_fold(fold_idx, train_idx, test_idx):
        x_train = embeddings[train_idx]
        y_train = labels[train_idx]
        x_test  = embeddings[test_idx]
        y_test  = labels[test_idx]

        x_train_fit = x_train
        y_train_fit = y_train

        probe = build_probe(x_train_fit, args)
        print(f"  Fold {fold_idx}/{args.cv_folds}: fitting SVC on "
              f"{len(x_train_fit)} samples...", flush=True)
        probe.fit(x_train_fit, y_train_fit)

        train_preds = probe.predict(x_train)
        test_preds  = probe.predict(x_test)
        print(f"  Fold {fold_idx}/{args.cv_folds}: done.", flush=True)
        n_kept = len(keep_classes)
        return {
            "fold": fold_idx,
            "train_size": int(len(x_train)),
            "train_fit_size": int(len(x_train_fit)),
            "test_size": int(len(x_test)),
            "train": compute_metrics(y_train, train_preds),
            "test":  compute_metrics(y_test,  test_preds),
            "probe_steps": list(probe.named_steps.keys()),
            "test_per_class": {
                "f1":        f1_score(y_test, test_preds, average=None,
                                      labels=np.arange(n_kept), zero_division=0).tolist(),
                "precision": precision_score(y_test, test_preds, average=None,
                                             labels=np.arange(n_kept), zero_division=0).tolist(),
                "recall":    recall_score(y_test, test_preds, average=None,
                                          labels=np.arange(n_kept), zero_division=0).tolist(),
                "support":   [int((y_test == c).sum()) for c in range(n_kept)],
            },
        }

    fold_metrics = Parallel(n_jobs=n_fold_jobs, backend="threading")(
        delayed(_run_fold)(fold_idx, train_idx, test_idx)
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1)
    )

    mean_metrics = {}
    for split in ["train", "test"]:
        split_metrics = {}
        for key in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
            split_metrics[key] = float(np.mean([
                fold[split][key] for fold in fold_metrics
            ]))
        mean_metrics[split] = split_metrics

    # Aggregate per-class metrics across folds (mean ± std)
    n_kept = len(keep_classes)
    per_class_cv = []
    for c in range(n_kept):
        fold_f1s   = [fm["test_per_class"]["f1"][c]        for fm in fold_metrics]
        fold_precs = [fm["test_per_class"]["precision"][c] for fm in fold_metrics]
        fold_recs  = [fm["test_per_class"]["recall"][c]    for fm in fold_metrics]
        fold_sups  = [fm["test_per_class"]["support"][c]   for fm in fold_metrics]
        per_class_cv.append({
            "class_idx":      int(c),
            "mean_f1":        float(np.mean(fold_f1s)),
            "std_f1":         float(np.std(fold_f1s)),
            "mean_precision": float(np.mean(fold_precs)),
            "std_precision":  float(np.std(fold_precs)),
            "mean_recall":    float(np.mean(fold_recs)),
            "std_recall":     float(np.std(fold_recs)),
            "mean_support":   float(np.mean(fold_sups)),
        })

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "per_class_cv": per_class_cv,
        "kept_classes": [int(x) for x in keep_classes.tolist()],
        "dropped_classes": dropped_info,
        "n_samples_after_filter": int(len(labels)),
    }


def save_fold_metrics(path, cv_result):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold",
            "train_size",
            "train_fit_size",
            "test_size",
            "train_accuracy",
            "test_accuracy",
            "train_macro_f1",
            "test_macro_f1",
        ])
        for fold in cv_result["fold_metrics"]:
            writer.writerow([
                fold["fold"],
                fold["train_size"],
                fold["train_fit_size"],
                fold["test_size"],
                fold["train"]["accuracy"],
                fold["test"]["accuracy"],
                fold["train"]["macro_f1"],
                fold["test"]["macro_f1"],
            ])


def main():
    args = parse_args()
    # Auto-resolve dataset-specific fields from registry (enables wandb sweep)
    if args.dataset_id in DATASET_REGISTRY:
        cfg = DATASET_REGISTRY[args.dataset_id]
        args.h5ad    = cfg["h5ad"]
        args.n_class = cfg["n_class"]
        if cfg["preprocess"]:
            args.preprocess = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.run_name or time.strftime("probe_%Y%m%d_%H%M%S")
    run_name = f"{run_name}_{args.dataset_id}"
    out_dir  = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output directory: {out_dir}")

    # ── Weights & Biases init ───────────────────────────────────────────
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )

    # ── HVG selection (scGPT paper Tutorial_Cell_Embedding protocol) ──────
    # Done here (not in dataset.py) to keep scGPT source untouched.
    # Writes a temp h5ad and points load_data at it.
    h5ad_path_for_load = args.h5ad
    _tmp_h5ad_to_cleanup = None
    if args.n_hvg and args.n_hvg > 0:
        import scanpy as sc
        import tempfile
        import atexit
        print(f"\nApplying HVG selection (n_top_genes={args.n_hvg}) before load_data ...")
        adata = sc.read_h5ad(args.h5ad)
        print(f"  Input shape: {adata.shape}")
        if args.n_hvg < adata.shape[1]:
            flavor = "seurat_v3" if args.preprocess else "seurat"
            print(f"  HVG flavor: {flavor} (preprocess={args.preprocess})")
            sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg, flavor=flavor)
            adata = adata[:, adata.var.highly_variable].copy()
            print(f"  HVG subset shape: {adata.shape}")
            _tmp_h5ad_to_cleanup = tempfile.NamedTemporaryFile(
                suffix=".h5ad", prefix=f"scgpt_hvg_{args.dataset_id}_", delete=False
            ).name
            adata.write_h5ad(_tmp_h5ad_to_cleanup)
            atexit.register(lambda p=_tmp_h5ad_to_cleanup:
                            os.path.exists(p) and os.unlink(p))
            h5ad_path_for_load = _tmp_h5ad_to_cleanup
            print(f"  Wrote HVG-filtered h5ad: {_tmp_h5ad_to_cleanup}")
        else:
            print(f"  Skipping HVG: n_hvg={args.n_hvg} >= n_vars={adata.shape[1]}")
        del adata

    # load_data from scGPT/celltype/dataset.py returns 7 values
    train_loader, val_loader, class_names, type2idx, _, vocab, pad_token_id = load_data(
        h5ad_path=h5ad_path_for_load,
        model_dir=args.model_dir,
        train_size=args.train_size,
        random_state=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        preprocess=args.preprocess,
        symbol_map=args.symbol_map or None,
    )
    n_class = len(class_names)
    assert n_class == args.n_class, (
        f"Expected {args.n_class} classes but found {n_class}. Update --n_class."
    )

    model = build_model(
        model_dir=args.model_dir,
        n_class=n_class,
        vocab=vocab,
        pad_token_id=pad_token_id,
        frozenmore=not args.no_frozenmore,
        device=device,
    )
    # Probe only needs frozen embeddings — freeze everything and discard cls_decoder
    for p in model.parameters():
        p.requires_grad = False
    del model.transformer.cls_decoder

    print("Extracting validation embeddings...")
    x_val, y_val = extract_embeddings(model, val_loader, device)
    print(f"Embedding shape: {x_val.shape[1]} dims")
    print(f"Validation samples: {len(y_val)}")
    print("Extracting training embeddings...")
    x_train, y_train = extract_embeddings(model, train_loader, device)
    print(f"Training samples: {len(y_train)}")

    cv_result = run_svc_cv(x_val, y_val, args)

    result = {
        "metrics": cv_result["mean_metrics"],
        "fold_metrics": cv_result["fold_metrics"],
        "embedding_dim": int(x_val.shape[1]),
        "class_names":  class_names,
        "per_class_cv": cv_result["per_class_cv"],
        "type2idx":     type2idx,
        "kept_classes": cv_result["kept_classes"],
        "dropped_classes": cv_result["dropped_classes"],
        "n_samples_after_filter": cv_result["n_samples_after_filter"],
        "args": vars(args),
        "protocol": "val_embeddings_5fold_svc_cv",
    }
    with open(os.path.join(out_dir, "probe_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)
    save_fold_metrics(os.path.join(out_dir, "probe_fold_metrics.csv"), cv_result)

    if args.save_embeddings:
        np.save(os.path.join(out_dir, "embeddings_val.npy"),   x_val)
        np.save(os.path.join(out_dir, "labels_val.npy"),       y_val)
        np.save(os.path.join(out_dir, "embeddings_train.npy"), x_train)
        np.save(os.path.join(out_dir, "labels_train.npy"),     y_train)
        x_all = np.concatenate([x_train, x_val], axis=0)
        y_all = np.concatenate([y_train, y_val], axis=0)
        np.save(os.path.join(out_dir, "embeddings_all.npy"),   x_all)
        np.save(os.path.join(out_dir, "labels_all.npy"),       y_all)

    print("\nProbe metrics")
    for split in ["train", "test"]:
        m = cv_result["mean_metrics"][split]
        print(
            f"cv {split:>5}: acc={m['accuracy']:.4f} "
            f"bal_acc={m['balanced_accuracy']:.4f} "
            f"macro_f1={m['macro_f1']:.4f} "
            f"weighted_f1={m['weighted_f1']:.4f}"
        )
    if cv_result["dropped_classes"]:
        print(f"Dropped classes with < {args.cv_folds} samples: "
              f"{cv_result['dropped_classes']}")
    print(f"\nSaved to: {out_dir}")

    # ── Weights & Biases logging ───────────────────────────────────────
    if not args.no_wandb:
        mean = cv_result["mean_metrics"]
        wandb.log({
            "cv_train/accuracy":          mean["train"]["accuracy"],
            "cv_train/balanced_accuracy": mean["train"]["balanced_accuracy"],
            "cv_train/macro_f1":          mean["train"]["macro_f1"],
            "cv_train/weighted_f1":       mean["train"]["weighted_f1"],
            "cv_test/accuracy":           mean["test"]["accuracy"],
            "cv_test/balanced_accuracy":  mean["test"]["balanced_accuracy"],
            "cv_test/macro_f1":           mean["test"]["macro_f1"],
            "cv_test/weighted_f1":        mean["test"]["weighted_f1"],
            "embedding_dim":              int(x_val.shape[1]),
            "n_val_samples":              int(len(y_val)),
            "n_classes_used":             len(cv_result["kept_classes"]),
            "n_classes_dropped":          len(cv_result["dropped_classes"]),
        })
        fold_table = wandb.Table(
            columns=["fold", "train_size", "test_size",
                     "train_acc", "test_acc", "train_macro_f1", "test_macro_f1"]
        )
        for fold in cv_result["fold_metrics"]:
            fold_table.add_data(
                fold["fold"],
                fold["train_size"],
                fold["test_size"],
                fold["train"]["accuracy"],
                fold["test"]["accuracy"],
                fold["train"]["macro_f1"],
                fold["test"]["macro_f1"],
            )
        wandb.log({"fold_metrics": fold_table})
        # Per-class accuracy table (mean ± std across folds, test split)
        per_class_table = wandb.Table(
            columns=["class_name", "mean_f1", "std_f1",
                     "mean_recall", "mean_precision", "mean_support"]
        )
        kept = cv_result["kept_classes"]
        for entry in cv_result["per_class_cv"]:
            orig = kept[entry["class_idx"]]
            name = class_names[orig] if orig < len(class_names) else str(orig)
            per_class_table.add_data(
                name,
                round(entry["mean_f1"],        4),
                round(entry["std_f1"],         4),
                round(entry["mean_recall"],    4),
                round(entry["mean_precision"], 4),
                round(entry["mean_support"],   1),
            )
        wandb.log({"per_class_metrics": per_class_table})
        wandb.finish()


if __name__ == "__main__":
    main()
