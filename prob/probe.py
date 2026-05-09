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


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract scGPT embeddings and evaluate a LinearSVC probe"
    )
    p.add_argument("--model_dir", type=str,
                   default="/lichaohan/scGPT/scGPT_human")
    p.add_argument("--h5ad", type=str,
                   default="/lichaohan/readData/5w_allcelltype_anno_symbol.h5ad")
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
    p.add_argument("--output_dir", type=str, default="outputs_probe")
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

        if args.max_samples and len(x_train) > args.max_samples:
            sampled_idx = np.random.choice(
                len(x_train), args.max_samples, replace=False
            )
            x_train_fit = x_train[sampled_idx]
            y_train_fit = y_train[sampled_idx]
        else:
            x_train_fit = x_train
            y_train_fit = y_train

        probe = build_probe(x_train_fit, args)
        print(f"  Fold {fold_idx}/{args.cv_folds}: fitting SVC on "
              f"{len(x_train_fit)} samples...", flush=True)
        probe.fit(x_train_fit, y_train_fit)

        train_preds = probe.predict(x_train)
        test_preds  = probe.predict(x_test)
        print(f"  Fold {fold_idx}/{args.cv_folds}: done.", flush=True)
        return {
            "fold": fold_idx,
            "train_size": int(len(x_train)),
            "train_fit_size": int(len(x_train_fit)),
            "test_size": int(len(x_test)),
            "train": compute_metrics(y_train, train_preds),
            "test":  compute_metrics(y_test,  test_preds),
            "probe_steps": list(probe.named_steps.keys()),
        }

    fold_metrics = Parallel(n_jobs=n_fold_jobs, backend="loky")(
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

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.run_name or time.strftime("probe_%Y%m%d_%H%M%S")
    out_dir  = os.path.join(_PROB_DIR, args.output_dir, run_name)
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

    # load_data from scGPT/celltype/dataset.py returns 7 values
    _train_loader, val_loader, class_names, type2idx, _, vocab, pad_token_id = load_data(
        h5ad_path=args.h5ad,
        model_dir=args.model_dir,
        train_size=args.train_size,
        random_state=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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

    cv_result = run_svc_cv(x_val, y_val, args)

    result = {
        "metrics": cv_result["mean_metrics"],
        "fold_metrics": cv_result["fold_metrics"],
        "embedding_dim": int(x_val.shape[1]),
        "class_names": class_names,
        "type2idx": type2idx,
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
        np.save(os.path.join(out_dir, "embeddings_val.npy"), x_val)
        np.save(os.path.join(out_dir, "labels_val.npy"), y_val)

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
        wandb.finish()


if __name__ == "__main__":
    main()
