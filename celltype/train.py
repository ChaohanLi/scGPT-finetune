"""
scGPT cell-type annotation baseline — training script.

Experimental setup mirrors the scFoundation baseline exactly:
  - same dataset  : 5w_allcelltype_anno_symbol.h5ad  (29 classes)
  - same split    : stratified 80/20, seed=42
  - same metrics  : macro-F1 and accuracy (reported per epoch)
  - same optimizer: AdamW + CosineAnnealingLR

Usage
-----
# Default (linear probing: last 2 transformer layers + cls_decoder trainable)
python train.py

# Also unfreeze token/value embeddings
python train.py --no_frozenmore

Outputs
-------
outputs/<run_name>/
    best_model.pt       – checkpoint with best val macro-F1
    metrics.csv         – per-epoch train/val loss and macro-F1/acc
    class_names.json    – class index ↔ name mapping
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score

_CELLTYPE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CELLTYPE_DIR)
from dataset import load_data    # noqa: E402
from model   import build_model  # noqa: E402


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="scGPT cell-type annotation baseline")
    p.add_argument("--model_dir",   type=str,
                   default="/lichaohan/scGPT/scGPT_human")
    p.add_argument("--h5ad",        type=str,
                   default="/lichaohan/readData/5w_allcelltype_anno_symbol.h5ad")
    p.add_argument("--n_class",     type=int,   default=29)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=12)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--train_size",  type=float, default=0.8)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--max_length",  type=int,   default=1200,
                   help="Max number of gene tokens per cell (incl. CLS)")
    p.add_argument("--no_frozenmore", action="store_true",
                   help="Also unfreeze token/value embeddings")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir",  type=str, default="outputs")
    p.add_argument("--run_name",    type=str, default=None)
    return p.parse_args()


# ── Training / evaluation helpers ─────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(training):
        for batch in loader:
            # Move inputs to device
            batch_dev = {
                "gene": batch["gene"].to(device, non_blocking=True),
                "expr": batch["expr"].to(device, non_blocking=True),
            }
            targets = batch["targets"].to(device, non_blocking=True)

            logits = model(batch_dev)
            loss   = criterion(logits, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * targets.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(targets.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss   = total_loss / len(all_labels)
    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy   = accuracy_score(all_labels, all_preds)
    return avg_loss, macro_f1, accuracy


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Output directory ──────────────────────────────────────────────────
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir  = os.path.join(_CELLTYPE_DIR, args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ── Data ──────────────────────────────────────────────────────────────
    (train_loader, val_loader,
     class_names, type2idx, _, vocab, pad_token_id) = load_data(
        h5ad_path    = args.h5ad,
        model_dir    = args.model_dir,
        train_size   = args.train_size,
        random_state = args.seed,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        max_length   = args.max_length,
    )

    n_class = len(class_names)
    print(f"Effective n_class: {n_class}  (arg was {args.n_class})")
    assert n_class == args.n_class, (
        f"Expected {args.n_class} classes but found {n_class} in data. "
        "Update --n_class."
    )

    with open(os.path.join(out_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(
        model_dir    = args.model_dir,
        n_class      = n_class,
        vocab        = vocab,
        pad_token_id = pad_token_id,
        frozenmore   = not args.no_frozenmore,
        device       = device,
    )

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_f1  = -1.0
    metrics_path = os.path.join(out_dir, "metrics.csv")

    with open(metrics_path, "w") as f:
        f.write("epoch,train_loss,train_macro_f1,train_acc,val_loss,val_macro_f1,val_acc\n")

    print("\n" + "=" * 60)
    print(f"Training for {args.epochs} epochs  lr={args.lr}  bs={args.batch_size}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_f1, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        torch.cuda.empty_cache()
        val_loss, val_f1, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss={tr_loss:.4f} f1={tr_f1:.4f} acc={tr_acc:.4f} | "
              f"val   loss={val_loss:.4f} f1={val_f1:.4f} acc={val_acc:.4f} | "
              f"{elapsed:.1f}s")

        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{tr_loss:.6f},{tr_f1:.6f},{tr_acc:.6f},"
                    f"{val_loss:.6f},{val_f1:.6f},{val_acc:.6f}\n")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt_path   = os.path.join(out_dir, "best_model.pt")
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_macro_f1": val_f1,
                "val_acc":      val_acc,
                "class_names":  class_names,
                "type2idx":     type2idx,
                "args":         vars(args),
            }, ckpt_path)
            print(f"  ↑ New best val macro-F1={val_f1:.4f} acc={val_acc:.4f}  "
                  f"saved to {ckpt_path}")

    print("\n" + "=" * 60)
    print(f"Training complete.  Best val macro-F1: {best_val_f1:.4f}")
    print(f"Metrics saved to:   {metrics_path}")


if __name__ == "__main__":
    main()
