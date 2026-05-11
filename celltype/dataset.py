"""
Data loading and split for scGPT cell-type annotation baseline.

Input : /lichaohan/readData/5w_allcelltype_anno_symbol.h5ad
          - ~50,000 cells × 30,582 genes (var.index = Ensembl ID;
            gene symbols in var["gene_symbol"])
          - X: log1p normalized (float32)
          - obs['cell_type']: 29 classes

Pipeline:
  1. Map gene symbols to scGPT vocab indices; filter unmapped genes
  2. For each cell: collect nonzero-expression gene tokens, prepend <cls>
  3. Same stratified 80/20 split as the scFoundation baseline
  4. Return DataLoaders (using scGPT DataCollator for binning/padding)
     + class metadata + class weights (for reference; not used in default CE)
     + vocab + pad_token_id (needed by model builder)
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import issparse

# ── scGPT imports ──────────────────────────────────────────────────────────
_SCGPT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SCGPT_DIR)

# scgpt/__init__.py imports scbank → databank → 'datasets' (not installed).
# Inject a stub before the package loads to prevent ModuleNotFoundError.
import types as _types
for _m in ("datasets", "scgpt.scbank", "scgpt.scbank.databank"):
    sys.modules.setdefault(_m, _types.ModuleType(_m))

from scgpt.tokenizer import GeneVocab        # noqa: E402
from scgpt.data_collator import DataCollator  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────
H5AD_PATH      = "/lichaohan/readData/5w_allcelltype_anno_symbol.h5ad"
MODEL_DIR      = "/lichaohan/scGPT/scGPT_human"
PAD_TOKEN      = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
PAD_VALUE      = -2   # from args.json
MAX_LENGTH     = 1200  # from args.json


# ── Exact replica of scFoundation baseline's stratified split ──────────────
def _stratified_train_val_split(
    barcodes: np.ndarray,
    labels: np.ndarray,
    train_size: float = 0.8,
    random_state: int = 42,
):
    barcodes = np.asarray(barcodes)
    labels   = np.asarray(labels)

    rng = np.random.default_rng(int(random_state))
    train_parts, val_parts = [], []

    for lab in np.unique(labels):
        idx  = np.flatnonzero(labels == lab)
        perm = rng.permutation(idx)
        n_tr = int(np.floor(len(idx) * train_size))
        if len(idx) >= 2:
            n_tr = min(max(n_tr, 1), len(idx) - 1)
        else:
            n_tr = len(idx)
        train_parts.append(perm[:n_tr])
        val_parts.append(perm[n_tr:])

    train_idx = np.concatenate(train_parts)
    val_idx   = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return barcodes[train_idx], barcodes[val_idx]


# ── PyTorch Dataset ────────────────────────────────────────────────────────
class CellTypeDataset(Dataset):
    """
    Each item is a dict:
        'id'          : int (row index)
        'genes'       : LongTensor  – vocab IDs, CLS token prepended
        'expressions' : FloatTensor – raw log1p values, PAD_VALUE for CLS
        'targets'     : LongTensor  – class label
    The DataCollator will bin the expression values and pad/sample to
    MAX_LENGTH tokens.
    """

    def __init__(
        self,
        X: np.ndarray,           # (n_cells, n_filtered_genes)
        gene_ids: np.ndarray,    # vocab IDs for each column in X
        labels: np.ndarray,      # integer class labels
        cls_token_id: int,
    ):
        self.X           = X
        self.gene_ids    = gene_ids
        self.labels      = torch.tensor(labels, dtype=torch.long)
        self.cls_token_id = cls_token_id

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.X[idx]
        nonzero_idx = np.nonzero(row)[0]
        values      = row[nonzero_idx].astype(np.float32)
        genes       = self.gene_ids[nonzero_idx]

        # Prepend <cls> token (expression = PAD_VALUE, left unchanged by DataCollator)
        genes  = np.insert(genes,  0, self.cls_token_id)
        values = np.insert(values, 0, float(PAD_VALUE))

        return {
            "id":          idx,
            "genes":       torch.from_numpy(genes).long(),
            "expressions": torch.from_numpy(values).float(),
            "targets":     self.labels[idx],
        }


def _collate_with_labels(collator: DataCollator):
    """Wrap DataCollator so it also passes through the 'targets' label."""
    def collate_fn(examples):
        targets = torch.stack([ex["targets"] for ex in examples])
        batch   = collator(examples)  # handles 'id', 'genes', 'expressions'
        batch["targets"] = targets
        return batch
    return collate_fn


# ── Main entry point ────────────────────────────────────────────────────────
def load_data(
    h5ad_path:       str   = H5AD_PATH,
    model_dir:       str   = MODEL_DIR,
    train_size:      float = 0.8,
    random_state:    int   = 42,
    batch_size:      int   = 12,
    num_workers:     int   = 4,
    max_length:      int   = MAX_LENGTH,
    preprocess:      bool  = False,
    symbol_map:      str   = None,
):
    import scanpy as sc

    print(f"Loading h5ad: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)

    # -- Optional preprocessing (normalize raw counts to log1p) -----------
    if preprocess:
        print("  Preprocessing: normalize_total + log1p (raw counts -> log1p)")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # ── Load vocabulary ───────────────────────────────────────────────────
    vocab_file = Path(model_dir) / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab[PAD_TOKEN])
    cls_token_id = vocab["<cls>"]
    pad_token_id = vocab[PAD_TOKEN]

    # ── Map genes to vocab ────────────────────────────────────────────────
    # Resolve gene names: prefer explicit column, then symbol_map, then var_names
    if "gene_symbol" in adata.var.columns:
        gene_names = list(adata.var["gene_symbol"].values)
    elif symbol_map:
        import pandas as _pd
        sym_df = _pd.read_csv(symbol_map, sep="\t", index_col=0)
        gene_names = [sym_df.loc[g, "gene_symbol"] if g in sym_df.index
                      else g for g in adata.var_names]
        n_mapped = sum(1 for g in adata.var_names if g in sym_df.index)
        print(f"  symbol_map: {n_mapped}/{len(adata.var_names)} Ensembl IDs resolved to symbols")
    else:
        gene_names = list(adata.var_names)
    id_in_vocab = np.array([
        vocab[g] if g in vocab else -1 for g in gene_names
    ])
    mask = id_in_vocab >= 0
    n_total   = len(gene_names)
    n_matched = int(mask.sum())
    print(f"Gene vocab match: {n_matched}/{n_total} genes")

    adata_filt = adata[:, mask]
    gene_ids   = id_in_vocab[mask]              # vocab IDs for filtered genes

    # ── Extract dense expression matrix ───────────────────────────────────
    X = adata_filt.X
    if issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    # ── Build label encoding ──────────────────────────────────────────────
    cell_types = adata_filt.obs["cell_type"].values
    classes    = sorted(set(cell_types))
    type2idx   = {c: i for i, c in enumerate(classes)}
    labels     = np.array([type2idx[c] for c in cell_types], dtype=np.int64)

    # ── Stratified split (identical to scFoundation baseline) ────────────
    barcodes = np.array(adata_filt.obs_names)
    train_bc, val_bc = _stratified_train_val_split(
        barcodes, labels, train_size=train_size, random_state=random_state
    )

    bc2idx     = {bc: i for i, bc in enumerate(barcodes)}
    train_idx  = np.array([bc2idx[bc] for bc in train_bc])
    val_idx    = np.array([bc2idx[bc] for bc in val_bc])

    print(f"Train cells: {len(train_idx)} | Val cells: {len(val_idx)}")

    # ── Class weights (for reference; not used in default CE loss) ────────
    train_labels = labels[train_idx]
    counts       = np.bincount(train_labels, minlength=len(classes)).astype(float)
    class_weights = (counts.sum() / (len(classes) * (counts + 1e-8))).astype(np.float32)

    # ── DataCollator (binning + padding, no MLM during training/val) ──────
    collator = DataCollator(
        do_padding=True,
        pad_token_id=pad_token_id,
        pad_value=PAD_VALUE,
        do_mlm=False,
        do_binning=True,
        max_length=max_length,
        sampling=True,
        keep_first_n_tokens=1,
    )
    collate_fn = _collate_with_labels(collator)

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = CellTypeDataset(X[train_idx], gene_ids, labels[train_idx], cls_token_id)
    val_ds   = CellTypeDataset(X[val_idx],   gene_ids, labels[val_idx],   cls_token_id)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False,
    )

    return train_loader, val_loader, classes, type2idx, class_weights, vocab, pad_token_id
