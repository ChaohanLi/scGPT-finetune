# scGPT Probe

Frozen-encoder linear probe for **scGPT** (51M params, 512-dim CLS token).  
Extracts val-set CLS embeddings, then evaluates a **LinearSVC** via 5-fold cross-validation.

---

## Supported Datasets

| `--dataset_id`   | h5ad path                                                 | Cells | Classes | Notes               |
|------------------|-----------------------------------------------------------|-------|---------|---------------------|
| `5w_symbol`      | `readData/5w_allcelltype_anno_symbol.h5ad`               | 50 k  | 29      | log1p normalized    |
| `5w_GSE196830`   | `readData/5w_PBMC_GSE196830/5w_allcelltype.h5ad`         | 50 k  | 29      | raw counts → add `--preprocess` |
| `GSE96583`       | `readData/GSE96583_PBMC/GSE96583_merged_dedup.h5ad`      | 41 k  | 8       | raw counts, gene-symbol var_names → add `--preprocess` |
| `10w_GSE196830`  | `readData/10w_PBMC_GSE196830/10w_allcelltype.h5ad`       | 100 k | 29      | raw counts → add `--preprocess` |
| `20w_GSE196830`  | `readData/20w_PBMC_GSE196830/20w_allcelltype.h5ad`       | 200 k | 29      | raw counts → add `--preprocess` |

> **`--preprocess`**: applies `sc.pp.normalize_total(target_sum=1e4)` + `sc.pp.log1p()` before embedding extraction. Required for raw-count datasets.

Gene-to-vocab mapping: if the h5ad has a `gene_symbol` column in `var`, it is used; otherwise `var_names` are treated as gene symbols directly (covers GSE96583).

---

## Quick Start

### Interactive (foreground)

```bash
bash run_probe.sh
```

Edit `run_probe.sh` to select the dataset (uncomment the block you want).

### Background (nohup)

```bash
nohup bash run_probe.sh > run_probe.log 2>&1 &
tail -f run_probe.log
```

### Manual CLI

```bash
# Original log1p dataset
python probe.py \
    --h5ad /lichaohan/readData/5w_allcelltype_anno_symbol.h5ad \
    --dataset_id 5w_symbol \
    --n_class 29 \
    --run_name my_run \
    --wandb_project scgpt-probe

# Raw count dataset
python probe.py \
    --h5ad /lichaohan/readData/5w_PBMC_GSE196830/5w_allcelltype.h5ad \
    --dataset_id 5w_GSE196830 \
    --n_class 29 \
    --preprocess \
    --run_name my_run \
    --wandb_project scgpt-probe
```

---

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--h5ad` | `5w_allcelltype_anno_symbol.h5ad` | Input h5ad path |
| `--dataset_id` | `5w_symbol` | Tag appended to wandb run name |
| `--n_class` | `29` | Expected number of cell types |
| `--preprocess` | off | Normalize raw counts before embedding |
| `--model_dir` | `scGPT_human` | Path to pre-trained scGPT checkpoint directory |
| `--run_name` | auto timestamp | wandb / output folder name prefix |
| `--wandb_project` | `scgpt-probe` | wandb project name |
| `--n_jobs` | `16` | CPU cores for parallel fold evaluation |
| `--pca_dim` | `100` | PCA before SVC (applied to 512-dim CLS embeddings) |
| `--no_wandb` | off | Disable wandb logging |
| `--save_embeddings` | off | Save embeddings as `.npz` in output dir |

---

## Protocol

```
val embeddings (512-dim CLS token, no cls_decoder)
  └─ 5-fold StratifiedKFold (shuffle, seed=42)
       └─ StandardScaler → PCA(100) → LinearSVC(dual=False, max_iter=2000)
            folds are run in parallel (--n_jobs controls core count)
            if >5000 train samples per fold → subsampled to 5000
```

---

## Output

Results are saved to `outputs_probe/<run_name>/`:
- `cv_summary.json` — per-fold and aggregated metrics

Metrics logged to wandb: `cv_macro_f1_mean`, `cv_macro_f1_std`, `cv_acc_mean`.
