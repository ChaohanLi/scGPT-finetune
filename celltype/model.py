"""
Cell-type classifier built on top of scGPT's pretrained TransformerModel.

Architecture
------------
  scGPT TransformerModel  (frozen, except last 2 transformer layers + cls_decoder)
    → CLS token embedding via cell_emb_style="cls"  (position 0, dim=512)
    → scGPT built-in ClsDecoder (nlayers=3):
        [Linear(512→512) + ReLU + LayerNorm(512)] × (nlayers-1=2)
        + Linear(512 → n_class)

Uses the official scGPT classification pipeline:
  model(src, values, src_key_padding_mask, CLS=True)["cls_output"]
which internally calls self.cls_decoder(cell_emb) — identical to Tutorial_Annotation.

The cls_decoder is ALWAYS trainable (it is not in the pretrained checkpoint
since that was trained with n_cls=1; load_pretrained will skip mismatched keys).
"""

import json
import os
import sys
from pathlib import Path

import torch
from torch import nn

_SCGPT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SCGPT_DIR)
from scgpt.model import TransformerModel        # noqa: E402
from scgpt.utils  import load_pretrained        # noqa: E402


class CellTypeClassifier(nn.Module):
    """
    scGPT encoder + built-in ClsDecoder for cell-type annotation.

    Parameters
    ----------
    model_dir    : directory containing vocab.json, args.json, best_model.pt
    n_class      : number of cell types (29 for our PBMC dataset)
    vocab        : GeneVocab object (must be loaded before calling build())
    pad_token_id : int, vocab ID of the <pad> token
    frozenmore   : if True (default), also freeze token/value embeddings;
                   set False to allow full fine-tuning of embeddings too
    """

    def __init__(
        self,
        model_dir: str,
        n_class: int,
        vocab,
        pad_token_id: int,
        frozenmore: bool = True,
    ):
        super().__init__()
        self.model_dir    = model_dir
        self.n_class      = n_class
        self.vocab        = vocab
        self.pad_token_id = pad_token_id
        self.frozenmore   = frozenmore
        self._built       = False

    def build(self, device: str = "cpu"):
        model_dir   = Path(self.model_dir)
        config_file = model_dir / "args.json"
        model_file  = model_dir / "best_model.pt"

        with open(config_file) as f:
            cfg = json.load(f)

        self.model_configs = cfg
        n_layers_cls = cfg.get("n_layers_cls", 3)

        # ── Build TransformerModel with CLS support ────────────────────────
        # Classification head (cls_decoder) source:
        #   scGPT/scgpt/model/model.py — class ClsDecoder (line 884)
        #   Instantiated inside TransformerModel.__init__() as:
        #     self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)
        #   Usage confirmed in: scGPT/tutorials/Tutorial_Annotation.ipynb
        #     TransformerModel(nlayers_cls=3, n_cls=num_types, ...)
        # n_cls=self.n_class so that cls_decoder is sized correctly from init.
        # cell_emb_style="cls" → _get_cell_emb_from_layer returns layer[:,0,:].
        # do_mvc=False / do_dab=False → no extra decoder heads.
        self.transformer = TransformerModel(
            ntoken               = len(self.vocab),
            d_model              = cfg["embsize"],
            nhead                = cfg["nheads"],
            d_hid                = cfg["d_hid"],
            nlayers              = cfg["nlayers"],
            nlayers_cls          = n_layers_cls,
            n_cls                = self.n_class,   # ← official: sized for actual task
            vocab                = self.vocab,
            dropout              = cfg["dropout"],
            pad_token            = cfg["pad_token"],
            pad_value            = cfg["pad_value"],
            do_mvc               = False,
            do_dab               = False,
            use_batch_labels     = False,
            domain_spec_batchnorm= False,
            input_emb_style      = "continuous",   # matches pretraining args.json
            cell_emb_style       = "cls",          # ← official annotation setting
            explicit_zero_prob   = False,
            use_fast_transformer = False,           # avoid flash-attn dependency
            pre_norm             = False,
        )

        # ── Load pretrained weights (skip cls_decoder: shape mismatch n_cls) ─
        # load_pretrained uses strict=False by default, so mismatched keys are
        # silently skipped — cls_decoder stays randomly initialized, which is
        # correct (it will be trained from scratch for our 29-class task).
        state = torch.load(model_file, map_location="cpu")
        load_pretrained(self.transformer, state, verbose=False)

        # ── Freeze strategy ───────────────────────────────────────────────
        # Step 1: freeze the entire model
        for p in self.transformer.parameters():
            p.requires_grad = False

        # Step 2: unfreeze last 2 transformer encoder layers
        n_layers = cfg["nlayers"]
        for layer_idx in [n_layers - 2, n_layers - 1]:
            for p in self.transformer.transformer_encoder.layers[layer_idx].parameters():
                p.requires_grad = True

        # Step 3: always unfreeze cls_decoder (randomly initialized for n_cls=29)
        for p in self.transformer.cls_decoder.parameters():
            p.requires_grad = True

        # Step 4: optionally unfreeze token/value embeddings
        if not self.frozenmore:
            for p in self.transformer.encoder.parameters():
                p.requires_grad = True
            for p in self.transformer.value_encoder.parameters():
                p.requires_grad = True

        total     = sum(p.numel() for p in self.transformer.parameters())
        trainable = sum(p.numel() for p in self.transformer.parameters()
                        if p.requires_grad)
        print(f"TransformerModel: {total:,} total | {trainable:,} trainable "
              f"({100*trainable/total:.1f}%)")

        self._built = True

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : dict from DataCollator (after _collate_with_labels), keys:
                'gene'  (B, seq_len) – vocab token IDs
                'expr'  (B, seq_len) – binned expression values

        Returns
        -------
        logits : (B, n_class)
        """
        assert self._built, "Call model.build() before forward()"

        gene_ids = batch["gene"]   # (B, seq_len)
        expr     = batch["expr"]   # (B, seq_len)

        src_key_padding_mask = gene_ids.eq(self.pad_token_id)  # (B, seq_len)

        # Official scGPT forward with CLS=True:
        #   → _encode(src, values, mask) → transformer_output
        #   → cell_emb = _get_cell_emb_from_layer(transformer_output)
        #                = transformer_output[:, 0, :]   (cell_emb_style="cls")
        #   → cls_output = cls_decoder(cell_emb)
        output_dict = self.transformer(
            gene_ids,
            expr,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
            CLS=True,
            CCE=False,
            MVC=False,
            ECS=False,
        )
        return output_dict["cls_output"]   # (B, n_class)


def build_model(
    model_dir: str,
    n_class: int,
    vocab,
    pad_token_id: int,
    frozenmore: bool = True,
    device: str = "cuda",
) -> CellTypeClassifier:
    model = CellTypeClassifier(
        model_dir=model_dir,
        n_class=n_class,
        vocab=vocab,
        pad_token_id=pad_token_id,
        frozenmore=frozenmore,
    )
    model.build(device=device)
    model.to(device)
    return model
