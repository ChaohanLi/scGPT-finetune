"""
Cell-type classifier built on top of scGPT's pretrained TransformerModel.

Same architecture as scGPT/celltype/model.py, with an additional encode()
method that returns the raw CLS-token embedding (B, 512) without passing
through cls_decoder — used by the LinearSVC probe.
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
    frozenmore   : if True (default), also freeze token/value embeddings
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
        # cell_emb_style="cls" → output["cell_emb"] = layer_output[:, 0, :]
        self.transformer = TransformerModel(
            ntoken               = len(self.vocab),
            d_model              = cfg["embsize"],
            nhead                = cfg["nheads"],
            d_hid                = cfg["d_hid"],
            nlayers              = cfg["nlayers"],
            nlayers_cls          = n_layers_cls,
            n_cls                = self.n_class,
            vocab                = self.vocab,
            dropout              = cfg["dropout"],
            pad_token            = cfg["pad_token"],
            pad_value            = cfg["pad_value"],
            do_mvc               = False,
            do_dab               = False,
            use_batch_labels     = False,
            domain_spec_batchnorm= False,
            input_emb_style      = "continuous",
            cell_emb_style       = "cls",
            explicit_zero_prob   = False,
            use_fast_transformer = False,
            pre_norm             = False,
        )

        state = torch.load(model_file, map_location="cpu")
        load_pretrained(self.transformer, state, verbose=False)

        # ── Freeze strategy ───────────────────────────────────────────────
        for p in self.transformer.parameters():
            p.requires_grad = False

        n_layers = cfg["nlayers"]
        for layer_idx in [n_layers - 2, n_layers - 1]:
            for p in self.transformer.transformer_encoder.layers[layer_idx].parameters():
                p.requires_grad = True

        for p in self.transformer.cls_decoder.parameters():
            p.requires_grad = True

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

    def encode(self, batch: dict) -> torch.Tensor:
        """
        Return the raw CLS-token embedding (B, d_model=512) without cls_decoder.

        Uses CLS=False so TransformerModel skips cls_decoder entirely.
        output["cell_emb"] = transformer_output[:, 0, :]  (cell_emb_style="cls")
        """
        assert self._built, "Call model.build() before encode()"

        gene_ids = batch["gene"]   # (B, seq_len)
        expr     = batch["expr"]   # (B, seq_len)

        src_key_padding_mask = gene_ids.eq(self.pad_token_id)

        output = self.transformer(
            gene_ids,
            expr,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
            CLS=False,   # skip cls_decoder — we want cell_emb directly
            CCE=False,
            MVC=False,
            ECS=False,
        )
        return output["cell_emb"]   # (B, 512)

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

        src_key_padding_mask = gene_ids.eq(self.pad_token_id)

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
