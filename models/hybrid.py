"""
Arm C — Hybrid Mamba-Transformer extractor (5:1 ratio).
Bottom 5 layers: Mamba SSM, Top 1 layer: Transformer attention.
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba
from .base import BaseExtractor
from .mamba_model import MambaBlock


class HybridExtractor(BaseExtractor):

    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = 256,
        num_labels: int = 7,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        mamba_layers: int = 5,
        attn_layers: int = 1,
        n_heads: int = 8,
        d_ff: int = 1024,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_crf: bool = False,
        class_weights: torch.Tensor | None = None,
    ):
        self.mamba_layers = mamba_layers
        self.attn_layers = attn_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self._dropout = dropout
        super().__init__(vocab_size, d_model, num_labels, max_len, dropout, pad_token_id, use_crf, class_weights)

    def _build_encoder(self):
        self.mamba_encoder = nn.ModuleList([
            MambaBlock(self.d_model, self.d_state, self.d_conv, self.expand)
            for _ in range(self.mamba_layers)
        ])
        self.attn_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=self.n_heads,
                dim_feedforward=self.d_ff, dropout=self._dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(self.attn_layers)
        ])
        self.final_norm = nn.LayerNorm(self.d_model)

    def encode(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.mamba_encoder:
            x = block(x)
        padding_mask = None
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
        for layer in self.attn_encoder:
            x = layer(x, src_key_padding_mask=padding_mask)
        return self.final_norm(x)
