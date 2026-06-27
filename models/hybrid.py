"""
Arm C — Hybrid Mamba-Transformer extractor (5:1 ratio).

Bottom 5 layers: Mamba SSM blocks (local sequential processing)
Top 1 layer:     Transformer attention (global cross-referencing)

Rationale: Mamba layers efficiently build up token representations
by reading left-to-right through receipt text, then the final
attention layer lets the model cross-reference all positions at once
(e.g., linking "TOTAL" to its value elsewhere on the receipt).
"""

import torch
import torch.nn as nn

from mamba_ssm import Mamba

from .base import BaseExtractor
from .mamba_model import MambaBlock


class HybridExtractor(BaseExtractor):
    """
    Hybrid Mamba + Transformer for token classification (NER).

    Encoder: 5 × MambaBlock + 1 × TransformerEncoderLayer
    """

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
    ):
        self.mamba_layers = mamba_layers
        self.attn_layers = attn_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        super().__init__(vocab_size, d_model, num_labels, max_len, dropout, pad_token_id)

    def _build_encoder(self):
        # Bottom: Mamba blocks (local sequential)
        self.mamba_encoder = nn.ModuleList([
            MambaBlock(self.d_model, self.d_state, self.d_conv, self.expand)
            for _ in range(self.mamba_layers)
        ])

        # Top: Transformer attention layer(s) (global)
        self.attn_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.n_heads,
                dim_feedforward=self.d_ff,
                dropout=self.pos_encoder.dropout.p,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(self.attn_layers)
        ])

        self.final_norm = nn.LayerNorm(self.d_model)

    def encode(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            attention_mask: (batch, seq_len) — 1=real, 0=pad

        Returns:
            (batch, seq_len, d_model)
        """
        # Mamba layers (ignore attention mask — sequential by nature)
        for block in self.mamba_encoder:
            x = block(x)

        # Attention layer(s) (use padding mask)
        padding_mask = None
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)

        for layer in self.attn_encoder:
            x = layer(x, src_key_padding_mask=padding_mask)

        return self.final_norm(x)
