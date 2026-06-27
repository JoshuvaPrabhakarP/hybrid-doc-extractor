"""
Arm B — Pure Mamba SSM extractor.

6 Mamba blocks, each wrapped with pre-LayerNorm and a residual connection.
Uses the mamba_ssm library (2.2.4) which is already patched for sm_120.
"""

import torch
import torch.nn as nn

from mamba_ssm import Mamba

from .base import BaseExtractor


class MambaBlock(nn.Module):
    """Single Mamba layer with pre-norm and residual connection."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) → (batch, seq_len, d_model)"""
        return x + self.mamba(self.norm(x))


class MambaExtractor(BaseExtractor):
    """
    Pure Mamba SSM for token classification (NER).

    Encoder: 6 × MambaBlock(d_model=256, d_state=16, d_conv=4, expand=2)

    Note: Mamba is inherently sequential (no attention mask needed).
    Padding is handled by the shared forward() via ignore_index=-100 in loss.
    """

    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = 256,
        num_labels: int = 7,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        super().__init__(vocab_size, d_model, num_labels, max_len, dropout, pad_token_id)

    def _build_encoder(self):
        self.encoder = nn.ModuleList([
            MambaBlock(self.d_model, self.d_state, self.d_conv, self.expand)
            for _ in range(self.n_layers)
        ])
        self.final_norm = nn.LayerNorm(self.d_model)

    def encode(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            attention_mask: ignored (Mamba doesn't use attention masks,
                           padding handled via loss masking)

        Returns:
            (batch, seq_len, d_model)
        """
        for block in self.encoder:
            x = block(x)
        return self.final_norm(x)
