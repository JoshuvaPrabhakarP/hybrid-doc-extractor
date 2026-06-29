"""
Arm A — Pure Transformer extractor.
6 standard Transformer encoder layers (multi-head self-attention + FFN).
"""

import torch
import torch.nn as nn
from .base import BaseExtractor


class TransformerExtractor(BaseExtractor):

    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = 256,
        num_labels: int = 7,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        use_crf: bool = False,
        class_weights: torch.Tensor | None = None,
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self._dropout = dropout
        super().__init__(vocab_size, d_model, num_labels, max_len, dropout, pad_token_id, use_crf, class_weights)

    def _build_encoder(self):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self._dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
            norm=nn.LayerNorm(self.d_model),
        )

    def encode(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        padding_mask = None
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
        return self.encoder(x, src_key_padding_mask=padding_mask)
