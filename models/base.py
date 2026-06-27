"""
Base model for document field extraction.

All three arms (Transformer, Mamba, Hybrid) inherit from BaseExtractor,
which provides the shared embedding, positional encoding, and
classification head. Subclasses only implement the encoder stack.
"""

import math
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape: (1, max_len, d_model) — broadcasts over batch
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class BaseExtractor(nn.Module, ABC):
    """
    Abstract base for all three extraction arms.

    Shared components:
        - Token embedding (vocab_size → d_model)
        - Sinusoidal positional encoding
        - Classification head (d_model → num_labels)

    Subclasses implement `_build_encoder()` and `encode()`.
    """

    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = 256,
        num_labels: int = 7,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id

        # --- Shared layers ---
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.classifier = nn.Linear(d_model, num_labels)

        # Subclass builds the encoder stack
        self._build_encoder()

        # Init weights
        self._init_weights()

    @abstractmethod
    def _build_encoder(self):
        """Subclass creates self.encoder (or equivalent layers) here."""
        ...

    @abstractmethod
    def encode(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Run the encoder stack.

        Args:
            x: (batch, seq_len, d_model) — embedded + position-encoded input
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            (batch, seq_len, d_model) — encoded representations
        """
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass: embed → encode → classify.

        Args:
            input_ids:      (batch, seq_len) — subword token IDs
            attention_mask:  (batch, seq_len) — 1 for real, 0 for pad
            labels:          (batch, seq_len) — BIO tag IDs (optional, for loss)

        Returns:
            dict with 'logits' (batch, seq_len, num_labels)
            and optionally 'loss' if labels provided.
        """
        # Embed + positional encoding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Encode (subclass-specific)
        x = self.encode(x, attention_mask)

        # Classify each token
        logits = self.classifier(x)  # (batch, seq_len, num_labels)

        output = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            # Reshape: (batch*seq_len, num_labels) vs (batch*seq_len,)
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            output["loss"] = loss

        return output

    def _init_weights(self):
        """Xavier uniform for embeddings and linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
