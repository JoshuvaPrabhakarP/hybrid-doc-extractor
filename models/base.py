"""
Base model for document field extraction.

All three arms inherit from BaseExtractor, which provides shared
embedding, positional encoding, classification head, and optional CRF.

CRF operates at word level: first-subword hidden states are gathered
into a dense sequence, then the CRF models tag transitions between words.
"""

import math
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

try:
    from torchcrf import CRF
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False


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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def _gather_first_subwords(
    hidden: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Gather first-subword hidden states into a dense word-level tensor.

    Args:
        hidden: (batch, seq_len, d_model)
        mask:   (batch, seq_len) — True for first-subword positions
        labels: (batch, seq_len) — optional, gathered in parallel

    Returns:
        word_hidden: (batch, max_words, d_model)
        word_mask:   (batch, max_words) — True for real words, False for padding
        word_labels: (batch, max_words) — if labels provided, else None
    """
    batch_size, seq_len, d_model = hidden.shape
    device = hidden.device

    # Count words per sample
    word_counts = mask.sum(dim=1)  # (batch,)
    max_words = word_counts.max().item()

    word_hidden = torch.zeros(batch_size, max_words, d_model, device=device)
    word_mask_out = torch.zeros(batch_size, max_words, dtype=torch.bool, device=device)
    word_labels_out = None
    if labels is not None:
        word_labels_out = torch.full(
            (batch_size, max_words), fill_value=0, dtype=torch.long, device=device
        )

    for i in range(batch_size):
        indices = mask[i].nonzero(as_tuple=True)[0]  # positions where mask is True
        n_words = indices.shape[0]
        word_hidden[i, :n_words] = hidden[i, indices]
        word_mask_out[i, :n_words] = True
        if labels is not None:
            word_labels_out[i, :n_words] = labels[i, indices]

    return word_hidden, word_mask_out, word_labels_out


class BaseExtractor(nn.Module, ABC):
    """
    Abstract base for all three extraction arms.

    Shared: embedding, positional encoding, classifier head, optional CRF.
    Subclasses implement _build_encoder() and encode().
    """

    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = 256,
        num_labels: int = 7,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        use_crf: bool = False,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id
        self.use_crf = use_crf and CRF_AVAILABLE

        # Shared layers
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.classifier = nn.Linear(d_model, num_labels)

        # Optional CRF
        if self.use_crf:
            self.crf = CRF(num_labels, batch_first=True)

        # Class weights for CrossEntropyLoss (when CRF is off)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        # Subclass builds the encoder
        self._build_encoder()
        self._init_weights()

    @abstractmethod
    def _build_encoder(self):
        ...

    @abstractmethod
    def encode(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        first_subword_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass: embed → encode → classify (with optional CRF).
        """
        # Embed + positional encoding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Encode
        x = self.encode(x, attention_mask)

        if self.use_crf and first_subword_mask is not None:
            return self._forward_crf(x, labels, first_subword_mask)
        else:
            return self._forward_ce(x, labels)

    def _forward_ce(
        self, x: torch.Tensor, labels: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        """Standard cross-entropy forward (with class weights)."""
        logits = self.classifier(x)
        output = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(
                weight=self.class_weights, ignore_index=-100
            )
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            output["loss"] = loss

        return output

    def _forward_crf(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor | None,
        first_subword_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        CRF forward: gather word-level states, apply CRF for loss/decode.
        """
        # Gather first-subword hidden states → word level
        word_hidden, word_mask, word_labels = _gather_first_subwords(
            hidden, first_subword_mask, labels
        )

        # Emissions at word level
        emissions = self.classifier(word_hidden)  # (batch, max_words, num_labels)

        output = {"logits": emissions}

        if word_labels is not None:
            # CRF negative log-likelihood loss
            log_likelihood = self.crf(emissions, word_labels, mask=word_mask, reduction="mean")
            output["loss"] = -log_likelihood

        # Decode best tag sequences (Viterbi)
        decoded = self.crf.decode(emissions, mask=word_mask)  # list of lists

        # Pad decoded sequences to same length for batching
        max_words = emissions.shape[1]
        padded_preds = []
        for seq in decoded:
            padded_preds.append(seq + [0] * (max_words - len(seq)))
        output["word_preds"] = torch.tensor(padded_preds, device=hidden.device)
        output["word_mask"] = word_mask

        return output

    def _init_weights(self):
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
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
