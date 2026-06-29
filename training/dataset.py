"""
Dataset and data loading for NER training.

Loads JSONL records (tokens + BIO tags), applies BPE tokenization
word-by-word with first-subword pooling, and returns padded tensor batches.

Includes class weight computation for handling tag imbalance (O dominates).
"""

import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset
from tokenizers import Tokenizer


# Label vocabulary — must match configs/experiment.yaml
LABEL2ID = {
    "O": 0,
    "B-DATE": 1, "I-DATE": 2,
    "B-AMOUNT": 3, "I-AMOUNT": 4,
    "B-NAME": 5, "I-NAME": 6,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


def compute_class_weights(jsonl_path: str | Path) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from training data.

    Scans all tags in the training JSONL, counts per-class frequency,
    and returns weights inversely proportional to frequency.
    This ensures rare entity tags (B-DATE, B-AMOUNT, etc.) get
    higher weight in the loss, countering the O-tag dominance.

    Returns:
        torch.Tensor of shape (NUM_LABELS,) — one weight per tag class
    """
    counts = Counter()
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            for tag in rec["tags"]:
                counts[LABEL2ID[tag]] += 1

    total = sum(counts.values())
    weights = torch.zeros(NUM_LABELS)
    for label_id in range(NUM_LABELS):
        freq = counts.get(label_id, 1)
        # Inverse frequency, normalized so weights sum to NUM_LABELS
        weights[label_id] = total / (NUM_LABELS * freq)

    return weights


def align_tags_to_subwords(
    tokenizer: Tokenizer,
    words: list[str],
    tags: list[str],
) -> tuple[list[int], list[int], list[bool]]:
    """
    Tokenize words with BPE and align BIO tags using first-subword pooling.

    Only the FIRST subword of each word gets the real tag.
    All subsequent subword pieces get -100 (ignored in loss & eval).

    Also returns a first_subword_mask indicating which positions
    are first subwords (needed for CRF word-level gathering).

    Returns:
        (subword_ids, tag_ids, first_subword_mask)
    """
    sub_ids = []
    tag_ids = []
    first_subword_mask = []

    for word, tag in zip(words, tags):
        encoded = tokenizer.encode(word, add_special_tokens=False)
        piece_ids = encoded.ids

        if not piece_ids:
            piece_ids = [tokenizer.token_to_id("[UNK]")]

        tag_id = LABEL2ID[tag]

        for i, pid in enumerate(piece_ids):
            sub_ids.append(pid)
            if i == 0:
                tag_ids.append(tag_id)
                first_subword_mask.append(True)
            else:
                tag_ids.append(-100)
                first_subword_mask.append(False)

    return sub_ids, tag_ids, first_subword_mask


class NERDataset(Dataset):
    """
    Dataset for document field extraction (NER).

    Each item is a dict with:
        - input_ids:          list[int] — [CLS] + subword IDs + [SEP]
        - labels:             list[int] — [-100] + tag IDs + [-100]
        - first_subword_mask: list[bool] — [False] + mask + [False]
          (False for [CLS], [SEP], and continuation subwords)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: Tokenizer,
        max_seq_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.pad_id = tokenizer.token_to_id("[PAD]")

        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                self.samples.append((rec["tokens"], rec["tags"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        words, tags = self.samples[idx]

        sub_ids, tag_ids, fsw_mask = align_tags_to_subwords(
            self.tokenizer, words, tags
        )

        # Truncate to max_seq_len - 2 (room for [CLS] and [SEP])
        max_content = self.max_seq_len - 2
        sub_ids = sub_ids[:max_content]
        tag_ids = tag_ids[:max_content]
        fsw_mask = fsw_mask[:max_content]

        # Wrap with special tokens
        input_ids = [self.cls_id] + sub_ids + [self.sep_id]
        labels = [-100] + tag_ids + [-100]
        first_subword_mask = [False] + fsw_mask + [False]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "first_subword_mask": first_subword_mask,
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Pad a batch of variable-length sequences to the longest in the batch.
    """
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []
    fsw_mask_batch = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        input_ids_batch.append(item["input_ids"] + [0] * pad_len)
        attention_mask_batch.append([1] * seq_len + [0] * pad_len)
        labels_batch.append(item["labels"] + [-100] * pad_len)
        fsw_mask_batch.append(item["first_subword_mask"] + [False] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
        "first_subword_mask": torch.tensor(fsw_mask_batch, dtype=torch.bool),
    }
