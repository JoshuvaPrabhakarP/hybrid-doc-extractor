"""
Training utilities — seeding, metrics, checkpointing.
"""

import os
import random
import torch
import numpy as np
from pathlib import Path

from .dataset import ID2LABEL


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Entity-level F1
# ---------------------------------------------------------------------------

def extract_entities(tag_ids: list[int]) -> set[tuple[str, int, int]]:
    """
    Extract entity spans from a BIO tag ID sequence with first-subword pooling.

    With first-subword pooling, -100 appears at continuation subword positions
    (not just padding). These must be SKIPPED without closing the current entity,
    since they sit between the first subwords of consecutive words.

    Example (first-subword pooling):
        tags = [-100, B-NAME, -100, -100, I-NAME, -100, O, B-DATE, -100, -100]
                [CLS] McDonald  "      s   Restaurant      ...  2024   -     03
        → {("NAME", 1, 4), ("DATE", 7, 7)}
    """
    entities = set()
    current_type = None
    current_start = None
    last_real_pos = None  # track last non-ignored position

    for i, tid in enumerate(tag_ids):
        if tid == -100:
            # Skip continuation subwords and special tokens
            # Do NOT close the current entity
            continue

        label = ID2LABEL.get(tid, "O")

        if label.startswith("B-"):
            # Close previous entity if any
            if current_type is not None:
                entities.add((current_type, current_start, last_real_pos))
            # Start new entity
            current_type = label[2:]
            current_start = i
            last_real_pos = i

        elif label.startswith("I-"):
            itype = label[2:]
            if current_type == itype:
                # Continue current entity
                last_real_pos = i
            else:
                # I- without matching B- → close previous, treat as B-
                if current_type is not None:
                    entities.add((current_type, current_start, last_real_pos))
                current_type = itype
                current_start = i
                last_real_pos = i

        else:
            # O tag — close any open entity
            if current_type is not None:
                entities.add((current_type, current_start, last_real_pos))
                current_type = None
            last_real_pos = i

    # Close final entity if sequence ends mid-entity
    if current_type is not None and last_real_pos is not None:
        entities.add((current_type, current_start, last_real_pos))

    return entities


def compute_entity_f1(
    all_preds: list[list[int]],
    all_labels: list[list[int]],
) -> dict[str, float]:
    """
    Compute entity-level precision, recall, and F1.

    An entity is correct only if both the type and exact span match.
    Also computes per-field F1 for DATE, AMOUNT, NAME.

    Args:
        all_preds:  list of predicted tag ID sequences (one per document)
        all_labels: list of gold tag ID sequences (one per document)

    Returns:
        dict with overall and per-field metrics:
            entity_f1, entity_precision, entity_recall,
            DATE_f1, AMOUNT_f1, NAME_f1
    """
    total_pred = 0
    total_gold = 0
    total_correct = 0

    # Per-field counters
    field_pred = {"DATE": 0, "AMOUNT": 0, "NAME": 0}
    field_gold = {"DATE": 0, "AMOUNT": 0, "NAME": 0}
    field_correct = {"DATE": 0, "AMOUNT": 0, "NAME": 0}

    for preds, golds in zip(all_preds, all_labels):
        pred_entities = extract_entities(preds)
        gold_entities = extract_entities(golds)

        total_pred += len(pred_entities)
        total_gold += len(gold_entities)
        total_correct += len(pred_entities & gold_entities)

        for field in field_pred:
            p_field = {e for e in pred_entities if e[0] == field}
            g_field = {e for e in gold_entities if e[0] == field}
            field_pred[field] += len(p_field)
            field_gold[field] += len(g_field)
            field_correct[field] += len(p_field & g_field)

    # Overall
    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result = {
        "entity_f1": f1,
        "entity_precision": precision,
        "entity_recall": recall,
    }

    # Per-field
    for field in field_pred:
        p = field_correct[field] / field_pred[field] if field_pred[field] > 0 else 0.0
        r = field_correct[field] / field_gold[field] if field_gold[field] > 0 else 0.0
        result[f"{field}_f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return result


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_f1: float,
    path: str | Path,
):
    """Save model checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_f1": val_f1,
    }, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Load model checkpoint. Returns checkpoint metadata."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return {"epoch": ckpt["epoch"], "val_f1": ckpt["val_f1"]}
