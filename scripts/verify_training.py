#!/usr/bin/env python3
"""
Phase 5 pre-flight — verify training pipeline components before full training.

Tests:
    1. NERDataset loads train.jsonl correctly
    2. collate_fn produces correct tensor shapes
    3. Model forward + backward on a real batch
    4. Entity F1 computation works
    5. Checkpoint save/load round-trips

Usage:
    cd ~/hybrid-doc-extractor
    python scripts/verify_training.py
"""

import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from models import build_model
from training.dataset import NERDataset, collate_fn, NUM_LABELS
from training.utils import set_seed, compute_entity_f1, save_checkpoint, load_checkpoint

PASS = "\033[92m✅\033[0m"
FAIL = "\033[91m❌\033[0m"
results = []

def check(name, ok, msg=""):
    results.append((name, ok))
    print(f"  {PASS if ok else FAIL}  {name}" + (f" — {msg}" if msg else ""))
    return ok


def main():
    print(f"\n{'='*60}")
    print(f"  Phase 5 Pre-flight Check")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Load tokenizer
    tok_path = PROJECT_ROOT / "tokenizer_training" / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))
    check("Tokenizer loaded", True)

    # 1. Dataset loading
    train_path = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
    val_path = PROJECT_ROOT / "data" / "processed" / "val.jsonl"

    train_ds = NERDataset(train_path, tokenizer, max_seq_len=512)
    val_ds = NERDataset(val_path, tokenizer, max_seq_len=512)

    check("Train dataset loaded", len(train_ds) == 2947, f"{len(train_ds)} samples")
    check("Val dataset loaded", len(val_ds) == 368, f"{len(val_ds)} samples")

    # Check a single sample
    sample = train_ds[0]
    check("Sample has input_ids", "input_ids" in sample)
    check("Sample has labels", "labels" in sample)
    check("Lengths match", len(sample["input_ids"]) == len(sample["labels"]),
          f"ids={len(sample['input_ids'])}, labels={len(sample['labels'])}")
    check("Starts with [CLS]", sample["input_ids"][0] == tokenizer.token_to_id("[CLS]"))
    check("Ends with [SEP]", sample["input_ids"][-1] == tokenizer.token_to_id("[SEP]"))
    check("[CLS] label is -100", sample["labels"][0] == -100)
    check("[SEP] label is -100", sample["labels"][-1] == -100)

    # 2. Collation
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))

    check("Batch has input_ids tensor", batch["input_ids"].shape[0] == 8)
    check("Batch has attention_mask", batch["attention_mask"].shape == batch["input_ids"].shape)
    check("Batch has labels", batch["labels"].shape == batch["input_ids"].shape)
    print(f"    Batch shape: {tuple(batch['input_ids'].shape)}")

    # 3. Forward + backward with real data
    set_seed(42)
    model = build_model("transformer", vocab_size=8000, d_model=256,
                        num_labels=NUM_LABELS, n_heads=8, d_ff=1024).to(device)

    batch_device = {k: v.to(device) for k, v in batch.items()}
    output = model(**batch_device)

    check("Forward on real batch", output["logits"].shape[0] == 8)
    check("Loss computed", output["loss"].item() > 0, f"loss={output['loss'].item():.4f}")

    output["loss"].backward()
    grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0
                  for p in model.parameters() if p.requires_grad)
    check("Backward pass", grad_ok)

    # 4. Entity F1 computation
    # Simulate some predictions
    with torch.no_grad():
        preds = output["logits"].argmax(dim=-1).cpu().tolist()
        labels = batch["labels"].cpu().tolist()

    metrics = compute_entity_f1(preds, labels)
    check("Entity F1 computes", "entity_f1" in metrics, f"F1={metrics['entity_f1']:.4f}")
    check("Per-field F1 present", all(f"{f}_f1" in metrics for f in ["DATE", "AMOUNT", "NAME"]))

    # 5. Checkpoint save/load
    ckpt_path = PROJECT_ROOT / "checkpoints" / "test_ckpt.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    save_checkpoint(model, optimizer, epoch=1, val_f1=0.5, path=ckpt_path)
    check("Checkpoint saved", ckpt_path.exists())

    model2 = build_model("transformer", vocab_size=8000, d_model=256,
                         num_labels=NUM_LABELS, n_heads=8, d_ff=1024)
    meta = load_checkpoint(ckpt_path, model2)
    check("Checkpoint loaded", meta["val_f1"] == 0.5, f"epoch={meta['epoch']}, f1={meta['val_f1']}")

    # Clean up test checkpoint
    ckpt_path.unlink()
    ckpt_path.parent.rmdir()

    # Summary
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"  Result: {passed}/{total} checks passed")
    if passed == total:
        print(f"  🚀 Training pipeline verified! Ready to train.")
        print(f"\n  To train all arms:")
        print(f"    python -m training.train")
        print(f"\n  To train a single arm:")
        print(f"    python -m training.train --arm transformer --seed 42")
    else:
        print(f"  ⚠️  Fix failures before training.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
