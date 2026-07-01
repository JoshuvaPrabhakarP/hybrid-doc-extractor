#!/usr/bin/env python3
"""
Phase 6, Step 1 — Test set evaluation.

Loads the best checkpoint for each arm x seed (18 total), runs on the
held-out test set (369 docs, never touched during training or validation
model selection), and reports entity F1, precision, recall, and per-field
breakdown.

This is the first unbiased estimate of each architecture's performance —
all Phase 5 checkpoints were selected using validation F1 only.

Usage:
    cd ~/hybrid-doc-extractor
    python -m evaluation.evaluate
    python -m evaluation.evaluate --arm transformer   # single arm, all seeds
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model
from training.dataset import NERDataset, collate_fn, NUM_LABELS, compute_class_weights
from training.utils import compute_entity_f1, load_checkpoint


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_model_kwargs(cfg, arm_name, use_crf, class_weights):
    """Same logic as training/train.py — kept in sync."""
    model_cfg = cfg["model"]
    kwargs = {
        "vocab_size": cfg["tokenizer"]["vocab_size"],
        "d_model": model_cfg["d_model"],
        "num_labels": NUM_LABELS,
        "max_len": model_cfg["max_len"],
        "dropout": model_cfg["dropout"],
        "pad_token_id": 0,
        "n_layers": model_cfg["n_layers"],
        "use_crf": use_crf,
        "class_weights": class_weights,
    }
    if arm_name == "transformer":
        kwargs["n_heads"] = model_cfg["n_heads"]
        kwargs["d_ff"] = model_cfg["d_ff"]
    elif arm_name == "mamba":
        kwargs["d_state"] = model_cfg["mamba"]["d_state"]
        kwargs["d_conv"] = model_cfg["mamba"]["d_conv"]
        kwargs["expand"] = model_cfg["mamba"]["expand"]
    elif arm_name.startswith("hybrid_"):
        del kwargs["n_layers"]
        kwargs["mamba_layers"] = model_cfg["arms"][arm_name]["mamba_layers"]
        kwargs["attn_layers"] = model_cfg["arms"][arm_name]["attn_layers"]
        kwargs["n_heads"] = model_cfg["n_heads"]
        kwargs["d_ff"] = model_cfg["d_ff"]
        kwargs["d_state"] = model_cfg["mamba"]["d_state"]
        kwargs["d_conv"] = model_cfg["mamba"]["d_conv"]
        kwargs["expand"] = model_cfg["mamba"]["expand"]
    return kwargs


@torch.no_grad()
def evaluate_on_test(model, loader, device, use_crf):
    """Run model on test set, return entity F1 metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        fsw_mask = batch["first_subword_mask"].to(device)

        output = model(
            input_ids, attention_mask=attention_mask,
            labels=labels, first_subword_mask=fsw_mask,
        )

        if use_crf and "word_preds" in output:
            word_preds = output["word_preds"]
            word_mask = output["word_mask"]
            for i in range(word_preds.shape[0]):
                n_words = word_mask[i].sum().item()
                pred_seq = word_preds[i, :n_words].cpu().tolist()
                lbl_indices = fsw_mask[i].nonzero(as_tuple=True)[0][:n_words]
                label_seq = labels[i, lbl_indices].cpu().tolist()
                all_preds.append(pred_seq)
                all_labels.append(label_seq)
        else:
            preds = output["logits"].argmax(dim=-1)
            for pred_seq, label_seq in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                all_preds.append(pred_seq)
                all_labels.append(label_seq)

    return compute_entity_f1(all_preds, all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default=None,
                        help="Single arm to evaluate. Default: all 6 arms")
    parser.add_argument("--config", type=str,
                        default=str(PROJECT_ROOT / "configs" / "experiment.yaml"))
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    device = cfg["experiment"]["device"]
    use_crf = True  # all final checkpoints trained with CRF

    tok_path = PROJECT_ROOT / cfg["tokenizer"]["path"]
    tokenizer = Tokenizer.from_file(str(tok_path))

    train_jsonl = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
    class_weights = compute_class_weights(train_jsonl).to(device)

    test_jsonl = PROJECT_ROOT / "data" / "processed" / "test.jsonl"
    test_ds = NERDataset(test_jsonl, tokenizer, cfg["model"]["max_len"])
    test_loader = DataLoader(
        test_ds, batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False, collate_fn=collate_fn,
    )

    all_arms = list(cfg["model"]["arms"].keys())
    arms = [args.arm] if args.arm else all_arms
    seeds = cfg["experiment"]["seed"]

    print(f"\n{'='*70}")
    print(f"  Phase 6 — Test Set Evaluation (held-out, {len(test_ds)} docs)")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Arms:   {arms}")
    print(f"  Seeds:  {seeds}\n")

    all_results = []

    for arm_name in arms:
        seed_f1s = []
        print(f"  --- {arm_name} ---")

        for seed in seeds:
            ckpt_path = PROJECT_ROOT / "checkpoints" / f"{arm_name}_seed{seed}" / "best.pt"
            if not ckpt_path.exists():
                print(f"    seed={seed}: ⚠️  checkpoint not found, skipping")
                continue

            model_kwargs = get_model_kwargs(cfg, arm_name, use_crf, class_weights)
            model = build_model(arm_name, **model_kwargs).to(device)
            meta = load_checkpoint(ckpt_path, model)

            metrics = evaluate_on_test(model, test_loader, device, use_crf)
            test_f1 = metrics["entity_f1"]
            seed_f1s.append(test_f1)

            print(
                f"    seed={seed}: test_F1={test_f1:.4f} "
                f"(val_F1 was {meta['val_f1']:.4f}) | "
                f"DATE={metrics.get('DATE_f1', 0):.3f} "
                f"AMT={metrics.get('AMOUNT_f1', 0):.3f} "
                f"NAME={metrics.get('NAME_f1', 0):.3f}"
            )

            all_results.append({
                "arm": arm_name,
                "seed": seed,
                "test_entity_f1": round(test_f1, 4),
                "test_precision": round(metrics["entity_precision"], 4),
                "test_recall": round(metrics["entity_recall"], 4),
                "test_DATE_f1": round(metrics.get("DATE_f1", 0), 4),
                "test_AMOUNT_f1": round(metrics.get("AMOUNT_f1", 0), 4),
                "test_NAME_f1": round(metrics.get("NAME_f1", 0), 4),
                "val_f1_at_checkpoint": round(meta["val_f1"], 4),
            })

        if seed_f1s:
            avg_f1 = sum(seed_f1s) / len(seed_f1s)
            print(f"    avg test_F1 = {avg_f1:.4f}\n")

    # Summary table
    print(f"{'='*70}")
    print(f"  Test Set Summary (avg across seeds)")
    print(f"{'='*70}")
    print(f"  {'Arm':<15s} {'Avg Test F1':>12s} {'Avg Precision':>14s} {'Avg Recall':>12s}")
    print(f"  {'-'*55}")

    arm_summary = {}
    for arm_name in arms:
        arm_results = [r for r in all_results if r["arm"] == arm_name]
        if not arm_results:
            continue
        avg_f1 = sum(r["test_entity_f1"] for r in arm_results) / len(arm_results)
        avg_p = sum(r["test_precision"] for r in arm_results) / len(arm_results)
        avg_r = sum(r["test_recall"] for r in arm_results) / len(arm_results)
        arm_summary[arm_name] = {"avg_f1": avg_f1, "avg_precision": avg_p, "avg_recall": avg_r}
        print(f"  {arm_name:<15s} {avg_f1:>12.4f} {avg_p:>14.4f} {avg_r:>12.4f}")

    # Save results
    output_path = PROJECT_ROOT / "evaluation" / "test_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"per_run": all_results, "summary": arm_summary}, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
