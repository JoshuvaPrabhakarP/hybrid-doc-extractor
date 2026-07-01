#!/usr/bin/env python3
"""
Phase 6, Step 3 — Sequence length ablation.

Tests throughput and memory at 256, 512, and 1024 tokens to see whether
Mamba's theoretical O(n) scaling (vs Transformer's O(n^2) attention)
shows up in practice as sequence length grows.

Since our checkpoints were trained at max_seq_len=512, we cannot validly
test F1 at other lengths (the model's positional encoding and learned
patterns are calibrated to 512). Instead this step uses SYNTHETIC inputs
of varying length to isolate pure compute/memory scaling behavior,
independent of trained weights or real data content.

Usage:
    cd ~/hybrid-doc-extractor
    python -m evaluation.seqlen_ablation
"""

import json
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model
from training.dataset import NUM_LABELS, compute_class_weights
from evaluation.evaluate import get_model_kwargs, load_config


@torch.no_grad()
def benchmark_seqlen(model, seq_len, batch_size, device, n_warmup=10, n_measure=30):
    """
    Benchmark a model at a fixed synthetic sequence length.

    Generates random token IDs (no real text needed — we're measuring
    pure compute/memory scaling, not accuracy) and times forward passes.
    """
    model.eval()
    vocab_size = 8000

    # Build a fixed synthetic batch (reused across warmup + measure)
    input_ids = torch.randint(5, vocab_size - 1, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    # first_subword_mask: alternate pattern (every other token is a "first subword")
    # to make CRF gathering produce a realistic word-level sequence length
    fsw_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    fsw_mask[:, ::2] = True  # every other position is a first-subword

    # Warmup
    for _ in range(n_warmup):
        _ = model(input_ids, attention_mask=attention_mask, first_subword_mask=fsw_mask)
        if device == "cuda":
            torch.cuda.synchronize()

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    total_elapsed = 0.0
    for _ in range(n_measure):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(input_ids, attention_mask=attention_mask, first_subword_mask=fsw_mask)
        if device == "cuda":
            torch.cuda.synchronize()
        total_elapsed += time.perf_counter() - t0

    docs_per_sec = (batch_size * n_measure) / total_elapsed if total_elapsed > 0 else 0
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0

    return {
        "docs_per_sec": round(docs_per_sec, 2),
        "peak_gpu_mb": round(peak_mem_mb, 1),
        "ms_per_batch": round((total_elapsed / n_measure) * 1000, 2),
    }


def main():
    cfg = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    device = cfg["experiment"]["device"]
    use_crf = True

    train_jsonl = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
    class_weights = compute_class_weights(train_jsonl).to(device)

    seq_lengths = [256, 512, 1024]
    batch_size = 8  # smaller batch since 1024-len sequences use more memory
    seed = cfg["experiment"]["seed"][0]
    arms = list(cfg["model"]["arms"].keys())

    print(f"\n{'='*70}")
    print(f"  Phase 6 — Sequence Length Ablation (synthetic inputs)")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Seq lengths: {seq_lengths}, batch_size: {batch_size}")
    print(f"  Note: tests pure compute/memory scaling, not accuracy\n")

    results = {}

    for seq_len in seq_lengths:
        print(f"  --- seq_len={seq_len} ---")

        for arm_name in arms:
            ckpt_path = PROJECT_ROOT / "checkpoints" / f"{arm_name}_seed{seed}" / "best.pt"
            if not ckpt_path.exists():
                print(f"    {arm_name}: ⚠️ checkpoint not found")
                continue

            # Build model with max_len large enough to cover all tested lengths
            model_kwargs = get_model_kwargs(cfg, arm_name, use_crf, class_weights)
            model_kwargs["max_len"] = max(seq_lengths)  # ensure pos encoding covers 1024
            model = build_model(arm_name, **model_kwargs).to(device)

            # Load only matching weights (pos encoding buffer size differs from
            # checkpoint's original max_len=512). strict=False skips missing keys
            # but still errors on SHAPE mismatches, so filter those out manually —
            # the positional encoding is a fixed sinusoidal table we can regenerate
            # at any length; it doesn't need to be loaded from the checkpoint.
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt["model_state_dict"]
            model_state = model.state_dict()
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if k in model_state and v.shape == model_state[k].shape
            }
            skipped = set(state_dict.keys()) - set(filtered_state_dict.keys())
            model.load_state_dict(filtered_state_dict, strict=False)

            try:
                bench = benchmark_seqlen(model, seq_len, batch_size, device)
                key = f"{arm_name}_seqlen{seq_len}"
                results[key] = {**bench, "arm": arm_name, "seq_len": seq_len}
                print(
                    f"    {arm_name:<15s} | {bench['docs_per_sec']:>7.2f} docs/sec | "
                    f"{bench['ms_per_batch']:>7.2f} ms/batch | "
                    f"peak mem: {bench['peak_gpu_mb']:>8.1f} MB"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"    {arm_name:<15s} | ⚠️ OOM at seq_len={seq_len}")
                results[f"{arm_name}_seqlen{seq_len}"] = {"oom": True, "arm": arm_name, "seq_len": seq_len}
                torch.cuda.empty_cache()

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        print()

    # Save
    output_path = PROJECT_ROOT / "evaluation" / "seqlen_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {output_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
