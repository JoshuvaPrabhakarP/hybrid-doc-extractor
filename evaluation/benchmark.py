#!/usr/bin/env python3
"""
Phase 6, Step 2 — Throughput and memory benchmark.

Measures inference speed (docs/sec) and peak GPU memory for each architecture.
This matters because Mamba's theoretical selling point is efficiency
(linear-time sequences) even though it lost on F1 — this step checks
whether that efficiency advantage actually shows up in practice.

Two measurement modes:
    - batch=1: realistic single-document deployment latency
    - batch=16: throughput under the training batch size

Usage:
    cd ~/hybrid-doc-extractor
    python -m evaluation.benchmark
"""

import json
import sys
import time
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model
from training.dataset import NERDataset, collate_fn, NUM_LABELS, compute_class_weights
from training.utils import load_checkpoint
from evaluation.evaluate import get_model_kwargs, load_config


@torch.no_grad()
def benchmark_arm(model, loader, device, n_warmup=10, n_measure=30):
    """
    Measure throughput (docs/sec) and peak GPU memory.

    Runs n_warmup batches to let CUDA kernels compile/cache, then
    measures over n_measure batches for a stable estimate.

    Uses per-batch CUDA synchronization during warmup to ensure kernels
    are fully compiled before timing starts (avoids async-queue artifacts
    that distort small-batch / batch=1 measurements).
    """
    model.eval()

    batches = []
    for i, batch in enumerate(loader):
        batches.append(batch)
        if len(batches) >= n_warmup + n_measure:
            break

    if len(batches) < n_warmup + 1:
        n_warmup = max(0, len(batches) - 1)
        n_measure = len(batches) - n_warmup

    # Warmup — fully synchronize after each batch so kernel compilation
    # and CUDA caching allocator warmup don't leak into the timed region
    for batch in batches[:n_warmup]:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        fsw_mask = batch["first_subword_mask"].to(device)
        _ = model(input_ids, attention_mask=attention_mask, first_subword_mask=fsw_mask)
        if device == "cuda":
            torch.cuda.synchronize()

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Measure — synchronize once per batch to get accurate per-call timing
    # without the overhead of synchronizing inside the hot loop affecting
    # GPU pipelining too much (still accurate since we sum wall-clock deltas)
    total_docs = 0
    total_elapsed = 0.0

    for batch in batches[n_warmup:n_warmup + n_measure]:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        fsw_mask = batch["first_subword_mask"].to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        _ = model(input_ids, attention_mask=attention_mask, first_subword_mask=fsw_mask)

        if device == "cuda":
            torch.cuda.synchronize()
        total_elapsed += time.perf_counter() - t0
        total_docs += input_ids.shape[0]

    docs_per_sec = total_docs / total_elapsed if total_elapsed > 0 else 0

    peak_mem_mb = 0
    if device == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "docs_per_sec": round(docs_per_sec, 2),
        "peak_gpu_mb": round(peak_mem_mb, 1),
        "total_docs_measured": total_docs,
        "elapsed_s": round(total_elapsed, 3),
    }


def main():
    cfg = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    device = cfg["experiment"]["device"]
    use_crf = True

    tok_path = PROJECT_ROOT / cfg["tokenizer"]["path"]
    tokenizer = Tokenizer.from_file(str(tok_path))

    train_jsonl = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
    class_weights = compute_class_weights(train_jsonl).to(device)

    test_jsonl = PROJECT_ROOT / "data" / "processed" / "test.jsonl"
    test_ds = NERDataset(test_jsonl, tokenizer, cfg["model"]["max_len"])

    arms = list(cfg["model"]["arms"].keys())
    seed = cfg["experiment"]["seed"][0]  # use first seed's checkpoint as representative

    print(f"\n{'='*70}")
    print(f"  Phase 6 — Throughput & Memory Benchmark")
    print(f"{'='*70}")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print(f"  Using seed={seed} checkpoint as representative for each arm\n")

    results = {}

    for batch_size, label in [(1, "batch=1 (single-doc latency)"), (16, "batch=16 (training-scale throughput)")]:
        print(f"  --- {label} ---")
        loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        for arm_name in arms:
            ckpt_path = PROJECT_ROOT / "checkpoints" / f"{arm_name}_seed{seed}" / "best.pt"
            if not ckpt_path.exists():
                print(f"    {arm_name}: ⚠️ checkpoint not found")
                continue

            model_kwargs = get_model_kwargs(cfg, arm_name, use_crf, class_weights)
            model = build_model(arm_name, **model_kwargs).to(device)
            load_checkpoint(ckpt_path, model)

            bench = benchmark_arm(model, loader, device)
            key = f"{arm_name}_bs{batch_size}"
            results[key] = {**bench, "arm": arm_name, "batch_size": batch_size}

            print(
                f"    {arm_name:<15s} | {bench['docs_per_sec']:>7.2f} docs/sec | "
                f"peak mem: {bench['peak_gpu_mb']:>8.1f} MB"
            )

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        print()

    # Save
    output_path = PROJECT_ROOT / "evaluation" / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {output_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
