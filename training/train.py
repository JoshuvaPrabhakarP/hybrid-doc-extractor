#!/usr/bin/env python3
"""
Phase 5 — Training pipeline with all fixes applied.

Fixes:
  1. Class-weighted loss (inverse frequency weights for tag imbalance)
  2. Lower LR (1e-4), more epochs (50), higher dropout (0.2), early stopping
  3. CRF layer (word-level Viterbi decoding for valid BIO transitions)

Usage:
    cd ~/hybrid-doc-extractor
    python -m training.train                           # all arms × all seeds
    python -m training.train --arm mamba --seed 42     # single run
    python -m training.train --no-crf                  # disable CRF
"""

import argparse
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
from training.utils import set_seed, compute_entity_f1, save_checkpoint


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        fsw_mask = batch["first_subword_mask"].to(device)

        optimizer.zero_grad()
        output = model(
            input_ids, attention_mask=attention_mask,
            labels=labels, first_subword_mask=fsw_mask,
        )
        loss = output["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, device, use_crf=False):
    model.eval()
    total_loss = 0.0
    n_batches = 0
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
        total_loss += output["loss"].item()
        n_batches += 1

        if use_crf and "word_preds" in output:
            # CRF: predictions are at word level
            word_preds = output["word_preds"]
            word_mask = output["word_mask"]
            # Gather word-level labels for comparison
            for i in range(word_preds.shape[0]):
                n_words = word_mask[i].sum().item()
                pred_seq = word_preds[i, :n_words].cpu().tolist()
                # Get word-level labels from the batch
                lbl_indices = fsw_mask[i].nonzero(as_tuple=True)[0][:n_words]
                label_seq = labels[i, lbl_indices].cpu().tolist()
                all_preds.append(pred_seq)
                all_labels.append(label_seq)
        else:
            # Standard: predictions at subword level
            preds = output["logits"].argmax(dim=-1)
            for pred_seq, label_seq in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                all_preds.append(pred_seq)
                all_labels.append(label_seq)

    metrics = compute_entity_f1(all_preds, all_labels)
    metrics["val_loss"] = total_loss / n_batches
    return metrics


def get_model_kwargs(cfg, arm_name, use_crf, class_weights):
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


def train_single_run(arm_name, seed, cfg, tokenizer, device, use_crf, class_weights):
    set_seed(seed)

    data_dir = PROJECT_ROOT / "data" / "processed"
    train_cfg = cfg["training"]
    patience = train_cfg.get("early_stopping_patience", 7)

    # Datasets
    train_ds = NERDataset(data_dir / "train.jsonl", tokenizer, cfg["model"]["max_len"])
    val_ds = NERDataset(data_dir / "val.jsonl", tokenizer, cfg["model"]["max_len"])

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"],
        shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"] * 2,
        shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # Model
    model_kwargs = get_model_kwargs(cfg, arm_name, use_crf, class_weights)
    model = build_model(arm_name, **model_kwargs).to(device)
    n_params = model.count_parameters()

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"],
    )
    total_steps = len(train_loader) * train_cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=train_cfg["lr"], total_steps=total_steps,
        pct_start=0.1, anneal_strategy="cos",
    )

    ckpt_dir = PROJECT_ROOT / "checkpoints" / f"{arm_name}_seed{seed}"
    ckpt_path = ckpt_dir / "best.pt"

    run_id = f"{arm_name}/seed={seed}"
    crf_tag = " +CRF" if use_crf else ""
    print(f"\n{'='*60}")
    print(f"  {run_id} | {n_params:,} params | {train_cfg['epochs']} epochs{crf_tag}")
    print(f"{'='*60}")

    best_f1 = 0.0
    best_epoch = -1
    epochs_no_improve = 0
    epoch_logs = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, train_cfg["grad_clip"]
        )
        val_metrics = validate(model, val_loader, device, use_crf=use_crf)
        elapsed = time.time() - t0

        val_f1 = val_metrics["entity_f1"]

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_metrics["val_loss"], 4),
            "entity_f1": round(val_f1, 4),
            "DATE_f1": round(val_metrics.get("DATE_f1", 0), 4),
            "AMOUNT_f1": round(val_metrics.get("AMOUNT_f1", 0), 4),
            "NAME_f1": round(val_metrics.get("NAME_f1", 0), 4),
            "time_s": round(elapsed, 1),
        }
        epoch_logs.append(log_entry)

        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_f1, ckpt_path)
        else:
            epochs_no_improve += 1

        marker = " ★" if is_best else ""
        print(
            f"  Epoch {epoch:>2d}/{train_cfg['epochs']} | "
            f"loss={train_loss:.4f} | "
            f"F1={val_f1:.4f} | "
            f"DATE={val_metrics.get('DATE_f1', 0):.3f} "
            f"AMT={val_metrics.get('AMOUNT_f1', 0):.3f} "
            f"NAME={val_metrics.get('NAME_f1', 0):.3f} | "
            f"{elapsed:.1f}s{marker}"
        )

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\n  ⏹ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    log_path = ckpt_dir / "training_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(epoch_logs, f, indent=2)

    print(f"\n  Best: epoch {best_epoch}, val_F1={best_f1:.4f}")
    print(f"  Checkpoint: {ckpt_path}")

    return {
        "arm": arm_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_f1": round(best_f1, 4),
        "params": n_params,
        "epochs_trained": epoch,
        "crf": use_crf,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-crf", action="store_true", help="Disable CRF layer")
    parser.add_argument("--config", type=str,
                        default=str(PROJECT_ROOT / "configs" / "experiment.yaml"))
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    device = cfg["experiment"]["device"]
    use_crf = not args.no_crf

    # Load tokenizer
    tok_path = PROJECT_ROOT / cfg["tokenizer"]["path"]
    tokenizer = Tokenizer.from_file(str(tok_path))

    # Compute class weights from training data
    train_jsonl = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
    class_weights = compute_class_weights(train_jsonl).to(device)

    arms = [args.arm] if args.arm else list(cfg["model"]["arms"].keys())
    seeds = [args.seed] if args.seed is not None else cfg["experiment"]["seed"]

    print(f"\n{'='*60}")
    print(f"  Hybrid Document Extractor — Training")
    print(f"{'='*60}")
    print(f"  Device:   {device} ({torch.cuda.get_device_name(0)})")
    print(f"  Arms:     {arms}")
    print(f"  Seeds:    {seeds}")
    print(f"  Epochs:   {cfg['training']['epochs']} (early stop patience={cfg['training'].get('early_stopping_patience', 7)})")
    print(f"  LR:       {cfg['training']['lr']}, Dropout: {cfg['model']['dropout']}")
    print(f"  CRF:      {'ON' if use_crf else 'OFF'}")
    print(f"  Weights:  {[f'{w:.2f}' for w in class_weights.cpu().tolist()]}")

    all_results = []
    for arm_name in arms:
        for seed in seeds:
            result = train_single_run(arm_name, seed, cfg, tokenizer, device, use_crf, class_weights)
            all_results.append(result)

    print(f"\n{'='*60}")
    print(f"  Training Summary")
    print(f"{'='*60}")
    print(f"  {'Arm':<15s} {'Seed':>5s} {'Params':>10s} {'Epoch':>6s} {'Val F1':>8s}")
    print(f"  {'-'*50}")
    for r in all_results:
        print(f"  {r['arm']:<15s} {r['seed']:>5d} {r['params']:>10,} {r['best_epoch']:>6d} {r['best_val_f1']:>8.4f}")

    results_path = PROJECT_ROOT / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
