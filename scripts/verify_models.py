#!/usr/bin/env python3
"""
Phase 4 verification — test all three model arms.

Run from project root:
    python scripts/verify_models.py

Tests:
    1. All three arms instantiate correctly
    2. Forward pass produces correct output shapes
    3. Loss computes when labels are provided
    4. Parameter counts are reported for fair comparison
    5. GPU forward pass works (if CUDA available)
"""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import build_model, MODEL_REGISTRY

PASS = "\033[92m✅\033[0m"
FAIL = "\033[91m❌\033[0m"

results = []

def check(name, ok, msg=""):
    results.append((name, ok))
    print(f"  {PASS if ok else FAIL}  {name}" + (f" — {msg}" if msg else ""))
    return ok


def test_arm(arm_name: str, device: str = "cpu", **extra_kwargs):
    """Test a single model arm."""
    print(f"\n  --- {arm_name.upper()} ---")

    # Build model
    try:
        model = build_model(arm_name, **extra_kwargs)
        model = model.to(device)
        check(f"[{arm_name}] instantiate", True)
    except Exception as e:
        check(f"[{arm_name}] instantiate", False, str(e))
        return

    # Parameter count
    n_params = model.count_parameters()
    check(f"[{arm_name}] param count", n_params > 0, f"{n_params:,} parameters")

    # Forward pass (no labels)
    batch_size, seq_len = 4, 128
    input_ids = torch.randint(5, 7995, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    # Simulate padding on last 20 tokens
    attention_mask[:, -20:] = 0

    try:
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)
        logits = output["logits"]
        expected_shape = (batch_size, seq_len, 7)
        check(f"[{arm_name}] forward shape", logits.shape == expected_shape,
              f"{tuple(logits.shape)}")
    except Exception as e:
        check(f"[{arm_name}] forward shape", False, str(e))
        return

    # Forward pass with labels (loss computation)
    labels = torch.randint(0, 7, (batch_size, seq_len), device=device)
    # Mark padding positions with -100 (ignore in loss)
    labels[:, -20:] = -100

    try:
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = output["loss"]
        check(f"[{arm_name}] loss computes", loss.item() > 0, f"loss={loss.item():.4f}")
    except Exception as e:
        check(f"[{arm_name}] loss computes", False, str(e))

    # Backward pass
    try:
        output["loss"].backward()
        grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in model.parameters() if p.requires_grad)
        check(f"[{arm_name}] backward pass", grad_ok)
    except Exception as e:
        check(f"[{arm_name}] backward pass", False, str(e))


def main():
    print(f"\n{'='*60}")
    print(f"  Phase 4 — Model Architecture Verification")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Test all three arms
    test_arm("transformer", device=device)
    test_arm("mamba", device=device)
    test_arm("hybrid_5_1", device=device,
             mamba_layers=5, attn_layers=1)

    # Summary with param comparison
    print(f"\n{'='*60}")
    print(f"  Parameter comparison (fair experiment check):")
    print(f"{'='*60}")

    for arm_name in MODEL_REGISTRY:
        kwargs = {}
        if arm_name == "hybrid_5_1":
            kwargs = {"mamba_layers": 5, "attn_layers": 1}
        model = build_model(arm_name, **kwargs)
        n = model.count_parameters()
        print(f"  {arm_name:>15s}: {n:>10,} params")

    # Final result
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"  Result: {passed}/{total} checks passed")
    if passed == total:
        print(f"  🚀 All three arms verified! Ready for Phase 5 (training).")
    else:
        print(f"  ⚠️  Fix failures before proceeding.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
