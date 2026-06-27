#!/usr/bin/env python3
"""Phase 3 pre-flight check — run on WSL2 before tokenizer training."""

import sys, os, json
from pathlib import Path

PASS, FAIL, WARN = "\033[92m✅\033[0m", "\033[91m❌\033[0m", "\033[93m⚠️\033[0m"
results = []

def check(name, ok, msg=""):
    results.append((name, ok))
    print(f"  {PASS if ok else FAIL}  {name}" + (f" — {msg}" if msg else ""))
    return ok

print("\n" + "="*60)
print("  Phase 3 Pre-flight Check")
print("="*60 + "\n")

# --- 1. Python version ---
v = sys.version_info
check("Python 3.12.x", v.major == 3 and v.minor == 12, f"{v.major}.{v.minor}.{v.micro}")

# --- 2. tokenizers library ---
try:
    import tokenizers
    check("tokenizers importable", True, f"v{tokenizers.__version__}")
    check("tokenizers == 0.19.1", tokenizers.__version__ == "0.19.1",
          tokenizers.__version__)
except ImportError:
    check("tokenizers importable", False, "pip install tokenizers==0.19.1")

# --- 3. PyTorch + CUDA (not needed for tokenizer, but confirm stack intact) ---
try:
    import torch
    check("torch importable", True, f"v{torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    check("torch.cuda.is_available()", cuda_ok)
    if cuda_ok:
        name = torch.cuda.get_device_name(0)
        check("GPU is RTX 5060 (Blackwell)", "5060" in name, name)
except ImportError:
    check("torch importable", False)

# --- 4. Data files exist ---
project_root = Path.cwd()
train_path = project_root / "data" / "processed" / "train.jsonl"
val_path   = project_root / "data" / "processed" / "val.jsonl"
test_path  = project_root / "data" / "processed" / "test.jsonl"

check("train.jsonl exists", train_path.exists(), str(train_path))
check("val.jsonl exists",   val_path.exists(),   str(val_path))
check("test.jsonl exists",  test_path.exists(),  str(test_path))

# --- 5. Data format sanity ---
if train_path.exists():
    with open(train_path) as f:
        lines = f.readlines()
    check("train.jsonl has ~2947 records", 2900 <= len(lines) <= 3000, f"{len(lines)} lines")
    
    sample = json.loads(lines[0])
    has_tokens = "tokens" in sample and isinstance(sample["tokens"], list)
    has_tags   = "tags" in sample and isinstance(sample["tags"], list)
    check("Record has 'tokens' (list)", has_tokens)
    check("Record has 'tags' (list)",   has_tags)
    
    if has_tokens and has_tags:
        check("tokens/tags same length", len(sample["tokens"]) == len(sample["tags"]),
              f"tokens={len(sample['tokens'])}, tags={len(sample['tags'])}")
        
        # Check BIO tag vocabulary
        all_tags = set()
        for line in lines[:200]:  # sample first 200
            rec = json.loads(line)
            all_tags.update(rec["tags"])
        expected = {"O", "B-DATE", "I-DATE", "B-AMOUNT", "I-AMOUNT", "B-NAME", "I-NAME"}
        check("BIO tags match expected set", all_tags == expected,
              f"found: {sorted(all_tags)}")
        
        # Token stats for tokenizer planning
        token_counts = [len(json.loads(l)["tokens"]) for l in lines]
        avg_tok = sum(token_counts) / len(token_counts)
        max_tok = max(token_counts)
        print(f"\n  📊 Token stats (train): avg={avg_tok:.0f}, max={max_tok}, "
              f"total unique words={len(set(t for l in lines for t in json.loads(l)['tokens']))}")

# --- 6. Output directory ---
tok_dir = project_root / "tokenizer_training"
check("tokenizer_training/ dir exists", tok_dir.exists())

# --- Summary ---
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"\n{'='*60}")
print(f"  Result: {passed}/{total} checks passed")
if passed == total:
    print("  🚀 Environment ready for Phase 3!")
else:
    print("  ⚠️  Fix the failures above before proceeding.")
print("="*60 + "\n")
