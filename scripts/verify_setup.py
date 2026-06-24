#!/usr/bin/env python3
"""
verify_setup.py — Run after install to confirm all dependencies work.
Exit 0 = ready to proceed to Phase 2.  Exit 1 = something's broken.
"""

import sys

def check(name, test_fn):
    try:
        result = test_fn()
        print(f"  ✓ {name}: {result}")
        return True
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return False

def main():
    print("=" * 60)
    print("Phase 1 · Setup Verification")
    print("=" * 60)
    results = []

    # --- Group 1: Core ML ---
    print("\n[Group 1] Core ML stack")
    results.append(check("Python version", lambda: sys.version.split()[0]))
    results.append(check("PyTorch", lambda: __import__("torch").__version__))
    results.append(check("CUDA available", lambda: __import__("torch").cuda.is_available()))
    results.append(check(
        "GPU name",
        lambda: __import__("torch").cuda.get_device_name(0)
        if __import__("torch").cuda.is_available() else "NO GPU — will use CPU (slow)"
    ))
    results.append(check("tokenizers", lambda: __import__("tokenizers").__version__))
    results.append(check("datasets", lambda: __import__("datasets").__version__))
    results.append(check("scikit-learn", lambda: __import__("sklearn").__version__))

    # --- Group 2: Mamba-specific ---
    print("\n[Group 2] Mamba-specific")
    results.append(check("causal-conv1d", lambda: __import__("causal_conv1d").__version__))
    results.append(check("mamba-ssm", lambda: __import__("mamba_ssm").__version__))
    results.append(check("einops", lambda: __import__("einops").__version__))

    # Critical: verify Mamba layer actually runs on GPU
    def test_mamba_forward():
        import torch
        from mamba_ssm import Mamba
        device = "cuda" if torch.cuda.is_available() else "cpu"
        layer = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).to(device)
        x = torch.randn(1, 32, 64).to(device)  # (batch, seq_len, d_model)
        out = layer(x)
        return f"output shape {tuple(out.shape)} on {device}"

    results.append(check("Mamba forward pass", test_mamba_forward))

    # --- Group 3: Research tooling ---
    print("\n[Group 3] Research tooling")
    results.append(check("wandb", lambda: __import__("wandb").__version__))
    def get_seqeval_version():
        from importlib.metadata import version
        return version("seqeval")
    results.append(check("seqeval", get_seqeval_version))
    results.append(check("matplotlib", lambda: __import__("matplotlib").__version__))
    results.append(check("streamlit", lambda: __import__("streamlit").__version__))

    # --- Summary ---
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"Result: {passed}/{total} checks passed")
    if all(results):
        print("✓ Setup complete — ready for Phase 2 (data pipeline)")
    else:
        print("✗ Fix the failures above before proceeding")
    print("=" * 60)
    sys.exit(0 if all(results) else 1)

if __name__ == "__main__":
    main()