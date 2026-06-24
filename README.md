# Hybrid Transformer-Mamba for Document Field Extraction

An empirical comparison of pure Transformer, pure Mamba SSM, and hybrid Transformer-Mamba architectures for structured field extraction (NER) from invoices, receipts, and forms.

## Research Questions

1. How do Transformer, Mamba, and hybrid architectures compare on entity-level F1 for document NER when trained from scratch?
2. What is the optimal Mamba:Attention layer ratio for token classification tasks?
3. Do SSM layers provide meaningful efficiency gains at typical document lengths (256-1024 tokens)?

## Quick Start

    bash install.sh
    python scripts/verify_setup.py

If verify_setup.py reports 15/15 checks passed (including a live Mamba GPU forward pass), your environment is ready. Read the Environment Setup Journey section below before troubleshooting anything yourself.

## Project Structure

See graphify-out/GRAPH_REPORT.md for the full module dependency map.

## Datasets (Phase 2, upcoming)

- SROIE - receipt entity recognition
- CORD - consolidated receipt dataset
- FUNSD - form understanding

---

## Environment Setup Journey (read this before debugging anything)

If you have a Blackwell-generation NVIDIA GPU (RTX 50-series: 5060, 5070, 5080, 5090, including laptop variants), you will hit multiple real, confirmed issues getting mamba-ssm working. This documents everything already solved.

### Confirmed Working Stack

    Hardware:        NVIDIA RTX 5060 Laptop GPU (Blackwell, sm_120) - CUDA 12.8 minimum required
    OS:              Ubuntu 24.04 (via WSL2 - native Windows does NOT work)
    Python:          3.12.3
    PyTorch:         2.7.0+cu128
    CUDA Toolkit:    12.8
    Host compiler:   gcc-11 / g++-11
    causal-conv1d:   1.5.0.post8, built from source with manual sm_120 patch
    mamba-ssm:       2.2.4, built from source with manual sm_120 patch
    transformers:    4.44.2 (pinned - do not use 5.x)
    tokenizers:      0.19.1 (pinned to match transformers 4.44.2)
    TORCH_CUDA_ARCH_LIST: "12.0"

Run bash install.sh to get this exact stack automatically.

### Issue 1: Native Windows is not viable

Building mamba-ssm on native Windows fails with a Windows-only bug: the build looks for a selective_scan.cpp file that does not exist (only .cu exists). Unfixed upstream since 2024 (see state-spaces/mamba issue 745). Solution: use WSL2 with Ubuntu.

### Issue 2: GPU architecture sm_120 needs CUDA 12.8 minimum

Blackwell GPUs use compute capability sm_120. CUDA 12.4 and earlier have no compiled kernels for this architecture at all. Code fails at runtime with "CUDA error: no kernel image is available for execution on the device" even if basic tensor ops appear to work. Need CUDA Toolkit 12.8+ and matching PyTorch build.

### Issue 3: gcc/g++ version conflicts with CUDA 12.8 headers

Newer Ubuntu ships gcc-13+ or gcc-15 by default. These conflict with CUDA 12.8 math headers via a noexcept specifier mismatch on functions like cospi, sinpi, rsqrt. Solution: install gcc-11/g++-11 and set CC, CXX, CUDAHOSTCXX environment variables before building.

### Issue 4: transformers 5.x breaks mamba-ssm 2.2.4

mamba-ssm 2.2.4 generation.py imports classes removed in transformers 5.x. Pin transformers==4.44.2 and tokenizers==0.19.1.

### Issue 5: Neither causal-conv1d nor mamba-ssm officially support sm_120 yet

Published packages do not compile sm_120 into their CUDA kernels by default. Install succeeds, import succeeds, but forward pass fails. Must manually patch each setup.py to add the sm_120 gencode target before building from source, then verify with cuobjdump that sm_120 actually appears in the compiled .so file.

### Issue 6: ABI compatibility between causal-conv1d and mamba-ssm is strict

causal-conv1d==1.5.0.post8 plus mamba-ssm==2.2.4 is the confirmed-compatible pair for this project.

### The Golden Rule

A package installing without error does not mean it works. Always run scripts/verify_setup.py and confirm a real GPU forward pass succeeds, not just that import succeeds.

---

## License

MIT
