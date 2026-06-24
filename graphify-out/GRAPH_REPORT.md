# GRAPH_REPORT.md - Module Dependency Map
Auto-updated as new modules are added. Consult before navigating the codebase.

## Project Status

- Phase 1 (Setup): COMPLETE - 15/15 verification checks passed, including live Mamba GPU forward pass
- Phases 2-8: PENDING

## Confirmed Working Stack (DO NOT casually upgrade - hard-won, fully patched)

    Hardware:        RTX 5060 Laptop GPU (Blackwell, sm_120) - requires CUDA 12.8 minimum
    OS:              Ubuntu 24.04 (WSL2)
    Python:          3.12.3
    PyTorch:         2.7.0+cu128
    CUDA Toolkit:    12.8 (CUDA_HOME=/usr/local/cuda-12.8, PATH/LD_LIBRARY_PATH set in ~/.bashrc)
    Host compiler:   gcc-11 / g++-11 (CC, CXX, CUDAHOSTCXX env vars - newer gcc/glibc breaks
                     CUDA math headers via noexcept mismatches on cospi/sinpi/rsqrt)
    causal-conv1d:   1.5.0.post8 - built from source WITH manual sm_120 patch (see below)
    mamba-ssm:       2.2.4 - built from source WITH manual sm_120 patch (see below)
    transformers:    4.44.2 (pinned - 5.x removed GreedySearchDecoderOnlyOutput that
                     mamba-ssm 2.2.4 generation.py imports)
    tokenizers:      0.19.1 (pinned to match transformers 4.44.2)
    TORCH_CUDA_ARCH_LIST: "12.0"

## Required Manual Source Patches (sm_120 / Blackwell)

Neither causal-conv1d nor mamba-ssm officially support sm_120 yet. Both needed this
exact patch added to their setup.py (in the cc_flag / gencode list):

    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_120,code=sm_120")

Applied to:
- causal-conv1d/setup.py
- mamba/setup.py (mamba-ssm)

Verify a build actually has sm_120 compiled in with:

    cuobjdump --list-elf causal_conv1d_cuda.so   (or selective_scan_cuda.so)

If sm_120 does not appear in the output, the patch did not take - rebuild.

## Key Lessons (read before touching the environment again)

1. CUDA 12.4 is a hard incompatibility with this GPU - Blackwell (sm_120) support
   starts at CUDA 12.8. Confirmed via GitHub issue state-spaces/mamba#745.
2. Native Windows is not viable for mamba-ssm - Windows-only missing
   selective_scan.cpp bug, unfixed upstream since 2024.
3. gcc-11 required as host compiler - gcc-15 (Ubuntu default) conflicts with
   CUDA 12.8 math headers.
4. Package installing successfully does not mean it works. mamba-ssm/causal-conv1d can
   install without error yet still lack sm_120 in their compiled kernels - always verify
   with a real GPU forward pass, not just import.
5. ABI compatibility between mamba-ssm and causal-conv1d is strict - version
   mismatches cause incompatible function arguments at runtime even when both
   individually work. 2.2.4 + 1.5.0.post8 is the confirmed-compatible pair.
6. Always rebuild causal-conv1d and mamba-ssm together after any PyTorch/CUDA change.
7. Use --no-build-isolation --no-cache-dir to avoid stale/mismatched cached wheels.

## Current Files

    requirements.txt               - pinned to the confirmed-working versions above
    install.sh                     - automated setup: WSL2 + gcc-11 + CUDA 12.8 + sm_120 patches
    configs/experiment.yaml        - shared hyperparams for all 3 arms
    scripts/verify_setup.py        - ALL 15 CHECKS PASS - working verification script
    graphify-out/GRAPH_REPORT.md   - this file
    README.md                      - research questions + quick start + full setup journey

## External: claude.md

    /home/claude/claude.md - standing workflow instructions (Graphify-first navigation)

## Next Steps

    DONE: Phase 1 - environment fully verified on GPU with sm_120 patches confirmed via cuobjdump
    NEXT: Push to GitHub now that there is a real working Phase 1 to commit
    THEN: Phase 2 - data pipeline (SROIE, CORD, FUNSD download + BIO tagging)

## Empty Directories (awaiting future phases)

    data/                - Phase 2: SROIE, CORD, FUNSD download + BIO preprocessing
    tokenizer_training/  - Phase 3: BPE tokenizer (8k vocab)
    models/              - Phase 4: transformer.py, mamba_model.py, hybrid.py, base.py
    training/            - Phase 5: train.py, utils.py
    evaluation/          - Phase 6: evaluate.py, ablation.py, efficiency.py
    app/                 - Phase 7: Streamlit dashboard
    paper/               - Phase 8: LaTeX source

## Dependency Flow

    configs/experiment.yaml
      (read by all modules)
    data/ -> tokenizer_training/ -> models/ -> training/ -> evaluation/ -> app/
