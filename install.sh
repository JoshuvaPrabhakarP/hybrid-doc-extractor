#!/bin/bash
# install.sh - Phase 1 setup for hybrid-doc-extractor
# CONFIRMED WORKING on: Ubuntu 24.04 (WSL2), RTX 5060 Laptop GPU (Blackwell, sm_120)
# If your GPU is a different architecture, check compute capability with:
#   python -c "import torch; print(torch.cuda.get_device_capability(0))"
# and adjust the sm_120 patch / TORCH_CUDA_ARCH_LIST below accordingly.

set -e

echo "============================================================"
echo "Phase 1: Setting up hybrid-doc-extractor environment"
echo "============================================================"

echo ""
echo "[1/8] Installing build tools + gcc-11..."
sudo apt update
sudo apt install -y build-essential gcc-11 g++-11 python3.12 python3.12-venv python3.12-dev

echo ""
echo "[2/8] Checking for CUDA 12.8 toolkit..."
if [ ! -d "/usr/local/cuda-12.8" ]; then
    echo "ERROR: /usr/local/cuda-12.8 not found. Install CUDA Toolkit 12.8 first."
    echo "See graphify-out/GRAPH_REPORT.md for the install procedure."
    exit 1
fi
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

echo ""
echo "[3/8] Creating virtual environment..."
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel packaging ninja

echo ""
echo "[4/8] Setting gcc-11 as host compiler..."
export CC=gcc-11
export CXX=g++-11
export CUDAHOSTCXX=g++-11
export TORCH_CUDA_ARCH_LIST="12.0"

echo ""
echo "[5/8] Installing PyTorch 2.7.0+cu128..."
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "[6/8] Building causal-conv1d 1.5.0.post8 with sm_120 patch..."
git clone --branch v1.5.0.post8 https://github.com/Dao-AILab/causal-conv1d.git /tmp/causal-conv1d-src
cd /tmp/causal-conv1d-src
sed -i "/cc_flag.append(.arch=compute_90,code=sm_90.)/a\\    cc_flag.append(\"-gencode\")\n    cc_flag.append(\"arch=compute_120,code=sm_120\")" setup.py
pip install . --no-build-isolation --no-cache-dir
cd -
rm -rf /tmp/causal-conv1d-src

echo ""
echo "[7/8] Building mamba-ssm 2.2.4 with sm_120 patch..."
git clone --branch v2.2.4 https://github.com/state-spaces/mamba.git /tmp/mamba-src
cd /tmp/mamba-src
sed -i "/cc_flag.append(.arch=compute_90,code=sm_90.)/a\\    cc_flag.append(\"-gencode\")\n    cc_flag.append(\"arch=compute_120,code=sm_120\")" setup.py
pip install . --no-build-isolation --no-cache-dir
cd -
rm -rf /tmp/mamba-src

echo ""
echo "[8/8] Installing remaining dependencies..."
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "Running verification..."
echo "============================================================"
python scripts/verify_setup.py
