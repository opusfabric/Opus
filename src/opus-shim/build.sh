#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! python -c "import torch" >/dev/null 2>&1; then
    echo "Error: the Opus shim requires a CUDA-enabled PyTorch installation in the active environment." >&2
    exit 2
fi

# --- Clean old builds ---
rm -rf build/ dist/ *.egg-info *.so

# --- Set environment for PyTorch CUDA libs ---
export TORCH_LIB=$(python -c "import torch; import os; print(os.path.join(torch.__path__[0],'lib'))")
export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LIBRARY_PATH="${TORCH_LIB}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# --- Show nvcc for sanity check ---
echo "Using nvcc at: $(which nvcc)"

# --- Rebuild extension linking only against Torch libs ---
python "${SCRIPT_DIR}/setup.py" build_ext --inplace
