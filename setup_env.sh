#!/bin/bash
# setup_env.sh — Create isolated venv for quake3-worldmodel
# Run from the quake3-worldmodel directory: bash setup_env.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure uv and caches are available
export PATH="/workspace/.local/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"

echo "=== Setting up quake3-worldmodel environment ==="

# Create project-specific venv
if [ ! -d ".venv" ]; then
    echo "--- Creating .venv ---"
    uv venv .venv --python 3.11
else
    echo "--- .venv already exists, skipping creation ---"
fi

source .venv/bin/activate

# Auto-detect GPU and install PyTorch accordingly
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "--- GPU detected, installing PyTorch with CUDA ---"
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
    echo "--- No GPU detected, installing PyTorch CPU ---"
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install remaining deps
echo "--- Installing project dependencies ---"
uv pip install -r requirements.txt

echo ""
echo "=== quake3-worldmodel environment ready ==="
echo "Activate with: source $SCRIPT_DIR/.venv/bin/activate"
