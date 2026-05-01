#!/bin/bash

# Create venv on scratch
python -m venv /scratch/.venv
source /scratch/.venv/bin/activate

# Point caches to scratch
export PIP_CACHE_DIR=/scratch/.cache/pip
export HF_HOME=/scratch/.cache/huggingface
mkdir -p $PIP_CACHE_DIR $HF_HOME

# Install packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Uninstall flash attention first, then install vllm
pip uninstall flash-attn -y
pip install vllm==0.19.1 transformers tqdm bitsandbytes antlr4-python3-runtime==4.11.1 accelerate ipykernel

# Register Jupyter kernel
python -m ipykernel install --user --name scratch-venv --display-name "Python (scratch)"

echo "Done! Switch your kernel to Python (scratch)"
