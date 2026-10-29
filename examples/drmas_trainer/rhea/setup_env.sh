#!/bin/bash
# One-time environment setup for running Dr. MAS (Math task) on Rhea.
#
# Run this inside an interactive GPU job so flash-attn can find CUDA, e.g.
#   srun -p general --gres=gpu:A6000:1 -c 8 --mem=64G -t 2:00:00 --pty bash
#   bash examples/drmas_trainer/rhea/setup_env.sh
set -euo pipefail

# Keep large caches off $HOME (adjust to your scratch / data directory on Rhea)
export DRMAS_CACHE=${DRMAS_CACHE:-$HOME/drmas_cache}
export HF_HOME=$DRMAS_CACHE/hf
export PIP_CACHE_DIR=$DRMAS_CACHE/pip
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"

# module load cuda/12.4   # uncomment / adjust if Rhea needs a CUDA module (check `module avail cuda`)

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^DrMAS "; then
    conda create -n DrMAS python==3.12 -y
fi
conda activate DrMAS

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO_ROOT"

pip3 install -r requirements_sglang.txt
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip3 install -e .

# Quick sanity check
python -c "import torch, sglang, flash_attn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# Prepare the math dataset -> ~/data/drmas_math/{train,test,test_sampled}.parquet
python examples/data_preprocess/drmas_math.py
ls -lh ~/data/drmas_math
