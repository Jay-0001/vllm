#!/usr/bin/env bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --job-name=vllm-a100-setup
#SBATCH --output=/u/jsaravananindira/vllm/jay_outputs/setup-%j.out
#SBATCH --error=/u/jsaravananindira/vllm/jay_outputs/setup-%j.err

set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO: $BASH_COMMAND" >&2' ERR

cd /u/jsaravananindira/vllm
export PATH="$HOME/.local/bin:$PATH"

echo "=== uv and GPU ==="
command -v uv
uv --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
git rev-parse HEAD

echo "=== Virtual environment ==="
uv venv --python 3.12 --seed .venv
source .venv/bin/activate

echo "=== Editable vLLM installation ==="
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto

echo "=== Precompiled kernels ==="
vllm download-kernels

echo "=== Import and GPU check ==="
.venv/bin/python -c '
import torch
import vllm
print("vLLM:", vllm.__version__)
print("PyTorch:", torch.__version__)
print("Visible CUDA GPUs:", torch.cuda.device_count())
assert torch.cuda.device_count() == 1
'

echo "=== Inference smoke test ==="
vllm bench latency \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype float16 \
  --max-model-len 512 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.75 \
  --input-len 64 \
  --output-len 1 \
  --batch-size 1 \
  --num-iters-warmup 1 \
  --num-iters 2 \
  --compilation-config '{"cudagraph_mode":"NONE","compile_sizes":[]}'

echo "=== Setup completed ==="