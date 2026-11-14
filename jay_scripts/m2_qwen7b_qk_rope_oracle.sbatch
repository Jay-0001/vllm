#!/usr/bin/env bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:04:30
#SBATCH --array=0-1
#SBATCH --job-name=m2-qk-rope
#SBATCH --output=/u/jsaravananindira/vllm/jay_outputs/m2_qwen7b_last_chance/qk-rope-%A_%a.out

set -Eeuo pipefail

repo=/u/jsaravananindira/vllm
cd "$repo"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch.' >&2; exit 2; }
[[ -x .venv/bin/python ]] || { echo 'Missing Delta venv.' >&2; exit 2; }

case "${SLURM_ARRAY_TASK_ID:?}" in
  0) shapes=decode ;;
  1) shapes=prefill ;;
  *) echo 'Unexpected array task.' >&2; exit 2 ;;
esac

output="$repo/jay_outputs/m2_qwen7b_last_chance/qk_rope_${shapes}_${SLURM_JOB_ID}"
mkdir -p "$output/torch_compile_debug" "$output/inductor_cache"
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR="$output/torch_compile_debug"
export TORCHINDUCTOR_CACHE_DIR="$output/inductor_cache"
{
  date -u +%FT%TZ
  git rev-parse HEAD
  git status --short
  .venv/bin/python -c 'import torch, triton, vllm; print(torch.__version__, triton.__version__, vllm.__version__)'
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
} > "$output/environment.txt"

/usr/bin/time -f 'process_wall_seconds=%e' -o "$output/process.time" \
  .venv/bin/python benchmarks/kernels/m2_fusion_boundary/qk_rope_oracle.py \
    --shapes "$shapes" --output "$output" \
    > "$output/run.log" 2>&1

printf 'Results: %s\n' "$output"
