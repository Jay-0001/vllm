#!/usr/bin/env bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:03:00
#SBATCH --job-name=m2-7b-graph
#SBATCH --output=/u/jsaravananindira/vllm/jay_outputs/m2_qwen7b_last_chance/slurm-%j.out

set -Eeuo pipefail

repo=/u/jsaravananindira/vllm
cd "$repo"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch.' >&2; exit 2; }
[[ -x .venv/bin/python ]] || { echo 'Missing Delta venv.' >&2; exit 2; }

root="$repo/jay_outputs/m2_qwen7b_last_chance"
case_dir="$root/compile_${SLURM_JOB_ID}"
mkdir -p "$case_dir" "$case_dir/cache" "$case_dir/debug" "$case_dir/graphs"
export VLLM_CACHE_ROOT="$case_dir/cache"
export TORCHINDUCTOR_CACHE_DIR="$case_dir/cache/inductor"
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR="$case_dir/debug"
export TORCH_LOGS=graph_breaks,recompiles

{
  date -u +%FT%TZ
  git rev-parse HEAD
  .venv/bin/python -c 'import torch, vllm; print(torch.__version__, vllm.__version__, torch.cuda.get_device_name())'
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
} > "$case_dir/environment.txt"

compile_config="$(printf \
  '{"cudagraph_mode":"NONE","compile_sizes":[],"debug_dump_path":"%s"}' \
  "$case_dir/graphs")"

/usr/bin/time -f 'process_wall_seconds=%e' -o "$case_dir/process.time" \
  .venv/bin/vllm bench latency \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dtype bfloat16 \
    --seed 0 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.80 \
    --input-len 512 \
    --output-len 1 \
    --batch-size 4 \
    --num-iters-warmup 1 \
    --num-iters 2 \
    --compilation-config "$compile_config" \
    --output-json "$case_dir/result.json"

printf 'Graph dump: %s\n' "$case_dir"
