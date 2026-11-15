#!/usr/bin/env bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --array=0-1%1
#SBATCH --job-name=m8-tp-profile
#SBATCH --output=/u/jsaravananindira/vllm_m8/jay_outputs/m8_distributed_overlap/profile-%A_%a.out

set -Eeuo pipefail

repo=/u/jsaravananindira/vllm_m8
cd "$repo"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch.' >&2; exit 2; }
[[ -x .venv/bin/python ]] || { echo 'M8 Delta venv is missing.' >&2; exit 2; }
case "${SLURM_ARRAY_TASK_ID:?}" in
  0) batch=1 ;;
  1) batch=16 ;;
  *) exit 2 ;;
esac

root="$repo/jay_outputs/m8_distributed_overlap"
output="$root/profile_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_b${batch}"
mkdir -p "$output/cache" "$output/traces"
export VLLM_CACHE_ROOT="$output/cache"
export PYTHONUNBUFFERED=1
{
  date -u +%FT%TZ
  hostname
  git rev-parse HEAD
  .venv/bin/python -c 'import torch, vllm; print(torch.__version__, vllm.__version__, vllm.__file__); assert torch.cuda.device_count() == 2'
  nvidia-smi topo -m
} > "$output/environment.txt"

/usr/bin/time -f 'process_wall_seconds=%e' -o "$output/process.time" \
  .venv/bin/python -m vllm.entrypoints.cli.main bench latency \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dtype bfloat16 --seed 1 --tensor-parallel-size 2 \
    --max-model-len 4096 --max-num-batched-tokens 4096 \
    --max-num-seqs 16 --gpu-memory-utilization 0.80 \
    --input-len 128 --output-len 8 --batch-size "$batch" \
    --num-iters-warmup 2 --num-iters 1 --profile \
    --profiler-config.profiler torch \
    --profiler-config.torch_profiler_dir "$output/traces" \
    > "$output/run.log" 2>&1

printf 'Trace directory: %s\n' "$output/traces"
