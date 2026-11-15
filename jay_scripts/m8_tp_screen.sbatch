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
#SBATCH --array=0-3%1
#SBATCH --job-name=m8-tp-screen
#SBATCH --output=/u/jsaravananindira/vllm_m8/jay_outputs/m8_distributed_overlap/slurm-%A_%a.out

set -Eeuo pipefail

repo=/u/jsaravananindira/vllm_m8
cd "$repo"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch.' >&2; exit 2; }
[[ -x .venv/bin/python ]] || { echo 'M8 Delta venv is missing.' >&2; exit 2; }
case "${SLURM_ARRAY_TASK_ID:?}" in
  0|3) arm=default; arguments=() ;;
  1|2) arm=nccl; arguments=(--disable-custom-all-reduce) ;;
  *) exit 2 ;;
esac

root="$repo/jay_outputs/m8_distributed_overlap"
output="$root/screen_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${arm}"
mkdir -p "$output/cache"
export VLLM_CACHE_ROOT="$output/cache"
export PYTHONUNBUFFERED=1

{
  date -u +%FT%TZ
  hostname
  git rev-parse HEAD
  git status --short
  printf 'arm=%s job=%s visible=%s\n' "$arm" "$SLURM_JOB_ID" "${CUDA_VISIBLE_DEVICES:-unset}"
  .venv/bin/python -c 'import torch, triton, vllm; print(torch.__version__, triton.__version__, vllm.__version__, vllm.__file__); assert torch.cuda.device_count() == 2'
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
  nvidia-smi topo -m
} > "$output/environment.txt"

/usr/bin/time -f 'process_wall_seconds=%e' -o "$output/process.time" \
  .venv/bin/python jay_scripts/m8_tp_probe.py \
    --output "$output/results.json" "${arguments[@]}" \
    > "$output/run.log" 2>&1

printf 'Results: %s\n' "$output"
