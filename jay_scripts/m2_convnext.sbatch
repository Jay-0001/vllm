#!/usr/bin/env bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --array=0-1%1
#SBATCH --job-name=m2-convnext
#SBATCH --output=/u/jsaravananindira/vllm/jay_outputs/m2_convnext_scout/slurm-%A_%a.out

set -Eeuo pipefail
cd /u/jsaravananindira/vllm
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch.' >&2; exit 2; }
[[ -x .venv/bin/python ]] || { echo 'Missing Delta venv.' >&2; exit 2; }

case "${SLURM_ARRAY_TASK_ID:?}" in
  0) batch=1; layout=1 ;;
  1) batch=1; layout=0 ;;
  2) batch=4; layout=1 ;;
  3) batch=4; layout=0 ;;
  *) echo 'Invalid array index.' >&2; exit 2 ;;
esac

root="$PWD/jay_outputs/m2_convnext_scout"
case_dir="$root/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_b${batch}_layout${layout}"
mkdir -p "$case_dir" "$case_dir/cache" "$case_dir/debug"
export TORCHINDUCTOR_LAYOUT_OPTIMIZATION="$layout"
export TORCHINDUCTOR_CACHE_DIR="$case_dir/cache"
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR="$case_dir/debug"
export TORCH_LOGS=graph_breaks,recompiles
{
  date -u
  git rev-parse HEAD
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
} > "$case_dir/environment.txt"

exec .venv/bin/python jay_scripts/m2_model_scout.py \
  --model convnext --batch "$batch" --output "$case_dir"
