#!/usr/bin/env bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:03:30
#SBATCH --array=0-3%1
#SBATCH --job-name=m2-7b-model
#SBATCH --output=/u/jsaravananindira/vllm/jay_outputs/m2_qwen7b_last_chance/model-%A_%a.out

set -Eeuo pipefail

repo=/u/jsaravananindira/vllm
cd "$repo"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch.' >&2; exit 2; }
[[ -x .venv/bin/python ]] || { echo 'Missing Delta venv.' >&2; exit 2; }
case "${SLURM_ARRAY_TASK_ID:?}" in
  0|3) arm=stock; export M2_QK_MODEL_ORACLE=0 ;;
  1|2) arm=oracle; export M2_QK_MODEL_ORACLE=1 ;;
  *) exit 2 ;;
esac

root="$repo/jay_outputs/m2_qwen7b_last_chance"
output="$root/model_${arm}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$output/torch_compile_debug" "$output/cache"
export PYTHONPATH="$repo/benchmarks/kernels/m2_fusion_boundary/model_oracle${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_CACHE_ROOT="$output/cache"
export TORCHINDUCTOR_CACHE_DIR="$output/cache/inductor"
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR="$output/torch_compile_debug"

{
  date -u +%FT%TZ
  git rev-parse HEAD
  printf 'arm=%s smoke=%s\n' "$arm" "${M2_SMOKE:-0}"
  .venv/bin/python -c 'import torch, triton, vllm; print(torch.__version__, triton.__version__, vllm.__version__)'
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
} > "$output/environment.txt"

arguments=()
if [[ "${M2_SMOKE:-0}" == 1 ]]; then arguments+=(--smoke); fi
/usr/bin/time -f 'process_wall_seconds=%e' -o "$output/process.time" \
  .venv/bin/python benchmarks/kernels/m2_fusion_boundary/model_latency.py \
    --output "$output" "${arguments[@]}" > "$output/run.log" 2>&1

if [[ "$arm" == oracle ]]; then
  if ! grep -Rl --include=output_code.py 'm2.qk_rope' \
      "$output/torch_compile_debug" > "$output/oracle_code_paths.txt"; then
    echo 'Oracle op absent from generated Inductor code.' >&2
    exit 1
  fi
fi
printf 'Results: %s\n' "$output"
