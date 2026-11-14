#!/usr/bin/env bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --job-name=vllm-shape64
#SBATCH --output=/u/jsaravananindira/vllm/jay_outputs/slurm-%j.out
#SBATCH --error=/u/jsaravananindira/vllm/jay_outputs/slurm-%j.err

set -Eeuo pipefail

repo=/u/jsaravananindira/vllm
root="$repo/specialization_experiments"
mode="${1:---plan}"
model="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
warmups="${WARMUPS:-15}"
iterations="${ITERATIONS:-60}"

case "$mode" in
  --plan|--run) ;;
  *) echo "Usage: $0 [--plan|--run]" >&2; exit 2 ;;
esac

cases=(
  'dynamic|64|1|[]|64_b1_a'
  'spec64|64|1|[64]|64_b1_a'
  'spec64|64|1|[64]|64_b1_b'
  'dynamic|64|1|[]|64_b1_b'
  'dynamic|63|1|[]|63_b1'
  'spec64|63|1|[64]|63_b1'
  'dynamic|65|1|[]|65_b1'
  'spec64|65|1|[64]|65_b1'
  'dynamic|64|2|[]|64_b2'
  'spec128|64|2|[128]|64_b2'
  'dynamic|64|4|[]|64_b4'
  'spec256|64|4|[256]|64_b4'
)

echo "Model: $model; warmups: $warmups; measured iterations: $iterations"
echo 'All cases use output_len=1 and cudagraph_mode=NONE.'
printf '%-9s %-9s %-6s %-14s %s\n' 'arm' 'input_len' 'batch' 'compile_sizes' 'case'
for row in "${cases[@]}"; do
  IFS='|' read -r arm input_len batch sizes label <<< "$row"
  printf '%-9s %-9s %-6s %-14s %s\n' "$arm" "$input_len" "$batch" "$sizes" "$label"
done

if [[ "$mode" == --plan ]]; then
  echo 'Plan only. Submit this file with sbatch and pass --run to execute.'
  exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Run this sweep through sbatch.' >&2; exit 1; }
cd "$repo"
[[ -x .venv/bin/vllm ]] || { echo 'Missing .venv/bin/vllm in checkout.' >&2; exit 1; }
command -v nvidia-smi >/dev/null
command -v /usr/bin/time >/dev/null

mkdir -p "$root"/{results,logs,cache,metadata}
run_id="$(date -u +%Y%m%dT%H%M%SZ)-job${SLURM_JOB_ID}"
mkdir -p "$root/results/$run_id" "$root/logs/$run_id" "$root/metadata/$run_id"

{
  echo "run_id=$run_id"
  echo "model=$model"
  echo "warmups=$warmups"
  echo "iterations=$iterations"
  echo "git_head=$(git rev-parse HEAD)"
  echo "vllm_executable=$repo/.venv/bin/vllm"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  .venv/bin/python -c 'import torch, vllm; print("torch=" + torch.__version__); print("vllm=" + vllm.__version__); print("visible_gpu_count=" + str(torch.cuda.device_count()))'
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
} > "$root/metadata/$run_id/environment.txt"

[[ "$(.venv/bin/python -c 'import torch; print(torch.cuda.device_count())')" == 1 ]] || {
  echo 'Expected exactly one visible CUDA GPU.' >&2
  exit 1
}

printf 'label\tarm\tinput_len\tbatch_size\tcompile_sizes\tcache_root\n' \
  > "$root/metadata/$run_id/cases.tsv"

for row in "${cases[@]}"; do
  IFS='|' read -r arm input_len batch sizes label <<< "$row"
  name="${arm}_${label}"
  cache_root="$root/cache/$arm"
  mkdir -p "$cache_root"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$arm" "$input_len" "$batch" "$sizes" "$cache_root" \
    >> "$root/metadata/$run_id/cases.tsv"
  echo "Running $name at $(date -u +%FT%TZ)"

  (
    export VLLM_CACHE_ROOT="$cache_root"
    export TORCH_LOGS=recompiles
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    /usr/bin/time -f 'process_wall_seconds=%e' \
      -o "$root/metadata/$run_id/$name.time" \
      .venv/bin/vllm bench latency \
        --model "$model" \
        --dtype float16 \
        --seed 0 \
        --max-model-len 512 \
        --max-num-batched-tokens 512 \
        --max-num-seqs 8 \
        --gpu-memory-utilization 0.75 \
        --input-len "$input_len" \
        --output-len 1 \
        --batch-size "$batch" \
        --num-iters-warmup "$warmups" \
        --num-iters "$iterations" \
        --compilation-config "{\"cudagraph_mode\":\"NONE\",\"compile_sizes\":$sizes}" \
        --output-json "$root/results/$run_id/$name.json" \
        2>&1 | tee "$root/logs/$run_id/$name.log"
  )
done

echo "Completed: $run_id"
echo "Results: $root/results/$run_id"
echo "Logs: $root/logs/$run_id"
echo "Metadata: $root/metadata/$run_id"
