#!/usr/bin/env bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --job-name=vllm-7b-shape2048
#SBATCH --output=/u/jsaravananindira/vllm/jay_outputs/shape2048-%j.out

set -Eeuo pipefail

cd /u/jsaravananindira/vllm
source .venv/bin/activate

root="$PWD/specialization_experiments"
run_id="qwen7b-512x4-${SLURM_JOB_ID}"
mkdir -p "$root"/{results,logs,cache,metadata}
mkdir -p "$root/results/$run_id" "$root/logs/$run_id" \
         "$root/cache/$run_id" "$root/metadata/$run_id"

{
  date -u
  git rev-parse HEAD
  vllm --version
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  .venv/bin/python -c 'import torch; print("Visible CUDA GPUs:", torch.cuda.device_count()); assert torch.cuda.device_count() == 1'
} > "$root/metadata/$run_id/environment.txt"

run_case() {
  local label="$1"
  local arm="$2"
  local compile_sizes="$3"
  local output_len="$4"
  local warmups="$5"
  local iterations="$6"
  local cache_root="$root/cache/$run_id/$arm"

  mkdir -p "$cache_root"
  echo "Starting $label at $(date -u +%FT%TZ)"

  VLLM_CACHE_ROOT="$cache_root" TORCH_LOGS=recompiles \
    /usr/bin/time -f 'process_wall_seconds=%e' \
      -o "$root/metadata/$run_id/$label.time" \
    vllm bench latency \
      --model Qwen/Qwen2.5-7B-Instruct \
      --dtype bfloat16 \
      --seed 0 \
      --max-model-len 4096 \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 8 \
      --gpu-memory-utilization 0.80 \
      --input-len 512 \
      --output-len "$output_len" \
      --batch-size 4 \
      --num-iters-warmup "$warmups" \
      --num-iters "$iterations" \
      --compilation-config "{\"cudagraph_mode\":\"NONE\",\"compile_sizes\":$compile_sizes}" \
      --output-json "$root/results/$run_id/$label.json" \
      2>&1 | tee "$root/logs/$run_id/$label.log"
}

# Prefill-focused measurement. The reverse-order pair reuses each arm's cache.
run_case prefill_dynamic_a dynamic '[]'     1 10 40
run_case prefill_spec_a    spec2048 '[2048]' 1 10 40
run_case prefill_spec_b    spec2048 '[2048]' 1 10 40
run_case prefill_dynamic_b dynamic '[]'     1 10 40

# Same prompts and batch, now with substantial decode work.
run_case generation_dynamic_a dynamic '[]'     128 5 20
run_case generation_spec_a    spec2048 '[2048]' 128 5 20
run_case generation_spec_b    spec2048 '[2048]' 128 5 20
run_case generation_dynamic_b dynamic '[]'     128 5 20

echo "Results: $root/results/$run_id"
echo "Logs: $root/logs/$run_id"
echo "Compilation caches: $root/cache/$run_id"