#!/usr/bin/env bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:05:00
#SBATCH --job-name=m2-7b-shapes
#SBATCH --output=/u/jsaravananindira/vllm/jay_outputs/m2_qwen7b_last_chance/burst-%j.out

set -Eeuo pipefail

repo=/u/jsaravananindira/vllm
cd "$repo"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch.' >&2; exit 2; }
root="$repo/jay_outputs/m2_qwen7b_last_chance"
output="$root/burst_shapes_${SLURM_JOB_ID}"
mkdir -p "$output/cache"
export VLLM_CACHE_ROOT="$output/cache"
port=$((20000 + SLURM_JOB_ID % 20000))
server_pid=
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
{
  date -u +%FT%TZ
  git rev-parse HEAD
  .venv/bin/python -c 'import torch, vllm; print(torch.__version__, vllm.__version__)'
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
} > "$output/environment.txt"

.venv/bin/vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 --port "$port" --dtype bfloat16 --seed 0 \
  --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 32 \
  --gpu-memory-utilization 0.80 --enable-logging-iteration-details \
  --compilation-config '{"cudagraph_mode":"NONE","compile_sizes":[]}' \
  > "$output/server.log" 2>&1 &
server_pid=$!
ready=0
for _ in {1..90}; do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -60 "$output/server.log" >&2
    exit 1
  fi
  sleep 2
done
[[ "$ready" == 1 ]] || { echo 'Server did not become ready.' >&2; exit 1; }

/usr/bin/time -f 'client_wall_seconds=%e' -o "$output/client.time" \
  .venv/bin/vllm bench serve --backend vllm --host 127.0.0.1 \
    --port "$port" --endpoint /v1/completions \
    --model Qwen/Qwen2.5-7B-Instruct --dataset-name random \
    --random-input-len 256 --random-output-len 64 \
    --random-range-ratio 0.5 --request-rate 16 --burstiness 1.0 \
    --max-concurrency 32 --num-prompts 128 --seed 0 --ignore-eos \
    --save-result --result-dir "$output" --result-filename serve.json \
    > "$output/client.log" 2>&1

.venv/bin/python \
  benchmarks/kernels/m2_fusion_boundary/summarize_qk_shape_histogram.py \
  --log "$output/server.log" --output "$output" \
  > "$output/histogram.log"
printf 'Results: %s\n' "$output"
