#!/usr/bin/env bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=vllm-shape-dist
#SBATCH --output=/u/jsaravananindira/vllm/jay_outputs/shape-dist-%j.out
#SBATCH --error=/u/jsaravananindira/vllm/jay_outputs/shape-dist-%j.err

set -Eeuo pipefail

repo=/u/jsaravananindira/vllm
root="$repo/specialization_experiments"
model="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
request_rate="${REQUEST_RATE:-1.0}"
num_prompts="${NUM_PROMPTS:-128}"
port=$((20000 + SLURM_JOB_ID % 20000))
run_id="shape-dist-$(date -u +%Y%m%dT%H%M%SZ)-job${SLURM_JOB_ID}"

cd "$repo"
source .venv/bin/activate

server_log="$root/logs/$run_id/server.log"
client_log="$root/logs/$run_id/client.log"
export VLLM_CACHE_ROOT="$root/cache/$run_id"

{
  echo "run_id=$run_id"
  echo "model=$model"
  echo "request_rate=$request_rate"
  echo "num_prompts=$num_prompts"
  echo "git_head=$(git rev-parse HEAD)"
  echo "vllm=$(vllm --version)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
} > "$root/metadata/$run_id/environment.txt"

server_pid=
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

vllm serve "$model" \
  --host 127.0.0.1 \
  --port "$port" \
  --dtype float16 \
  --seed 0 \
  --max-model-len 2048 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.80 \
  --enable-logging-iteration-details \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","compile_sizes":[]}' \
  > "$server_log" 2>&1 &
server_pid=$!

ready=0
for attempt in {1..240}; do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -80 "$server_log" >&2
    exit 1
  fi
  sleep 2
done
[[ "$ready" == 1 ]] || {
  echo "Server did not become ready." >&2
  tail -80 "$server_log" >&2
  exit 1
}

vllm bench serve \
  --backend vllm \
  --host 127.0.0.1 \
  --port "$port" \
  --endpoint /v1/completions \
  --model "$model" \
  --dataset-name random \
  --random-input-len 256 \
  --random-output-len 64 \
  --random-range-ratio 0.5 \
  --request-rate "$request_rate" \
  --burstiness 1.0 \
  --max-concurrency 16 \
  --num-prompts "$num_prompts" \
  --seed 0 \
  --ignore-eos \
  --save-result \
  --result-dir "$root/results/$run_id" \
  --result-filename serve.json \
  > "$client_log" 2>&1

sleep 2
cleanup
server_pid=

python - "$server_log" \
  "$root/results/$run_id/iterations.csv" \
  "$root/results/$run_id/shape_histogram.csv" <<'PY'
import csv
import re
import sys
from collections import Counter

pattern = re.compile(
    r"Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, "
    r"(\d+) generation requests, (\d+) generation tokens, "
    r"iteration elapsed time: ([\d.]+) ms"
)
rows = []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if "(dummy)" in line:
        continue
    match = pattern.search(line)
    if not match:
        continue
    iteration, ctx_req, ctx_tok, gen_req, gen_tok, cpu_ms = match.groups()
    ctx_req, ctx_tok, gen_req, gen_tok = map(
        int, (ctx_req, ctx_tok, gen_req, gen_tok)
    )
    phase = "mixed" if ctx_req and gen_req else (
        "prefill" if ctx_req else "decode"
    )
    rows.append((
        int(iteration), phase, ctx_req, ctx_tok,
        gen_req, gen_tok, ctx_tok + gen_tok, float(cpu_ms)
    ))

if not rows:
    raise SystemExit("No iteration details found; inspect server.log")

with open(sys.argv[2], "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow((
        "iteration", "phase", "context_requests", "context_tokens",
        "generation_requests", "generation_tokens",
        "scheduled_tokens", "logged_cpu_ms"
    ))
    writer.writerows(rows)

counts = Counter((row[1], row[6]) for row in rows)
with open(sys.argv[3], "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(("phase", "scheduled_tokens", "iterations"))
    for (phase, tokens), count in sorted(counts.items()):
        writer.writerow((phase, tokens, count))

print(f"Recorded {len(rows)} scheduler iterations")
print("Most frequent phase/shape pairs:")
for (phase, tokens), count in counts.most_common(20):
    print(f"  {phase:7s} {tokens:4d} tokens: {count} iterations")
PY

echo "Results: $root/results/$run_id"
echo "Logs: $root/logs/$run_id"