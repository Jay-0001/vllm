#!/usr/bin/env bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=m4-decision
#SBATCH --output=jay_outputs/m4_fullstack_decision/slurm-%j.out

set -Eeuo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch.' >&2; exit 2; }
cd "${SLURM_SUBMIT_DIR:?Submit from the M4 repository root}"
[[ -f jay_scripts/m4_fullstack_decision.sbatch ]] || exit 2
python_bin="${M4_PYTHON:-$PWD/.venv/bin/python}"
[[ -x "$python_bin" ]] || { echo 'Missing CUDA Python environment.' >&2; exit 2; }
command -v setsid >/dev/null || { echo 'setsid is required.' >&2; exit 2; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

root="$PWD/jay_outputs/m4_fullstack_decision/$SLURM_JOB_ID"
mkdir -p "$root"
{
  date -u
  git rev-parse HEAD
  git status --short
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  "$python_bin" -c 'import torch,triton,vllm; print(torch.__version__,triton.__version__,torch.version.cuda,vllm.__version__,vllm.__file__)'
} > "$root/environment.txt"

server_pid=""
sampler_pid=""
cleanup() {
  if [[ -n "$sampler_pid" ]]; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
    sampler_pid=""
  fi
  if [[ -n "$server_pid" ]]; then
    kill -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap cleanup EXIT

model="Qwen/Qwen2.5-7B-Instruct"
for utilization in 0.96 0.99; do
  if [[ "$utilization" == 0.96 ]]; then
    arms=(stock_default stock_eager stream_eager)
  else
    arms=(stream_eager stock_eager stock_default)
  fi
  for arm in "${arms[@]}"; do
    case_dir="$root/${utilization}_${arm}"
    mkdir -p "$case_dir"
    export M4_STREAM_QWEN_MLP=0
    extra=()
    if [[ "$arm" != stock_default ]]; then extra+=(--enforce-eager); fi
    if [[ "$arm" == stream_eager ]]; then export M4_STREAM_QWEN_MLP=1; fi

    printf 'starting\n' > "$case_dir/status.txt"
    echo "arm=$arm utilization=$utilization stream=$M4_STREAM_QWEN_MLP" \
      > "$case_dir/config.txt"
    launch_epoch=$(date +%s)
    setsid "$python_bin" -m vllm.entrypoints.cli.main serve "$model" \
      --dtype bfloat16 --tensor-parallel-size 1 \
      --max-model-len 4096 --max-num-batched-tokens 2048 \
      --max-num-seqs 16 --gpu-memory-utilization "$utilization" \
      --host 127.0.0.1 --port 8000 "${extra[@]}" \
      > "$case_dir/server.log" 2>&1 &
    server_pid=$!

    ready=0
    for ((second=0; second<900; second++)); do
      if "$python_bin" -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/health",timeout=1)' \
          >/dev/null 2>&1; then
        ready=1
        break
      fi
      if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
      sleep 1
    done
    if [[ "$ready" != 1 ]]; then
      printf 'server_failed\n' > "$case_dir/status.txt"
      cleanup
      continue
    fi
    printf 'server_ready\n' > "$case_dir/status.txt"
    printf 'startup_seconds=%s\n' "$(( $(date +%s) - launch_epoch ))" \
      >> "$case_dir/config.txt"

    for trace in long_prefill mixed; do
      if [[ "$trace" == long_prefill ]]; then
        input_len=2048; output_len=64; range_ratio=0.0
        prompts=96; rate=inf
      else
        input_len=1536; output_len=128; range_ratio=0.5
        prompts=128; rate=4
      fi
      nvidia-smi --query-gpu=memory.used \
        --format=csv,noheader,nounits --loop-ms=50 \
        > "$case_dir/${trace}_device_mib.txt" 2>/dev/null &
      sampler_pid=$!
      if "$python_bin" -m vllm.entrypoints.cli.main bench serve \
          --backend openai --host 127.0.0.1 --port 8000 \
          --model "$model" --dataset-name random \
          --random-input-len "$input_len" \
          --random-output-len "$output_len" \
          --random-range-ratio "$range_ratio" \
          --num-prompts "$prompts" --request-rate "$rate" \
          --max-concurrency 16 --num-warmups 4 \
          --seed 20261115 --temperature 0 --ignore-eos \
          --save-result --save-detailed --result-dir "$case_dir" \
          --result-filename "$trace.json" \
          > "$case_dir/${trace}_client.log" 2>&1; then
        printf '%s success\n' "$trace" >> "$case_dir/status.txt"
      else
        printf '%s failed\n' "$trace" >> "$case_dir/status.txt"
      fi
      kill "$sampler_pid" 2>/dev/null || true
      wait "$sampler_pid" 2>/dev/null || true
      sampler_pid=""
    done
    cleanup
    for ((second=0; second<60; second++)); do
      if ! "$python_bin" -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/health",timeout=1)' \
          >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    if "$python_bin" -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/health",timeout=1)' \
        >/dev/null 2>&1; then
      echo 'Previous server still owns port 8000; stopping.' >&2
      exit 3
    fi
  done
done

"$python_bin" jay_scripts/m4_fullstack_decide.py --root "$root" \
  > "$root/decision_stdout.txt"
