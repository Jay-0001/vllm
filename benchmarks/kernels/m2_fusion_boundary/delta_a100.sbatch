#!/usr/bin/env bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=m2-fusion-screen
#SBATCH --output=/u/jsaravananindira/vllm/jay_outputs/m2-screen-%j.out

set -Eeuo pipefail

cd /u/jsaravananindira/vllm
source .venv/bin/activate

model_size="${MODEL_SIZE:-0.5b}"
tokens_list="${TOKENS:-1 4 64}"
repeats="${REPEATS:-1}"
profile="${PROFILE:-1}"
arms_list="${ARMS:-stock split_norm split_activation benchmark_fusion}"
root="$PWD/m2_experiments"
run_id="${SLURM_JOB_ID}"
mkdir -p "$root/results/$run_id" "$root/logs/$run_id" \
  "$root/cache/$run_id" "$root/metadata/$run_id"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
[[ "$gpu_name" == *A100* ]] || {
  echo "Expected an A100, got: $gpu_name" >&2
  exit 1
}

{
  date -u
  git rev-parse HEAD
  git branch --show-current
  sha256sum benchmarks/kernels/m2_fusion_boundary/screen.py
  .venv/bin/python -c 'import torch,triton; print("torch",torch.__version__,"CUDA",torch.version.cuda,"triton",triton.__version__); print("visible GPUs",torch.cuda.device_count()); assert torch.cuda.device_count()==1'
  vllm --version
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
} > "$root/metadata/$run_id/environment.txt"

benchmark_fusion_available=1
if ! .venv/bin/python -c 'from torch._inductor import config; assert hasattr(config, "benchmark_fusion")' \
  >/dev/null 2>&1; then
  benchmark_fusion_available=0
  echo "Installed PyTorch lacks benchmark_fusion; omitting that arm" \
    | tee -a "$root/metadata/$run_id/environment.txt"
fi

run_case() {
  local tokens="$1"
  local arm="$2"
  local repeat="$3"
  local label="${model_size}-t${tokens}-${arm}-r${repeat}"
  local cache="$root/cache/$run_id/$label"
  mkdir -p "$cache/inductor" "$cache/triton"
  local fusion=0
  [[ "$arm" == "benchmark_fusion" ]] && fusion=1
  local profile_arg=()
  [[ "$profile" == 1 ]] && profile_arg=(--profile)
  echo "Starting $label at $(date -u +%FT%TZ)"
  TORCHINDUCTOR_CACHE_DIR="$cache/inductor" \
  TRITON_CACHE_DIR="$cache/triton" \
  TORCH_COMPILE_DEBUG_DIR="$cache/debug" \
  TORCHINDUCTOR_BENCHMARK_FUSION="$fusion" \
  TORCH_COMPILE_DEBUG=1 TORCH_LOGS=fusion,recompiles \
    .venv/bin/python benchmarks/kernels/m2_fusion_boundary/screen.py \
      --model-size "$model_size" --tokens "$tokens" --arm "$arm" \
      --output "$root/results/$run_id/$label.json" \
      "${profile_arg[@]}" \
      > "$root/logs/$run_id/$label.log" 2>&1
  cat "$root/results/$run_id/$label.json"
}

for repeat in $(seq 1 "$repeats"); do
  for tokens in $tokens_list; do
    read -r -a arms <<< "$arms_list"
    if (( ! benchmark_fusion_available )); then
      filtered=()
      for arm in "${arms[@]}"; do
        [[ "$arm" == benchmark_fusion ]] || filtered+=("$arm")
      done
      arms=("${filtered[@]}")
    fi
    (( ${#arms[@]} > 0 )) || { echo "No arms selected" >&2; exit 1; }
    if (( repeat % 2 == 0 )); then
      reversed=()
      for (( i=${#arms[@]}-1; i>=0; i-- )); do
        reversed+=("${arms[i]}")
      done
      arms=("${reversed[@]}")
    fi
    for arm in "${arms[@]}"; do
      run_case "$tokens" "$arm" "$repeat"
    done
  done
done

echo "Results: $root/results/$run_id"
echo "Logs: $root/logs/$run_id"
echo "Compiler caches: $root/cache/$run_id"
