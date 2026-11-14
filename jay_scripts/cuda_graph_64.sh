    #!/usr/bin/env bash
    #SBATCH --account=bcpt-delta-gpu
    #SBATCH --partition=gpuA100x4-interactive
    #SBATCH --qos=bcpt-delta-gpu
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --gpus-per-node=1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=64G
    #SBATCH --time=00:10:00
    #SBATCH --job-name=vllm-cg64
    #SBATCH --output=/u/jsaravananindira/vllm/jay_outputs/slurm-cg64-%j.out
    #SBATCH --error=/u/jsaravananindira/vllm/jay_outputs/slurm-cg64-%j.err

set -Eeuo pipefail

repo=/u/jsaravananindira/vllm
root="$repo/specialization_experiments"
model="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
warmups="${WARMUPS:-15}"
iterations="${ITERATIONS:-60}"

cd "$repo"
[[ -x .venv/bin/vllm ]] || {
  echo 'Missing .venv/bin/vllm in checkout.' >&2
  exit 1
}

command -v nvidia-smi >/dev/null
command -v /usr/bin/time >/dev/null

run_id="cg-piecewise64-$(date -u +%Y%m%dT%H%M%SZ)-job${SLURM_JOB_ID}"
mkdir -p \
  "$root/results/$run_id" \
  "$root/logs/$run_id" \
  "$root/metadata/$run_id"

{
  echo "run_id=$run_id"
  echo "model=$model"
  echo "warmups=$warmups"
  echo "iterations=$iterations"
  echo "cudagraph_mode=PIECEWISE"
  echo "cudagraph_capture_sizes=[64]"
  echo "git_head=$(git rev-parse HEAD)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  .venv/bin/python -c 'import torch, vllm; print("torch=" + torch.__version__); print("vllm=" + vllm.__version__); print("visible_gpu_count=" + str(torch.cuda.device_count()))'
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
} > "$root/metadata/$run_id/environment.txt"

[[ "$(.venv/bin/python -c 'import torch; print(torch.cuda.device_count())')" == 1 ]] || {
  echo 'Expected exactly one visible CUDA GPU.' >&2
  exit 1
}

cases=(
  'dynamic|[]|64_b1_a'
  'spec64|[64]|64_b1_a'
  'spec64|[64]|64_b1_b'
  'dynamic|[]|64_b1_b'
)

printf 'name\tarm\tcompile_sizes\tcache_root\n' \
  > "$root/metadata/$run_id/cases.tsv"

for row in "${cases[@]}"; do
  IFS='|' read -r arm sizes label <<< "$row"
  name="${arm}_${label}"
  cache_root="$root/cache/$run_id/$arm"
  mkdir -p "$cache_root"

  printf '%s\t%s\t%s\t%s\n' \
    "$name" "$arm" "$sizes" "$cache_root" \
    >> "$root/metadata/$run_id/cases.tsv"

  echo "Running $name at $(date -u +%FT%TZ)"

  (
    export VLLM_CACHE_ROOT="$cache_root"
    export TORCH_LOGS=recompiles

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
        --input-len 64 \
        --output-len 1 \
        --batch-size 1 \
        --num-iters-warmup "$warmups" \
        --num-iters "$iterations" \
        --compilation-config "{\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[64],\"compile_sizes\":$sizes}" \
        --output-json "$root/results/$run_id/$name.json" \
        2>&1 | tee "$root/logs/$run_id/$name.log"
  )
done

echo "Completed: $run_id"
echo "Results: $root/results/$run_id"
echo "Logs: $root/logs/$run_id"
echo "Metadata: $root/metadata/$run_id"