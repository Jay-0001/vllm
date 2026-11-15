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
#SBATCH --job-name=m4-stream
#SBATCH --output=jay_outputs/m4_stream_screen/slurm-%j.out

set -Eeuo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch.' >&2; exit 2; }
cd "${SLURM_SUBMIT_DIR:?Submit from the M4 repository root}"
[[ -f jay_scripts/m4_stream_screen.py ]] || {
  echo 'Submit from the M4 repository root.' >&2
  exit 2
}
python_bin="${M4_PYTHON:-$PWD/.venv/bin/python}"
[[ -x "$python_bin" ]] || { echo 'Missing CUDA Python environment.' >&2; exit 2; }

root="$PWD/jay_outputs/m4_stream_screen/$SLURM_JOB_ID"
for arm in stock stream; do
  case_dir="$root/$arm"
  mkdir -p "$case_dir/cache" "$case_dir/debug"
  export TORCHINDUCTOR_CACHE_DIR="$case_dir/cache"
  export TORCH_COMPILE_DEBUG=1
  export TORCH_COMPILE_DEBUG_DIR="$case_dir/debug"
  export TORCH_LOGS=graph_breaks,recompiles
  {
    date -u
    git rev-parse HEAD
    git branch --show-current
    echo "python_bin=$python_bin"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    "$python_bin" -c 'import torch,triton,vllm; print(torch.__version__,triton.__version__,torch.version.cuda,vllm.__version__)'
  } > "$case_dir/environment.txt"
  "$python_bin" jay_scripts/m4_stream_screen.py \
    --arm "$arm" --tokens 2048 --tile 2048 --repeats 25 \
    --output "$case_dir" 2>&1 | tee "$case_dir/run.log"
done
