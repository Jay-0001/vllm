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
#SBATCH --array=0-5%1
#SBATCH --job-name=m4-peak
#SBATCH --output=jay_outputs/m4_peak_screen/slurm-%A_%a.out

set -Eeuo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch.' >&2; exit 2; }
cd "${SLURM_SUBMIT_DIR:?Submit from the M4 repository root}"
[[ -f jay_scripts/m4_peak_screen.py ]] || {
  echo 'Submit from the M4 repository root.' >&2
  exit 2
}
python_bin="${M4_PYTHON:-$PWD/.venv/bin/python}"
[[ -x "$python_bin" ]] || { echo 'Missing CUDA Python environment.' >&2; exit 2; }

case "${SLURM_ARRAY_TASK_ID:?}" in
  0) tokens=64; arm=stock ;;
  1) tokens=64; arm=tiled ;;
  2) tokens=512; arm=stock ;;
  3) tokens=512; arm=tiled ;;
  4) tokens=2048; arm=stock ;;
  5) tokens=2048; arm=tiled ;;
  *) echo 'Invalid array index.' >&2; exit 2 ;;
esac

root="$PWD/jay_outputs/m4_peak_screen"
case_dir="$root/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${tokens}_${arm}"
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
  "$python_bin" -c 'import torch,triton; print("torch",torch.__version__,"triton",triton.__version__,"cuda",torch.version.cuda)'
  "$python_bin" -c 'import vllm; print("vllm",vllm.__version__,"path",vllm.__file__)'
} > "$case_dir/environment.txt"

exec "$python_bin" jay_scripts/m4_peak_screen.py \
  --tokens "$tokens" --arm "$arm" --output "$case_dir"
