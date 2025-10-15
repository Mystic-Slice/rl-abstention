#!/bin/bash

#SBATCH --account=jieyuz_1727
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH -o logs/%x-%j.out

module purge
module load gcc/13.3.0
module load cuda/12.4.0
module load python/3.11.9

nvidia-smi || true

nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
python - <<'PY'
import torch, platform
print("GPU:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
print("cuda.is_available:", torch.cuda.is_available())
print("torch.version:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("bf16 matmul allowed:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
print("dtype test:", torch.tensor([1.0], device="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else "no cuda")
print("Python:", platform.python_version())
PY


python -u eval.py
