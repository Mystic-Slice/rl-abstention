#!/bin/bash

#SBATCH --account=jieyuz_1727
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH -o logs/%x-%j.out

module purge
module load gcc/13.3.0
module load cuda/12.2
module load python/3.11.9

nvidia-smi || true

nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
python - <<'PY'
import torch, platform
print("cuda.is_available:", torch.cuda.is_available())
print("torch.version:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("bf16 matmul allowed:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
print("dtype test:", torch.tensor([1.0], device="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else "no cuda")
print("Python:", platform.python_version())
PY


python -u train.py
