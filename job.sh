#!/bin/bash

#SBATCH --account=jieyuz_1727
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH -o logs/%x-%j.out

module purge
module load gcc/13.3.0
module load python/3.11.9

export CUDA_VISIBLE_DEVICES=""

python -u train.py
