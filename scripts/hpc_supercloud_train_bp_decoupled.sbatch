#!/bin/bash

# run mkdir -p logs/hpc to get the log info.

#SBATCH -J pfagcn
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -p mit_normal_gpu
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH -t 6:00:00
#SBATCH -o logs/hpc/pfagcn_%j.log
#SBATCH -e logs/hpc/pfagcn_%j.err

set -euo pipefail

mkdir -p logs/hpc

module load deprecated-modules
module load python/3.10.8-x86_64

source pfagcn_env/bin/activate

cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-20}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-20}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-20}
export HYDRA_FULL_ERROR=1

srun python -u scripts/main.py train --config-path configs  --config-name hpc_full_config --aspect bp \
    experiment_name=pfagcn_supercloud_1