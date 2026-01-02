#!/bin/bash

# run mkdir -p logs/hpc to get the log info.

#SBATCH -J pf_decoupled_bp_train
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -p mit_preemptable
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH -t 24:00:00
#SBATCH -o /orcd/home/002/lerchen/code/cafa_proj/logs/hpc/pf_dc_bp_train%j.log
#SBATCH -e /orcd/home/002/lerchen/code/cafa_proj/logs/hpc/pf_dc_bp_train%j.err

set -euo pipefail

mkdir -p /orcd/home/002/lerchen/code/cafa_proj/pf-agcn-extension/logs/hpc

# module load deprecated-modules
# module load python/3.10.8-x86_64

source /orcd/home/002/lerchen/code/cafa_proj/pf_env/bin/activate

cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-20}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-20}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-20}
export HYDRA_FULL_ERROR=1

srun python -u /orcd/home/002/lerchen/code/cafa_proj/pf-agcn-extension/scripts/main.py train --config-path configs  --config-name hpc_full_config --aspect bp \
    experiment_name=pfagcn
