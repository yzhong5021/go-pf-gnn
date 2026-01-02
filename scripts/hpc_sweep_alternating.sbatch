#!/bin/bash

# run mkdir -p logs/hpc to get the log info.

#SBATCH -J pf_alternating_sweep
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -p mit_normal_gpu
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 6:00:00
#SBATCH -o /orcd/home/002/lerchen/code/cafa_proj/logs/pf_at_sweep%j.log
#SBATCH -e /orcd/home/002/lerchen/code/cafa_proj/logs/pf_at_sweep%j.err

set -euo pipefail

mkdir -p logs/hpc

module load deprecated-modules
module load python/3.10.8-x86_64

source /orcd/home/002/lerchen/code/cafa_proj/pf_env/bin/activate

cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-20}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-20}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-20}
export HYDRA_FULL_ERROR=1

for ASPECT in MF BP CC; do
  echo "=== Starting training for ASPECT=${ASPECT} ==="
  srun python -u /orcd/home/002/lerchen/code/cafa_proj/pf-agcn-extension/scripts/main.py train \
    --config-path configs \
    --config-name hpc_full_config \
    --aspect "${ASPECT}" \
    --multirun \
    experiment_name=pfagcn_supercloud_test \
    run_name=pfagcn_supercloud_test_alternating_${ASPECT} \ 
    model.graph.structure=alternating
done
