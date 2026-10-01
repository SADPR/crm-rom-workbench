#!/usr/bin/env bash
# Collect the results of one batch's start-by-form sweep.

#SBATCH --job-name=naca5d_sweep_metrics
#SBATCH --partition=cfarhat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=Sobol5DRuns/logs/sweep_metrics_%j.out
#SBATCH --error=Sobol5DRuns/logs/sweep_metrics_%j.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"
count=${1:?Usage: sbatch submit_sobol5d_sweep_metrics.sbatch COUNT [NAME]}
name=${2:-sweep}

module purge
module load cmake/3.24.2 gcc/10.1.0 openmpi/4.1.2 imkl/2019

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

# Merges a training HDM's wall Cp if its metrics never did.
export SOWER=/home/groups/cfarhat/bin/sower

python3 -B sobol5d_sweep.py metrics --pod "${count}" --name "${name}"
