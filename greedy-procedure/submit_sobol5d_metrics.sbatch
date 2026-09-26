#!/usr/bin/env bash
# Merge surface fields and compute one batch's PROM errors over the 5D test set.

#SBATCH --job-name=crm5d_metrics
#SBATCH --partition=cfarhat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=Sobol5DRuns/logs/metrics_%j.out
#SBATCH --error=Sobol5DRuns/logs/metrics_%j.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"
count=${1:?Usage: sbatch submit_sobol5d_metrics.sbatch COUNT}

module purge
module load cmake/3.24.2 gcc/10.1.0 openmpi/4.1.2 imkl/2019

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

export SOWER=/home/groups/cfarhat/bin/sower

python3 -B sobol5d_test.py metrics --pod "${count}"
python3 -B sobol5d_test.py summary
