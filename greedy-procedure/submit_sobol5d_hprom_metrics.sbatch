#!/usr/bin/env bash
# Compare every HPROM of one batch with the HDM and the reference PROM.

#SBATCH --job-name=naca5d_hprom_metrics
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=Sobol5DRuns/logs/hprom_metrics_%j.out
#SBATCH --error=Sobol5DRuns/logs/hprom_metrics_%j.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"
count=${1:?Usage: sbatch submit_sobol5d_hprom_metrics.sbatch COUNT [NAME]}
name=${2:-hprom}

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

python3 -B sobol5d_hprom.py metrics --pod "${count}" --name "${name}"
