#!/usr/bin/env bash
# Copy one batch's POD to the HPROM working directory and freeze its ECSW training cases.

#SBATCH --job-name=naca5d_hprom_prep
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=Sobol5DRuns/logs/hprom_prepare_%j.out
#SBATCH --error=Sobol5DRuns/logs/hprom_prepare_%j.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"
count=${1:?Usage: sbatch submit_sobol5d_hprom_prepare.sbatch COUNT [CASES] [NAME] [ROWS_FROM]}
cases=${2:-32}
name=${3:-hprom}
# ROWS_FROM: another working directory whose training rows this one reuses (e.g. hprom).
rows_from=${4:-}

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

python3 -B sobol5d_hprom.py prepare --pod "${count}" --cases "${cases}" --name "${name}" \
    ${rows_from:+--rows-from "${rows_from}"}
