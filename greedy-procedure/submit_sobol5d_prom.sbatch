#!/usr/bin/env bash
# Run one batch's frozen-POD PROM at one 5D test point per array index.

#SBATCH --job-name=crm5d_prom
#SBATCH --partition=cfarhat
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=24
#SBATCH --time=04:00:00
#SBATCH --output=Sobol5DRuns/logs/prom_%A_%a.out
#SBATCH --error=Sobol5DRuns/logs/prom_%A_%a.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"
count=${1:?Usage: sbatch --array=1-32%3 submit_sobol5d_prom.sbatch COUNT}
index=${SLURM_ARRAY_TASK_ID:?Submit this script as an array, even for one index.}

[[ "${SLURM_NTASKS}" -eq 120 ]] || {
    echo "Expected 120 MPI tasks, received ${SLURM_NTASKS}." >&2
    exit 1
}

module purge
module load cmake/3.24.2 gcc/10.1.0 openmpi/4.1.2 imkl/2019

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

export SOWER=/home/groups/cfarhat/bin/sower
export AEROF=/scratch/users/sadpr/Code3Aug/aero-f/build-scalapack/bin/aerof.opt
export MPI=srun
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

[[ -x "${AEROF}" ]] || {
    echo "Missing AERO-F executable: ${AEROF}" >&2
    exit 1
}

python3 -B sobol5d_test.py prom --pod "${count}" --test-index "${index}"
