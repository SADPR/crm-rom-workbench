#!/usr/bin/env bash
# Run the frozen 32-state clean CRM PROM at the independent holdout point.

#SBATCH --job-name=crm_holdout_prom
#SBATCH --partition=cfarhat
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=24
#SBATCH --time=01:00:00
#SBATCH --output=clean_holdout_prom_%j.out
#SBATCH --error=clean_holdout_prom_%j.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"

[[ "${SLURM_NTASKS}" -eq 120 ]] || {
    echo "Expected 120 MPI tasks, received ${SLURM_NTASKS}." >&2
    exit 1
}

module purge
module load cmake/3.24.2 gcc/10.1.0 openmpi/4.1.2 imkl/2019

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

export AEROF=/scratch/users/sadpr/Code3Aug/aero-f/build-scalapack/bin/aerof.opt
export SOWER=/home/groups/cfarhat/bin/sower
export CRM_CONDA_BASE=/scratch/users/sadpr/Code3Aug/miniconda3
export CRM_LAPLACE_ENV=CRM_Laplace
export MPI=srun
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 -B clean_laplace_holdout.py prepare-prom
python3 -B clean_laplace_holdout.py run-prom
