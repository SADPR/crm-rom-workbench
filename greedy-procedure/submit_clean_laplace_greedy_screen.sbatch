#!/usr/bin/env bash
# Full-PROM residual screen for one clean Laplace greedy iteration.

#SBATCH --job-name=crm_greedy_screen
#SBATCH --partition=cfarhat
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=24
#SBATCH --time=02:00:00
#SBATCH --output=CleanLaplaceGreedy40/screening/screen_%A_%a.out
#SBATCH --error=CleanLaplaceGreedy40/screening/screen_%A_%a.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"
iteration=${GREEDY_ITERATION:?GREEDY_ITERATION is required.}
candidate_number=${SLURM_ARRAY_TASK_ID:?Missing Slurm array index.}

[[ "${SLURM_NTASKS}" -eq 120 ]] || {
    echo "Expected 120 MPI tasks, received ${SLURM_NTASKS}." >&2
    exit 1
}
[[ -f "CleanLaplaceGreedy40/screening/iteration$(printf '%03d' "${iteration}")/candidates.json" ]] || {
    echo "Initialize the candidate catalog before submitting the screen array." >&2
    exit 1
}

module purge
module load cmake/3.24.2 gcc/10.1.0 openmpi/4.1.2 imkl/2019

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

export SOWER=/home/groups/cfarhat/bin/sower
export PARTMESH=/home/groups/cfarhat/bin/partnmesh
export CD2TET=/home/groups/cfarhat/bin/cd2tet
export CRM_CONDA_BASE=/scratch/users/sadpr/Code3Aug/miniconda3
export CRM_LAPLACE_ENV=CRM_Laplace
export AEROF=/scratch/users/sadpr/Code3Aug/aero-f/build-scalapack/bin/aerof.opt
export MPI=srun
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

[[ -x "${AEROF}" ]] || {
    echo "Missing AERO-F executable: ${AEROF}" >&2
    exit 1
}

python3 -B clean_laplace_greedy.py screen \
    --iteration "${iteration}" --candidate-number "${candidate_number}"
