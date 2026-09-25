#!/usr/bin/env bash
# Recompute the nominal CRM baseline with the corrected Laplace shift.

#SBATCH --job-name=crm_clean_base
#SBATCH --partition=cfarhat
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=24
#SBATCH --time=04:00:00
#SBATCH --output=CleanLaplacePrecompute/baseline_%j.out
#SBATCH --error=CleanLaplacePrecompute/baseline_%j.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"

[[ "${SLURM_NTASKS}" -eq 120 ]] || {
    echo "Expected 120 MPI tasks, received ${SLURM_NTASKS}." >&2
    exit 1
}
[[ -f CleanLaplaceRuns/campaign.json ]] || {
    echo "Initialize the clean campaign before submitting this baseline." >&2
    exit 1
}
[[ ! -e CleanLaplaceBaseline ]] || {
    echo "Refusing to overwrite CleanLaplaceBaseline." >&2
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

python3 -B clean_laplace_baseline.py prepare
python3 -B clean_laplace_baseline.py run
