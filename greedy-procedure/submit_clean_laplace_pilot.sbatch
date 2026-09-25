#!/usr/bin/env bash
# Build the clean static data and run Sobol point 1 with the corrected shift.

#SBATCH --job-name=crm_clean_pilot
#SBATCH --partition=cfarhat
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=24
#SBATCH --time=04:00:00
#SBATCH --output=CleanLaplacePrecompute/pilot_%j.out
#SBATCH --error=CleanLaplacePrecompute/pilot_%j.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
repo_dir=$(cd "${work_dir}/.." && pwd)
cd "${work_dir}"

[[ "${SLURM_NTASKS}" -eq 120 ]] || {
    echo "Expected 120 MPI tasks, received ${SLURM_NTASKS}." >&2
    exit 1
}
[[ -f CleanLaplaceRuns/campaign.json ]] || {
    echo "Initialize the clean campaign before submitting this job." >&2
    exit 1
}
[[ ! -e CleanLaplacePrecompute/HDMrun001 ]] || {
    echo "Refusing to overwrite CleanLaplacePrecompute/HDMrun001." >&2
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

{
    printf 'job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'run_index=1\n'
    printf 'nodes=%s\n' "${SLURM_JOB_NUM_NODES}"
    printf 'tasks=%s\n' "${SLURM_NTASKS}"
    printf 'aerof=%s\n' "${AEROF}"
    (cd "${repo_dir}" && git rev-parse HEAD)
    module -t list 2>&1
    python3 --version
} > "CleanLaplacePrecompute/pilot_${SLURM_JOB_ID}.env"

python3 -B clean_laplace_initial.py prepare --run-index 1
python3 -B clean_laplace_initial.py run --run-index 1
