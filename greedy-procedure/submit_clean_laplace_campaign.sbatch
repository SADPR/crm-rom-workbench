#!/usr/bin/env bash
# Run clean Sobol points 2--32; at most three 5-node HDMs run concurrently.

#SBATCH --job-name=crm_clean_hdm
#SBATCH --partition=cfarhat
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=24
#SBATCH --time=04:00:00
#SBATCH --array=2-32%3
#SBATCH --output=CleanLaplacePrecompute/campaign_%A_%a.out
#SBATCH --error=CleanLaplacePrecompute/campaign_%A_%a.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
repo_dir=$(cd "${work_dir}/.." && pwd)
cd "${work_dir}"
run_index=${SLURM_ARRAY_TASK_ID:?Missing Slurm array index.}

if (( run_index < 2 || run_index > 32 )); then
    echo "Expected a Sobol index in [2, 32], received ${run_index}." >&2
    exit 2
fi
[[ "${SLURM_NTASKS}" -eq 120 ]] || {
    echo "Expected 120 MPI tasks, received ${SLURM_NTASKS}." >&2
    exit 1
}
[[ -f CleanLaplaceRuns/campaign.json ]] || {
    echo "Initialize the clean campaign before submitting this array." >&2
    exit 1
}
[[ -f CleanLaplaceRuns/data/fluidmodel.con ]] || {
    echo "The clean pilot must complete before this array starts." >&2
    exit 1
}
[[ ! -e "CleanLaplacePrecompute/HDMrun$(printf '%03d' "${run_index}")" ]] || {
    echo "Refusing to overwrite CleanLaplacePrecompute/HDMrun$(printf '%03d' "${run_index}")." >&2
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
    printf 'array_index=%s\n' "${run_index}"
    printf 'nodes=%s\n' "${SLURM_JOB_NUM_NODES}"
    printf 'tasks=%s\n' "${SLURM_NTASKS}"
    printf 'aerof=%s\n' "${AEROF}"
    (cd "${repo_dir}" && git rev-parse HEAD)
    module -t list 2>&1
    python3 --version
} > "CleanLaplacePrecompute/campaign_${SLURM_JOB_ID}_${run_index}.env"

python3 -B clean_laplace_initial.py prepare --run-index "${run_index}"
python3 -B clean_laplace_initial.py run --run-index "${run_index}"
