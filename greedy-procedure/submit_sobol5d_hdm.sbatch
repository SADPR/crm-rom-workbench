#!/usr/bin/env bash
# Prepare and run one 5D HDM per array index, for the training design or the test set.

#SBATCH --job-name=crm5d_hdm
#SBATCH --partition=cfarhat
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=24
#SBATCH --time=06:00:00
#SBATCH --output=Sobol5DRuns/logs/hdm_%A_%a.out
#SBATCH --error=Sobol5DRuns/logs/hdm_%A_%a.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
repo_dir=$(cd "${work_dir}/.." && pwd)
cd "${work_dir}"
set_name=${1:?Usage: sbatch --array=LIST%3 submit_sobol5d_hdm.sbatch train|test}
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
    printf 'job_id=%s\narray_index=%s\nset=%s\n' "${SLURM_JOB_ID}" "${index}" "${set_name}"
    printf 'nodes=%s\naerof=%s\n' "${SLURM_JOB_NUM_NODES}" "${AEROF}"
    (cd "${repo_dir}" && git rev-parse HEAD)
    module -t list 2>&1
} > "Sobol5DRuns/logs/hdm_${SLURM_ARRAY_JOB_ID}_${index}.env"

case "${set_name}" in
    train)
        python3 -B sobol5d_campaign.py prepare --run-index "${index}"
        python3 -B sobol5d_campaign.py run --run-index "${index}"
        ;;
    test)
        python3 -B sobol5d_test.py prepare-hdm --test-index "${index}"
        python3 -B sobol5d_test.py run-hdm --test-index "${index}"
        ;;
    *)
        echo "Unknown set ${set_name}; use train or test." >&2
        exit 2
        ;;
esac
