#!/usr/bin/env bash
# Restart the steady solver from the final unsteady state of one HDM per array index, to see
# whether the steady solver keeps that state (Section 16 of the handoff).

#SBATCH --job-name=naca5d_resteady
#SBATCH --partition=cfarhat
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=24
#SBATCH --time=04:00:00
#SBATCH --output=Sobol5DRuns/logs/resteady_%A_%a.out
#SBATCH --error=Sobol5DRuns/logs/resteady_%A_%a.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
repo_dir=$(cd "${work_dir}/.." && pwd)
cd "${work_dir}"
index=${SLURM_ARRAY_TASK_ID:?Usage: sbatch --array=LIST submit_sobol5d_resteady.sbatch}

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

{
    printf 'job_id=%s\narray_index=%s\n' "${SLURM_JOB_ID}" "${index}"
    printf 'nodes=%s\naerof=%s\n' "${SLURM_JOB_NUM_NODES}" "${AEROF}"
    (cd "${repo_dir}" && git rev-parse HEAD)
    module -t list 2>&1
} > "Sobol5DRuns/logs/resteady_${SLURM_ARRAY_JOB_ID}_${index}.env"

python3 -B sobol5d_unsteady.py resteady --run-index "${index}"
