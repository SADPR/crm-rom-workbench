#!/usr/bin/env bash
# Run the prepared first non-nominal steady HDM sample.

#SBATCH --job-name=crm_hdm_pilot
#SBATCH --partition=cfarhat
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=24
#SBATCH --time=04:00:00
#SBATCH --output=InitialHDMruns/pilot_%j.out
#SBATCH --error=InitialHDMruns/pilot_%j.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
repo_dir=$(cd "${work_dir}/.." && pwd)
cd "${work_dir}"

run_dir=InitialHDMruns/HDMrun001
[[ -f "${run_dir}/input1" && -f "${run_dir}/input2" ]] || {
    echo "Missing prepared pilot inputs; run initial_hdm_pilot.py prepare first." >&2
    exit 1
}
[[ -f "${run_dir}/Laplace-bin/ushift.bin001" && -f "${run_dir}/Laplace-bin/ushift.bin120" ]] || {
    echo "Missing the 120-part pilot Laplace shift." >&2
    exit 1
}
[[ "${SLURM_NTASKS}" -eq 120 ]] || {
    echo "Expected 120 MPI tasks, received ${SLURM_NTASKS}." >&2
    exit 1
}

module purge
module load cmake/3.24.2 gcc/10.1.0 openmpi/4.1.2 imkl/2019

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

export AEROF=/scratch/users/sadpr/Code3Aug/aero-f/build-core/bin/aerof.opt
export MPI=srun
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

[[ -x "${AEROF}" ]] || {
    echo "Missing AERO-F executable: ${AEROF}" >&2
    exit 1
}

{
    printf 'job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'nodes=%s\n' "${SLURM_JOB_NUM_NODES}"
    printf 'tasks=%s\n' "${SLURM_NTASKS}"
    printf 'aerof=%s\n' "${AEROF}"
    # Sherlock's Git does not support -C; enter the repository explicitly.
    (cd "${repo_dir}" && git rev-parse HEAD)
    module -t list 2>&1
    python3 --version
} > "${run_dir}/job_${SLURM_JOB_ID}.env"

python3 -B initial_hdm_pilot.py run
