#!/usr/bin/env bash
# Run initial Sobol HDM samples 2--32; each task owns one output directory.

#SBATCH --job-name=crm_init_hdm
#SBATCH --partition=cfarhat
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=24
#SBATCH --time=04:00:00
#SBATCH --array=2-32%3
#SBATCH --output=InitialHDMruns/campaign_%A_%a.out
#SBATCH --error=InitialHDMruns/campaign_%A_%a.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
repo_dir=$(cd "${work_dir}/.." && pwd)
cd "${work_dir}"
run_index=${SLURM_ARRAY_TASK_ID:?Missing Slurm array index.}

if (( run_index < 2 || run_index > 32 )); then
    echo "Expected a Sobol index in [2, 32], received ${run_index}." >&2
    exit 2
fi
[[ -f GreedyRuns/data/fluidmodel.con ]] || {
    echo "Missing pilot-prepared static CRM data." >&2
    exit 1
}
[[ -f GreedyRuns/data/fluidmodel.msh120 ]] || {
    echo "Missing 120-way pilot mesh decomposition." >&2
    exit 1
}
[[ ! -e "InitialHDMruns/HDMrun$(printf '%03d' "${run_index}")" ]] || {
    echo "Refusing to overwrite InitialHDMruns/HDMrun$(printf '%03d' "${run_index}")." >&2
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

export CRM_CONDA_BASE=/scratch/users/sadpr/Code3Aug/miniconda3
export CRM_LAPLACE_ENV=CRM_Laplace
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
    printf 'array_index=%s\n' "${run_index}"
    printf 'nodes=%s\n' "${SLURM_JOB_NUM_NODES}"
    printf 'tasks=%s\n' "${SLURM_NTASKS}"
    printf 'aerof=%s\n' "${AEROF}"
    (cd "${repo_dir}" && git rev-parse HEAD)
    module -t list 2>&1
    python3 --version
} > "InitialHDMruns/campaign_${SLURM_JOB_ID}_${run_index}.env"

python3 -B initial_hdm_campaign.py prepare --run-index "${run_index}"
python3 -B initial_hdm_campaign.py run --run-index "${run_index}"
