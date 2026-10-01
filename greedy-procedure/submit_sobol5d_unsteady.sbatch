#!/usr/bin/env bash
# Run one unsteady (URANS) HDM per array index, from the steady state of that training HDM.

#SBATCH --job-name=crm5d_unst
#SBATCH --partition=cfarhat
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=24
#SBATCH --time=12:00:00
# AERO-F ends a run cleanly on SIGUSR1 (Main.C, TsDesc::outputToDisk): it writes the final
# fields and restart files, so a run cut by the time limit keeps its last state.
#SBATCH --signal=USR1@900
#SBATCH --output=Sobol5DRuns/logs/unsteady_%A_%a.out
#SBATCH --error=Sobol5DRuns/logs/unsteady_%A_%a.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
repo_dir=$(cd "${work_dir}/.." && pwd)
cd "${work_dir}"
# Extra arguments, e.g. --dt 5e-5 --max-time 0.2 --newton 5, go to prepare.
index=${SLURM_ARRAY_TASK_ID:?Usage: sbatch --array=LIST%4 submit_sobol5d_unsteady.sbatch [--dt S --max-time S --newton N]}

[[ "${SLURM_NTASKS}" -eq 120 ]] || {
    echo "Expected 120 MPI tasks, received ${SLURM_NTASKS}." >&2
    exit 1
}

module purge
module load cmake/3.24.2 gcc/10.1.0 openmpi/4.1.2 imkl/2019

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

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
} > "Sobol5DRuns/logs/unsteady_${SLURM_ARRAY_JOB_ID}_${index}.env"

python3 -B sobol5d_unsteady.py prepare --run-index "${index}" "$@"
python3 -B sobol5d_unsteady.py run --run-index "${index}"
python3 -B sobol5d_unsteady.py summary --run-index "${index}"
