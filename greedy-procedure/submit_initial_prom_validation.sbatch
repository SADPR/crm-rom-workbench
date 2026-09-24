#!/usr/bin/env bash
# Validate the 32-state global PROM at its first training point.

#SBATCH --job-name=crm_prom_val
#SBATCH --partition=cfarhat
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=24
#SBATCH --time=01:00:00
#SBATCH --output=GreedyRuns/validation_prom_%j.out
#SBATCH --error=GreedyRuns/validation_prom_%j.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"

[[ "${SLURM_NTASKS}" -eq 120 ]] || {
    echo "Expected 120 MPI tasks, received ${SLURM_NTASKS}." >&2
    exit 1
}
[[ -f GreedyRuns/reductionrun032/nonlinearrom/cluster0/state.rob120 ]] || {
    echo "Missing the completed 32-state POD basis." >&2
    exit 1
}

module purge
module load cmake/3.24.2 gcc/10.1.0 openmpi/4.1.2 imkl/2019

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

export AEROF=/scratch/users/sadpr/Code3Aug/aero-f/build-core/bin/aerof.opt
export SOWER=/home/groups/cfarhat/bin/sower
export MPI=srun
export CRM_CONDA_BASE=/scratch/users/sadpr/Code3Aug/miniconda3
export CRM_LAPLACE_ENV=CRM_Laplace
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

[[ -x "${AEROF}" ]] || {
    echo "Missing AERO-F executable: ${AEROF}" >&2
    exit 1
}
[[ -x "${SOWER}" ]] || {
    echo "Missing SOWER executable: ${SOWER}" >&2
    exit 1
}

python3 -B validate_initial_prom.py prepare
python3 -B validate_initial_prom.py run
