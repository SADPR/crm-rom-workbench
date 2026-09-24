#!/usr/bin/env bash
# Compare short PROM solver variants without modifying GreedyRuns results.

#SBATCH --job-name=crm_prom_diag
#SBATCH --partition=cfarhat
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=24
#SBATCH --output=GreedyRuns/prom_diagnostics_%j.out
#SBATCH --error=GreedyRuns/prom_diagnostics_%j.err
#SBATCH --error=CRMDiagnostics/prom_diagnostics_%j.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"

[[ "${SLURM_NTASKS}" -eq 120 ]] || {
    echo "Expected 120 MPI tasks, received ${SLURM_NTASKS}." >&2
    exit 1
}

module purge
module load cmake/3.24.2 gcc/10.1.0 openmpi/4.1.2 imkl/2019

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

export AEROF=/scratch/users/sadpr/Code3Aug/aero-f/build-scalapack/bin/aerof.opt
export SOWER=/home/groups/cfarhat/bin/sower
export MPI=srun
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

mode=${1:-full}
case_name=${2:-all}
if [[ "${mode}" == full ]]; then
    python3 -B diagnose_initial_prom.py prepare
elif [[ "${mode}" != resume ]]; then
    echo "Usage: sbatch $0 [full|resume] [all|one_newton|unscaled|line_search]" >&2
    exit 1
fi

python3 -B diagnose_initial_prom.py run --case "${case_name}"
