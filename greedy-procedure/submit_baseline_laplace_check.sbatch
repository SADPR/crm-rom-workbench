#!/usr/bin/env bash
# Verify that AERO-F reads the nominal geometry-specific Laplace shift.

#SBATCH --job-name=crm_laplace_check
#SBATCH --partition=cfarhat
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=24
#SBATCH --time=00:30:00
#SBATCH --output=BaselineRuns/HDMrun001/laplace/aerof-check/slurm_%j.out
#SBATCH --error=BaselineRuns/HDMrun001/laplace/aerof-check/slurm_%j.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"

input_file=BaselineRuns/HDMrun001/laplace/aerof-check/input_laplace
shift_prefix=BaselineRuns/HDMrun001/laplace/Laplace-bin/ushift.bin
aerof=/scratch/users/sadpr/Code3Aug/aero-f/build-core/bin/aerof.opt

[[ -f "${input_file}" ]] || {
    echo "Missing ${input_file}; run benchmark_laplace_shift.py prepare-aerof-check first." >&2
    exit 1
}
[[ -f "${shift_prefix}001" && -f "${shift_prefix}120" ]] || {
    echo "Missing the 120-part Laplace shift: ${shift_prefix}" >&2
    exit 1
}
[[ -x "${aerof}" ]] || {
    echo "Missing AERO-F executable: ${aerof}" >&2
    exit 1
}
[[ "${SLURM_NTASKS}" -eq 120 ]] || {
    echo "Expected 120 MPI tasks, received ${SLURM_NTASKS}." >&2
    exit 1
}

module purge
module load cmake/3.24.2 gcc/10.1.0 openmpi/4.1.2 imkl/2019

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun "${aerof}" "${input_file}"
