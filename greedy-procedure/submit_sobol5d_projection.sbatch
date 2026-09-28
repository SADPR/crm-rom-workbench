#!/usr/bin/env bash
# Start one batch's PROM from the projection of the truth at a few test points.

#SBATCH --job-name=crm5d_proj
#SBATCH --partition=cfarhat
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=24
#SBATCH --time=04:00:00
#SBATCH --output=Sobol5DRuns/logs/projection_%j.out
#SBATCH --error=Sobol5DRuns/logs/projection_%j.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"
count=${1:?Usage: sbatch submit_sobol5d_projection.sbatch COUNT [TEST_INDEX ...]}
shift
# By default, four POD-128 points whose PROM misplaces the shock and one that it predicts well.
if (( $# > 0 )); then
    indices=("$@")
else
    indices=(3 19 23 30 5)
fi

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

python3 -B sobol5d_projection.py prepare --pod "${count}"
python3 -B sobol5d_projection.py preprocess --pod "${count}"
# One failed PROM must not hide the others; metrics reports it as unavailable.
for index in "${indices[@]}"; do
    python3 -B sobol5d_projection.py prom --pod "${count}" --test-index "${index}" \
        || echo "Projected-truth PROM at test point ${index} failed." >&2
done
python3 -B sobol5d_projection.py metrics --pod "${count}"
