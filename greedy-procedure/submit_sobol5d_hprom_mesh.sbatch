#!/usr/bin/env bash
# ECSW NNLS over all the HPROM training rows, then the reduced mesh decomposed for 24 ranks.

#SBATCH --job-name=naca5d_hprom_mesh
#SBATCH --partition=cfarhat
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=24
#SBATCH --time=48:00:00
#SBATCH --output=Sobol5DRuns/logs/hprom_mesh_%j.out
#SBATCH --error=Sobol5DRuns/logs/hprom_mesh_%j.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"
count=${1:?Usage: sbatch submit_sobol5d_hprom_mesh.sbatch COUNT [TOLERANCE]}
tolerance=${2:-}

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
export AEROF=/scratch/users/sadpr/Code3Aug/aero-f/build-scalapack/bin/aerof.opt
export MPI=srun
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

[[ -x "${AEROF}" ]] || {
    echo "Missing AERO-F executable: ${AEROF}" >&2
    exit 1
}

python3 -B sobol5d_hprom.py mesh --pod "${count}" ${tolerance:+--tolerance "${tolerance}"}
