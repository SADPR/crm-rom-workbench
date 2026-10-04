#!/usr/bin/env bash
# Build one batch's clustered POD and initialize its Delaunay PROM sweep over the test set.

#SBATCH --job-name=naca5d_local
#SBATCH --partition=cfarhat
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=24
#SBATCH --time=02:00:00
#SBATCH --output=Sobol5DRuns/logs/local_%j.out
#SBATCH --error=Sobol5DRuns/logs/local_%j.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"
usage='Usage: sbatch [--export=ALL,AEROF_POD=EXE] submit_sobol5d_local.sbatch COUNT CLUSTERS [OVERLAP] [MIN_CLUSTER_SIZE] [SUFFIX]'
count=${1:?${usage}}
clusters=${2:?${usage}}
# HGV2 uses a 20 percent overlap between neighboring clusters.
overlap=${3:-20}
# Optional MinClusterSize: clusters with fewer own snapshots merge into their nearest one.
min_size=${4:-}
if [[ -n "${min_size}" ]]; then
    tag="m${min_size}"
    size_option=(--min-size "${min_size}")
    basis="reductionrun${count}-c${clusters}-m${min_size}"
else
    tag=""
    size_option=()
    basis="reductionrun${count}-c${clusters}"
fi
# Optional directory tag, e.g. "regress" for a regression build of an existing POD.
suffix=${5:-}
if [[ -n "${suffix}" ]]; then
    basis="${basis}-${suffix}"
    tag="${tag}${suffix}"
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
export PARTMESH=/home/groups/cfarhat/bin/partnmesh
export CD2TET=/home/groups/cfarhat/bin/cd2tet
# AEROF_POD selects another build for this offline step (e.g. one carrying a clustering fix);
# local.json records the executable and its commit.
export AEROF=${AEROF_POD:-/scratch/users/sadpr/Code3Aug/aero-f/build-scalapack/bin/aerof.opt}
export MPI=srun
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

[[ -x "${AEROF}" ]] || {
    echo "Missing AERO-F executable: ${AEROF}" >&2
    exit 1
}

python3 -B sobol5d_local.py pod --count "${count}" --clusters "${clusters}" --overlap "${overlap}" \
    ${size_option[@]+"${size_option[@]}"} --suffix "${suffix}"
# Every test point with the start restricted to the PROM's starting cluster, so the start lies in
# its local basis (a plain Delaunay start mixes clusters; see the handoff), NonDescriptor, 5 x 30.
python3 -B sobol5d_sweep.py init --pod "${count}" --name "local${clusters}${tag}ic_" --points 1-32 \
    --starts delaunay-cluster --forms nondescriptor --basis "${basis}"
