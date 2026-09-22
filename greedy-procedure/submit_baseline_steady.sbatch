#!/usr/bin/env bash
# Submit from greedy-procedure after baseline_steady.py prepare has succeeded.

#SBATCH --job-name=crm_steady
#SBATCH --partition=cfarhat
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=24
#SBATCH --time=04:00:00
#SBATCH --output=BaselineRuns/baseline_%j.out
#SBATCH --error=BaselineRuns/baseline_%j.err

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work_dir="${repo_dir}/greedy-procedure"
cd "${work_dir}"

[[ -f BaselineRuns/HDMrun001/input1 ]] || {
    echo 'Missing BaselineRuns/HDMrun001/input1; run baseline_steady.py prepare first.' >&2
    exit 1
}
[[ -f BaselineRuns/HDMrun001/input2 ]] || {
    echo 'Missing BaselineRuns/HDMrun001/input2; run baseline_steady.py prepare first.' >&2
    exit 1
}

module purge
module load cmake/3.24.2 gcc/10.1.0 openmpi/4.1.2 imkl/2019

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

export AEROF=/scratch/users/sadpr/Code3Aug/aero-f/build-core/bin/aerof.opt
export PARTMESH=/home/groups/cfarhat/bin/partnmesh
export SOWER=/home/groups/cfarhat/bin/sower
export CD2TET=/home/groups/cfarhat/bin/cd2tet
export MPI=srun
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

[[ -x "${AEROF}" ]] || {
    echo "Missing AERO-F executable: ${AEROF}" >&2
    exit 1
}

repo_commit=$(cd "${repo_dir}" && git rev-parse HEAD)

[[ "${SLURM_NTASKS}" -eq 120 ]] || {
    echo "Expected 120 MPI tasks, received ${SLURM_NTASKS}." >&2
    exit 1
}

{
    printf 'job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'nodes=%s\n' "${SLURM_JOB_NUM_NODES}"
    printf 'tasks=%s\n' "${SLURM_NTASKS}"
    printf 'aerof=%s\n' "${AEROF}"
    printf 'aerof_commit=%s\n' "${repo_commit}"
    module -t list 2>&1
    python3 --version
    srun --version | head -1
} > "BaselineRuns/job_${SLURM_JOB_ID}.env"

python3 -B baseline_steady.py run
