#!/usr/bin/env bash
# Shock-start PROMs with 5 outer x 5 Gauss-Newton iterations, one sweep after the other so that no
# more than THROTTLE 5-node PROMs run at once. The full residual plateaus after the first outer
# iteration, and the local POD-384 PROMs, which ran 5 x 5 by accident (their sweep took the Newton
# count of the 30 x 5 batch-384 inputs), matched the global POD-384 PROM at a sixth of the cost.
# Each sweep differs from the one it is compared with only in its iteration counts:
#   POD 384 and 256 global (30 x 5: shock_384, shock_256), POD 128 global (5 x 30: shock_128),
#   local POD 256 (5 x 30: local8m10shock_256) and its PROM-RBF n = 24 (5 x 30: local8m10rbf24_256).
# Submit it from greedy-procedure/, after any running chain of 5-node jobs:
#   sbatch [--dependency=afterany:JOB] submit_sobol5d_5x5.sbatch [THROTTLE]

#SBATCH --job-name=naca5d_5x5
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=Sobol5DRuns/logs/5x5_%j.out
#SBATCH --error=Sobol5DRuns/logs/5x5_%j.err

set -uo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"
throttle=${1:-3}

module purge
module load cmake/3.24.2 gcc/10.1.0 openmpi/4.1.2 imkl/2019

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

export SOWER=/home/groups/cfarhat/bin/sower
# Jobs submitted from here inherit this job's environment; an inherited memory request clashes with
# a PROM job's own and srun then refuses to start AERO-F (see submit_sobol5d_shock_night.sbatch).
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

# Initialize a 5 x 5 sweep over the 32 test points, run its PROMs (sbatch --wait returns when the
# whole array has ended), then its metrics. A failed init is reported and skipped.
sweep() {
    local count=$1 name=$2
    shift 2
    if ! python3 -B sobol5d_sweep.py init --pod "${count}" --points 1-32 --forms nondescriptor \
            --its 5 --inner 5 --name "${name}" "$@"; then
        echo "Sweep ${name}${count}: init failed, skipped." >&2
        return
    fi
    local array
    array=$(sbatch --parsable --wait --array=1-32%"${throttle}" submit_sobol5d_sweep.sbatch "${count}" "${name}")
    echo "Sweep ${name}${count}: PROMs ${array} ended with exit code $?."
    echo "Sweep ${name}${count}: metrics $(sbatch --parsable submit_sobol5d_sweep_metrics.sbatch "${count}" "${name}")."
}

sweep 384 shock5x5_ --starts shock
sweep 256 shock5x5_ --starts shock
sweep 128 shock5x5_ --starts shock
sweep 256 local8m10shock5x5_ --starts shock-cluster --basis reductionrun256-c8-m10
sweep 256 local8m10rbf24_5x5_ --starts shock-cluster --basis reductionrun256-c8-m10 --rbf rbf256-c8-m10-n24
