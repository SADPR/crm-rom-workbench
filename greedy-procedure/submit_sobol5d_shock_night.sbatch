#!/usr/bin/env bash
# Overnight shock-start sweeps, one after the other, so that no more than THROTTLE 5-node PROMs run
# at once. Each mirrors the PROMs it is compared with, so only the IC changes:
#   1. POD 384, global, 30 x 5: the batch-384 PROMs (romruns384);
#   2. POD 128, global, 5 x 30: the POD-128 PROMs of the results table (romruns128);
#   3. POD 256, local k = 8 with MinClusterSize 10, 5 x 30: local8m10ic_.
# The first sweep copies the batch-384 PROM inputs, so submit it from greedy-procedure/ after that
# array:
#   sbatch --dependency=afterany:POD384_PROMS submit_sobol5d_shock_night.sbatch [THROTTLE]

#SBATCH --job-name=naca5d_night
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --mem=16G
#SBATCH --output=Sobol5DRuns/logs/night_%j.out
#SBATCH --error=Sobol5DRuns/logs/night_%j.err

set -uo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"
throttle=${1:-3}

module purge
module load cmake/3.24.2 gcc/10.1.0 openmpi/4.1.2 imkl/2019

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

# The shock start merges a training HDM's wall Cp if nothing did before (an accepted restart).
export SOWER=/home/groups/cfarhat/bin/sower

# One sweep: init, its 32 PROMs, then its metrics. sbatch --wait returns when the whole array has
# ended. A failed init is reported and the night moves on to the next sweep.
sweep() {
    local count=$1 name=$2 limit=$3
    shift 3
    if ! python3 -B sobol5d_sweep.py init --pod "${count}" --points 1-32 --forms nondescriptor \
            --name "${name}" "$@"; then
        echo "Sweep ${name}${count}: init failed, skipped." >&2
        return
    fi
    local array metrics
    array=$(sbatch --parsable --wait --time="${limit}" --array=1-32%"${throttle}" \
        submit_sobol5d_sweep.sbatch "${count}" "${name}")
    echo "Sweep ${name}${count}: PROMs ${array} ended with exit code $?."
    metrics=$(sbatch --parsable submit_sobol5d_sweep_metrics.sbatch "${count}" "${name}")
    echo "Sweep ${name}${count}: metrics ${metrics}."
}

# A POD-384 PROM takes about 45 min, close to the sweep script's 1 h limit.
sweep 384 shock_ 02:00:00 --starts shock --its 30 --inner 5
sweep 128 shock_ 01:00:00 --starts shock --its 5
sweep 256 local8m10shock_ 01:00:00 --starts shock-cluster --its 5 --basis reductionrun256-c8-m10
