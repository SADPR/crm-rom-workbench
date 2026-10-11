#!/usr/bin/env bash
# Global PROM-RBFs from the shock IC at 5 x 5 iterations, so they compare with the 5 x 5 PROMs and
# local PROM-RBFs (submit_sobol5d_5x5.sbatch): n = 12 (closures rbf256-n12, rbf384-n12) and
# n = 24 (closures trained here), on POD 256 and POD 384, one sweep after the other so that no more
# than THROTTLE 5-node PROMs run at once. Submit it from greedy-procedure/:
#   sbatch [--dependency=afterany:JOB] submit_sobol5d_rbf_5x5.sbatch [THROTTLE]

#SBATCH --job-name=naca5d_rbf5x5
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=Sobol5DRuns/logs/rbf5x5_%j.out
#SBATCH --error=Sobol5DRuns/logs/rbf5x5_%j.err

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

# Initialize a 5 x 5 shock-start sweep over the 32 test points, run its PROMs (sbatch --wait returns
# when the whole array has ended), then its metrics. A failed init is reported and skipped.
sweep() {
    local count=$1 name=$2
    shift 2
    if ! python3 -B sobol5d_sweep.py init --pod "${count}" --points 1-32 --forms nondescriptor \
            --starts shock --its 5 --inner 5 --name "${name}" "$@"; then
        echo "Sweep ${name}${count}: init failed, skipped." >&2
        return
    fi
    local array
    array=$(sbatch --parsable --wait --array=1-32%"${throttle}" submit_sobol5d_sweep.sbatch "${count}" "${name}")
    echo "Sweep ${name}${count}: PROMs ${array} ended with exit code $?."
    echo "Sweep ${name}${count}: metrics $(sbatch --parsable submit_sobol5d_sweep_metrics.sbatch "${count}" "${name}")."
}

for count in 256 384; do
    python3 -B sobol5d_rbf.py train --pod "${count}" --dim 24 \
        || echo "RBF training of rbf${count}-n24 failed; its sweep will be skipped." >&2
    for n in 12 24; do
        sweep "${count}" "rbf${n}shock5x5_" --rbf "rbf${count}-n${n}"
    done
done
