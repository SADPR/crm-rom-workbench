#!/usr/bin/env bash
# Overnight PROM-RBF tests on POD 256 and POD 384, one sweep after the other, so that no more than
# THROTTLE 5-node jobs run at once (the local POD build runs alone):
#   256: global PROM-RBFs (closures rbf256-n6/8/12) from the shock IC; local closures on
#        reductionrun256-c8-m10 (n = 6, 12, 24) and their PROM-RBFs from the shock-cluster and
#        delaunay-cluster ICs.
#   384: global closures (n = 6, 8, 12) and their PROM-RBFs from the shock and Delaunay ICs; the
#        local POD (k = 8, MinClusterSize 10) with the AERO-F overlap fix; its PROMs without RBF
#        (delaunay-cluster, shock-cluster); local closures (n = 6, 12, 24) and their PROM-RBFs.
# Global PROM-RBFs run 30 x 5, like rbf6_256 and romruns384; local ones 5 x 30, like local8m10ic_256.
# Submit it from greedy-procedure/, after any running chain of 5-node jobs:
#   sbatch --dependency=afterany:JOB submit_sobol5d_rbf_night.sbatch [THROTTLE]

#SBATCH --job-name=naca5d_rbf_night
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --time=40:00:00
#SBATCH --output=Sobol5DRuns/logs/rbf_night_%j.out
#SBATCH --error=Sobol5DRuns/logs/rbf_night_%j.err

set -uo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"
throttle=${1:-3}

module purge
module load cmake/3.24.2 gcc/10.1.0 openmpi/4.1.2 imkl/2019

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

# The shock starts merge a training HDM's wall Cp if nothing did before (an accepted restart).
export SOWER=/home/groups/cfarhat/bin/sower
# Jobs submitted from here inherit this job's environment; an inherited memory request clashes with
# a PROM job's own and srun then refuses to start AERO-F (see submit_sobol5d_shock_night.sbatch).
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
# The clustered POD needs the overlap fix (aerof-patches/0001); the PROMs use the production build.
aerof_pod=/scratch/users/sadpr/Code3Aug/aerof-overlapfix/build-scalapack/bin/aerof.opt

# Run an initialized sweep's PROMs, then its metrics. sbatch --wait returns when the whole array has
# ended; the array size comes from the sweep, which may hold several starts.
run_sweep() {
    local count=$1 name=$2 limit=$3 runs array
    if ! runs=$(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["runs"]))' \
            "Sobol5DRuns/evaluate/${name}${count}/sweep.json"); then
        echo "Sweep ${name}${count}: no sweep.json, skipped." >&2
        return
    fi
    array=$(sbatch --parsable --wait --time="${limit}" --array=1-"${runs}"%"${throttle}" \
        submit_sobol5d_sweep.sbatch "${count}" "${name}")
    echo "Sweep ${name}${count}: ${runs} PROMs in ${array}, exit code $?."
    echo "Sweep ${name}${count}: metrics $(sbatch --parsable submit_sobol5d_sweep_metrics.sbatch "${count}" "${name}")."
}

# Initialize a sweep over the 32 test points, then run it. A failed init is reported and skipped.
sweep() {
    local count=$1 name=$2 limit=$3
    shift 3
    if ! python3 -B sobol5d_sweep.py init --pod "${count}" --points 1-32 --forms nondescriptor \
            --name "${name}" "$@"; then
        echo "Sweep ${name}${count}: init failed, skipped." >&2
        return
    fi
    run_sweep "${count}" "${name}" "${limit}"
}

train() {
    python3 -B sobol5d_rbf.py "$@" || echo "RBF training '$*' failed; its sweep will be skipped." >&2
}

# POD 256.
for n in 6 12 24; do train train-local --pod 256 --basis reductionrun256-c8-m10 --dim "${n}"; done
for n in 6 8 12; do
    sweep 256 "rbf${n}shock_" 01:00:00 --starts shock --its 30 --inner 5 --rbf "rbf256-n${n}"
done
for n in 6 12 24; do
    sweep 256 "local8m10rbf${n}_" 01:00:00 --starts shock-cluster,delaunay-cluster --its 5 \
        --basis reductionrun256-c8-m10 --rbf "rbf256-c8-m10-n${n}"
done

# POD 384. A global POD-384 PROM takes about 45 min, close to the sweep script's 1 h limit.
for n in 6 8 12; do train train --pod 384 --dim "${n}"; done
for n in 6 8 12; do
    sweep 384 "rbf${n}_" 02:00:00 --starts shock,delaunay --its 30 --inner 5 --rbf "rbf384-n${n}"
done
# The local POD job also initializes local8m10ic_384 (delaunay-cluster, 5 x 30).
if sbatch --wait --export=ALL,AEROF_POD="${aerof_pod}" submit_sobol5d_local.sbatch 384 8 20 10; then
    echo "Local POD 384 built."
    run_sweep 384 local8m10ic_ 01:00:00
    sweep 384 local8m10shock_ 01:00:00 --starts shock-cluster --its 5 --basis reductionrun384-c8-m10
    for n in 6 12 24; do train train-local --pod 384 --basis reductionrun384-c8-m10 --dim "${n}"; done
    for n in 6 12 24; do
        sweep 384 "local8m10rbf${n}_" 01:00:00 --starts shock-cluster,delaunay-cluster --its 5 \
            --basis reductionrun384-c8-m10 --rbf "rbf384-c8-m10-n${n}"
    done
else
    echo "Local POD 384 failed; its sweeps were skipped." >&2
fi
