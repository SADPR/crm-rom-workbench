#!/usr/bin/env bash
# After the last restarts of a batch: accept the listed HDMs whose forces are stationary (if they
# still miss HDMtol2), rebuild the global POD, evaluate it, then build and evaluate the local POD
# (CLUSTERS = 0 skips the local POD).

#SBATCH --job-name=naca5d_accept
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --output=Sobol5DRuns/logs/accept_%j.out
#SBATCH --error=Sobol5DRuns/logs/accept_%j.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"
usage='Usage: sbatch --dependency=afterany:RESTARTS submit_sobol5d_accept_tail.sbatch COUNT CLUSTERS HDM...'
count=${1:?${usage}}
clusters=${2:?${usage}}
shift 2

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

for index in "$@"; do
    # An HDM that converged in its restart is refused here, which is fine.
    python3 -B sobol5d_campaign.py accept --run-index "${index}" \
        --reason "stationary lift and drag after its restarts" || echo "HDM ${index} not accepted."
done
python3 -B sobol5d_campaign.py audit --count "${count}"

# The global chain first, then the local one, so no more than three 5-node jobs run at once.
pod=$(sbatch --parsable submit_sobol5d_pod.sbatch "${count}")
prom=$(sbatch --parsable --dependency=afterok:${pod} --array=1-32%3 submit_sobol5d_prom.sbatch \
    "${count}")
metrics=$(sbatch --parsable --dependency=afterok:${pod},afterany:${prom} --time=08:00:00 \
    submit_sobol5d_metrics.sbatch "${count}")
echo "POD ${pod}, PROMs ${prom}, metrics ${metrics}"
if (( clusters > 0 )); then
    local=$(sbatch --parsable --dependency=afterok:${pod},afterany:${prom} submit_sobol5d_local.sbatch \
        "${count}" "${clusters}" 20)
    sweep=$(sbatch --parsable --dependency=afterok:${local} --array=1-32%3 submit_sobol5d_sweep.sbatch \
        "${count}" "local${clusters}_")
    local_metrics=$(sbatch --parsable --dependency=afterok:${local},afterany:${sweep} \
        submit_sobol5d_sweep_metrics.sbatch "${count}" "local${clusters}_")
    echo "local POD ${local}, PROMs ${sweep}, metrics ${local_metrics}"
fi
