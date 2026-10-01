#!/usr/bin/env bash
# After a training batch: stage 3 for its unconverged HDMs, then assemble and POD, the test
# PROMs with the Delaunay start (the POD-128 sweep winner), and the metrics, as one chain.

#SBATCH --job-name=naca5d_tail
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --output=Sobol5DRuns/logs/tail_%j.out
#SBATCH --error=Sobol5DRuns/logs/tail_%j.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"
count=${1:?Usage: sbatch --dependency=afterany:HDM_ARRAY submit_sobol5d_batch_tail.sbatch COUNT}

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

python3 -B sobol5d_campaign.py audit --count "${count}"
extend=$(python3 -B sobol5d_campaign.py extend-list --count "${count}" | sed -n 's/^EXTEND: //p')

dependency=""
if [[ -n "${extend}" ]]; then
    stage3=$(sbatch --parsable --array="${extend}%3" submit_sobol5d_hdm.sbatch train-extend)
    dependency="--dependency=afterany:${stage3}"
    echo "Stage 3 for HDMs ${extend}: job ${stage3}"
fi
# The POD job assembles the batch; HDMs that stay unconverged are excluded and listed.
pod=$(sbatch --parsable ${dependency} submit_sobol5d_pod.sbatch "${count}")
prom=$(sbatch --parsable --dependency=afterok:${pod} --array=1-32%3 submit_sobol5d_prom.sbatch \
    "${count}" --start delaunay --its 5)
# afterok on the POD too: without it, a failed POD still runs metrics on no PROMs.
metrics=$(sbatch --parsable --dependency=afterok:${pod},afterany:${prom} --time=08:00:00 submit_sobol5d_metrics.sbatch \
    "${count}")
echo "POD ${pod}, PROMs ${prom}, metrics ${metrics}"
