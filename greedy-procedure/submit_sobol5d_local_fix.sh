#!/usr/bin/env bash
# Local-PROM fix test on POD 256 with its 4 k-means clusters (reductionrun256-c4):
#   local4ic_      cluster-restricted start, 5 outer x 30 Gauss-Newton iterations;
#   local4ic30x5_  cluster-restricted start, 30 outer x 5, so the cluster can switch more often;
#   global30x5_    the global POD 256 with 30 x 5, the control for the iteration split.
# Each PROM array waits for the previous one, so at most four 5-node PROMs run at once.
# Run it on a login node from greedy-procedure/, with the GreedyAEROF environment active.

set -euo pipefail

after=${1:?Usage: ./submit_sobol5d_local_fix.sh JOB_TO_WAIT_FOR}
cd "$(dirname "$0")"

common=(--pod 256 --points 1-32 --forms nondescriptor)
python3 -B sobol5d_sweep.py init "${common[@]}" --name local4ic_ --starts delaunay-cluster \
    --basis reductionrun256-c4
python3 -B sobol5d_sweep.py init "${common[@]}" --name local4ic30x5_ --starts delaunay-cluster \
    --basis reductionrun256-c4 --its 30 --inner 5
python3 -B sobol5d_sweep.py init "${common[@]}" --name global30x5_ --starts delaunay --its 30 --inner 5

previous=${after}
for name in local4ic_ local4ic30x5_ global30x5_; do
    array=$(sbatch --parsable --dependency=afterany:${previous} --array=1-32%4 \
        submit_sobol5d_sweep.sbatch 256 "${name}")
    metrics=$(sbatch --parsable --dependency=afterany:${array} submit_sobol5d_sweep_metrics.sbatch 256 "${name}")
    echo "${name}: PROMs ${array}, metrics ${metrics}"
    previous=${array}
done
