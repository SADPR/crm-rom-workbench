#!/usr/bin/env bash
# Global PROM-RBF test on POD 256: train the RBF closures with n = 6, 8 and 12 primary
# coordinates (sobol5d_rbf.py), then run 32 PROM-RBFs for each with the final PROM
# configuration (Delaunay start, NonDescriptor, 30 x 5). The three arrays run one after the
# other, so at most four 5-node PROMs run at once. Run it on a login node from
# greedy-procedure/, with GreedyAEROF active (the trainer needs scikit-learn).

set -euo pipefail

cd "$(dirname "$0")"
python3 -c 'import sklearn' 2>/dev/null || {
    echo 'Activate GreedyAEROF first: the RBF trainer needs scikit-learn.' >&2
    exit 1
}
# Optional: a job to wait for before the first array.
previous=${1:-}

for n in 6 8 12; do
    python3 -B sobol5d_rbf.py train --pod 256 --dim "${n}"
    python3 -B sobol5d_sweep.py init --pod 256 --points 1-32 --starts delaunay --forms nondescriptor \
        --its 30 --inner 5 --name "rbf${n}_" --rbf "rbf256-n${n}"
done

for n in 6 8 12; do
    dependency=${previous:+--dependency=afterany:${previous}}
    array=$(sbatch --parsable ${dependency} --array=1-32%4 submit_sobol5d_sweep.sbatch 256 "rbf${n}_")
    metrics=$(sbatch --parsable --dependency=afterany:${array} submit_sobol5d_sweep_metrics.sbatch 256 "rbf${n}_")
    echo "n = ${n}: PROMs ${array}, metrics ${metrics}"
    previous=${array}
done
