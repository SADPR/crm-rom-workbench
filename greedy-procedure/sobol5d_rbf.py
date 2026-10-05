#!/usr/bin/env python3
"""Train the RBF closure of a global PROM-RBF on one POD batch.

The PROM-RBF state is u = u_ref + V q + Vbar N(q). V holds the first n POD modes, Vbar the
remaining ones, and N is an RBF map from the n primary coordinates to the secondary ones,
read by AERO-F's general manifold (GeneralManifoldRbfName). As in HGV2, the training data are
the coordinates of every distinct training state in the full POD basis. They are AERO-F's own
IC products V^T (u_j - u_ref) (state.basisUicProducts), the quantities the online start uses;
alias catalog entries, which repeat a state, are dropped. The fit is the supplied
rbf_trainer.py (Gaussian kernel, min-max scaling, epsilon by a 90/10 split, refit on all data).
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import sobol5d_campaign as campaign
import sobol5d_sweep as sweep

AERO_F_FILES = ('rbf_precomputations.txt', 'rbf_xTrain.txt', 'rbf_stdscaling.txt', 'rbf_hyper.txt')
# The trainer's default eps range, [0.5, 10], misses the optimum here (about 0.04 to 0.13 on
# POD 256): its validation error exceeded 100%. Search a wider range, with the two kernels of
# the HGV2 trainer.
EPS_RANGE = (0.01, 10.0, 120)
KERNELS = 'gaussian,imq'


def rbf_dir(settings, count, dim):
    """Return the directory of one batch's RBF closure with n primary coordinates."""
    return Path(settings.MasterDir) / 'rbf{:03d}-n{}'.format(count, dim)


def training_coordinates(pod):
    """Return the full-basis coordinates of the distinct training states, in catalog order."""
    products = sweep.cluster_products(pod)
    if len(products) != 1:
        raise RuntimeError('{} is clustered; the PROM-RBF here is global.'.format(pod))
    rows, runs = products[0], sweep.catalog_runs(pod)
    first = {}
    for entry, run in enumerate(runs):
        first.setdefault(run, entry)
    keep = sorted(first.values())
    return rows[keep], [runs[entry] for entry in keep]


def header_size(path):
    """Return the two integers of an AERO-F RBF text file's first line."""
    return tuple(int(value) for value in path.read_text().split('\n', 1)[0].split())


def train(count, dim):
    """Write the training coordinates and fit the RBF closure with n = dim."""
    settings = campaign.configure_settings()
    pod = campaign.pod_dir(settings, count)
    target = rbf_dir(settings, count, dim)
    if target.exists():
        raise RuntimeError('{} exists; refusing to overwrite an RBF closure.'.format(target))
    coords, runs = training_coordinates(pod)
    states, basis = coords.shape
    if states != basis:
        raise RuntimeError('{} distinct states but a basis of {}; expected a full-rank POD.'.format(
            states, basis))
    if not 1 <= dim < basis:
        raise ValueError('n must be in [1, {}).'.format(basis))
    target.mkdir(parents=True)
    data = target / 'state.coords'
    with open(data, 'w') as handle:
        handle.write('# coordinates of the {} distinct training states in the POD-{} basis\n'.format(
            states, count))
        np.savetxt(handle, coords, fmt='%.16e', delimiter=',')
    log = target / 'train.log'
    with open(log, 'w') as handle:
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('rbf_trainer.py')),
                                 '--data_file', str(data), '--dimV', str(dim),
                                 '--output_path', '{}/'.format(target.as_posix()),
                                 '--eps_min', str(EPS_RANGE[0]), '--eps_max', str(EPS_RANGE[1]),
                                 '--n_eps', str(EPS_RANGE[2]), '--kernels', KERNELS],
                                stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError('rbf_trainer.py failed; see {}.'.format(log))
    missing = [name for name in AERO_F_FILES if not (target / name).is_file()]
    if missing:
        raise RuntimeError('rbf_trainer.py did not write {}.'.format(', '.join(missing)))
    sizes = {name: header_size(target / name) for name in AERO_F_FILES}
    if (sizes['rbf_xTrain.txt'] != (states, dim) or sizes['rbf_precomputations.txt'] != (states, basis - dim)
            or sizes['rbf_stdscaling.txt'] != (dim, basis - dim)):
        raise RuntimeError('Unexpected RBF file sizes: {}.'.format(sizes))
    best = re.search(r'\[Best split\] ker=(\S+)\s+eps=(\S+)\s+RelL2=([\d.]+)%', log.read_text())
    campaign.write_json(target / 'rbf.json', {
        'pod': str(pod),
        'primary_dimension': dim,
        'secondary_dimension': basis - dim,
        'training_states': states,
        'training_runs': runs,
        'kernel': best.group(1),
        'epsilon': float(best.group(2)),
        'validation_rel_l2_percent': float(best.group(3)),
        'epsilon_search': {'min': EPS_RANGE[0], 'max': EPS_RANGE[1], 'count': EPS_RANGE[2], 'kernels': KERNELS},
        'source': 'state.basisUicProducts of {}, aliases dropped'.format(pod),
    })
    print('RBF closure {}: n = {}, nbar = {}, {} kernel, eps = {}, validation error {}% of q_bar.'.format(
        target, dim, basis - dim, best.group(1), best.group(2), best.group(3)))


def main():
    """Train one RBF closure."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('train',))
    parser.add_argument('--pod', type=int, required=True)
    parser.add_argument('--dim', type=int, required=True, help='primary dimension n')
    args = parser.parse_args()
    train(args.pod, args.dim)


if __name__ == '__main__':
    main()
