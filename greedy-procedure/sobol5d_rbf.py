#!/usr/bin/env python3
"""Train the RBF closures of a PROM-RBF on one POD batch: global, or one per cluster.

The PROM-RBF state is u = u_ref + V q + Vbar N(q). V holds the first n POD modes, Vbar the
remaining ones, and N is an RBF map from the n primary coordinates to the secondary ones,
read by AERO-F's general manifold (GeneralManifoldRbfName). As in HGV2, the training data are
the coordinates of every distinct training state in the full POD basis. They are AERO-F's own
IC products V^T (u_j - u_ref) (state.basisUicProducts), the quantities the online start uses;
alias catalog entries, which repeat a state, are dropped. The fit is the supplied
rbf_trainer.py (Gaussian kernel, min-max scaling, epsilon by a 90/10 split, refit on all data).
A local closure (train-local) fits one map per cluster of a clustered POD, on the coordinates of
the cluster's member states in its own basis.
"""

import argparse
import json
import os
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


def fit(target, coords, dim, description):
    """Fit one RBF closure on full-basis coordinates (one row per state) in a new directory."""
    states, basis = coords.shape
    if states != basis:
        raise RuntimeError('{} states but a basis of {}; expected a full-rank POD.'.format(states, basis))
    if not 1 <= dim < basis:
        raise ValueError('n must be in [1, {}).'.format(basis))
    target.mkdir(parents=True)
    data = target / 'state.coords'
    with open(data, 'w') as handle:
        handle.write('# {}\n'.format(description))
        np.savetxt(handle, coords, fmt='%.16e', delimiter=',')
    log = target / 'train.log'
    # Unbuffered, so a killed trainer still leaves its output; one thread, since the fit is small
    # and login nodes limit what a process may use.
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with open(log, 'w') as handle:
        result = subprocess.run([sys.executable, '-u', '-B', str(Path(__file__).with_name('rbf_trainer.py')),
                                 '--data_file', str(data), '--dimV', str(dim),
                                 '--output_path', '{}/'.format(target.as_posix()),
                                 '--eps_min', str(EPS_RANGE[0]), '--eps_max', str(EPS_RANGE[1]),
                                 '--n_eps', str(EPS_RANGE[2]), '--kernels', KERNELS],
                                stdout=handle, stderr=subprocess.STDOUT, check=False, env=environment)
    if result.returncode != 0:
        # A negative code is the signal that stopped it (e.g. -9 when a node limit kills it).
        raise RuntimeError('rbf_trainer.py exited with {}; see {}.'.format(result.returncode, log))
    missing = [name for name in AERO_F_FILES if not (target / name).is_file()]
    if missing:
        raise RuntimeError('rbf_trainer.py did not write {}.'.format(', '.join(missing)))
    sizes = {name: header_size(target / name) for name in AERO_F_FILES}
    if (sizes['rbf_xTrain.txt'] != (states, dim) or sizes['rbf_precomputations.txt'] != (states, basis - dim)
            or sizes['rbf_stdscaling.txt'] != (dim, basis - dim)):
        raise RuntimeError('Unexpected RBF file sizes in {}: {}.'.format(target, sizes))
    best = re.search(r'\[Best split\] ker=(\S+)\s+eps=(\S+)\s+RelL2=([\d.]+)%', log.read_text())
    return {
        'primary_dimension': dim,
        'secondary_dimension': basis - dim,
        'training_states': states,
        'kernel': best.group(1),
        'epsilon': float(best.group(2)),
        'validation_rel_l2_percent': float(best.group(3)),
    }


def train(count, dim):
    """Write the training coordinates and fit the RBF closure with n = dim."""
    settings = campaign.configure_settings()
    pod = campaign.pod_dir(settings, count)
    target = rbf_dir(settings, count, dim)
    if target.exists():
        raise RuntimeError('{} exists; refusing to overwrite an RBF closure.'.format(target))
    coords, runs = training_coordinates(pod)
    fitted = fit(target, coords, dim, 'coordinates of the {} distinct training states in the POD-{} basis'.format(
        len(coords), count))
    campaign.write_json(target / 'rbf.json', dict(
        {'pod': str(pod)}, **fitted, training_runs=runs,
        epsilon_search={'min': EPS_RANGE[0], 'max': EPS_RANGE[1], 'count': EPS_RANGE[2], 'kernels': KERNELS},
        source='state.basisUicProducts of {}, aliases dropped'.format(pod)))
    print('RBF closure {}: n = {}, nbar = {}, {} kernel, eps = {}, validation error {}% of q_bar.'.format(
        target, dim, fitted['secondary_dimension'], fitted['kernel'], fitted['epsilon'],
        fitted['validation_rel_l2_percent']))


def local_rbf_dir(settings, count, basis, dim):
    """Return the directory of the local RBF closures of one clustered POD with n primary coordinates."""
    prefix = 'reductionrun{:03d}-'.format(count)
    if not basis.startswith(prefix):
        raise ValueError('{} is not a clustered POD of batch {}.'.format(basis, count))
    return Path(settings.MasterDir) / 'rbf{:03d}-{}-n{}'.format(count, basis[len(prefix):], dim)


def local_training_coordinates(pod, local, clusters):
    """Per cluster, the coordinates of its distinct member states in its own basis, in catalog order.

    The members are the cluster's snapshots, overlap included, so their coordinates span the whole
    local basis; alias catalog entries, which repeat a state, are dropped.
    """
    products = sweep.cluster_products(local)
    runs = sweep.catalog_runs(pod)
    members = sweep.cluster_members(local, clusters)
    if len(products) != clusters:
        raise RuntimeError('{} has {} clusters of IC products, not {}.'.format(local, len(products), clusters))
    data = []
    for k in range(clusters):
        first = {}
        for entry, run in enumerate(runs):
            if run in members[k]:
                first.setdefault(run, entry)
        keep = sorted(first.values())
        coords = products[k][keep]
        if len(keep) != len(members[k]) or np.linalg.matrix_rank(coords) != coords.shape[1]:
            raise RuntimeError('Cluster {} of {}: {} member entries for {} snapshots, coordinates of rank {} '
                               'in a basis of {}.'.format(k, local, len(keep), len(members[k]),
                                                          np.linalg.matrix_rank(coords), coords.shape[1]))
        data.append((coords, [runs[entry] for entry in keep]))
    return data


def train_local(count, basis, dim):
    """Fit one RBF closure per cluster of a clustered POD, each with n = dim on its own basis.

    AERO-F reads them from GeneralManifoldRbfName = .../cluster0/, replacing that component with each
    cluster's directory, and sizes each cluster's V and Vbar from its own closure.
    """
    settings = campaign.configure_settings()
    pod = campaign.pod_dir(settings, count)
    local = Path(settings.MasterDir) / basis
    record = local / 'local.json'
    if not record.is_file():
        raise RuntimeError('Missing {}; build the clustered POD first.'.format(record))
    clusters = json.loads(record.read_text())['clusters']
    target = local_rbf_dir(settings, count, basis, dim)
    staging = target.with_name(target.name + '.partial')
    for path in (target, staging):
        if path.exists():
            raise RuntimeError('{} exists; refusing to overwrite an RBF closure.'.format(path))
    data = local_training_coordinates(pod, local, clusters)
    sizes = [coords.shape[1] for coords, _ in data]
    if dim >= min(sizes):
        raise ValueError('n = {} must be below the smallest cluster basis of {} ({}).'.format(dim, basis, sizes))
    fitted = []
    for k, (coords, runs) in enumerate(data):
        result = fit(staging / 'cluster{}'.format(k), coords, dim,
                     'coordinates of the {} member states of cluster {} in its basis of {}'.format(len(coords), k, basis))
        fitted.append(dict(result, training_runs=runs))
        print('  cluster {}: n = {}, nbar = {}, {} kernel, eps = {}, validation error {}% of q_bar.'.format(
            k, dim, result['secondary_dimension'], result['kernel'], result['epsilon'],
            result['validation_rel_l2_percent']), flush=True)
    campaign.write_json(staging / 'rbf.json', {
        'pod': str(pod),
        'basis': str(local),
        'clusters': clusters,
        'primary_dimension': dim,
        'secondary_dimension': [result['secondary_dimension'] for result in fitted],
        'per_cluster': fitted,
        'epsilon_search': {'min': EPS_RANGE[0], 'max': EPS_RANGE[1], 'count': EPS_RANGE[2], 'kernels': KERNELS},
        'source': 'per-cluster state.basisUicProducts of {} at its member states, aliases dropped'.format(local),
    })
    staging.rename(target)
    print('Local RBF closures {}: {} clusters, n = {}.'.format(target, clusters, dim))


def main():
    """Train one global RBF closure, or one per cluster of a clustered POD."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('train', 'train-local'))
    parser.add_argument('--pod', type=int, required=True)
    parser.add_argument('--dim', type=int, required=True, help='primary dimension n')
    parser.add_argument('--basis', help='for train-local, a clustered POD, e.g. reductionrun256-c8-m10')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args.pod, args.dim)
    else:
        if not args.basis:
            raise ValueError('train-local needs --basis.')
        train_local(args.pod, args.basis, args.dim)


if __name__ == '__main__':
    main()
