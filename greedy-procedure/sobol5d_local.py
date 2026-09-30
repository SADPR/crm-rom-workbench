#!/usr/bin/env python3
"""Build a clustered (local) POD of one training batch from the input of its global POD.

The global input is reused with only the number of clusters, the snapshot overlap between
neighboring clusters, and a fixed k-means seed changed (AERO-F seeds from the clock by
default). AERO-F may merge small clusters, so the actual count is read back from the output
and recorded in local.json; the PROMs must use that count.
"""

import argparse
import re
import shutil
from pathlib import Path

import sobol5d_campaign as campaign
import sobol5d_projection as projection


KMEANS_SEED = 20260930


def local_dir(settings, count, clusters):
    """Return the reduction directory of one batch's clustered POD."""
    return Path(settings.MasterDir) / 'reductionrun{:03d}-c{}'.format(count, clusters)


def cluster_summary(directory):
    """Return the actual clusters, their snapshot counts, and their basis sizes from a POD run."""
    clusters = sorted(int(path.name[7:]) for path in (directory / 'nonlinearrom').glob('cluster[0-9]*')
                      if (path / 'state.rob001').is_file())
    log = (directory / 'log.pod').read_text()
    snapshots = [int(value) for value in re.findall(r'Reading (\d+) snapshots from cluster \d+', log)]
    bases = [int(value) for value in re.findall(r'Retaining (\d+) of \d+ vectors', log)]
    return clusters, snapshots, bases


def build(count, clusters, overlap, seed=KMEANS_SEED):
    """Run AERO-F's clustered POD of one assembled batch."""
    settings = campaign.configure_settings()
    pod = campaign.pod_dir(settings, count)
    target = local_dir(settings, count, clusters)
    if target.exists():
        raise RuntimeError('{} exists; refusing to overwrite a POD.'.format(target))
    for path in (pod / 'input.pod', pod / 'statesnapdata.txt', pod / 'parsoldata.txt'):
        if not path.is_file():
            raise RuntimeError('Missing {}; build the global POD {} first.'.format(path, count))
    # The root catalogs follow the latest batch; this batch's copies do not.
    states, solutions = (pod / 'statesnapdata.txt').as_posix(), (pod / 'parsoldata.txt').as_posix()
    text = projection.patch((pod / 'input.pod').read_text(), [
        ('MultipleSolutionsData = "{}parsoldata.txt";'.format(settings.MasterDir),
         'MultipleSolutionsData = "{}";'.format(solutions), 1),
        ('ProjectionErrorSnapshotData = "{}statesnapdata.txt";'.format(settings.MasterDir),
         'ProjectionErrorSnapshotData = "{}";'.format(states), 1),
        ('StateSnapshotData = "{}statesnapdata.txt";'.format(settings.MasterDir),
         'StateSnapshotData = "{}";'.format(states), 1),
        ('Prefix = "{}/";'.format(pod.as_posix()), 'Prefix = "{}/";'.format(target.as_posix()), 1),
        ('NumClusters = 1;', 'NumClusters = {};'.format(clusters), 1),
        ('         PercentOverlap = {};\n'.format(settings.PercentOverlap),
         '         PercentOverlap = {};\n         KMeansRandomSeed = {};\n'.format(overlap, seed), 1),
    ], pod / 'input.pod')
    target.mkdir()
    input_file = target / 'input.pod'
    input_file.write_text(text)
    shutil.copy2(pod / 'statesnapdata.txt', target / 'statesnapdata.txt')
    shutil.copy2(pod / 'parsoldata.txt', target / 'parsoldata.txt')
    projection.run_aerof(settings, input_file, target / 'log.pod')
    actual, snapshots, bases = cluster_summary(target)
    if not actual or actual != list(range(len(actual))) or len(bases) != len(actual):
        raise RuntimeError('Unexpected clusters in {}: {}, bases {}.'.format(target, actual, bases))
    campaign.write_json(target / 'local.json', {
        'pod': str(pod),
        'clusters_requested': clusters,
        'clusters': len(actual),
        'percent_overlap': overlap,
        'kmeans_seed': seed,
        'snapshots_per_cluster': snapshots,
        'basis_sizes': bases,
    })
    print('Clustered POD {}: {} clusters, snapshots {}, basis sizes {}.'.format(
        target, len(actual), snapshots, bases))


def main():
    """Build one clustered POD."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('pod',))
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--clusters', type=int, required=True)
    parser.add_argument('--overlap', type=float, default=20.0, help='percent overlap (HGV2 uses 20)')
    args = parser.parse_args()
    if args.clusters < 2:
        raise ValueError('--clusters must be at least 2; the global POD is the one-cluster case.')
    build(args.count, args.clusters, args.overlap)


if __name__ == '__main__':
    main()
