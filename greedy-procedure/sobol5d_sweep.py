#!/usr/bin/env python3
"""Sweep the PROM start (IC) and residual form of one POD batch at a fixed set of test points.

Each run patches the original frozen-POD PROM input of its test point, so only the IC, the
residual form, the iteration count, and the unused output fields change. Iteration 0 is
written, so every run records where it starts as well as where it ends. A sweep has a name,
so later subsets (for example, the winning configuration on all test points) get their own
directory next to the first sweep.
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pyaeroopt

import sobol5d_campaign as campaign
import sobol5d_projection as projection
import sobol5d_test as test


# The 4 diagnostic points first, then the rest of the 11 whose POD-128 PROM misplaces the
# shock, then 5 points it predicts well, spread in Mach.
POINTS = (19, 23, 30, 5, 3, 6, 7, 10, 11, 14, 27, 31, 22, 26, 2, 32)
FLAGGED = (3, 6, 7, 10, 11, 14, 19, 23, 27, 30, 31)
# IDW keeps AERO-F's inverse-distance weights with exponent 2; the projection is a reference.
STARTS = {
    'idw100': 'IDW over the 100 nearest catalog states (the original PROM start)',
    'idw8': 'IDW over the 8 nearest catalog states',
    'delaunay': 'linear interpolation on the Delaunay simplex (LinearNDInterpolator, rescale=True)',
    'rbf': 'RBF interpolation, linear kernel with a degree-1 polynomial, unit-cube parameters',
    'projection': 'orthogonal projection of the truth (reference, not a practical start)',
}
FORMS = {'nondescriptor': 'NonDescriptor', 'descriptor': 'Descriptor', 'hybrid': 'Hybrid'}
SWEEP_ITS = 5
# Output fields the sweep metrics do not use; blanking them does not change the solve.
UNUSED_OUTPUTS = ('Mach = "Mach.bin";', 'Displacement = "Displacement.bin";',
                  'Velocity = "Velocity.bin";', 'FluxResidual = "FluxRes.bin";',
                  'ControlVolume = "ControlVolume.bin";')


def sweep_dir(settings, count, name='sweep'):
    """Return the root of one named sweep of one batch."""
    return Path(settings.MasterDir) / 'evaluate/{}{:03d}'.format(name, count)


def read_sweep(settings, count, name='sweep'):
    """Read the frozen run list of one named sweep."""
    path = sweep_dir(settings, count, name) / 'sweep.json'
    if not path.is_file():
        raise RuntimeError('Run the init action first; missing {}.'.format(path))
    return json.loads(path.read_text())


def weights_path(root, start, index):
    """Return the external IC weights file of one start at one test point."""
    return root / 'weights' / start / 'point{:03d}.txt'.format(index)


def initialize(count, name='sweep', points=POINTS, starts=tuple(STARTS), forms=tuple(FORMS),
               iterations=SWEEP_ITS, basis=None):
    """Freeze the run list and write the Delaunay and RBF weights of every point.

    `basis` names a clustered reduction directory next to the global one (sobol5d_local.py);
    its PROMs keep the global IDW catalog, whose IC products that POD also computed.
    """
    settings = campaign.configure_settings()
    pod = campaign.pod_dir(settings, count)
    root = sweep_dir(settings, count, name)
    if root.exists():
        raise RuntimeError('{} exists; refusing to replace a sweep.'.format(root))
    local = None
    if basis is not None:
        if 'projection' in starts:
            raise ValueError('The projection start uses the global bases; drop it with --basis.')
        record = Path(settings.MasterDir) / basis / 'local.json'
        if not record.is_file():
            raise RuntimeError('Missing {}; build the clustered POD first.'.format(record))
        local = {'directory': str(Path(settings.MasterDir) / basis),
                 'clusters': json.loads(record.read_text())['clusters']}
    if 'projection' in starts:
        diagnostic = projection.diagnostic_dir(settings, count)
        if not (diagnostic / 'log.projection').is_file():
            raise RuntimeError('The projection start needs {}; run its preprocessing first.'.format(
                diagnostic))
    test_points = test.read_points()
    for index in points:
        source = test.prom_paths(settings, count, index)[0] / 'input'
        if not source.is_file():
            raise RuntimeError('Missing {}; run the original PROM first.'.format(source))
    catalog = test.catalog_points(pod)
    root.mkdir(parents=True)
    for start in starts:
        if start not in ('delaunay', 'rbf'):
            continue
        (root / 'weights' / start).mkdir(parents=True)
        for index in points:
            test.write_weights(weights_path(root, start, index),
                               test.interpolation_weights(start, catalog, test_points[index - 1]))
    runs = []
    for index in points:
        for start in starts:
            for form in forms:
                runs.append({
                    'run': len(runs) + 1, 'point': index, 'start': start, 'form': form,
                    'directory': str(root / '{}-{}'.format(start, form) / 'point{:03d}'.format(index)),
                })
    campaign.write_json(root / 'sweep.json', {
        'name': name,
        'pod': count,
        'iterations': iterations,
        'points': list(points),
        'flagged': list(FLAGGED),
        'starts': {start: STARTS[start] for start in starts},
        'forms': {form: FORMS[form] for form in forms},
        'basis': local,
        'runs': runs,
    })
    print('Initialized {} with {} runs.'.format(root, len(runs)))


def sweep_input(settings, count, run, root, iterations, basis=None):
    """Return the original PROM input of the run's point with the sweep's changes applied."""
    index = run['point']
    pod = campaign.pod_dir(settings, count)
    original, original_laplace = test.prom_paths(settings, count, index)
    hdm = test.hdm_dir(index)
    directory = Path(run['directory'])
    source = (original / 'input').read_text()
    # Originals may already carry a start and an iteration count (sobol5d_test.py prom --start).
    outer = re.findall(r'   MaxIts = \d+;\n   Eps = 1e-10;', source)
    weights = re.findall(r'InterpICWeights = "[^"]*";', source)
    if len(outer) != 1 or len(weights) != 1:
        raise RuntimeError('Cannot find the outer MaxIts and InterpICWeights in {}.'.format(original))
    # The truth HDM holds this exact geometry and Laplace shift; the PROM only reads them.
    replacements = [
        ('Position = "{}/deform/Position.bin";'.format(original.as_posix()),
         'Position = "{}/deform/Position.bin";'.format(hdm.as_posix()), 1),
        ('WallDistance = "{}/deform/WallDistance.bin";'.format(original.as_posix()),
         'WallDistance = "{}/deform/WallDistance.bin";'.format(hdm.as_posix()), 1),
        ('LaplaceSnapshotData = "{}/ushift.bin";'.format(original_laplace.as_posix()),
         'LaplaceSnapshotData = "{}/Laplace-bin/ushift.bin";'.format(hdm.as_posix()), 1),
        ('{}/'.format(original.as_posix()), '{}/'.format(directory.as_posix()), None),
        (outer[0], '   MaxIts = {};\n   Eps = 1e-10;'.format(iterations), 1),
        # A positive frequency also writes iteration 0, the start.
        ('      Frequency = 0;', '      Frequency = {};'.format(iterations), 1),
        ('Form = NonDescriptor;', 'Form = {};'.format(FORMS[run['form']]), 1),
    ]
    if basis is not None:
        replacements += [
            ('Prefix = "{}/";'.format(pod.as_posix()), 'Prefix = "{}/";'.format(basis['directory']), 1),
            ('NumClusters = 1;', 'NumClusters = {};'.format(basis['clusters']), 1),
        ]
    replacements += [(line, '{} = "";'.format(line.split(' = ')[0]), 1) for line in UNUSED_OUTPUTS]
    if run['start'] == 'idw8':
        replacements.append(('MaxInterpolatedSolutions = {};'.format(settings.MaxIntSols),
                             'MaxInterpolatedSolutions = 8;', 1))
    # Only the Delaunay and RBF starts read external weights; the others must not inherit any.
    # This goes first: an inherited weights path lies in the original directory, which the
    # path replacement rewrites.
    if run['start'] in ('delaunay', 'rbf'):
        replacements.insert(0, (weights[0], 'InterpICWeights = "{}";'.format(
            weights_path(root, run['start'], index).as_posix()), 1))
    elif weights[0] != 'InterpICWeights = "";':
        replacements.insert(0, (weights[0], 'InterpICWeights = "";', 1))
    if run['start'] == 'projection':
        diagnostic = projection.diagnostic_dir(settings, count)
        replacements += [
            ('MultipleSolutionsData = "{}";'.format((pod / 'parsoldata.txt').as_posix()),
             'MultipleSolutionsData = "{}";'.format((diagnostic / 'testsoldata.txt').as_posix()), 1),
            ('Prefix = "{}/";'.format(pod.as_posix()), 'Prefix = "{}/";'.format(diagnostic.as_posix()), 1),
        ]
    text = projection.patch(source, replacements, original / 'input')
    if 'romruns{:03d}'.format(count) in text:
        raise RuntimeError('The sweep input of run {} still refers to romruns.'.format(run['run']))
    return text


def run_prom(count, number, name='sweep'):
    """Run one sweep PROM and merge its surface pressure."""
    settings = campaign.configure_settings()
    sweep = read_sweep(settings, count, name)
    if not 1 <= number <= len(sweep['runs']):
        raise ValueError('--run must be in [1, {}].'.format(len(sweep['runs'])))
    run = sweep['runs'][number - 1]
    directory = Path(run['directory'])
    if directory.exists():
        raise RuntimeError('{} exists; refusing to overwrite a PROM.'.format(directory))
    text = sweep_input(settings, count, run, sweep_dir(settings, count, name), sweep['iterations'],
                       sweep.get('basis'))
    original = test.prom_paths(settings, count, run['point'])[0]
    for folder in ('results', 'postpro'):
        (directory / folder).mkdir(parents=True)
    (directory / 'parameters.txt').write_text((original / 'parameters.txt').read_text())
    (directory / 'input').write_text(text)
    campaign.write_json(directory / 'run.json', run)
    projection.run_aerof(settings, directory / 'input', directory / 'log')
    frg = pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
    )
    test.merge_surface_fields(frg, directory, fields=('PressureCoefficient',))
    print('Completed sweep run {}: {} {} at test point {:03d}.'.format(
        number, run['start'], run['form'], run['point']))


def cluster_products(directory):
    """Return the precomputed IC coordinates of every cluster, one row per catalog entry.

    The ASCII file holds Dimension#1 (clusters), then per cluster Dimension#2 (entries), then
    per entry Dimension#3 (that cluster's basis size) and its values.
    """
    lines = iter(line for line in (directory / 'nonlinearrom/state.basisUicProducts').read_text()
                 .splitlines() if line.strip() and not line.startswith('MultiVecType'))
    size = lambda: int(next(lines).split(':')[1])
    clusters = []
    for _ in range(size()):
        entries = [[float(next(lines)) for _ in range(size())] for _ in range(size())]
        clusters.append(np.array(entries))
    return clusters


def reduced_history(path):
    """Return (cluster, coordinates) per outer iteration; the basis size changes with clusters."""
    history = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            values = line.split()
            history.append((int(float(values[1])), np.array([float(value) for value in values[3:]])))
    return history


def expected_start(settings, run, root, catalog, products, truth_products, cluster=0):
    """Return the reduced coordinates the run's iteration 0 must hold in its starting cluster."""
    index = run['point']
    if run['start'] == 'projection':
        return truth_products[index - 1]
    if run['start'] in ('delaunay', 'rbf'):
        lines = weights_path(root, run['start'], index).read_text().split()
        weights = np.array([float(value) for value in lines[1:]])
    else:
        neighbors = settings.MaxIntSols if run['start'] == 'idw100' else 8
        weights = test.idw_weights(catalog, test.read_points()[index - 1], neighbors, settings.DistExp)
    return weights @ products[cluster]


def run_metrics(settings, count, run, root, iterations, wall, catalog, products, truth_products):
    """Compare one sweep PROM's start and final states with the truth and the original PROM."""
    directory = Path(run['directory'])
    index = run['point']
    result = {key: run[key] for key in ('run', 'point', 'start', 'form')}
    result['flagged'] = index in FLAGGED
    if not directory.exists():
        return dict(result, status='missing')
    residual_file = directory / 'postpro/Residual.out'
    history = projection.table(residual_file) if residual_file.is_file() else np.empty((0, 1))
    if len(history) == 0 or int(history[-1, 0]) < iterations:
        return dict(result, status='crashed')
    xpost = directory / 'postpro/PressureCoefficient.xpost'
    if not xpost.is_file():
        return dict(result, status='unmerged')
    frames = projection.xpost_frames(xpost, wall)
    if len(frames) < 2 or not all(np.isfinite(frame).all() for frame in frames):
        return dict(result, status='invalid-output')
    hdm = test.hdm_dir(index)
    original = test.prom_paths(settings, count, index)[0]
    truth = test.wall_pressure(hdm, wall)
    truth_forces = projection.table(hdm / 'postpro/liftdrag.out')[-1]
    history = reduced_history(directory / 'postpro/ReducedCoords.out')
    clusters = [cluster for cluster, _ in history]
    start = expected_start(settings, run, root, catalog, products, truth_products, clusters[0])
    _, _, final_residual = test.prom_residual(directory)
    lift_start, drag_start = projection.force_errors(directory, truth_forces, 0)
    lift_final, drag_final = projection.force_errors(directory, truth_forces, -1)
    return dict(
        result,
        status='ok',
        frames=len(frames),
        start_check=float(np.abs(history[0][1] - start).max() / np.abs(start).max()),
        # A switch in the last iteration leaves no comparable pair; report it as not converged.
        convergence=(float(np.linalg.norm(history[-1][1] - history[-2][1])
                           / np.linalg.norm(history[-1][1]))
                     if clusters[-1] == clusters[-2] else float('inf')),
        start_cluster=clusters[0], final_cluster=clusters[-1],
        cluster_switches=sum(a != b for a, b in zip(clusters, clusters[1:])),
        cp_start=test.relative_error(frames[0], truth, 2),
        cp_final=test.relative_error(frames[-1], truth, 2),
        cp_final_vs_original=test.relative_error(frames[-1], test.wall_pressure(original, wall), 2),
        lift_start=lift_start, drag_start=drag_start, lift_final=lift_final, drag_final=drag_final,
        residual_start=projection.initial_residual(directory),
        residual_final=final_residual,
    )


def summarize(values):
    """Return the median, mean, and maximum of a list, or None if it is empty."""
    if not values:
        return None
    return {'median': float(np.median(values)), 'mean': float(np.mean(values)),
            'max': float(np.max(values))}


def metrics(count, name='sweep'):
    """Write per-run and per-configuration results of one named sweep."""
    settings = campaign.configure_settings()
    sweep = read_sweep(settings, count, name)
    root = sweep_dir(settings, count, name)
    pod = campaign.pod_dir(settings, count)
    raw_nodes = np.loadtxt('{}_nodes'.format(settings.TopFilePath), dtype=np.float64)
    wall = test.wall_nodes('{}.top'.format(settings.TopFilePath), raw_nodes)
    catalog = test.catalog_points(pod)
    products = cluster_products(Path(sweep['basis']['directory']) if sweep.get('basis') else pod)
    truth_products = (projection.ic_products(projection.diagnostic_dir(settings, count))
                      if 'projection' in sweep['starts'] else None)
    runs = [run_metrics(settings, count, run, root, sweep['iterations'], wall, catalog, products,
                        truth_products)
            for run in sweep['runs']]
    configurations = {}
    for start in sweep['starts']:
        for form in sweep['forms']:
            group = [run for run in runs if run['start'] == start and run['form'] == form]
            done = [run for run in group if run['status'] == 'ok']
            configurations['{}-{}'.format(start, form)] = {
                'ok': len(done),
                'failed': {'{:03d}'.format(run['point']): run['status']
                           for run in group if run['status'] != 'ok'},
                'cp_start': summarize([run['cp_start'] for run in done]),
                'cp_final': summarize([run['cp_final'] for run in done]),
                'cp_final_flagged': summarize([run['cp_final'] for run in done if run['flagged']]),
                'cp_final_good': summarize([run['cp_final'] for run in done if not run['flagged']]),
                'worst_convergence': max((run['convergence'] for run in done), default=None),
                'worst_start_check': max((run['start_check'] for run in done), default=None),
            }
    campaign.write_json(root / 'metrics.json', {
        'name': sweep.get('name', name),
        'pod': count,
        'iterations': sweep['iterations'],
        'unit': 'percent relative L2 error of the z = 0 wall Cp against the truth HDM; '
                'residuals are in each run\'s own form',
        'configurations': configurations,
        'runs': runs,
    })
    print('{:26s} {:>5s} {:>8s} {:>8s} {:>8s} {:>8s} {:>7s}'.format(
        'configuration', 'ok', 'start', 'final', 'flagged', 'good', 'max'))
    for label, value in configurations.items():
        if value['cp_final'] is None:
            print('{:26s} {:>5d}'.format(label, value['ok']))
            continue
        good = value['cp_final_good']['median'] if value['cp_final_good'] else float('nan')
        flagged = value['cp_final_flagged']['median'] if value['cp_final_flagged'] else float('nan')
        print('{:26s} {:>5d} {:8.1f} {:8.1f} {:8.1f} {:8.1f} {:7.1f}'.format(
            label, value['ok'], value['cp_start']['median'], value['cp_final']['median'],
            flagged, good, value['cp_final']['max']))


def index_list(text):
    """Parse test indices such as '1-32' or '3,11,19,23'."""
    indices = []
    for part in text.split(','):
        low, _, high = part.partition('-')
        indices.extend(range(int(low), int(high or low) + 1))
    return tuple(indices)


def main():
    """Run one explicit stage of a named sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('init', 'prom', 'metrics'))
    parser.add_argument('--pod', type=int, required=True)
    parser.add_argument('--name', default='sweep', help='sweep name; its root is evaluate/NAMEnnn')
    parser.add_argument('--run', type=int)
    parser.add_argument('--points', type=index_list, default=POINTS, help="for init, e.g. '1-32'")
    parser.add_argument('--starts', default=','.join(STARTS), help='for init, comma-separated')
    parser.add_argument('--forms', default=','.join(FORMS), help='for init, comma-separated')
    parser.add_argument('--its', type=int, default=SWEEP_ITS, help='for init, outer iterations')
    parser.add_argument('--basis', help='for init, a clustered reduction directory, e.g. reductionrun256-c4')
    args = parser.parse_args()

    if args.mode == 'init':
        starts, forms = tuple(args.starts.split(',')), tuple(args.forms.split(','))
        unknown = [value for value in starts if value not in STARTS] + [
            value for value in forms if value not in FORMS]
        if unknown or not all(1 <= index <= test.TEST_COUNT for index in args.points):
            raise ValueError('Unknown start, form, or test index: {}.'.format(unknown or args.points))
        initialize(args.pod, args.name, args.points, starts, forms, args.its, args.basis)
    elif args.mode == 'prom':
        if args.run is None:
            raise ValueError('--run is required for prom.')
        run_prom(args.pod, args.run, args.name)
    else:
        metrics(args.pod, args.name)


if __name__ == '__main__':
    main()
