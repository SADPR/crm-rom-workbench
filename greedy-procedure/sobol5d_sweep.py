#!/usr/bin/env python3
"""Sweep the PROM start (IC) and residual form of one POD batch at a fixed set of test points.

Each run patches the original frozen-POD PROM input of its test point, so only the IC, the
residual form, the iteration count, and the unused output fields change. Iteration 0 is
written, so every run records where it starts as well as where it ends.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyaeroopt
import scipy.interpolate

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


def sweep_dir(settings, count):
    """Return the root of one batch's sweep."""
    return Path(settings.MasterDir) / 'evaluate/sweep{:03d}'.format(count)


def read_sweep(settings, count):
    """Read the frozen run list of one batch's sweep."""
    path = sweep_dir(settings, count) / 'sweep.json'
    if not path.is_file():
        raise RuntimeError('Run the init action first; missing {}.'.format(path))
    return json.loads(path.read_text())


def catalog_points(pod):
    """Return the parameters of the batch's IDW catalog, in catalog order."""
    lines = (pod / 'parsoldata.txt').read_text().split('\n')
    count, size = int(lines[0]), int(lines[1])
    return np.array([[float(value) for value in lines[3 + entry * (size + 1):2 + (entry + 1) * (size + 1)]]
                     for entry in range(count)])


def unit_cube(catalog, point):
    """Normalize a point with the catalog's min/max per parameter, as AERO-F does."""
    lower, upper = catalog.min(axis=0), catalog.max(axis=0)
    return (np.asarray(point, dtype=float) - lower) / (upper - lower)


def idw_weights(catalog, point, neighbors, exponent=2.0):
    """Reproduce AERO-F's IDW weights over the nearest catalog entries."""
    distance = np.linalg.norm(unit_cube(catalog, catalog) - unit_cube(catalog, point), axis=1)
    distance = distance ** exponent
    keep = np.argsort(distance, kind='stable')[:neighbors]
    weights = np.zeros(len(catalog))
    weights[keep] = 1.0 / distance[keep]
    return weights / weights.sum()


def interpolation_weights(start, catalog, point):
    """Return the external IC weights of a Delaunay or RBF start, one per catalog entry."""
    if start == 'delaunay':
        interpolator = scipy.interpolate.LinearNDInterpolator(catalog, np.eye(len(catalog)),
                                                              rescale=True)
        weights = interpolator(np.asarray(point, dtype=float))[0]
    else:
        interpolator = scipy.interpolate.RBFInterpolator(unit_cube(catalog, catalog),
                                                         np.eye(len(catalog)), kernel='linear',
                                                         degree=1)
        weights = interpolator(unit_cube(catalog, point)[None, :])[0]
    if np.isnan(weights).any() or abs(weights.sum() - 1.0) > 1e-8:
        raise RuntimeError('Invalid {} weights at {}.'.format(start, point))
    return weights


def weights_path(root, start, index):
    """Return the external IC weights file of one start at one test point."""
    return root / 'weights' / start / 'point{:03d}.txt'.format(index)


def initialize(count):
    """Freeze the run list and write the Delaunay and RBF weights of every point."""
    settings = campaign.configure_settings()
    pod = campaign.pod_dir(settings, count)
    root = sweep_dir(settings, count)
    if root.exists():
        raise RuntimeError('{} exists; refusing to replace a sweep.'.format(root))
    diagnostic = projection.diagnostic_dir(settings, count)
    if not (diagnostic / 'log.projection').is_file():
        raise RuntimeError('The projection start needs {}; run its preprocessing first.'.format(
            diagnostic))
    points = test.read_points()
    for index in POINTS:
        source = test.prom_paths(settings, count, index)[0] / 'input'
        if not source.is_file():
            raise RuntimeError('Missing {}; run the original PROM first.'.format(source))
    catalog = catalog_points(pod)
    root.mkdir(parents=True)
    for start in ('delaunay', 'rbf'):
        (root / 'weights' / start).mkdir(parents=True)
        for index in POINTS:
            weights = interpolation_weights(start, catalog, points[index - 1])
            weights_path(root, start, index).write_text(
                '{}\n'.format(len(weights)) + ''.join('{:.17e}\n'.format(value) for value in weights))
    runs = []
    for index in POINTS:
        for start in STARTS:
            for form in FORMS:
                runs.append({
                    'run': len(runs) + 1, 'point': index, 'start': start, 'form': form,
                    'directory': str(root / '{}-{}'.format(start, form) / 'point{:03d}'.format(index)),
                })
    campaign.write_json(root / 'sweep.json', {
        'pod': count,
        'iterations': SWEEP_ITS,
        'points': list(POINTS),
        'flagged': list(FLAGGED),
        'starts': STARTS,
        'forms': FORMS,
        'runs': runs,
    })
    print('Initialized {} with {} runs.'.format(root, len(runs)))


def sweep_input(settings, count, run):
    """Return the original PROM input of the run's point with the sweep's changes applied."""
    index = run['point']
    pod = campaign.pod_dir(settings, count)
    root = sweep_dir(settings, count)
    original, original_laplace = test.prom_paths(settings, count, index)
    hdm = test.hdm_dir(index)
    directory = Path(run['directory'])
    # The truth HDM holds this exact geometry and Laplace shift; the PROM only reads them.
    replacements = [
        ('Position = "{}/deform/Position.bin";'.format(original.as_posix()),
         'Position = "{}/deform/Position.bin";'.format(hdm.as_posix()), 1),
        ('WallDistance = "{}/deform/WallDistance.bin";'.format(original.as_posix()),
         'WallDistance = "{}/deform/WallDistance.bin";'.format(hdm.as_posix()), 1),
        ('LaplaceSnapshotData = "{}/ushift.bin";'.format(original_laplace.as_posix()),
         'LaplaceSnapshotData = "{}/Laplace-bin/ushift.bin";'.format(hdm.as_posix()), 1),
        ('{}/'.format(original.as_posix()), '{}/'.format(directory.as_posix()), None),
        ('   MaxIts = {};\n   Eps = 1e-10;'.format(settings.MaxItsHROM),
         '   MaxIts = {};\n   Eps = 1e-10;'.format(SWEEP_ITS), 1),
        # A positive frequency also writes iteration 0, the start.
        ('      Frequency = 0;', '      Frequency = {};'.format(SWEEP_ITS), 1),
        ('Form = NonDescriptor;', 'Form = {};'.format(FORMS[run['form']]), 1),
    ]
    replacements += [(line, '{} = "";'.format(line.split(' = ')[0]), 1) for line in UNUSED_OUTPUTS]
    if run['start'] == 'idw8':
        replacements.append(('MaxInterpolatedSolutions = {};'.format(settings.MaxIntSols),
                             'MaxInterpolatedSolutions = 8;', 1))
    elif run['start'] in ('delaunay', 'rbf'):
        replacements.append(('InterpICWeights = "";', 'InterpICWeights = "{}";'.format(
            weights_path(root, run['start'], index).as_posix()), 1))
    elif run['start'] == 'projection':
        diagnostic = projection.diagnostic_dir(settings, count)
        replacements += [
            ('MultipleSolutionsData = "{}";'.format((pod / 'parsoldata.txt').as_posix()),
             'MultipleSolutionsData = "{}";'.format((diagnostic / 'testsoldata.txt').as_posix()), 1),
            ('Prefix = "{}/";'.format(pod.as_posix()), 'Prefix = "{}/";'.format(diagnostic.as_posix()), 1),
        ]
    text = projection.patch((original / 'input').read_text(), replacements, original / 'input')
    if 'romruns{:03d}'.format(count) in text:
        raise RuntimeError('The sweep input of run {} still refers to romruns.'.format(run['run']))
    return text


def run_prom(count, number):
    """Run one sweep PROM and merge its surface pressure."""
    settings = campaign.configure_settings()
    sweep = read_sweep(settings, count)
    if not 1 <= number <= len(sweep['runs']):
        raise ValueError('--run must be in [1, {}].'.format(len(sweep['runs'])))
    run = sweep['runs'][number - 1]
    directory = Path(run['directory'])
    if directory.exists():
        raise RuntimeError('{} exists; refusing to overwrite a PROM.'.format(directory))
    text = sweep_input(settings, count, run)
    original = test.prom_paths(settings, count, run['point'])[0]
    for name in ('results', 'postpro'):
        (directory / name).mkdir(parents=True)
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


def expected_start(settings, count, run, catalog, products, truth_products):
    """Return the reduced coordinates the run's iteration 0 must hold."""
    index = run['point']
    if run['start'] == 'projection':
        return truth_products[index - 1]
    if run['start'] in ('delaunay', 'rbf'):
        lines = weights_path(sweep_dir(settings, count), run['start'], index).read_text().split()
        weights = np.array([float(value) for value in lines[1:]])
    else:
        neighbors = settings.MaxIntSols if run['start'] == 'idw100' else 8
        weights = idw_weights(catalog, test.read_points()[index - 1], neighbors, settings.DistExp)
    return weights @ products


def run_metrics(settings, count, run, wall, catalog, products, truth_products):
    """Compare one sweep PROM's start and final states with the truth and the original PROM."""
    directory = Path(run['directory'])
    index = run['point']
    result = {key: run[key] for key in ('run', 'point', 'start', 'form')}
    result['flagged'] = index in FLAGGED
    if not directory.exists():
        return dict(result, status='missing')
    residual_file = directory / 'postpro/Residual.out'
    history = projection.table(residual_file) if residual_file.is_file() else np.empty((0, 1))
    if len(history) == 0 or int(history[-1, 0]) < SWEEP_ITS:
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
    coordinates = projection.table(directory / 'postpro/ReducedCoords.out')[:, 3:]
    start = expected_start(settings, count, run, catalog, products, truth_products)
    _, _, final_residual = test.prom_residual(directory)
    lift_start, drag_start = projection.force_errors(directory, truth_forces, 0)
    lift_final, drag_final = projection.force_errors(directory, truth_forces, -1)
    return dict(
        result,
        status='ok',
        frames=len(frames),
        start_check=float(np.abs(coordinates[0] - start).max() / np.abs(start).max()),
        convergence=float(np.linalg.norm(coordinates[-1] - coordinates[-2])
                          / np.linalg.norm(coordinates[-1])),
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


def metrics(count):
    """Write per-run and per-configuration results of one batch's sweep."""
    settings = campaign.configure_settings()
    sweep = read_sweep(settings, count)
    pod = campaign.pod_dir(settings, count)
    raw_nodes = np.loadtxt('{}_nodes'.format(settings.TopFilePath), dtype=np.float64)
    wall = test.wall_nodes('{}.top'.format(settings.TopFilePath), raw_nodes)
    catalog = catalog_points(pod)
    products = projection.ic_products(pod)
    truth_products = projection.ic_products(projection.diagnostic_dir(settings, count))
    runs = [run_metrics(settings, count, run, wall, catalog, products, truth_products)
            for run in sweep['runs']]
    configurations = {}
    for start in STARTS:
        for form in FORMS:
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
    campaign.write_json(sweep_dir(settings, count) / 'metrics.json', {
        'pod': count,
        'unit': 'percent relative L2 error of the z = 0 wall Cp against the truth HDM; '
                'residuals are in each run\'s own form',
        'configurations': configurations,
        'runs': runs,
    })
    print('{:26s} {:>5s} {:>8s} {:>8s} {:>8s} {:>8s} {:>7s}'.format(
        'configuration', 'ok', 'start', 'final', 'flagged', 'good', 'max'))
    for name, value in configurations.items():
        if value['cp_final'] is None:
            print('{:26s} {:>5d}'.format(name, value['ok']))
            continue
        good = value['cp_final_good']['median'] if value['cp_final_good'] else float('nan')
        flagged = value['cp_final_flagged']['median'] if value['cp_final_flagged'] else float('nan')
        print('{:26s} {:>5d} {:8.1f} {:8.1f} {:8.1f} {:8.1f} {:7.1f}'.format(
            name, value['ok'], value['cp_start']['median'], value['cp_final']['median'],
            flagged, good, value['cp_final']['max']))


def main():
    """Run one explicit stage of the sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('init', 'prom', 'metrics'))
    parser.add_argument('--pod', type=int, required=True)
    parser.add_argument('--run', type=int)
    args = parser.parse_args()

    if args.mode == 'init':
        initialize(args.pod)
    elif args.mode == 'prom':
        if args.run is None:
            raise ValueError('--run is required for prom.')
        run_prom(args.pod, args.run)
    else:
        metrics(args.pod)


if __name__ == '__main__':
    main()
