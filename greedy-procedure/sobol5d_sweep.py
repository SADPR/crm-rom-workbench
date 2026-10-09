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
import scipy.interpolate
import scipy.spatial

import sobol5d_campaign as campaign
import sobol5d_projection as projection
import sobol5d_test as test
from sobol5d_unsteady import cp_star


# The 4 diagnostic points first, then the rest of the 11 whose POD-128 PROM misplaces the
# shock, then 5 points it predicts well, spread in Mach.
POINTS = (19, 23, 30, 5, 3, 6, 7, 10, 11, 14, 27, 31, 22, 26, 2, 32)
FLAGGED = (3, 6, 7, 10, 11, 14, 19, 23, 27, 30, 31)
# IDW keeps AERO-F's inverse-distance weights with exponent 2; the projection is a reference.
STARTS = {
    'idw100': 'IDW over the 100 nearest catalog states (the original PROM start)',
    'idw8': 'IDW over the 8 nearest catalog states',
    'delaunay': 'linear interpolation on the Delaunay simplex (LinearNDInterpolator, rescale=True)',
    'delaunay-cluster': ('Delaunay over the training states of the cluster the PROM starts in, so the '
                         'start lies in that local basis (IDW over its nearest states outside their hull)'),
    'rbf': 'RBF interpolation, linear kernel with a degree-1 polynomial, unit-cube parameters',
    'projection': 'orthogonal projection of the truth (reference, not a practical start)',
    'shock': ('Delaunay (IDW outside their hull) over the training states with the shock structure '
              'predicted at the point by cubic RBFs: shocked near the predicted position, or unshocked'),
    'shock-cluster': ('the shock start over the training states of the cluster the PROM starts in, so the '
                      'start lies in that local basis'),
}
FORMS = {'nondescriptor': 'NonDescriptor', 'descriptor': 'Descriptor', 'hybrid': 'Hybrid'}
SWEEP_ITS = 5
# Starts read from an external weights file (InterpICWeights).
WEIGHTED_STARTS = ('delaunay', 'rbf', 'delaunay-cluster', 'shock', 'shock-cluster')
# Fallback of the cluster start outside the hull of its cluster's parameters: as many
# neighbors as a 5D simplex has vertices.
CLUSTER_NEIGHBORS = 6
# Shock start. A shock is the steepest Cp rise behind a supersonic region on 0.1 < x/c < 0.95,
# counted when dCp/d(x/c) reaches SHOCK_STRENGTH (the detector of the result slides). The window
# around the predicted position widens until it holds SHOCK_NEIGHBORS training states.
SHOCK_STRENGTH = 3.0
SHOCK_WINDOWS = (0.02, 0.03, 0.04, 0.05, 0.06)
SHOCK_NEIGHBORS = 6
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
               iterations=SWEEP_ITS, basis=None, inner=None, rbf=None, galerkin=False):
    """Freeze the run list and write the external IC weights of every point.

    `basis` names a clustered reduction directory next to the global one (sobol5d_local.py);
    its PROMs keep the global IDW catalog, whose IC products that POD also computed.
    `inner` replaces the Gauss-Newton iterations per outer iteration (30 in runs.py).
    `rbf` names an RBF closure (sobol5d_rbf.py) that turns the PROM into a PROM-RBF: a global one,
    or, with `basis`, the local closures of that clustered POD.
    `galerkin` replaces LSPG with Galerkin projection, V^T R = 0, solved by Newton.
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
    for start in ('delaunay-cluster', 'shock-cluster'):
        if start in starts and local is None:
            raise ValueError('The {} start needs a clustered --basis.'.format(start))
    closure = None
    if galerkin and rbf is not None:
        raise ValueError('The PROM-RBF here is LSPG only; drop --galerkin or --rbf.')
    if rbf is not None:
        closure = rbf_closure(settings, rbf, local)
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
    cluster_starts, shock_starts = {}, {}
    for start in starts:
        if start not in WEIGHTED_STARTS:
            continue
        (root / 'weights' / start).mkdir(parents=True)
        if start in ('delaunay-cluster', 'shock-cluster'):
            directory = Path(local['directory'])
            runs_of = catalog_runs(pod)
            members = cluster_members(directory, local['clusters'])
            distances = read_clustered(directory / 'nonlinearrom/state.ucUicDist')
        if start in ('shock', 'shock-cluster'):
            raw_nodes = np.loadtxt('{}_nodes'.format(settings.TopFilePath), dtype=np.float64)
            wall = test.wall_nodes('{}.top'.format(settings.TopFilePath), raw_nodes)
            frg = pyaeroopt.interface.Frg(
                top='{}.top'.format(settings.TopFilePath),
                geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
            )
            shocks = training_shocks(catalog, catalog_runs(pod), wall, wall_surfaces(raw_nodes, wall), frg)
        for index in points:
            if start == 'delaunay-cluster':
                weights, record = cluster_start(catalog, test_points[index - 1], runs_of, members, distances)
                cluster_starts['{:03d}'.format(index)] = record
            elif start == 'shock':
                weights, record = shock_weights(catalog, test_points[index - 1], shocks)
                shock_starts['{:03d}'.format(index)] = record
            elif start == 'shock-cluster':
                weights, record = shock_cluster_start(catalog, test_points[index - 1], shocks, runs_of,
                                                      members, distances)
                shock_starts['{:03d}'.format(index)] = record
            else:
                weights = test.interpolation_weights(start, catalog, test_points[index - 1])
            test.write_weights(weights_path(root, start, index), weights)
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
        'inner': inner,
        'projection': 'galerkin' if galerkin else 'lspg',
        'points': list(points),
        'flagged': list(FLAGGED),
        'starts': {start: STARTS[start] for start in starts},
        'forms': {form: FORMS[form] for form in forms},
        'basis': local,
        'rbf': closure,
        'cluster_starts': cluster_starts,
        'shock_starts': shock_starts,
        'runs': runs,
    })
    print('Initialized {} with {} runs.'.format(root, len(runs)))
    for index, record in cluster_starts.items():
        print('  point {}: start in cluster {} ({}, consistent: {}; plain Delaunay picks {})'.format(
            index, record['cluster'], record['method'], record['consistent'],
            record['plain_delaunay_cluster']))
    for index, record in shock_starts.items():
        where = '' if 'cluster' not in record else ' (cluster {}, consistent: {})'.format(
            record['cluster'], record['consistent'])
        if record['structure'] == 'no-shock':
            print('  point {}: no shock predicted; {} over the {} unshocked states{}'.format(
                index, record['method'], record['candidates'], where))
        else:
            print('  point {}: {} shock predicted at x/c = {:.3f}; {} over the {} states within {:.3f}c{}'.format(
                index, record['surface'], record['position'], record['method'], record['candidates'],
                record['window'], where))


def rbf_closure(settings, rbf, local):
    """Return the RBF closure record of a sweep: global, or local to the sweep's clustered basis.

    AERO-F replaces the cluster0 component of a local closure's path with each cluster's own
    directory, so a local sweep points it at .../cluster0.
    """
    record = Path(settings.MasterDir) / rbf / 'rbf.json'
    if not record.is_file():
        raise RuntimeError('Missing {}; train the RBF closure first.'.format(record))
    meta = json.loads(record.read_text())
    directory = Path(settings.MasterDir) / rbf
    if 'clusters' in meta:
        if local is None or meta['basis'] != local['directory']:
            raise ValueError('{} holds local closures of {}; pass that POD with --basis.'.format(rbf, meta['basis']))
        directory = directory / 'cluster0'
    elif local is not None:
        raise ValueError('{} is a global closure; drop --basis, or train local ones (sobol5d_rbf.py '
                         'train-local).'.format(rbf))
    return {'directory': str(directory), 'primary_dimension': meta['primary_dimension'],
            'secondary_dimension': meta['secondary_dimension']}

def sweep_input(settings, count, run, root, iterations, basis=None, inner=None, rbf=None,
                galerkin=False):
    """Return the original PROM input of the run's point with the sweep's changes applied."""
    index = run['point']
    pod = campaign.pod_dir(settings, count)
    original, original_laplace = test.prom_paths(settings, count, index)
    hdm = test.hdm_dir(index)
    directory = Path(run['directory'])
    source = (original / 'input').read_text()
    # Originals may already carry a start and an iteration count (sobol5d_test.py prom --start).
    outer = re.findall(r'   MaxIts = \d+;\n   Eps = 1e-10;', source)
    newton = re.findall(r'      under Newton \{\n         MaxIts = \d+;', source)
    weights = re.findall(r'InterpICWeights = "[^"]*";', source)
    if len(outer) != 1 or len(newton) != 1 or len(weights) != 1:
        raise RuntimeError('Cannot find the outer and Newton MaxIts and InterpICWeights in {}.'.format(
            original))
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
    if inner is not None:
        replacements.append((newton[0], '      under Newton {{\n         MaxIts = {};'.format(inner), 1))
    if galerkin:
        # AERO-F solves the square V^T A V by LU for Galerkin and warns unless the solver is
        # NormalEquations; QR applies to LSPG only.
        replacements += [
            ('Projection = LeastSquaresPetrovGalerkin;', 'Projection = Galerkin;', 1),
            ('LeastSquaresSolver = QR;', 'LeastSquaresSolver = NormalEquations;', 1),
        ]
    if rbf is not None:
        # AERO-F takes the first n POD modes as V and the next nbar as Vbar, and reads the
        # closure from this prefix (it must end in a slash).
        replacements += [
            ('UseGeneralManifold = False;', 'UseGeneralManifold = True;', 1),
            ('GeneralManifoldRbfName = "";', 'GeneralManifoldRbfName = "{}/";'.format(rbf['directory']), 1),
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
    if run['start'] in WEIGHTED_STARTS:
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
                       sweep.get('basis'), sweep.get('inner'), sweep.get('rbf'),
                       sweep.get('projection') == 'galerkin')
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


def read_clustered(path):
    """Read one of AERO-F's per-cluster ASCII products as one 2D array per cluster.

    The file holds Dimension#1 (clusters), then per cluster Dimension#2 (rows), then per row
    Dimension#3 (its length) and its values.
    """
    lines = iter(line for line in path.read_text().splitlines()
                 if line.strip() and not line.startswith('MultiVecType'))
    size = lambda: int(next(lines).split(':')[1])
    clusters = []
    for _ in range(size()):
        rows = [[float(next(lines)) for _ in range(size())] for _ in range(size())]
        clusters.append(np.array(rows))
    return clusters


def cluster_products(directory):
    """Return the precomputed IC coordinates of every cluster, one row per catalog entry."""
    return read_clustered(directory / 'nonlinearrom/state.basisUicProducts')


def catalog_runs(pod):
    """Return the training HDM directory behind every catalog entry; aliases name their original."""
    lines = (pod / 'parsoldata.txt').read_text().split('\n')
    count, size = int(lines[0]), int(lines[1])
    return [re.match(r'(\S*/HDMrun\d{3})/', lines[2 + entry * (size + 1)]).group(1)
            for entry in range(count)]


def wall_surfaces(raw_nodes, wall):
    """Return, per surface, the positions within `wall` sorted by x and their x/c (reference mesh)."""
    x = raw_nodes[wall, 1]
    xc = (x - x.min()) / (x.max() - x.min())
    surfaces = {}
    for name, side in (('upper', raw_nodes[wall, 2] >= 0), ('lower', raw_nodes[wall, 2] < 0)):
        rows = np.nonzero(side)[0]
        rows = rows[np.argsort(xc[rows])]
        surfaces[name] = (rows, xc[rows])
    return surfaces


def wall_shock(cp, mach, rows, xc):
    """Return (x/c, steepness) of the shock on one surface, or (nan, 0) without a supersonic region."""
    keep = (xc > 0.1) & (xc < 0.95)
    x, values = xc[keep], cp[rows][keep]
    if len(x) < 5 or values.min() >= cp_star(mach):
        return float('nan'), 0.0
    gradient = np.gradient(values, x)
    gradient[~np.maximum.accumulate(values < cp_star(mach))] = 0.0
    k = int(np.argmax(gradient))
    return float(x[k]), float(gradient[k])


def training_shocks(catalog, runs, wall, surfaces, frg):
    """Return the shock of every catalog entry per surface: (x/c, steepness), one row per entry.

    A training HDM whose surface Cp was never merged (an accepted restart, for instance) is merged
    here; a restarted HDM's merged file holds its final state only.
    """
    cache = {}
    shocks = {name: np.zeros((len(runs), 2)) for name in surfaces}
    for entry, run in enumerate(runs):
        if run not in cache:
            test.merge_surface_fields(frg, Path(run), fields=('PressureCoefficient',))
            cache[run] = test.wall_pressure(Path(run), wall)
        for name, (rows, xc) in surfaces.items():
            shocks[name][entry] = wall_shock(cache[run], catalog[entry][0], rows, xc)
    return shocks


def shock_weights(catalog, point, shocks, allowed=None):
    """Return IC weights over the training states with the shock structure predicted at the point.

    Per surface, a cubic RBF of the shock steepness over all entries says whether the point has a
    shock, and a cubic RBF of the position over the shocked entries says where; a position outside
    the detector's 0.1 < x/c < 0.95 counts as no shock. With a predicted shock, the candidates are
    the entries shocked on the surface of the steeper one, within the smallest window of its
    position that holds SHOCK_NEIGHBORS of them (the nearest positions beyond the last window).
    Without one, they are the entries with no shock on either surface. The weights interpolate
    over the candidates only: Delaunay inside their hull, IDW outside (cluster_weights).
    `allowed` restricts the candidates to some entries (a cluster's); without any left, the
    weights are None.
    """
    unit = test.unit_cube(catalog, catalog)
    target = test.unit_cube(catalog, point)[None, :]
    predicted = {}
    for name, rows in shocks.items():
        shocked = rows[:, 1] >= SHOCK_STRENGTH
        steepness = scipy.interpolate.RBFInterpolator(unit, rows[:, 1], kernel='cubic', degree=1)(target)[0]
        position = scipy.interpolate.RBFInterpolator(unit[shocked], rows[shocked, 0], kernel='cubic',
                                                     degree=1)(target)[0]
        predicted[name] = (float(position), float(steepness))
    record = {'predicted': predicted}
    surfaces = [name for name, (position, steepness) in predicted.items()
                if steepness >= SHOCK_STRENGTH and 0.1 < position < 0.95]
    if allowed is None:
        allowed = np.ones(len(catalog), dtype=bool)
    if surfaces:
        surface = max(surfaces, key=lambda name: predicted[name][1])
        position = predicted[surface][0]
        rows = shocks[surface]
        shocked = np.nonzero((rows[:, 1] >= SHOCK_STRENGTH) & allowed)[0]
        if not len(shocked):
            return None, record
        gap = np.abs(rows[shocked, 0] - position)
        window = next((w for w in SHOCK_WINDOWS if np.sum(gap <= w) >= SHOCK_NEIGHBORS), None)
        if window is None:
            window = float(np.sort(gap)[min(SHOCK_NEIGHBORS, len(gap)) - 1])
        inside = np.zeros(len(catalog), dtype=bool)
        inside[shocked[gap <= window]] = True
        record.update(structure='shock', surface=surface, position=position, window=window)
    else:
        inside = np.all([rows[:, 1] < SHOCK_STRENGTH for rows in shocks.values()], axis=0) & allowed
        if not inside.any():
            return None, record
        record.update(structure='no-shock')
    weights, method = cluster_weights(catalog, point, inside)
    record.update(method=method, candidates=int(inside.sum()),
                  neighbors={int(entry): float(weights[entry]) for entry in np.nonzero(weights)[0]})
    return weights, record

def cluster_members(local, clusters):
    """Return the training HDM directories of every cluster's snapshots, overlap included."""
    return [set(re.findall(r'(\S*/HDMrun\d{3})/', (local / 'nonlinearrom/cluster{}/state.snaps'.format(k))
                           .read_text()))
            for k in range(clusters)]


def starting_cluster(weights, distances):
    """Return the cluster AERO-F starts in for these IC weights.

    ImplicitRomTsDesc::formInitialCondition picks the centroid nearest to sum_j w_j u_j, using
    ucUicDist[k][i][j] = (u_i - c_k).(u_j - c_k), so the squared distance is w' D_k w.
    """
    return int(np.argmin([weights @ distance @ weights for distance in distances]))


def cluster_weights(catalog, point, inside):
    """Return Delaunay weights over the catalog entries marked inside, zero elsewhere.

    Outside the hull of those entries, inverse-distance weights over the nearest ones, with the
    parameters normalized by the whole catalog as AERO-F does.
    """
    weights = np.zeros(len(catalog))
    members = catalog[inside]
    # A cluster with too few or degenerate parameter points has no 5D triangulation at all.
    try:
        local = scipy.interpolate.LinearNDInterpolator(members, np.eye(len(members)), rescale=True)(
            np.asarray(point, dtype=float))[0]
    except (scipy.spatial.QhullError, ValueError):
        local = np.full(len(members), np.nan)
    method = 'delaunay'
    if np.isnan(local).any():
        distance = np.linalg.norm(test.unit_cube(catalog, members) - test.unit_cube(catalog, point),
                                  axis=1) ** 2
        keep = np.argsort(distance, kind='stable')[:CLUSTER_NEIGHBORS]
        local = np.zeros(len(members))
        local[keep] = 1.0 / distance[keep]
        local /= local.sum()
        method = 'idw{}'.format(CLUSTER_NEIGHBORS)
    weights[inside] = local
    if abs(weights.sum() - 1.0) > 1e-8:
        raise RuntimeError('Cluster weights at {} do not sum to one.'.format(point))
    return weights, method


def cluster_start(catalog, point, runs, members, distances):
    """Return weights whose states all belong to the cluster AERO-F will start the PROM in.

    The cluster AERO-F picks with the plain Delaunay start is tried first, then the others by
    distance. A cluster is kept when its own restricted start still makes AERO-F pick it.
    """
    full = test.interpolation_weights('delaunay', catalog, point)
    order = np.argsort([full @ distance @ full for distance in distances])
    for cluster in order:
        inside = np.array([run in members[cluster] for run in runs])
        weights, method = cluster_weights(catalog, point, inside)
        if starting_cluster(weights, distances) == cluster:
            return weights, {'cluster': int(cluster), 'method': method, 'consistent': True,
                             'plain_delaunay_cluster': int(order[0])}
    inside = np.array([run in members[order[0]] for run in runs])
    weights, method = cluster_weights(catalog, point, inside)
    return weights, {'cluster': int(order[0]), 'method': method, 'consistent': False,
                     'plain_delaunay_cluster': int(order[0])}


def shock_cluster_start(catalog, point, shocks, runs, members, distances):
    """Return shock-start weights whose states all belong to the cluster AERO-F will start in.

    As cluster_start: the clusters are tried by their distance to the unrestricted shock start,
    and one is kept when its restricted start still makes AERO-F pick it. A cluster with no state
    of the predicted shock structure is skipped.
    """
    full, _ = shock_weights(catalog, point, shocks)
    order = np.argsort([full @ distance @ full for distance in distances])
    first = None
    for cluster in order:
        inside = np.array([run in members[cluster] for run in runs])
        weights, record = shock_weights(catalog, point, shocks, inside)
        if weights is None:
            continue
        record = dict(record, cluster=int(cluster), plain_shock_cluster=int(order[0]))
        if starting_cluster(weights, distances) == cluster:
            return weights, dict(record, consistent=True)
        if first is None:
            first = (weights, dict(record, consistent=False))
    if first is None:
        raise RuntimeError('No cluster holds a state of the predicted shock structure at {}.'.format(point))
    return first

def reduced_history(path):
    """Return (cluster, coordinates) per outer iteration; the basis size changes with clusters."""
    history = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            values = line.split()
            history.append((int(float(values[1])), np.array([float(value) for value in values[3:]])))
    return history


def start_weights(settings, run, root, catalog):
    """Return the run's IC weights over the catalog, or None for the projection start."""
    index = run['point']
    if run['start'] == 'projection':
        return None
    if run['start'] in WEIGHTED_STARTS:
        lines = weights_path(root, run['start'], index).read_text().split()
        return np.array([float(value) for value in lines[1:]])
    neighbors = settings.MaxIntSols if run['start'] == 'idw100' else 8
    return test.idw_weights(catalog, test.read_points()[index - 1], neighbors, settings.DistExp)


def expected_start(settings, run, root, catalog, products, truth_products, cluster=0):
    """Return the reduced coordinates the run's iteration 0 must hold in its starting cluster."""
    weights = start_weights(settings, run, root, catalog)
    if weights is None:
        return truth_products[run['point'] - 1]
    return weights @ products[cluster]


def blended_start(weights, runs, training_cp):
    """Return sum_j w_j Cp_j over the training wall Cp: the start before any projection."""
    blend = 0.0
    for entry in np.nonzero(np.abs(weights) > 1e-14)[0]:
        blend = blend + weights[entry] * training_cp(runs[entry])
    return blend


def run_metrics(settings, count, run, root, iterations, wall, catalog, products, truth_products, runs,
                training_cp):
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
    # A PROM-RBF starts from the first n coordinates only; N(q0) supplies the rest.
    start = start[:len(history[0][1])]
    weights = start_weights(settings, run, root, catalog)
    # How much of the blended start its projection onto the starting basis loses. On the first
    # local sweep, starts within their cluster gave below 0.01%, and the deformed ones 2-45%.
    # For a PROM-RBF it is the gap between the manifold start u(q0) and the blended start.
    start_loss = (test.relative_error(frames[0], blended_start(weights, runs, training_cp), 2)
                  if weights is not None else None)
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
        start_loss=start_loss,
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
    frg = pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
    )
    cache = {}

    def training_cp(directory):
        if directory not in cache:
            test.merge_surface_fields(frg, Path(directory), fields=('PressureCoefficient',))
            cache[directory] = test.wall_pressure(Path(directory), wall)
        return cache[directory]

    catalog_dirs = catalog_runs(pod)
    runs = [run_metrics(settings, count, run, root, sweep['iterations'], wall, catalog, products,
                        truth_products, catalog_dirs, training_cp)
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
                'start_loss': summarize([run['start_loss'] for run in done
                                         if run['start_loss'] is not None]),
            }
    campaign.write_json(root / 'metrics.json', {
        'name': sweep.get('name', name),
        'pod': count,
        'iterations': sweep['iterations'],
        'inner': sweep.get('inner'),
        'unit': 'percent relative L2 error of the z = 0 wall Cp against the truth HDM; '
                'residuals are in each run\'s own form',
        'configurations': configurations,
        'runs': runs,
    })
    print('{:26s} {:>5s} {:>8s} {:>8s} {:>8s} {:>8s} {:>7s} {:>7s}'.format(
        'configuration', 'ok', 'start', 'final', 'flagged', 'good', 'max', 'loss'))
    for label, value in configurations.items():
        if value['cp_final'] is None:
            print('{:26s} {:>5d}'.format(label, value['ok']))
            continue
        good = value['cp_final_good']['median'] if value['cp_final_good'] else float('nan')
        flagged = value['cp_final_flagged']['median'] if value['cp_final_flagged'] else float('nan')
        loss = value['start_loss']['max'] if value['start_loss'] else float('nan')
        print('{:26s} {:>5d} {:8.1f} {:8.1f} {:8.1f} {:8.1f} {:7.1f} {:7.1f}'.format(
            label, value['ok'], value['cp_start']['median'], value['cp_final']['median'],
            flagged, good, value['cp_final']['max'], loss))


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
    parser.add_argument('--inner', type=int, help='for init, Gauss-Newton iterations per outer one')
    parser.add_argument('--rbf', help='for init, an RBF closure directory, e.g. rbf256-n6, or with '
                                      '--basis a local one, e.g. rbf256-c8-m10-n12')
    parser.add_argument('--basis', help='for init, a clustered reduction directory, e.g. reductionrun256-c4')
    parser.add_argument('--galerkin', action='store_true', help='for init, Galerkin instead of LSPG')
    args = parser.parse_args()

    if args.mode == 'init':
        starts, forms = tuple(args.starts.split(',')), tuple(args.forms.split(','))
        unknown = [value for value in starts if value not in STARTS] + [
            value for value in forms if value not in FORMS]
        if unknown or not all(1 <= index <= test.TEST_COUNT for index in args.points):
            raise ValueError('Unknown start, form, or test index: {}.'.format(unknown or args.points))
        initialize(args.pod, args.name, args.points, starts, forms, args.its, args.basis, args.inner,
                   args.rbf, args.galerkin)
    elif args.mode == 'prom':
        if args.run is None:
            raise ValueError('--run is required for prom.')
        run_prom(args.pod, args.run, args.name)
    else:
        metrics(args.pod, args.name)


if __name__ == '__main__':
    main()
