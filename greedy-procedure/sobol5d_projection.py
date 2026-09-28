#!/usr/bin/env python3
"""Start the PROM of one POD batch from the projection of the truth onto its trial space.

The HDM stores U - Ushift and the PROM uses U = Ushift + V q. With the test truths as its
IDW catalog, AERO-F gives the entry at the operating point weight one, so the PROM starts
from the orthogonal projection of the truth. Comparing that start, where the PROM goes
from it, and the original PROM tells
whether the basis cannot represent the transonic shock or the LSPG solve does not find it.
AERO-F precomputes the IC coordinates offline, so the diagnostic works on a copy of the
batch's reduction directory: prepare, preprocess, prom (one test point each), metrics.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyaeroopt

import initial_pod
import sobol5d_campaign as campaign
import sobol5d_test as test


DIAGNOSTIC_ITS = 5


def diagnostic_dir(settings, count):
    """Return the working copy of one batch's reduction directory."""
    return Path(settings.MasterDir) / 'evaluate/projection{:03d}'.format(count)


def patch(text, replacements, source):
    """Apply exact replacements; each must match `expected` times, or at least once if None."""
    for old, new, expected in replacements:
        found = text.count(old)
        if found == 0 or (expected is not None and found != expected):
            raise RuntimeError('Expected {!r} {} time(s) in {}, found {}.'.format(
                old, expected or 'one or more', source, found))
        text = text.replace(old, new)
    return text


def truth_entries(settings):
    """Return catalog entries of every test truth, honoring stage-3 restarts."""
    entries = []
    for index, point in enumerate(test.read_points(), start=1):
        directory = test.hdm_dir(index)
        campaign.final_residual(directory, settings.HDMtol2)
        entry = {'index': index, 'point': point, 'target': directory}
        if (directory / 'stage3.json').is_file():
            stage3 = json.loads((directory / 'stage3.json').read_text())
            entry['snapshot'], entry['snap_index'] = stage3['snapshot'], stage3['snap_index']
        entries.append(entry)
    return entries


def ic_products(target):
    """Return the precomputed IC coordinates of the first cluster, one row per catalog entry."""
    lines = (target / 'nonlinearrom/state.basisUicProducts').read_text().splitlines()
    dimensions = [int(line.split(':')[1]) for line in lines if line.startswith('Dimension#')]
    values = np.array([float(line) for line in lines if line.strip() and ':' not in line])
    return values[:dimensions[1] * dimensions[2]].reshape(dimensions[1], dimensions[2])


def projection_errors(target):
    """Read AERO-F's relative state projection error of each test truth, keyed by test index."""
    files = sorted(target.glob('nonlinearrom/**/state.proj'))
    if len(files) != 1:
        raise RuntimeError('Expected one state.proj under {}, found {}.'.format(target, files))
    errors = {}
    for line in files[0].read_text().splitlines()[1:]:
        fields = line.split()
        if fields:
            errors[int(fields[0]) + 1] = float(fields[1])
    return errors


def run_aerof(settings, input_file, log):
    """Run AERO-F on all HDM ranks and fail on a nonzero exit code."""
    aerof = os.environ.get('AEROF')
    if not aerof or not Path(aerof).is_file():
        raise RuntimeError('AEROF must name the built AERO-F executable.')
    hpc = pyaeroopt.interface.Hpc(machine='independence', batch=False, bg=False,
                                  nproc=settings.HDMnproc)
    hpc.mpi = os.environ.get('MPI', 'srun')
    command = hpc.execute_str(aerof, str(input_file))
    print(command, flush=True)
    with open(log, 'w') as log_file:
        result = subprocess.run(command, shell=True, stdout=log_file, stderr=subprocess.STDOUT,
                                check=False)
    if result.returncode != 0:
        raise RuntimeError('AERO-F failed; inspect {}.'.format(log))


def prepare(count):
    """Copy the batch's bases and write the truth catalogs and the preprocessing input."""
    settings = campaign.configure_settings()
    pod = campaign.pod_dir(settings, count)
    target = diagnostic_dir(settings, count)
    staging = target.with_name('{}.partial'.format(target.name))
    for path in (target, staging):
        if path.exists():
            raise RuntimeError('{} exists; refusing to overwrite it.'.format(path))
    for path in (pod / 'input.pod', pod / 'statesnapdata.txt', pod / 'parsoldata.txt',
                 pod / 'nonlinearrom/state.basisUicProducts'):
        if not path.is_file():
            raise RuntimeError('Missing {}; build POD {} first.'.format(path, count))
    entries = truth_entries(settings)

    prefix = '{}/'.format(target.as_posix())
    text = patch((pod / 'input.pod').read_text(), [
        ('MultipleSolutionsData = "{}parsoldata.txt";'.format(settings.MasterDir),
         'MultipleSolutionsData = "{}testsoldata.txt";'.format(prefix), 1),
        ('ProjectionErrorSnapshotData = "{}statesnapdata.txt";'.format(settings.MasterDir),
         'ProjectionErrorSnapshotData = "{}testsnapdata.txt";'.format(prefix), 1),
        # The root catalogs follow the latest batch; this batch's copies do not.
        ('StateSnapshotData = "{}statesnapdata.txt";'.format(settings.MasterDir),
         'StateSnapshotData = "{}";'.format((pod / 'statesnapdata.txt').as_posix()), 1),
        ('Prefix = "{}/";'.format(pod.as_posix()), 'Prefix = "{}";'.format(prefix), 1),
        ('under Clustering {\n', 'under Clustering {\n         UseExistingClusters = True;\n', 1),
        ('PreprocessForProjections = True;', 'PreprocessForProjections = False;', 1),
        ('ComputePOD = True;', 'ComputePOD = False;', 1),
        ('ComputeProjectionError = False;', 'ComputeProjectionError = True;', 1),
    ], pod / 'input.pod')

    # AERO-F rewrites the IC products in place, so the copy must not share files with the POD.
    staging.mkdir(parents=True)
    shutil.copytree(pod / 'nonlinearrom', staging / 'nonlinearrom')
    (staging / 'testsnapdata.txt').write_text(initial_pod.snapshot_catalog(settings, entries))
    (staging / 'testsoldata.txt').write_text(initial_pod.parameter_catalog(settings, entries))
    (staging / 'input.projection').write_text(text)
    campaign.write_json(staging / 'diagnostic.json', {
        'pod': str(pod),
        'truths': [str(entry['target']) for entry in entries],
        'prom_iterations': DIAGNOSTIC_ITS,
    })
    staging.rename(target)
    print('Prepared {} from {} with {} test truths.'.format(target, pod, len(entries)))


def preprocess(count):
    """Compute the truth projection errors and the IC coordinates of the truth catalog."""
    settings = campaign.configure_settings()
    target = diagnostic_dir(settings, count)
    log = target / 'log.projection'
    if not (target / 'input.projection').is_file():
        raise RuntimeError('Run the prepare action first.')
    if log.exists():
        raise RuntimeError('{} exists; refusing to repeat the preprocessing.'.format(log))
    run_aerof(settings, target / 'input.projection', log)
    truths = len(test.read_points())
    if ic_products(target).shape[0] != truths:
        raise RuntimeError('The IC products of {} do not hold the {} truths.'.format(target, truths))
    errors = projection_errors(target)
    if sorted(errors) != list(range(1, truths + 1)):
        raise RuntimeError('Incomplete projection errors in {}.'.format(target))
    print('Relative state projection error: median {:.3e}, max {:.3e}.'.format(
        float(np.median(list(errors.values()))), max(errors.values())))


def run_prom(count, index):
    """Rerun the original PROM input at one test point, starting from the projected truth."""
    settings = campaign.configure_settings()
    pod = campaign.pod_dir(settings, count)
    target = diagnostic_dir(settings, count)
    if not (target / 'log.projection').is_file():
        raise RuntimeError('Run the preprocess action first.')
    original, original_laplace = test.prom_paths(settings, count, index)
    source = original / 'input'
    if not source.is_file():
        raise RuntimeError('Missing {}; run the original PROM first.'.format(source))
    run = target / 'point{:03d}'.format(index)
    if run.exists():
        raise RuntimeError('{} exists; refusing to overwrite a PROM.'.format(run))

    text = patch(source.read_text(), [
        ('MultipleSolutionsData = "{}";'.format((pod / 'parsoldata.txt').as_posix()),
         'MultipleSolutionsData = "{}";'.format((target / 'testsoldata.txt').as_posix()), 1),
        ('LaplaceSnapshotData = "{}/ushift.bin";'.format(original_laplace.as_posix()),
         'LaplaceSnapshotData = "{}/Laplace-bin/ushift.bin";'.format(run.as_posix()), 1),
        ('{}/'.format(original.as_posix()), '{}/'.format(run.as_posix()), None),
        ('Prefix = "{}/";'.format(pod.as_posix()), 'Prefix = "{}/";'.format(target.as_posix()), 1),
        ('   MaxIts = {};\n   Eps = 1e-10;'.format(settings.MaxItsHROM),
         '   MaxIts = {};\n   Eps = 1e-10;'.format(DIAGNOSTIC_ITS), 1),
        # A positive frequency also writes iteration 0, which is the projected truth.
        ('      Frequency = 0;', '      Frequency = {};'.format(DIAGNOSTIC_ITS), 1),
    ], source)
    for stale in ('romruns{:03d}'.format(count), 'reductionrun{:03d}'.format(count)):
        if stale in text:
            raise RuntimeError('{} still refers to {}.'.format(source, stale))

    hdm = test.hdm_dir(index)
    for name in ('results', 'postpro', 'deform', 'Laplace-bin'):
        (run / name).mkdir(parents=True)
    shutil.copy2(original / 'parameters.txt', run / 'parameters.txt')
    for path in sorted((hdm / 'deform').iterdir()):
        shutil.copy2(path, run / 'deform' / path.name)
    for path in sorted((hdm / 'Laplace-bin').glob('ushift.bin*')):
        shutil.copy2(path, run / 'Laplace-bin' / path.name)
    (run / 'input').write_text(text)
    campaign.write_json(run / 'prom.json', {
        'source_input': str(source),
        'initial_condition': 'projection of {} onto POD {}'.format(hdm, count),
        'iterations': DIAGNOSTIC_ITS,
    })
    run_aerof(settings, run / 'input', run / 'log')
    if not (run / 'postpro/Residual.out').is_file():
        raise RuntimeError('PROM failed; inspect {}.'.format(run / 'log'))
    print('Completed the projected-truth PROM {} at test point {:03d}.'.format(count, index))


def xpost_frames(path, rows):
    """Return the given rows of every frame of a merged scalar XPOST file."""
    frames = []
    with open(path) as xpost:
        next(xpost)
        size = int(next(xpost))
        for line in xpost:
            if not line.strip():
                break
            values = np.array([float(next(xpost)) for _ in range(size)])
            frames.append(values[rows])
    return frames


def table(path):
    """Read a numeric AERO-F history file, skipping comments."""
    return np.array([[float(value) for value in line.split()]
                     for line in path.read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith('#')])


def initial_residual(run_directory):
    """Return the full residual norm of the PROM's initial condition."""
    for line in (run_directory / 'log').read_text().splitlines():
        if 'Spatial residual norm = ' in line:
            return float(line.split('Spatial residual norm = ')[1])
    raise RuntimeError('No initial residual in {}.'.format(run_directory / 'log'))


def force_errors(run_directory, truth, row):
    """Return the percent lift and drag errors of one liftdrag.out row against the truth."""
    values = table(run_directory / 'postpro/liftdrag.out')[row]
    return (100.0 * abs(values[5] - truth[5]) / abs(truth[5]),
            100.0 * abs(values[4] - truth[4]) / abs(truth[4]))


def point_diagnostic(settings, frg, wall, count, index, products, state_errors):
    """Compare the projected truth, the PROM started from it, and the original PROM."""
    target = diagnostic_dir(settings, count)
    run = target / 'point{:03d}'.format(index)
    original, _ = test.prom_paths(settings, count, index)
    hdm = test.hdm_dir(index)
    if not (run / 'postpro/Residual.out').is_file():
        return {'status': 'prom-unavailable'}
    test.merge_surface_fields(frg, run, fields=('PressureCoefficient',))
    frames = xpost_frames(run / 'postpro/PressureCoefficient.xpost', wall)
    truth = test.wall_pressure(hdm, wall)
    original_cp = test.wall_pressure(original, wall)
    truth_forces = table(hdm / 'postpro/liftdrag.out')[-1]
    coordinates = table(run / 'postpro/ReducedCoords.out')[:, 3:]
    start = products[index - 1]
    _, _, final_residual = test.prom_residual(run)
    _, _, original_residual = test.prom_residual(original)
    lift_projection, drag_projection = force_errors(run, truth_forces, 0)
    lift_final, drag_final = force_errors(run, truth_forces, -1)
    lift_original, drag_original = force_errors(original, truth_forces, -1)
    return {
        'status': 'ok',
        'frames': len(frames),
        'ic_is_projection': float(np.abs(coordinates[0] - start).max() / np.abs(start).max()),
        'state_projection_error': state_errors[index],
        'cp_projection': test.relative_error(frames[0], truth, 2),
        'cp_from_projection': test.relative_error(frames[-1], truth, 2),
        'cp_original': test.relative_error(original_cp, truth, 2),
        'cp_from_projection_vs_original': test.relative_error(frames[-1], original_cp, 2),
        'lift_projection': lift_projection, 'drag_projection': drag_projection,
        'lift_from_projection': lift_final, 'drag_from_projection': drag_final,
        'lift_original': lift_original, 'drag_original': drag_original,
        'residual_projection': initial_residual(run),
        'residual_from_projection': final_residual,
        'residual_original': original_residual,
        'movement': 100.0 * float(np.linalg.norm(coordinates[-1] - coordinates[0])
                                  / np.linalg.norm(coordinates[0])),
    }


def metrics(count):
    """Write the diagnostic of every projected-truth PROM of one batch."""
    settings = campaign.configure_settings()
    target = diagnostic_dir(settings, count)
    frg = pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
    )
    raw_nodes = np.loadtxt('{}_nodes'.format(settings.TopFilePath), dtype=np.float64)
    wall = test.wall_nodes('{}.top'.format(settings.TopFilePath), raw_nodes)
    products = ic_products(target)
    state_errors = projection_errors(target)
    points = {}
    for run in sorted(target.glob('point[0-9][0-9][0-9]')):
        index = int(run.name[5:])
        points[run.name[5:]] = point_diagnostic(settings, frg, wall, count, index, products,
                                                state_errors)
    campaign.write_json(target / 'metrics.json', {
        'pod': count,
        'unit': 'percent relative error against the truth HDM; cp over the z = 0 wall nodes; '
                'residuals are full-order norms',
        'state_projection_error': {'{:03d}'.format(key): value
                                   for key, value in sorted(state_errors.items())},
        'points': points,
    })
    print('point  state-proj  Cp: projection -> from it | original   '
          'residual: projection -> from it | original')
    for key, value in points.items():
        if value['status'] != 'ok':
            print('{}  {}'.format(key, value['status']))
            continue
        print('{}  {:9.2e}   {:10.1f} -> {:7.1f} | {:7.1f}   {:10.3e} -> {:9.3e} | {:9.3e}'.format(
            key, value['state_projection_error'], value['cp_projection'],
            value['cp_from_projection'], value['cp_original'], value['residual_projection'],
            value['residual_from_projection'], value['residual_original']))


def main():
    """Run one explicit stage of the projected-truth diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('prepare', 'preprocess', 'prom', 'metrics'))
    parser.add_argument('--pod', type=int, required=True)
    parser.add_argument('--test-index', type=int)
    args = parser.parse_args()

    if args.mode == 'prepare':
        prepare(args.pod)
    elif args.mode == 'preprocess':
        preprocess(args.pod)
    elif args.mode == 'prom':
        if args.test_index is None or not 1 <= args.test_index <= test.TEST_COUNT:
            raise ValueError('--test-index must be in [1, {}].'.format(test.TEST_COUNT))
        run_prom(args.pod, args.test_index)
    else:
        metrics(args.pod)


if __name__ == '__main__':
    main()
