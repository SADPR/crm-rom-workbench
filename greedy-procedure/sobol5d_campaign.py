#!/usr/bin/env python3
"""Run the 5D transonic Sobol HDM campaign and build its POD batches."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyaeroopt

import initial_hdm_campaign
import initial_pod
from runs import getRuns
from setup import Settings
from sobolGenerator import sobolGenerator


MASTER_DIR = 'Sobol5DRuns/'
PRECOMPUTE_DIR = 'Sobol5DPrecompute/'
LOG_DIR = 'Sobol5DRuns/logs/'
# Yihong Zhu's final box: [Mach, AoA, max-camber location, max camber, thickness].
LOWER_BOUNDS = [0.6, -2.0, 0.2, 0.0, 0.09]
UPPER_BOUNDS = [0.8, 2.0, 0.4, 0.03, 0.12]
# Reconstruction beta of the flow and turbulence Space blocks, not the inlet angle.
RECONSTRUCTION_BETA = 0.5
MAX_COUNT = 512
BATCH_COUNTS = (128, 256, 512)


def configure_settings():
    """Pin the 5D box, beta = 0.5, and the campaign roots in one place."""
    settings = Settings()
    settings.MasterDir = MASTER_DIR
    settings.InitHDMPreCompDir = PRECOMPUTE_DIR
    settings.RunGreedy = False
    settings.PODMethod = 'ScalapackSVD'
    settings.ParamsLowerBound = list(LOWER_BOUNDS)
    settings.ParamsUpperBound = list(UPPER_BOUNDS)
    settings.Beta = RECONSTRUCTION_BETA
    settings.numrand = MAX_COUNT
    settings.numSkip = 0
    return settings


def write_json(path, data):
    """Write small campaign metadata without exposing a partial file."""
    path = Path(path)
    temporary = path.with_name('{}.tmp'.format(path.name))
    temporary.write_text(json.dumps(data, indent=2, default=str) + '\n')
    os.replace(temporary, path)


def manifest_path():
    """Return the campaign manifest that freezes the design and batches."""
    return Path(MASTER_DIR) / 'campaign.json'


def read_manifest():
    """Read the frozen campaign manifest."""
    if not manifest_path().is_file():
        raise RuntimeError('Run the init action first.')
    return json.loads(manifest_path().read_text())


def design_points(settings):
    """Return the nested training design: 32 corners, then unscrambled Sobol points."""
    ranges = list(zip(settings.ParamsLowerBound, settings.ParamsUpperBound))
    points = sobolGenerator(ranges, settings.numrand, include_corners=True,
                            numToSkip=settings.numSkip, make_plot=False)
    return [[round(float(value), 6) for value in point] for point in points]


def run_dir(index, root=PRECOMPUTE_DIR):
    """Return the directory of one one-based design point below a campaign root."""
    return Path(root) / 'HDMrun{:03d}'.format(index)


def static_files(settings):
    """List the shared decomposition files required by every HDM sample."""
    prefix = '{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix)
    return [
        '{}.top'.format(settings.TopFilePath),
        '{}.top.dec.{}'.format(settings.TopFilePath, settings.HDMnclust),
        '{}.con'.format(prefix),
        '{}.msh001'.format(prefix),
        '{}.msh{:03d}'.format(prefix, settings.HDMnclust),
        '{}.dwall001'.format(prefix),
        '{}.dwall{:03d}'.format(prefix, settings.HDMnclust),
    ]


def initialize(settings):
    """Create the campaign roots and freeze the training design before any job runs."""
    master_dir = Path(settings.MasterDir)
    precompute_dir = Path(settings.InitHDMPreCompDir)
    if master_dir.exists() or precompute_dir.exists():
        raise RuntimeError('{} or {} already exists; refusing to reuse a campaign root.'.format(
            master_dir, precompute_dir
        ))
    master_dir.mkdir()
    precompute_dir.mkdir()
    Path(LOG_DIR).mkdir()
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    write_json(master_dir / 'settings.effective.json', vars(settings))
    write_json(manifest_path(), {
        'name': 'sobol5d-transonic-steady-crm',
        'source_commit': source_commit,
        'lower_bounds': settings.ParamsLowerBound,
        'upper_bounds': settings.ParamsUpperBound,
        'reconstruction_beta': settings.Beta,
        'shift_type': settings.ShiftType,
        'laplace_shift_each': settings.LaplaceShiftEach,
        'laplace_boundaries': {'InletFixed_2': 1, 'StickMoving_3': 0, 'Symmetry_1': 'natural'},
        'form': settings.Form,
        'pod_method': settings.PODMethod,
        'hdm_tolerance': settings.HDMtol2,
        'design': 'sobolGenerator(..., include_corners=True): 32 corners, then unscrambled Sobol',
        'batch_counts': list(BATCH_COUNTS),
        'points': design_points(settings),
        'batches': {},
    })
    print('Initialized {} and {} with {} design points.'.format(
        master_dir, precompute_dir, MAX_COUNT
    ))


def prepare_static_data(settings):
    """Build a fresh shared mesh decomposition for the campaign."""
    data_dir = Path(settings.MasterDir) / 'data'
    read_manifest()
    if data_dir.exists():
        missing = [path for path in static_files(settings) if not Path(path).is_file()]
        if missing:
            raise RuntimeError('Incomplete static data: {}.'.format(', '.join(missing)))
        return

    data_dir.mkdir()
    frg = pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
    )
    frg.part_mesh(settings.HDMnclust, log='/dev/null')
    frg.sower_fluid_top(
        [settings.HDMnclust, settings.HDMnproc, settings.HROMnproc, 1],
        settings.HDMnclust,
        log='/dev/null',
    )
    pyaeroopt.util.frg_util.run_cd2tet_fromtop(
        '{}.top'.format(settings.TopFilePath),
        '{}.sinus'.format(frg.geom_pre),
        log='/dev/null',
    )
    frg.sower_fluid_split(
        file2split='{}.sinus.dwall'.format(frg.geom_pre),
        out='{}.dwall'.format(frg.geom_pre),
        nclust=settings.HDMnclust,
        log='/dev/null',
    )

    missing = [path for path in static_files(settings) if not Path(path).is_file()]
    if missing:
        raise RuntimeError('Missing static data: {}.'.format(', '.join(missing)))
    print('Prepared static CRM data in {}.'.format(data_dir))


def record_shift_stats(run_directory):
    """Require the geometry-specific Laplace shift to be nonconstant."""
    shift_files = sorted((run_directory / 'xdmf_files').glob('*-u_shift.xpost'))
    if len(shift_files) != 1:
        raise RuntimeError('Expected one Laplace shift XPOST in {}, found {}.'.format(
            run_directory / 'xdmf_files', len(shift_files)
        ))
    _, values = pyaeroopt.util.frg_util.read_xpost(str(shift_files[0]))
    values = np.asarray(values, dtype=np.float64)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum - minimum <= 1.0e-12:
        raise RuntimeError('Laplace shift in {} is constant or non-finite.'.format(shift_files[0]))
    write_json(run_directory / 'laplace_shift.json', {
        'xpost': shift_files[0].name,
        'minimum': minimum,
        'maximum': maximum,
        'range': maximum - minimum,
    })
    print('Verified nonconstant Laplace shift: [{:.6e}, {:.6e}].'.format(minimum, maximum))


def final_residual(run_directory, tolerance):
    """Read the final HDM residual and require the campaign tolerance."""
    residual = initial_pod.read_final_residual(run_directory)
    if residual > tolerance:
        raise RuntimeError('Final residual {:.6e} exceeds {:.6e} in {}.'.format(
            residual, tolerance, run_directory
        ))
    return residual


def prepare(settings, run_index):
    """Prepare one frozen design point and its geometry-specific Laplace shift."""
    points = read_manifest()['points']
    if not 1 <= run_index <= len(points):
        raise ValueError('--run-index must be in [1, {}].'.format(len(points)))
    if run_index == 1:
        prepare_static_data(settings)
    point, _ = initial_hdm_campaign.get_point(settings, run_index)
    if point != points[run_index - 1]:
        raise RuntimeError('Design point {} differs from the frozen manifest.'.format(run_index))
    initial_hdm_campaign.prepare(settings, run_index)
    record_shift_stats(run_dir(run_index, settings.InitHDMPreCompDir))


def run(settings, run_index):
    """Run both HDM phases and require the campaign tolerance."""
    initial_hdm_campaign.run(settings, run_index)
    directory = run_dir(run_index, settings.InitHDMPreCompDir)
    residual = final_residual(directory, settings.HDMtol2)
    print('Validated {} with final residual {:.6e}.'.format(directory, residual))


def classify(settings, index, point):
    """Return the state, final residual, and directory of one design point."""
    locations = [run_dir(index, root) for root in (settings.MasterDir, settings.InitHDMPreCompDir)]
    existing = [location for location in locations if location.exists()]
    if len(existing) > 1:
        raise RuntimeError('HDM {:03d} exists in both campaign roots.'.format(index))
    if not existing:
        return 'missing', None, None
    directory = existing[0]
    required = [
        directory / 'parameters.json',
        directory / 'laplace_shift.json',
        directory / 'references/Restart.data',
        directory / 'snapshots/State.bin001',
        directory / 'snapshots/State.bin{:03d}'.format(settings.HDMnclust),
        directory / 'postpro/Residual.out',
    ]
    if not all(path.is_file() for path in required):
        return 'incomplete', None, directory
    if json.loads(required[0].read_text()).get('point') != point:
        raise RuntimeError('HDM {:03d} does not match the frozen design.'.format(index))
    if float(json.loads(required[1].read_text()).get('range', 0.0)) <= 1.0e-12:
        return 'constant-shift', None, directory
    residual = initial_pod.read_final_residual(directory)
    state = 'converged' if residual <= settings.HDMtol2 else 'unconverged'
    return state, residual, directory


def audit(settings, count):
    """Report every design point up to count without changing any directory."""
    points = read_manifest()['points'][:count]
    rows = [(index,) + classify(settings, index, point)
            for index, point in enumerate(points, start=1)]
    tally = Counter(row[1] for row in rows)
    print('Audited {} design points: {}.'.format(
        count, ', '.join('{} {}'.format(value, key) for key, value in sorted(tally.items()))
    ))
    for index, state, residual, _ in rows:
        if state != 'converged':
            suffix = '' if residual is None else ' ({:.6e})'.format(residual)
            print('  HDM {:03d}: {}{}'.format(index, state, suffix))
    return rows


def assemble(settings, count):
    """Move converged HDMs into the training root and write the batch catalogs."""
    manifest = read_manifest()
    rows = audit(settings, count)
    pending = [index for index, state, _, _ in rows if state in ('missing', 'incomplete')]
    if pending:
        raise RuntimeError('HDMs still missing or incomplete: {}.'.format(pending))
    entries = []
    for index, state, _, directory in rows:
        if state != 'converged':
            continue
        target = run_dir(index, settings.MasterDir)
        if directory != target:
            directory.rename(target)
        entries.append({'index': index, 'point': manifest['points'][index - 1], 'target': target})
    state_path, parameter_path = initial_pod.catalog_paths(settings)
    initial_pod.write_atomically(state_path, initial_pod.snapshot_catalog(settings, entries))
    initial_pod.write_atomically(parameter_path, initial_pod.parameter_catalog(settings, entries))
    # Unconverged or invalid HDMs stay in the precompute root and are listed, never dropped silently.
    manifest['batches'][str(count)] = {
        'included': [entry['index'] for entry in entries],
        'excluded': {str(index): state for index, state, _, _ in rows if state != 'converged'},
    }
    write_json(manifest_path(), manifest)
    print('Wrote catalogs for batch {} with {} snapshots.'.format(count, len(entries)))


def pod_dir(settings, count):
    """Return the reduction directory of one batch."""
    return Path(settings.MasterDir) / 'reductionrun{:03d}'.format(count)


def build_pod(settings, count):
    """Build the ScaLAPACK POD of one assembled batch and keep a copy of its catalogs."""
    manifest = read_manifest()
    batch = manifest['batches'].get(str(count))
    if batch is None:
        raise RuntimeError('Assemble batch {} before building its POD.'.format(count))
    state_path, parameter_path = initial_pod.catalog_paths(settings)
    if int(state_path.read_text().splitlines()[0]) != len(batch['included']):
        raise RuntimeError('The root catalogs do not describe batch {}.'.format(count))
    directory = pod_dir(settings, count)
    if directory.exists():
        raise RuntimeError('{} exists; refusing to overwrite a POD.'.format(directory))
    aerof = os.environ.get('AEROF')
    if not aerof or not Path(aerof).is_file():
        raise RuntimeError('AEROF must name the built AERO-F executable.')

    points = [manifest['points'][index - 1] for index in batch['included']]
    centroid = [sum(point[column] for point in points) / len(points)
                for column in range(len(points[0]))]
    frg = pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
    )
    pod = getRuns(settings)['POD'](frg, p=centroid, HDMind=count)
    hpc = pyaeroopt.interface.Hpc(machine='independence', batch=False, bg=False,
                                  nproc=settings.HDMnproc)
    hpc.mpi = os.environ.get('MPI', 'srun')
    pod.create_input_file()
    pod.writeInputFile()
    command = hpc.execute_str(pod.bin, pod.infile.fname)
    print(command, flush=True)
    with open(pod.infile.log, 'w') as log_file:
        result = subprocess.run(command, shell=True, stdout=log_file, stderr=subprocess.STDOUT,
                                check=False)
    if result.returncode != 0:
        raise RuntimeError('POD {} failed; inspect {}.'.format(count, pod.infile.log))
    required = [
        directory / 'nonlinearrom/cluster0/state.rob001',
        directory / 'nonlinearrom/cluster0/state.rob{:03d}'.format(settings.HDMnclust),
        directory / 'nonlinearrom/cluster0/state.ref001',
        directory / 'nonlinearrom/cluster0/state.ref{:03d}'.format(settings.HDMnclust),
        directory / 'nonlinearrom/cluster0/state.svals',
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError('POD {} is incomplete: {}.'.format(count, ', '.join(missing)))
    # Later batches rewrite the root catalogs; PROMs of this batch read these copies.
    shutil.copy2(state_path, directory / state_path.name)
    shutil.copy2(parameter_path, directory / parameter_path.name)
    print('Completed ScaLAPACK POD {} with {} snapshots.'.format(count, len(points)))


def main():
    """Run one explicit stage of the 5D Sobol campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('init', 'point', 'prepare', 'run', 'audit', 'assemble', 'pod'))
    parser.add_argument('--run-index', type=int)
    parser.add_argument('--count', type=int)
    args = parser.parse_args()

    settings = configure_settings()
    if args.mode == 'init':
        initialize(settings)
    elif args.mode in ('point', 'prepare', 'run'):
        if args.run_index is None:
            raise ValueError('--run-index is required for {}.'.format(args.mode))
        if args.mode == 'point':
            print('{}/{} {}'.format(args.run_index, MAX_COUNT, read_manifest()['points'][args.run_index - 1]))
        elif args.mode == 'prepare':
            prepare(settings, args.run_index)
        else:
            run(settings, args.run_index)
    else:
        if args.count is None or not 1 <= args.count <= MAX_COUNT:
            raise ValueError('--count must be in [1, {}].'.format(MAX_COUNT))
        if args.mode == 'audit':
            audit(settings, args.count)
        elif args.mode == 'assemble':
            assemble(settings, args.count)
        else:
            build_pod(settings, args.count)


if __name__ == '__main__':
    main()
