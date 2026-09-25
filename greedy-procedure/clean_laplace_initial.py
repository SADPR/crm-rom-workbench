#!/usr/bin/env python3
"""Run the isolated, clean 32-point CRM Laplace/HDM initialization."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyaeroopt

from initial_hdm_campaign import get_point, prepare as prepare_point, run as run_point
from initial_pod import initial_points
from setup import Settings


MASTER_DIR = 'CleanLaplaceRuns/'
PRECOMPUTE_DIR = 'CleanLaplacePrecompute/'
RUN_COUNT = 32


def configure_settings():
    """Use dedicated paths without starting the greedy algorithm."""
    settings = Settings()
    settings.MasterDir = MASTER_DIR
    settings.InitHDMPreCompDir = PRECOMPUTE_DIR
    settings.RunGreedy = False
    return settings


def run_dir(settings, run_index):
    """Return the isolated directory for one one-based Sobol sample."""
    return Path(settings.InitHDMPreCompDir) / 'HDMrun{:03d}'.format(run_index)


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


def write_json(path, data):
    """Write small campaign metadata with stable indentation."""
    with open(path, 'w') as output:
        json.dump(data, output, indent=2)
        output.write('\n')


def initialize(settings):
    """Create empty, dedicated campaign roots before Slurm writes logs there."""
    master_dir = Path(settings.MasterDir)
    precompute_dir = Path(settings.InitHDMPreCompDir)
    if master_dir.exists() or precompute_dir.exists():
        raise RuntimeError(
            '{} or {} already exists; refusing to reuse a clean campaign root.'.format(
                master_dir, precompute_dir
            )
        )

    master_dir.mkdir()
    precompute_dir.mkdir()
    points = initial_points(settings)
    if len(points) != RUN_COUNT:
        raise RuntimeError("Expected {} Sobol points, generated {}.".format(RUN_COUNT, len(points)))
    source_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True
    ).strip()
    write_json(master_dir / 'campaign.json', {
        'name': 'clean-laplace-steady-crm-initialization',
        'source_commit': source_commit,
        'run_count': RUN_COUNT,
        'master_dir': settings.MasterDir,
        'precompute_dir': settings.InitHDMPreCompDir,
        'shift_type': settings.ShiftType,
        'laplace_shift_each': settings.LaplaceShiftEach,
        'laplace_boundaries': {
            'InletFixed_2': 1,
            'StickMoving_3': 0,
            'Symmetry_1': 'natural',
        },
        'points': points,
    })
    print('Initialized {} and {}.'.format(master_dir, precompute_dir))


def prepare_static_data(settings):
    """Build a fresh shared mesh decomposition for the clean campaign."""
    master_dir = Path(settings.MasterDir)
    marker = master_dir / 'campaign.json'
    data_dir = master_dir / 'data'
    if not marker.is_file():
        raise RuntimeError('Run the init action before preparing clean HDMs.')
    if data_dir.exists():
        missing = [path for path in static_files(settings) if not Path(path).is_file()]
        if missing:
            raise RuntimeError('Incomplete clean static data: {}.'.format(', '.join(missing)))
        return

    data_dir.mkdir()
    shutil.copy2('setup.py', master_dir / 'settings.readonly')
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
        raise RuntimeError('Missing clean static data: {}.'.format(', '.join(missing)))
    print('Prepared clean static CRM data in {}.'.format(data_dir))


def record_shift_stats(run_directory):
    """Require the geometry-specific Laplace shift to be nonconstant."""
    shift_files = sorted((run_directory / 'xdmf_files').glob('*-u_shift.xpost'))
    if len(shift_files) != 1:
        raise RuntimeError(
            'Expected one Laplace shift XPOST in {}, found {}.'.format(
                run_directory / 'xdmf_files', len(shift_files)
            )
        )
    _, values = pyaeroopt.util.frg_util.read_xpost(str(shift_files[0]))
    values = np.asarray(values, dtype=np.float64)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum - minimum <= 1.0e-12:
        raise RuntimeError(
            'Laplace shift in {} is constant or non-finite.'.format(shift_files[0])
        )
    write_json(run_directory / 'laplace_shift.json', {
        'xpost': shift_files[0].name,
        'minimum': minimum,
        'maximum': maximum,
        'range': maximum - minimum,
    })
    print('Verified nonconstant Laplace shift: [{:.6e}, {:.6e}].'.format(
        minimum, maximum
    ))


def final_residual(run_directory, tolerance):
    """Read and validate the final residual from the restarted HDM phase."""
    residual_file = run_directory / 'postpro/Residual.out'
    if not residual_file.is_file():
        raise RuntimeError('Missing {}.'.format(residual_file))
    lines = [line.split() for line in residual_file.read_text().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError('{} is empty.'.format(residual_file))
    try:
        residual = float(lines[-1][2])
    except (IndexError, ValueError) as error:
        raise RuntimeError('Cannot read the final residual from {}.'.format(residual_file)) from error
    if residual > tolerance:
        raise RuntimeError(
            'Final residual {:.6e} exceeds {:.6e} in {}.'.format(
                residual, tolerance, run_directory
            )
        )
    return residual


def prepare(settings, run_index):
    """Prepare one exact Sobol point and its corrected Laplace field."""
    if run_index == 1:
        prepare_static_data(settings)
    else:
        missing = [path for path in static_files(settings) if not Path(path).is_file()]
        if missing:
            raise RuntimeError('Missing clean static data: {}.'.format(', '.join(missing)))

    prepare_point(settings, run_index)
    directory = run_dir(settings, run_index)
    if run_index == 1:
        shutil.copy2(directory / 'parameters.json', directory / 'pilot.json')
    record_shift_stats(directory)


def run(settings, run_index):
    """Run both HDM phases and require the requested final convergence."""
    run_point(settings, run_index)
    directory = run_dir(settings, run_index)
    residual = final_residual(directory, settings.HDMtol2)
    print('Validated {} with final residual {:.6e}.'.format(directory, residual))


def audit(settings):
    """Check every clean sample before it is assembled into the POD catalog."""
    points = initial_points(settings)
    if len(points) != RUN_COUNT:
        raise RuntimeError("Expected {} Sobol points, generated {}.".format(RUN_COUNT, len(points)))
    for run_index in range(1, RUN_COUNT + 1):
        directory = run_dir(settings, run_index)
        metadata = directory / ('pilot.json' if run_index == 1 else 'parameters.json')
        shift_metadata = directory / 'laplace_shift.json'
        if not metadata.is_file() or not shift_metadata.is_file():
            raise RuntimeError('Missing campaign metadata for {}.'.format(directory))
        recorded = json.loads(metadata.read_text())
        expected = points[run_index - 1]
        if recorded.get('point') != expected:
            raise RuntimeError('Unexpected Sobol point in {}.'.format(metadata))
        shift = json.loads(shift_metadata.read_text())
        if float(shift.get('range', 0.0)) <= 1.0e-12:
            raise RuntimeError('Invalid Laplace shift metadata in {}.'.format(shift_metadata))
        final_residual(directory, settings.HDMtol2)
    print('Audited {} clean Laplace HDMs.'.format(RUN_COUNT))


def main():
    """Run one explicit stage of the clean initial campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('init', 'point', 'prepare', 'run', 'audit'))
    parser.add_argument('--run-index', type=int)
    args = parser.parse_args()

    settings = configure_settings()
    if args.mode == 'init':
        if args.run_index is not None:
            raise ValueError('The init action does not take --run-index.')
        initialize(settings)
        return
    if args.mode == 'audit':
        if args.run_index is not None:
            raise ValueError('The audit action does not take --run-index.')
        audit(settings)
        return
    if args.run_index is None or not 1 <= args.run_index <= RUN_COUNT:
        raise ValueError('--run-index must be in [1, {}].'.format(RUN_COUNT))
    if args.mode == 'point':
        point, point_count = get_point(settings, args.run_index)
        print('{}/{} {}'.format(args.run_index, point_count, point))
    elif args.mode == 'prepare':
        prepare(settings, args.run_index)
    else:
        run(settings, args.run_index)


if __name__ == '__main__':
    main()
