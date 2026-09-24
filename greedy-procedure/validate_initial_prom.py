#!/usr/bin/env python3
"""Prepare or run one training-point validation of the initial global PROM."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np
import pyaeroopt

from setup import Settings
from runs import create_deformed_top_file, getRuns, solveCurrentLaplace


POD_INDEX = 32
POINT_INDEX = 1


def configure_settings():
    """Use the completed 32-snapshot global PROM without starting greedy."""
    settings = Settings()
    settings.MasterDir = 'GreedyRuns/'
    settings.RunGreedy = False
    settings.HyperReduced = False
    return settings


def validation_paths(settings):
    """Return the standard ROM and Laplace evaluation directories."""
    master = Path(settings.MasterDir)
    run_dir = master / 'evaluate/romruns{:03d}/point{:03d}'.format(
        POD_INDEX, POINT_INDEX
    )
    laplace_dir = master / 'evaluate/hromruns{:03d}/point{:03d}/Laplace-bin'.format(
        POD_INDEX, POINT_INDEX
    )
    return run_dir, laplace_dir


def load_training_point(settings):
    """Read the exact parameter point recorded by the first initial HDM."""
    metadata_file = Path(settings.MasterDir) / 'HDMrun001/pilot.json'
    if not metadata_file.is_file():
        raise RuntimeError('Missing {}'.format(metadata_file))
    metadata = json.loads(metadata_file.read_text())
    point = metadata.get('point')
    if metadata.get('index') != POINT_INDEX or not isinstance(point, list) or len(point) != 5:
        raise RuntimeError('{} does not describe HDMrun001.'.format(metadata_file))
    return [float(value) for value in point]


def make_frg(settings):
    """Build the full-mesh FRG interface used by the PROM validation."""
    return pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
    )


def check_pod(settings):
    """Require the first and last partitions of the completed global basis."""
    cluster_dir = (
        Path(settings.MasterDir)
        / 'reductionrun{:03d}/nonlinearrom/cluster0'.format(POD_INDEX)
    )
    required = [
        cluster_dir / 'state.rob001',
        cluster_dir / 'state.rob{:03d}'.format(settings.HDMnclust),
        cluster_dir / 'state.ref001',
        cluster_dir / 'state.ref{:03d}'.format(settings.HDMnclust),
        cluster_dir / 'state.svals',
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError('Initial POD is incomplete: {}.'.format(', '.join(missing)))


def prepare(settings):
    """Create geometry, Laplace shift, and AERO-F input for one PROM run."""
    check_pod(settings)
    point = load_training_point(settings)
    run_dir, laplace_dir = validation_paths(settings)
    if run_dir.exists() or laplace_dir.exists():
        raise RuntimeError('Validation output exists; refusing to overwrite it.')

    frg = make_frg(settings)
    rom = getRuns(settings)['ROM'](
        frg,
        p=point,
        HDMind=POD_INDEX,
        pind=POINT_INDEX,
        hyper=False,
        hrtest=False,
        evaluate=True,
    )
    ref_nodes = np.loadtxt(
        '{}_nodes'.format(settings.TopFilePath), dtype=np.float64
    )[:, 1:]
    ref_stick = np.loadtxt(
        '{}_stick'.format(settings.TopFilePath), dtype=np.int32
    )[:, 2:]
    rom.prep(ref_nodes, ref_stick)

    # Recompute the geometry-specific shift to validate the full online path.
    deformed_top = create_deformed_top_file(
        '{}/'.format(run_dir), settings.TopFilePath
    )
    solveCurrentLaplace(
        deformed_top,
        '{}/'.format(laplace_dir.parent),
        frg,
        settings,
        settings.LaplaceNumProc,
    )

    rom.create_input_file()
    rom.writeInputFile()
    required = [
        run_dir / 'input',
        run_dir / 'deform/Position.bin001',
        run_dir / 'deform/Position.bin{:03d}'.format(settings.HDMnclust),
        laplace_dir / 'ushift.bin001',
        laplace_dir / 'ushift.bin{:03d}'.format(settings.HDMnclust),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError('PROM preparation is incomplete: {}.'.format(', '.join(missing)))

    with open(run_dir / 'validation.json', 'w') as validation_file:
        json.dump({
            'pod_index': POD_INDEX,
            'point_index': POINT_INDEX,
            'point': point,
            'reference_hdm': 'GreedyRuns/HDMrun001',
            'hyper_reduced': False,
        }, validation_file, indent=2)
        validation_file.write('\n')
    print('Prepared initial PROM validation at {}.'.format(run_dir))


def last_numeric_row(path):
    """Return the final whitespace-separated numeric row from an output file."""
    if not path.is_file():
        raise RuntimeError('Missing {}'.format(path))
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError('{} is empty.'.format(path))
    try:
        return [float(value) for value in rows[-1]]
    except ValueError as error:
        raise RuntimeError('Cannot parse the final row of {}.'.format(path)) from error


def initial_residual(log_file):
    """Read the initial spatial residual reported by AERO-F."""
    pattern = re.compile(r'Spatial residual norm =\s*([+\-0-9.eE]+)')
    for line in log_file.read_text().splitlines():
        match = pattern.search(line)
        if match:
            return float(match.group(1))
    raise RuntimeError('Cannot find the initial spatial residual in {}.'.format(log_file))


def summarize(settings, run_dir, point):
    """Compare PROM convergence and forces with the training HDM."""
    prom_residual = last_numeric_row(run_dir / 'postpro/Residual.out')
    prom_liftdrag = last_numeric_row(run_dir / 'postpro/liftdrag.out')
    hdm_dir = Path(settings.MasterDir) / 'HDMrun001/postpro'
    hdm_liftdrag = last_numeric_row(hdm_dir / 'liftdrag.out')
    relative_residual = prom_residual[2]
    absolute_residual = relative_residual * initial_residual(run_dir / 'log')
    prom_drag, prom_lift = prom_liftdrag[4], prom_liftdrag[5]
    hdm_drag, hdm_lift = hdm_liftdrag[4], hdm_liftdrag[5]

    summary = {
        'pod_index': POD_INDEX,
        'point_index': POINT_INDEX,
        'point': point,
        'iterations': int(prom_residual[0]),
        'relative_residual': relative_residual,
        'absolute_residual': absolute_residual,
        'prom_drag': prom_drag,
        'hdm_drag': hdm_drag,
        'drag_relative_error': abs(prom_drag - hdm_drag) / abs(hdm_drag),
        'prom_lift': prom_lift,
        'hdm_lift': hdm_lift,
        'lift_relative_error': abs(prom_lift - hdm_lift) / abs(hdm_lift),
    }
    with open(run_dir / 'validation_summary.json', 'w') as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write('\n')

    print('PROM iterations: {}'.format(summary['iterations']))
    print('PROM relative residual: {:.6e}'.format(relative_residual))
    print('Drag: PROM {:.8e}, HDM {:.8e}, relative error {:.6e}'.format(
        prom_drag, hdm_drag, summary['drag_relative_error']))
    print('Lift: PROM {:.8e}, HDM {:.8e}, relative error {:.6e}'.format(
        prom_lift, hdm_lift, summary['lift_relative_error']))


def run(settings):
    """Execute the prepared PROM input and write its HDM comparison summary."""
    point = load_training_point(settings)
    run_dir, laplace_dir = validation_paths(settings)
    required = [
        run_dir / 'input',
        laplace_dir / 'ushift.bin001',
        laplace_dir / 'ushift.bin{:03d}'.format(settings.HDMnclust),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError('Missing prepared PROM files: {}.'.format(', '.join(missing)))

    aerof = os.environ.get('AEROF')
    if not aerof or not Path(aerof).is_file():
        raise RuntimeError('AEROF must name the built AERO-F executable.')
    command = shlex.split(os.environ.get('MPI', 'mpirun')) + [
        '-n', str(settings.HDMnproc), aerof, str(run_dir / 'input')
    ]
    print(' '.join(command), flush=True)
    with open(run_dir / 'log', 'w') as log_file:
        result = subprocess.run(
            command, stdout=log_file, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode != 0:
        raise RuntimeError('PROM validation failed; inspect {}/log.'.format(run_dir))
    summarize(settings, run_dir, point)
    print('Initial PROM validation completed: {}.'.format(run_dir))


def main():
    """Run one stage of the initial PROM validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    args = parser.parse_args()
    settings = configure_settings()
    if args.mode == 'prepare':
        prepare(settings)
    else:
        run(settings)


if __name__ == '__main__':
    main()
