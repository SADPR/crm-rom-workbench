#!/usr/bin/env python3
"""Prepare or run one initial Sobol steady HDM sample."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pyaeroopt

from setup import Settings
from runs import getRuns
from sobolGenerator import sobolGenerator


def configure_settings():
    """Use the production paths without starting the greedy algorithm."""
    settings = Settings()
    settings.MasterDir = 'GreedyRuns/'
    settings.InitHDMPreCompDir = 'InitialHDMruns/'
    settings.RunGreedy = False
    return settings


def get_point(settings, run_index):
    """Return one reproducible initial Sobol point without making a plot."""
    ranges = list(zip(settings.ParamsLowerBound, settings.ParamsUpperBound))
    points = sobolGenerator(
        ranges,
        settings.numrand,
        include_corners=True,
        numToSkip=settings.numSkip,
        make_plot=False,
    )
    if run_index < 1 or run_index > len(points):
        raise ValueError('Run index must be in [1, {}].'.format(len(points)))
    return [round(float(value), 6) for value in points[run_index - 1]], len(points)


def shared_files(settings):
    """Return the static decomposition files prepared by the pilot."""
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


def prepare(settings, run_index):
    """Prepare one isolated initial HDM run and its geometry-specific shift."""
    point, point_count = get_point(settings, run_index)
    run_dir = Path(settings.InitHDMPreCompDir) / 'HDMrun{:03d}'.format(run_index)
    if run_dir.exists():
        raise RuntimeError('{} exists; refusing to overwrite it.'.format(run_dir))

    missing = [path for path in shared_files(settings) if not Path(path).is_file()]
    if missing:
        raise RuntimeError('Missing pilot-prepared static data: {}'.format(', '.join(missing)))

    frg = pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
    )
    runs = getRuns(settings)
    hdm1 = runs['HDM'](frg, p=point, HDMind=run_index, step=1, maindir=True)
    hdm2 = runs['HDM'](frg, p=point, HDMind=run_index, step=2, maindir=True)
    ref_nodes = np.loadtxt('{}_nodes'.format(settings.TopFilePath), dtype=np.float64)[:, 1:]
    ref_stick = np.loadtxt('{}_stick'.format(settings.TopFilePath), dtype=np.int32)[:, 2:]

    hdm1.prep(ref_nodes, ref_stick)
    for hdm in (hdm1, hdm2):
        hdm.create_input_file()
        hdm.writeInputFile()

    for suffix in ('001', '{:03d}'.format(settings.HDMnclust)):
        shift = run_dir / 'Laplace-bin/ushift.bin{}'.format(suffix)
        if not shift.is_file():
            raise RuntimeError('Missing {}'.format(shift))

    with open(run_dir / 'parameters.json', 'w') as parameter_file:
        json.dump({
            'index': run_index,
            'point': point,
            'point_count': point_count,
            'source': 'sobolGenerator(..., include_corners=True)',
        }, parameter_file, indent=2)
        parameter_file.write('\n')
    print('Prepared {} for {}'.format(run_dir, point))


def run(settings, run_index):
    """Run the two prepared HDM phases for one initial Sobol point."""
    run_dir = Path(settings.InitHDMPreCompDir) / 'HDMrun{:03d}'.format(run_index)
    inputs = [run_dir / 'input1', run_dir / 'input2']
    missing = [str(input_file) for input_file in inputs if not input_file.is_file()]
    if missing:
        raise RuntimeError('Missing prepared inputs: {}'.format(', '.join(missing)))

    aerof = os.environ.get('AEROF')
    if not aerof:
        raise RuntimeError('AEROF must name the AERO-F executable.')
    hpc = pyaeroopt.interface.Hpc(
        machine='independence', batch=False, bg=False, nproc=settings.HDMnproc
    )
    hpc.mpi = os.environ.get('MPI', 'mpirun')

    for step, input_file in enumerate(inputs, start=1):
        command = hpc.execute_str(aerof, str(input_file))
        print(command, flush=True)
        with open(run_dir / 'log{}'.format(step), 'w') as log_file:
            result = subprocess.run(
                command, shell=True, stdout=log_file, stderr=subprocess.STDOUT, check=False
            )
        if result.returncode != 0:
            raise RuntimeError('AERO-F step {} failed; inspect log{}.'.format(step, step))

    for suffix in ('001', '{:03d}'.format(settings.HDMnclust)):
        state = run_dir / 'snapshots/State.bin{}'.format(suffix)
        if not state.is_file():
            raise RuntimeError('Missing {}'.format(state))
    print('Initial HDM completed: {}'.format(run_dir))


def main():
    """Parse one campaign action and its one-based Sobol run index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('point', 'prepare', 'run'))
    parser.add_argument('--run-index', required=True, type=int)
    args = parser.parse_args()

    settings = configure_settings()
    if args.mode == 'point':
        point, point_count = get_point(settings, args.run_index)
        print('{}/{} {}'.format(args.run_index, point_count, point))
    elif args.mode == 'prepare':
        prepare(settings, args.run_index)
    else:
        run(settings, args.run_index)


if __name__ == '__main__':
    main()
