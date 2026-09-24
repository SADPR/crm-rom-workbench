#!/usr/bin/env python3
"""Prepare or run the first non-nominal steady HDM initial sample."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyaeroopt

from setup import Settings
from runs import getRuns


DEFAULT_POINT = [0.4, -5.0, 0.3, 0.03, 0.12]
RUN_INDEX = 1


def configure_settings():
    """Use production HDM paths without starting the greedy algorithm."""
    settings = Settings()
    settings.MasterDir = 'GreedyRuns/'
    settings.InitHDMPreCompDir = 'InitialHDMruns/'
    settings.RunGreedy = False
    return settings


def prepare(settings, point):
    """Create the first Sobol corner's Laplace shift and two HDM inputs."""
    master_dir = Path(settings.MasterDir)
    run_dir = Path(settings.InitHDMPreCompDir) / 'HDMrun{:03d}'.format(RUN_INDEX)
    if master_dir.exists():
        raise RuntimeError(
            '{} exists; refusing to replace shared greedy data.'.format(master_dir)
        )
    if run_dir.exists():
        raise RuntimeError('{} already exists; refusing to overwrite it.'.format(run_dir))

    master_dir.mkdir()
    (master_dir / 'data').mkdir()
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

    runs = getRuns(settings)
    hdm1 = runs['HDM'](
        frg, p=point, HDMind=RUN_INDEX, step=1, maindir=True
    )
    hdm2 = runs['HDM'](
        frg, p=point, HDMind=RUN_INDEX, step=2, maindir=True
    )
    ref_nodes = np.loadtxt('{}_nodes'.format(settings.TopFilePath), dtype=np.float64)[:, 1:]
    ref_stick = np.loadtxt('{}_stick'.format(settings.TopFilePath), dtype=np.int32)[:, 2:]

    hdm1.prep(ref_nodes, ref_stick)
    for hdm in (hdm1, hdm2):
        hdm.create_input_file()
        hdm.writeInputFile()

    for suffix in ('001', '120'):
        shift = run_dir / 'Laplace-bin/ushift.bin{}'.format(suffix)
        if not shift.is_file():
            raise RuntimeError('Missing {}'.format(shift))

    with open(run_dir / 'pilot.json', 'w') as pilot_file:
        json.dump({
            'index': RUN_INDEX,
            'point': point,
            'role': 'first 32-point Sobol initialization corner',
        }, pilot_file, indent=2)
        pilot_file.write('\n')

    print('Prepared {}'.format(run_dir))
    print('Review input1 and input2 before submitting the pilot.')


def run(settings):
    """Run the prepared pilot without updating greedy snapshot catalogs."""
    run_dir = Path(settings.InitHDMPreCompDir) / 'HDMrun{:03d}'.format(RUN_INDEX)
    inputs = [run_dir / 'input1', run_dir / 'input2']
    missing = [str(input_file) for input_file in inputs if not input_file.is_file()]
    if missing:
        raise RuntimeError('Missing prepared input files: {}'.format(', '.join(missing)))

    aerof = os.environ.get('AEROF')
    if not aerof:
        raise RuntimeError('AEROF must name the AERO-F executable.')

    hpc = pyaeroopt.interface.Hpc(
        machine='independence',
        batch=False,
        bg=False,
        nproc=settings.HDMnproc,
    )
    hpc.mpi = os.environ.get('MPI', 'mpirun')

    for step, input_file in enumerate(inputs, start=1):
        command = hpc.execute_str(aerof, str(input_file))
        print(command, flush=True)
        with open(run_dir / 'log{}'.format(step), 'w') as log_file:
            result = subprocess.run(
                command,
                shell=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError('AERO-F step {} failed; inspect log{}.'.format(step, step))

    state = run_dir / 'snapshots/State.bin001'
    if not state.is_file():
        raise RuntimeError('Missing {}'.format(state))
    print('Pilot HDM completed: {}'.format(run_dir))


def main():
    """Parse the requested pilot action and parameter point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument(
        '--point',
        nargs=5,
        type=float,
        default=DEFAULT_POINT,
        metavar=('MACH', 'AOA', 'CAMBER_LOC', 'CAMBER', 'THICKNESS'),
    )
    args = parser.parse_args()

    settings = configure_settings()
    if args.mode == 'prepare':
        prepare(settings, args.point)
    else:
        run(settings)


if __name__ == '__main__':
    main()
