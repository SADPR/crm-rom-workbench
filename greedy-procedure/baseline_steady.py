#!/usr/bin/env python3
"""Prepare or run one isolated nominal steady HDM baseline."""

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


DEFAULT_POINT = [0.5, 0.0, 0.45, 0.03, 0.12]


def configure_settings():
    """Reuse the supplied steady settings with isolated baseline outputs."""
    settings = Settings()
    settings.MasterDir = 'BaselineRuns/'
    settings.RunGreedy = False
    return settings


def prepare(settings, point):
    """Create the deformed mesh and both HDM inputs without running AERO-F."""
    baseline_dir = Path(settings.MasterDir)
    if baseline_dir.exists():
        raise RuntimeError(
            '{} already exists; refusing to overwrite a baseline.'.format(baseline_dir)
        )

    baseline_dir.mkdir()
    (baseline_dir / 'data').mkdir()
    shutil.copy2('setup.py', baseline_dir / 'settings.readonly')

    with open(baseline_dir / 'baseline.json', 'w') as baseline_file:
        json.dump({
            'point': point,
            'problem_type': 'Steady',
            'hdm_partitions': settings.HDMnclust,
            'hdm_processes': settings.HDMnproc,
        }, baseline_file, indent=2)
        baseline_file.write('\n')

    frg = pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
    )

    # Mirror main.py's mesh setup without constructing a GreedyAlgorithmRun.
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
    hdm1 = runs['HDM'](frg, p=point, HDMind=1, step=1)
    hdm2 = runs['HDM'](frg, p=point, HDMind=1, step=2)

    # Match GreedyAlgorithm: discard mesh file identifiers before deformation.
    ref_nodes = np.loadtxt('{}_nodes'.format(settings.TopFilePath), dtype=np.float64)[:, 1:]
    ref_stick = np.loadtxt('{}_stick'.format(settings.TopFilePath), dtype=np.int32)[:, 2:]

    hdm1.prep(ref_nodes, ref_stick)
    for hdm in (hdm1, hdm2):
        hdm.create_input_file()
        hdm.writeInputFile()

    print('Prepared {}'.format(hdm1.SuperDir))
    print('Review input1 and input2 before running this baseline.')


def run(settings):
    """Run prepared HDM steps without updating the greedy snapshot catalogs."""
    run_dir = Path(settings.MasterDir) / 'HDMrun001'
    inputs = [run_dir / 'input1', run_dir / 'input2']
    missing_inputs = [str(input_file) for input_file in inputs if not input_file.is_file()]
    if missing_inputs:
        raise RuntimeError('Missing prepared input files: {}'.format(', '.join(missing_inputs)))

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
        log_file = run_dir / 'log{}'.format(step)
        command = hpc.execute_str(aerof, str(input_file))
        print(command, flush=True)
        with open(log_file, 'w') as output:
            result = subprocess.run(
                command,
                shell=True,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError('AERO-F step {} failed; inspect {}.'.format(step, log_file))


def main():
    """Parse the requested baseline action and parameter point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--point', nargs=5, type=float, default=DEFAULT_POINT,
                        metavar=('MACH', 'AOA', 'CAMBER_LOC', 'CAMBER', 'THICKNESS'))
    args = parser.parse_args()

    settings = configure_settings()
    if args.mode == 'prepare':
        prepare(settings, args.point)
    else:
        run(settings)


if __name__ == '__main__':
    main()
