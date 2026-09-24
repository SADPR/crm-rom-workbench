#!/usr/bin/env python3
"""Prepare or run one geometry-specific Laplace-shift benchmark."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess


DEFAULT_BASELINE = Path('BaselineRuns/HDMrun001')


def workspace(baseline):
    return baseline / 'laplace'


def prepare(baseline):
    """Create an isolated deformed mesh from the converged nominal baseline."""
    from runs import create_deformed_top_file
    from setup import Settings

    if not (baseline / 'deform/Position.xpost').is_file():
        raise RuntimeError('Missing {}/deform/Position.xpost'.format(baseline))

    run_dir = workspace(baseline)
    if run_dir.exists():
        raise RuntimeError('{} already exists; refusing to overwrite it.'.format(run_dir))

    (run_dir / 'deform').mkdir(parents=True)
    shutil.copy2(baseline / 'deform/Position.xpost', run_dir / 'deform/Position.xpost')

    settings = Settings()
    deformed_top = create_deformed_top_file('{}/'.format(run_dir), settings.TopFilePath)
    manifest = {
        'baseline': str(baseline),
        'deformed_top': deformed_top,
        'laplace_num_proc': settings.LaplaceNumProc,
        'shift_type': settings.ShiftType,
        'laplace_shift_each': settings.LaplaceShiftEach,
    }
    with open(run_dir / 'manifest.json', 'w') as output:
        json.dump(manifest, output, indent=2)
        output.write('\n')

    print('Prepared {}'.format(run_dir))
    print('Run: python3 -B benchmark_laplace_shift.py run')


def run(baseline):
    """Run FEniCS and partition its shift field for the nominal HDM mesh."""
    import pyaeroopt

    from setup import Settings
    from ushift_inject import ushiftXPOST

    run_dir = workspace(baseline)
    manifest_path = run_dir / 'manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError('Run prepare first; missing {}'.format(manifest_path))

    with open(manifest_path) as input_file:
        manifest = json.load(input_file)

    deformed_top = Path(manifest['deformed_top'])
    xdmf_dir = run_dir / 'xdmf_files'
    laplace_dir = run_dir / 'Laplace-bin'
    stem = deformed_top.stem
    u_star = xdmf_dir / '{}-u_star.xpost'.format(stem)
    u_shift = laplace_dir / '{}-u_shift.xpost'.format(stem)
    expected = laplace_dir / 'ushift.bin001'
    if expected.exists():
        raise RuntimeError('{} exists; refusing to overwrite it.'.format(expected))

    if u_star.is_file():
        print('Reusing {}'.format(u_star))
    else:
        subprocess.run(
            ['./run_laplace_shift.sh', str(deformed_top), str(xdmf_dir),
             str(manifest['laplace_num_proc'])],
            check=True,
        )

    laplace_dir.mkdir(exist_ok=True)
    if u_shift.is_file():
        print('Reusing {}'.format(u_shift))
    else:
        u_shift = Path(ushiftXPOST(
            str(u_star), 'SpalartAllmaras', folder_name='{}/'.format(laplace_dir)))

    settings = Settings()
    frg = pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='BaselineRuns/data/{}'.format(settings.GeometryPrefix),
    )
    frg.sower_fluid_split(
        file2split=str(u_shift),
        out='{}/ushift.bin'.format(laplace_dir),
        nclust=settings.HDMnclust,
        log='{}/log.sower'.format(run_dir),
    )

    if not expected.is_file():
        raise RuntimeError('Sower did not create {}'.format(expected))
    print('Laplace benchmark completed: {}'.format(laplace_dir / 'ushift.bin'))


def prepare_aerof_check(baseline):
    """Write an AERO-F initialization check that reads the nominal Laplace shift."""
    run_dir = workspace(baseline)
    shift = run_dir / 'Laplace-bin/ushift.bin001'
    if not shift.is_file():
        raise RuntimeError('Run the Laplace benchmark first; missing {}'.format(shift))

    check_dir = run_dir / 'aerof-check'
    if check_dir.exists():
        raise RuntimeError('{} already exists; refusing to overwrite it.'.format(check_dir))
    for name in ('postpro', 'references', 'results', 'snapshots'):
        (check_dir / name).mkdir(parents=True, exist_ok=True)

    source = baseline / 'input2'
    if not source.is_file():
        raise RuntimeError('Missing {}'.format(source))
    content = source.read_text()
    replacements = {
        'LaplaceSnapshotData = "";':
            'LaplaceSnapshotData = "{}/Laplace-bin/ushift.bin";'.format(run_dir),
        'Prefix = "{}/results/";'.format(baseline):
            'Prefix = "{}/results/";'.format(check_dir),
        'Prefix = "{}/references/";'.format(baseline):
            'Prefix = "{}/references/";'.format(check_dir),
        'Prefix = "{}/snapshots/";'.format(baseline):
            'Prefix = "{}/snapshots/";'.format(check_dir),
        'OutputShiftVectorType = None;': 'OutputShiftVectorType = Laplace;',
        'MaxIts = 9000;': 'MaxIts = 1;',
        'Eps = 5e-07;': 'Eps = 1e-05;',
    }
    for old, new in replacements.items():
        if content.count(old) != 1:
            raise RuntimeError('Expected one occurrence of {!r}'.format(old))
        content = content.replace(old, new)

    input_file = check_dir / 'input_laplace'
    input_file.write_text(content)
    print('Prepared {}'.format(input_file))
    print('Run from the greedy-procedure directory:')
    print('  srun -N 5 -n 120 "${{AEROF}}" {} > {}/log 2>&1'.format(
        input_file, check_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'prepare-aerof-check'))
    parser.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args()

    if args.mode == 'prepare':
        prepare(args.baseline)
    elif args.mode == 'run':
        run(args.baseline)
    else:
        prepare_aerof_check(args.baseline)


if __name__ == '__main__':
    main()
