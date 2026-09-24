#!/usr/bin/env python3
"""Audit, assemble, or build the POD for the completed initial HDMs."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from setup import Settings
from sobolGenerator import sobolGenerator


def configure_settings():
    """Use the production directories without starting the greedy loop."""
    settings = Settings()
    settings.MasterDir = 'GreedyRuns/'
    settings.InitHDMPreCompDir = 'InitialHDMruns/'
    settings.RunGreedy = False
    return settings


def initial_points(settings):
    """Recreate the ordered Sobol initialization used by the HDM campaign."""
    ranges = list(zip(settings.ParamsLowerBound, settings.ParamsUpperBound))
    return [
        [round(float(value), 6) for value in point]
        for point in sobolGenerator(
            ranges,
            settings.numrand,
            include_corners=True,
            numToSkip=settings.numSkip,
            make_plot=False,
        )
    ]


def run_name(index):
    """Return the fixed three-digit HDM directory name."""
    return 'HDMrun{:03d}'.format(index)


def metadata_path(run_dir, index):
    """Return the metadata file written by the pilot or campaign script."""
    return run_dir / ('pilot.json' if index == 1 else 'parameters.json')


def read_final_residual(run_dir):
    """Read the final nonlinear residual written by the second HDM phase."""
    residual_file = run_dir / 'postpro/Residual.out'
    if not residual_file.is_file():
        raise RuntimeError('Missing {}'.format(residual_file))

    lines = [line.split() for line in residual_file.read_text().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError('{} is empty.'.format(residual_file))
    try:
        return float(lines[-1][2])
    except (IndexError, ValueError) as error:
        raise RuntimeError('Cannot read the final residual from {}.'.format(residual_file)) from error


def check_point(metadata, expected, index):
    """Require metadata to identify the exact Sobol point for this HDM."""
    if metadata.get('index') != index:
        raise RuntimeError('HDM {:03d} has inconsistent metadata index.'.format(index))
    point = metadata.get('point')
    if not isinstance(point, list) or len(point) != len(expected):
        raise RuntimeError('HDM {:03d} has an invalid parameter point.'.format(index))
    if any(not math.isclose(float(value), target, abs_tol=1e-12)
           for value, target in zip(point, expected)):
        raise RuntimeError('HDM {:03d} does not match its Sobol point.'.format(index))


def inspect_run(run_dir, index, point, settings):
    """Validate one completed HDM before it can become a ROM snapshot."""
    metadata_file = metadata_path(run_dir, index)
    if not metadata_file.is_file():
        raise RuntimeError('Missing {}'.format(metadata_file))
    try:
        metadata = json.loads(metadata_file.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError('Cannot parse {}.'.format(metadata_file)) from error
    check_point(metadata, point, index)

    required_files = [
        run_dir / 'input1',
        run_dir / 'input2',
        run_dir / 'log1',
        run_dir / 'log2',
        run_dir / 'references/Restart.data',
        run_dir / 'snapshots/State.bin001',
        run_dir / 'snapshots/State.bin{:03d}'.format(settings.HDMnclust),
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError('HDM {:03d} is incomplete: {}.'.format(index, ', '.join(missing)))

    residual = read_final_residual(run_dir)
    if residual > settings.HDMtol2:
        raise RuntimeError(
            'HDM {:03d} residual {:.6e} exceeds {:.6e}.'.format(
                index, residual, settings.HDMtol2
            )
        )
    return residual


def snapshot_catalog(settings, entries):
    """Format the state snapshot catalog consumed by AERO-F POD."""
    lines = ['{}\n'.format(len(entries))]
    for entry in entries:
        lines.append(
            '{}/snapshots/State.bin {} {} 1 1 \n'.format(
                entry['target'].as_posix(), settings.SnapIndex, settings.SnapIndex
            )
        )
    return ''.join(lines)


def parameter_catalog(settings, entries):
    """Format the parametric-state catalog used for ROM initial conditions."""
    lines = ['{}\n'.format(len(entries)), '{}\n'.format(len(entries[0]['point']))]
    for entry in entries:
        lines.append(
            '{}/snapshots/State.bin {}\n'.format(
                entry['target'].as_posix(), settings.SnapIndex
            )
        )
        lines.extend('{}\n'.format(value) for value in entry['point'])
    return ''.join(lines)


def catalog_paths(settings):
    """Return the two catalogs maintained at the greedy-workflow root."""
    master_dir = Path(settings.MasterDir)
    return master_dir / 'statesnapdata.txt', master_dir / 'parsoldata.txt'


def check_catalogs(settings, entries):
    """Reject existing catalogs unless they exactly describe this campaign."""
    expected = (snapshot_catalog(settings, entries), parameter_catalog(settings, entries))
    paths = catalog_paths(settings)
    existing = [path.exists() for path in paths]
    if any(existing) and not all(existing):
        raise RuntimeError('Only one initial snapshot catalog exists.')
    if all(existing):
        contents = tuple(path.read_text() for path in paths)
        if contents != expected:
            raise RuntimeError('Existing snapshot catalogs do not match the 32 initial HDMs.')
    return paths, expected


def audit(settings):
    """Preflight every initial HDM without changing directories or files."""
    points = initial_points(settings)
    if len(points) != 32:
        raise RuntimeError('Expected 32 initial Sobol points, generated {}.'.format(len(points)))

    source_root = Path(settings.InitHDMPreCompDir)
    target_root = Path(settings.MasterDir)
    if not target_root.is_dir():
        raise RuntimeError('Missing {}'.format(target_root))

    entries = []
    for index, point in enumerate(points, start=1):
        source = source_root / run_name(index)
        target = target_root / run_name(index)
        if source.exists() == target.exists():
            state = 'both exist' if source.exists() else 'neither exists'
            raise RuntimeError('HDM {:03d}: {}.'.format(index, state))
        run_dir = target if target.exists() else source
        residual = inspect_run(run_dir, index, point, settings)
        entries.append({
            'index': index,
            'point': point,
            'source': source,
            'target': target,
            'residual': residual,
        })

    check_catalogs(settings, entries)
    print('Audited {} completed initial HDMs.'.format(len(entries)))
    print('Final residual range: {:.6e} to {:.6e}.'.format(
        min(entry['residual'] for entry in entries),
        max(entry['residual'] for entry in entries),
    ))
    return entries


def write_atomically(path, content):
    """Install a complete catalog so readers never see a partial file."""
    with tempfile.NamedTemporaryFile(
        mode='w', dir=path.parent, prefix='{}.tmp.'.format(path.name), delete=False
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def assemble(settings):
    """Move verified HDM directories into GreedyRuns and write their catalogs."""
    entries = audit(settings)
    paths, contents = check_catalogs(settings, entries)

    for entry in entries:
        if entry['source'].exists():
            entry['source'].rename(entry['target'])
            print('Moved {} to {}.'.format(entry['source'], entry['target']))

    if not all(path.exists() for path in paths):
        for path, content in zip(paths, contents):
            write_atomically(path, content)
        print('Wrote initial snapshot and parameter catalogs.')
    else:
        print('Reusing matching initial snapshot catalogs.')
    return entries


def build_pod(settings):
    """Run AERO-F POD over the assembled 32-state initial snapshot catalog."""
    import pyaeroopt

    from runs import getRuns

    entries = audit(settings)
    if any(entry['source'].exists() for entry in entries):
        raise RuntimeError('Run assemble before building the POD.')
    pod_dir = Path(settings.MasterDir) / 'reductionrun{:03d}'.format(len(entries))
    if pod_dir.exists():
        raise RuntimeError('{} already exists; refusing to overwrite it.'.format(pod_dir))

    aerof = os.environ.get('AEROF')
    if not aerof or not Path(aerof).is_file():
        raise RuntimeError('AEROF must name the built AERO-F executable.')

    centroid = [sum(entry['point'][column] for entry in entries) / len(entries)
                for column in range(len(entries[0]['point']))]
    frg = pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
    )
    pod = getRuns(settings)['POD'](frg, p=centroid, HDMind=len(entries))
    hpc = pyaeroopt.interface.Hpc(
        machine='independence', batch=False, bg=False, nproc=settings.HDMnproc
    )
    hpc.mpi = os.environ.get('MPI', 'mpirun')
    print('Building initial POD with {} ranks: {}.'.format(settings.HDMnproc, hpc.mpi))
    pod.create_input_file()
    pod.writeInputFile()
    command = hpc.execute_str(pod.bin, pod.infile.fname)
    print(command, flush=True)
    with open(pod.infile.log, 'w') as log_file:
        result = subprocess.run(
            command, shell=True, stdout=log_file, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode != 0:
        raise RuntimeError('Initial POD failed; inspect {}.'.format(pod.infile.log))
    print('Initial POD completed: {}.'.format(pod_dir))


def main():
    """Run the requested safe stage of initial PROM construction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('audit', 'assemble', 'pod'))
    args = parser.parse_args()

    settings = configure_settings()
    if args.mode == 'audit':
        audit(settings)
    elif args.mode == 'assemble':
        assemble(settings)
    else:
        build_pod(settings)


if __name__ == '__main__':
    main()
