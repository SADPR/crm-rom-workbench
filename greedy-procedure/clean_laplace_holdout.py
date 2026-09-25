#!/usr/bin/env python3
"""Run one isolated out-of-training CRM HDM/PROM holdout comparison."""

import argparse
import filecmp
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pyaeroopt

from initial_pod import initial_points
from setup import Settings
from sobolGenerator import sobolGenerator


TRAINING_DIR = Path('CleanLaplaceRuns')
HOLDOUT_DIR = Path('CleanLaplaceHoldout')
POD_INDEX = 32
PROM_POINT_INDEX = 901
HOLDOUT_SOBOL_INDEX = 33


def configure_settings():
    """Read the frozen clean database while writing the HDM outside it."""
    settings = Settings()
    settings.MasterDir = '{}/'.format(TRAINING_DIR)
    settings.InitHDMPreCompDir = '{}/'.format(HOLDOUT_DIR)
    settings.RunGreedy = False
    return settings


def holdout_point(settings):
    """Return the first Sobol continuation point after the 32 training samples."""
    ranges = list(zip(settings.ParamsLowerBound, settings.ParamsUpperBound))
    points = sobolGenerator(
        ranges,
        HOLDOUT_SOBOL_INDEX,
        include_corners=True,
        make_plot=False,
    )
    point = [round(float(value), 6) for value in points[HOLDOUT_SOBOL_INDEX - 1]]
    training = initial_points(settings)
    if point in training:
        raise RuntimeError('The proposed holdout is already in the POD catalog.')
    return point, training


def hdm_dir():
    """Return the HDM-only directory kept outside the frozen training root."""
    return HOLDOUT_DIR / 'HDMrun001'


def prom_dir():
    """Return the separate online PROM directory associated with the frozen POD."""
    return TRAINING_DIR / 'evaluate/romruns{:03d}/point{:03d}'.format(
        POD_INDEX, PROM_POINT_INDEX
    )


def prom_laplace_dir():
    """Return the online Laplace directory chosen by AERO-F's ROM convention."""
    return TRAINING_DIR / 'evaluate/hromruns{:03d}/point{:03d}'.format(
        POD_INDEX, PROM_POINT_INDEX
    ) / 'Laplace-bin'


def require_file(path):
    """Raise a specific error before consuming incomplete campaign data."""
    if not path.is_file():
        raise RuntimeError('Missing required file: {}.'.format(path))


def static_files(settings):
    """List the immutable clean mesh decomposition reused by both holdout runs."""
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


def sha256(path):
    """Return a stable fingerprint for a frozen POD input file."""
    digest = hashlib.sha256()
    with open(path, 'rb') as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_catalog_fingerprints():
    """Fingerprint the only catalogs that define the 32-state online ROM."""
    paths = {
        'parameter_catalog': TRAINING_DIR / 'parsoldata.txt',
        'snapshot_catalog': TRAINING_DIR / 'statesnapdata.txt',
        'singular_values': TRAINING_DIR / 'reductionrun032/nonlinearrom/cluster0/state.svals',
    }
    for path in paths.values():
        require_file(path)
    return {name: sha256(path) for name, path in paths.items()}


def read_manifest():
    """Read the immutable holdout definition created during HDM preparation."""
    path = HOLDOUT_DIR / 'holdout.json'
    require_file(path)
    return json.loads(path.read_text())


def write_manifest(manifest):
    """Write the compact record describing this independent validation."""
    (HOLDOUT_DIR / 'holdout.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n'
    )


def check_frozen_catalog(manifest):
    """Reject a holdout if its 32-sample POD inputs changed underneath it."""
    expected = manifest.get('frozen_catalog_sha256')
    actual = frozen_catalog_fingerprints()
    if expected != actual:
        raise RuntimeError('The frozen 32-state POD catalog changed during the holdout.')


def check_clean_pod(settings):
    """Require the full distributed ScaLAPACK POD before online evaluation."""
    cluster_dir = TRAINING_DIR / 'reductionrun032/nonlinearrom/cluster0'
    required = [
        TRAINING_DIR / 'parsoldata.txt',
        cluster_dir / 'state.rob001',
        cluster_dir / 'state.rob{:03d}'.format(settings.HDMnclust),
        cluster_dir / 'state.ref001',
        cluster_dir / 'state.ref{:03d}'.format(settings.HDMnclust),
        cluster_dir / 'state.svals',
    ]
    for path in required:
        require_file(path)


def read_final_hdm_residual(directory):
    """Read the reported final residual of the second HDM phase."""
    path = directory / 'postpro/Residual.out'
    require_file(path)
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    try:
        return float(rows[-1][2])
    except (IndexError, ValueError) as error:
        raise RuntimeError('Cannot read the final HDM residual from {}.'.format(path)) from error


def read_numeric_rows(path):
    """Read numerical AERO-F history rows while skipping comments and blanks."""
    require_file(path)
    rows = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            rows.append([float(value) for value in line.split()])
    if not rows:
        raise RuntimeError('{} has no numeric rows.'.format(path))
    return rows


def relative_error(value, reference):
    """Return a relative force error with an absolute fallback near zero."""
    difference = abs(value - reference)
    return difference / abs(reference) if abs(reference) > 1.0e-14 else difference


def shift_file(directory):
    """Locate the one merged Laplace shift created for an online geometry."""
    candidates = sorted((directory / 'xdmf_files').glob('*-u_shift.xpost'))
    if len(candidates) != 1:
        raise RuntimeError('Expected one Laplace shift in {}, found {}.'.format(
            directory / 'xdmf_files', len(candidates)
        ))
    return candidates[0]


def compare_geometry_and_shift(hdm, prom, settings):
    """Require both solvers to receive the same deformed mesh and Laplace field."""
    for name in ('Position.bin', 'WallDistance.bin'):
        for part in range(1, settings.HDMnclust + 1):
            hdm_file = hdm / 'deform/{}{:03d}'.format(name, part)
            prom_file = prom / 'deform/{}{:03d}'.format(name, part)
            require_file(hdm_file)
            require_file(prom_file)
            if not filecmp.cmp(hdm_file, prom_file, shallow=False):
                raise RuntimeError('{} differs at partition {:03d}.'.format(name, part))

    _, hdm_shift = pyaeroopt.util.frg_util.read_xpost(str(shift_file(hdm)))
    _, prom_shift = pyaeroopt.util.frg_util.read_xpost(
        str(shift_file(prom_laplace_dir().parent))
    )
    if hdm_shift.shape != prom_shift.shape:
        raise RuntimeError('HDM and PROM Laplace shifts have incompatible shapes.')
    difference = hdm_shift - prom_shift
    denominator = np.linalg.norm(hdm_shift)
    relative_l2 = np.linalg.norm(difference) / denominator if denominator else np.linalg.norm(difference)
    if relative_l2 > 1.0e-10:
        raise RuntimeError('HDM/PROM Laplace shifts differ by {:.6e}.'.format(relative_l2))
    return float(relative_l2)


def run_aerof(input_path, log_path, settings):
    """Run one prepared AERO-F input with the current Slurm allocation."""
    aerof = os.environ.get('AEROF')
    if not aerof or not Path(aerof).is_file():
        raise RuntimeError('AEROF must name the built AERO-F executable.')
    command = [
        os.environ.get('MPI', 'srun'), '-n', str(settings.HDMnproc), aerof, str(input_path)
    ]
    print(' '.join(command), flush=True)
    with open(log_path, 'w') as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError('AERO-F failed; inspect {}.'.format(log_path))


def prepare_hdm(settings):
    """Prepare both clean HDM phases at a point excluded from the POD catalog."""
    from runs import getRuns

    if HOLDOUT_DIR.exists():
        raise RuntimeError('{} exists; refusing to overwrite a holdout.'.format(HOLDOUT_DIR))
    missing = [path for path in static_files(settings) if not Path(path).is_file()]
    if missing:
        raise RuntimeError('The clean shared mesh is incomplete: {}.'.format(', '.join(missing)))

    point, training = holdout_point(settings)
    HOLDOUT_DIR.mkdir()
    manifest = {
        'name': 'clean-laplace-steady-crm-holdout',
        'point': point,
        'point_source': 'Sobol continuation index {}'.format(HOLDOUT_SOBOL_INDEX),
        'excluded_from_training_points': len(training),
        'included_in_pod': False,
        'training_root': str(TRAINING_DIR),
        'hdm_directory': str(hdm_dir()),
        'prom_directory': str(prom_dir()),
        'prom_laplace_directory': str(prom_laplace_dir()),
        'frozen_catalog_sha256': frozen_catalog_fingerprints(),
    }
    write_manifest(manifest)

    frg = pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
    )
    runs = getRuns(settings)
    hdm1 = runs['HDM'](frg, p=point, HDMind=1, step=1, maindir=True)
    hdm2 = runs['HDM'](frg, p=point, HDMind=1, step=2, maindir=True)
    ref_nodes = np.loadtxt('{}_nodes'.format(settings.TopFilePath), dtype=np.float64)[:, 1:]
    ref_stick = np.loadtxt('{}_stick'.format(settings.TopFilePath), dtype=np.int32)[:, 2:]
    hdm1.prep(ref_nodes, ref_stick)
    hdm1.create_input_file()
    hdm2.create_input_file()
    hdm1.writeInputFile()
    hdm2.writeInputFile()

    _, values = pyaeroopt.util.frg_util.read_xpost(str(shift_file(hdm_dir())))
    shift_range = float(np.max(values) - np.min(values))
    if not np.isfinite(shift_range) or shift_range <= 1.0e-12:
        raise RuntimeError('The holdout Laplace shift is constant or non-finite.')
    manifest['hdm_laplace_shift_range'] = shift_range
    write_manifest(manifest)
    print('Prepared holdout HDM at {} for {}.'.format(hdm_dir(), point))


def run_hdm(settings):
    """Run the two HDM phases without inserting any snapshot into the POD."""
    manifest = read_manifest()
    check_frozen_catalog(manifest)
    inputs = [hdm_dir() / 'input1', hdm_dir() / 'input2']
    for step, input_path in enumerate(inputs, start=1):
        require_file(input_path)
        run_aerof(input_path, hdm_dir() / 'log{}'.format(step), settings)
    residual = read_final_hdm_residual(hdm_dir())
    if residual > settings.HDMtol2:
        raise RuntimeError('Holdout HDM residual {:.6e} exceeds {:.6e}.'.format(
            residual, settings.HDMtol2
        ))
    manifest['hdm_final_reported_residual'] = residual
    write_manifest(manifest)
    print('Holdout HDM completed with residual {:.6e}.'.format(residual))


def prepare_prom(settings):
    """Prepare the frozen 32-state PROM at the exact completed HDM point."""
    from runs import create_deformed_top_file, getRuns, solveCurrentLaplace

    manifest = read_manifest()
    check_frozen_catalog(manifest)
    check_clean_pod(settings)
    if read_final_hdm_residual(hdm_dir()) > settings.HDMtol2:
        raise RuntimeError('The holdout HDM did not reach the required tolerance.')
    if prom_dir().exists() or prom_laplace_dir().parent.exists():
        raise RuntimeError('The holdout PROM directory already exists; refusing to overwrite it.')

    point = manifest['point']
    frg = pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
    )
    rom = getRuns(settings)['ROM'](
        frg, p=point, HDMind=POD_INDEX, pind=PROM_POINT_INDEX,
        hyper=False, hrtest=False, evaluate=True,
    )
    ref_nodes = np.loadtxt('{}_nodes'.format(settings.TopFilePath), dtype=np.float64)[:, 1:]
    ref_stick = np.loadtxt('{}_stick'.format(settings.TopFilePath), dtype=np.int32)[:, 2:]
    rom.prep(ref_nodes, ref_stick)
    deformed_top = create_deformed_top_file('{}/'.format(prom_dir()), settings.TopFilePath)
    solveCurrentLaplace(
        deformed_top, '{}/'.format(prom_laplace_dir().parent), frg,
        settings, settings.LaplaceNumProc,
    )
    rom.create_input_file()
    rom.writeInputFile()

    geometry_shift_error = compare_geometry_and_shift(hdm_dir(), prom_dir(), settings)
    manifest['hdm_prom_laplace_relative_l2_difference'] = geometry_shift_error
    write_manifest(manifest)
    print('Prepared frozen-POD holdout PROM at {}.'.format(prom_dir()))


def run_prom(settings):
    """Run the PROM and record diagnostics without imposing an HDM residual target."""
    manifest = read_manifest()
    check_frozen_catalog(manifest)
    require_file(prom_dir() / 'input')
    require_file(prom_laplace_dir() / 'ushift.bin001')
    require_file(prom_laplace_dir() / 'ushift.bin{:03d}'.format(settings.HDMnclust))
    if (prom_dir() / 'postpro/Residual.out').exists():
        raise RuntimeError('Holdout PROM results exist; refusing to overwrite them.')
    run_aerof(prom_dir() / 'input', prom_dir() / 'log', settings)

    hdm_liftdrag = read_numeric_rows(hdm_dir() / 'postpro/liftdrag.out')[-1]
    prom_liftdrag = read_numeric_rows(prom_dir() / 'postpro/liftdrag.out')[-1]
    prom_residual = read_numeric_rows(prom_dir() / 'postpro/Residual.out')[-1]
    manifest['prom_iterations'] = int(prom_residual[0])
    manifest['prom_final_relative_full_residual'] = prom_residual[2]
    manifest['drag_relative_error'] = relative_error(prom_liftdrag[4], hdm_liftdrag[4])
    manifest['lift_relative_error'] = relative_error(prom_liftdrag[5], hdm_liftdrag[5])
    check_frozen_catalog(manifest)
    write_manifest(manifest)
    print('Holdout PROM completed: drag error {:.6e}, lift error {:.6e}.'.format(
        manifest['drag_relative_error'], manifest['lift_relative_error']
    ))


def show_point(settings):
    """Print the exact independent point and prove it is not a training sample."""
    point, training = holdout_point(settings)
    print('Sobol holdout {}/{}: {}'.format(HOLDOUT_SOBOL_INDEX, len(training) + 1, point))


def main():
    """Execute exactly one holdout stage requested by the Slurm wrappers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'mode', choices=('point', 'prepare-hdm', 'run-hdm', 'prepare-prom', 'run-prom')
    )
    args = parser.parse_args()
    settings = configure_settings()
    if args.mode == 'point':
        show_point(settings)
    elif args.mode == 'prepare-hdm':
        prepare_hdm(settings)
    elif args.mode == 'run-hdm':
        run_hdm(settings)
    elif args.mode == 'prepare-prom':
        prepare_prom(settings)
    else:
        run_prom(settings)


if __name__ == '__main__':
    main()
