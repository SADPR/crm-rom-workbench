#!/usr/bin/env python3
"""Run the independent 5D test set: HDM truth, frozen-POD PROMs, and error metrics."""

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import types

import numpy as np
import pyaeroopt
from scipy.stats import qmc

import initial_hdm_campaign
import initial_pod
import sobol5d_campaign as campaign
from runs import getRuns


TEST_DIR = 'Sobol5DTest/'
TEST_COUNT = 32
TEST_SEED = 20260925
METRICS = ('cp_l2', 'cp_l1', 'sf_l2', 'sf_l1', 'drag', 'lift')


def test_settings():
    """Write test HDMs outside the training roots while reusing their mesh data."""
    settings = campaign.configure_settings()
    settings.InitHDMPreCompDir = TEST_DIR
    return settings


def manifest_path():
    """Return the manifest that freezes the test points."""
    return Path(TEST_DIR) / 'test.json'


def read_points():
    """Read the frozen test points."""
    if not manifest_path().is_file():
        raise RuntimeError('Run the init action first.')
    return json.loads(manifest_path().read_text())['points']


def hdm_dir(index):
    """Return the truth HDM directory of one one-based test point."""
    return campaign.run_dir(index, TEST_DIR)


def prom_paths(settings, count, index):
    """Return the PROM and Laplace directories that runs.py uses for this test point."""
    root = Path(settings.MasterDir)
    return (root / 'evaluate/romruns{:03d}/point{:03d}'.format(count, index),
            root / 'evaluate/hromruns{:03d}/point{:03d}/Laplace-bin'.format(count, index))


def initialize():
    """Freeze an independent scrambled-Sobol test set, disjoint from the training design."""
    if Path(TEST_DIR).exists():
        raise RuntimeError('{} exists; refusing to replace the test set.'.format(TEST_DIR))
    settings = campaign.configure_settings()
    lower = np.array(settings.ParamsLowerBound)
    upper = np.array(settings.ParamsUpperBound)
    unit = qmc.Sobol(d=len(lower), scramble=True, seed=TEST_SEED).random(TEST_COUNT)
    points = [[round(float(value), 6) for value in lower + row * (upper - lower)] for row in unit]
    training = campaign.design_points(settings)
    overlap = [point for point in points if point in training]
    if overlap:
        raise RuntimeError('Test points overlap the training design: {}.'.format(overlap))
    Path(TEST_DIR).mkdir()
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    campaign.write_json(manifest_path(), {
        'name': 'sobol5d-transonic-test-set',
        'source_commit': source_commit,
        'sequence': 'scipy.stats.qmc.Sobol(d=5, scramble=True, seed={})'.format(TEST_SEED),
        'points': points,
    })
    print('Initialized {} with {} test points.'.format(TEST_DIR, TEST_COUNT))


def prepare_hdm(index):
    """Prepare one truth HDM with the campaign geometry, Laplace, and HDM settings."""
    settings = test_settings()
    missing = [path for path in campaign.static_files(settings) if not Path(path).is_file()]
    if missing:
        raise RuntimeError('Run the training pilot first; missing {}.'.format(', '.join(missing)))
    initial_hdm_campaign.prepare(settings, index, point=read_points()[index - 1])
    campaign.record_shift_stats(hdm_dir(index))


def run_hdm(index):
    """Run both phases of one truth HDM and require the campaign tolerance."""
    settings = test_settings()
    initial_hdm_campaign.run(settings, index)
    residual = campaign.final_residual(hdm_dir(index), settings.HDMtol2)
    print('Validated {} with final residual {:.6e}.'.format(hdm_dir(index), residual))


def run_prom(count, index):
    """Run the frozen-POD PROM of one batch at one converged test point."""
    settings = campaign.configure_settings()
    pod = campaign.pod_dir(settings, count)
    catalog = pod / 'parsoldata.txt'
    if not catalog.is_file():
        raise RuntimeError('Missing {}; build POD {} first.'.format(catalog, count))
    hdm = hdm_dir(index)
    campaign.final_residual(hdm, settings.HDMtol2)
    rom_dir, laplace_dir = prom_paths(settings, count, index)
    if rom_dir.exists() or laplace_dir.parent.exists():
        raise RuntimeError('{} exists; refusing to overwrite a PROM.'.format(rom_dir))
    aerof = os.environ.get('AEROF')
    if not aerof or not Path(aerof).is_file():
        raise RuntimeError('AEROF must name the built AERO-F executable.')

    frg = pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
    )
    rom = getRuns(settings)['ROM'](frg, p=read_points()[index - 1], HDMind=count, pind=index,
                                   hyper=False, hrtest=False, evaluate=True)
    # The truth HDM already holds this exact deformed geometry and Laplace shift.
    for source in sorted((hdm / 'deform').iterdir()):
        shutil.copy2(source, rom_dir / 'deform' / source.name)
    laplace_dir.mkdir(parents=True)
    for source in sorted((hdm / 'Laplace-bin').glob('ushift.bin*')):
        shutil.copy2(source, laplace_dir / source.name)
    rom.create_input_file()
    rom.writeInputFile()
    # The root catalogs follow the latest batch; the IDW guess must use this batch's copy.
    input_file = rom_dir / 'input'
    text = input_file.read_text()
    root_catalog = 'MultipleSolutionsData = "{}parsoldata.txt";'.format(settings.MasterDir)
    if text.count(root_catalog) != 1:
        raise RuntimeError('Cannot find the IDW catalog entry in {}.'.format(input_file))
    input_file.write_text(text.replace(root_catalog,
                                       'MultipleSolutionsData = "{}";'.format(catalog.as_posix())))
    campaign.write_json(rom_dir / 'prom.json', {
        'pod': str(pod),
        'idw_catalog': str(catalog),
        'test_index': index,
        'geometry_and_laplace_from': str(hdm),
    })

    hpc = pyaeroopt.interface.Hpc(machine='independence', batch=False, bg=False,
                                  nproc=settings.HDMnproc)
    hpc.mpi = os.environ.get('MPI', 'srun')
    command = hpc.execute_str(aerof, str(input_file))
    print(command, flush=True)
    with open(rom_dir / 'log', 'w') as log_file:
        result = subprocess.run(command, shell=True, stdout=log_file, stderr=subprocess.STDOUT,
                                check=False)
    if result.returncode != 0 or not (rom_dir / 'postpro/Residual.out').is_file():
        raise RuntimeError('PROM failed; inspect {}.'.format(rom_dir / 'log'))
    print('Completed PROM {} at test point {:03d}.'.format(count, index))


def merge_surface_fields(frg, run_directory):
    """Merge the distributed Cp and skin-friction results once per run."""
    for field in ('PressureCoefficient', 'SkinFriction'):
        output = run_directory / 'postpro/{}.xpost'.format(field)
        if output.is_file() and output.stat().st_size > 0:
            continue
        frg.sower_fluid_merge('{}/results/{}.bin'.format(run_directory, field),
                              '{}/postpro/{}'.format(run_directory, field), field, log='/dev/null')
        if not output.is_file():
            raise RuntimeError('Merge did not produce {}.'.format(output))


def surface_curves(settings, run_directory, raw_nodes):
    """Return wall positions, Cp, and Sf with Yihong's extraction in runs.py getSkinForces."""
    run = types.SimpleNamespace(SuperDir='{}/'.format(run_directory))
    return getRuns(settings)['HDM'].getSkinForces(run, raw_nodes)


def relative_error(approximation, reference, order):
    """Return the relative vector error in percent."""
    return 100.0 * float(np.linalg.norm(approximation - reference, order)
                         / np.linalg.norm(reference, order))


def lift_drag(run_directory):
    """Read the final lift and drag, as runs.py getLD does."""
    line = (run_directory / 'postpro/liftdrag.out').read_text().splitlines()[-1].split()
    return float(line[5]), float(line[4])


def prom_residual(rom_dir):
    """Return the PROM iterations and its final relative and absolute full residuals."""
    rows = [line.split() for line in (rom_dir / 'postpro/Residual.out').read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')]
    relative = float(rows[-1][2])
    initial = None
    for line in (rom_dir / 'log').read_text().splitlines():
        if 'Spatial residual norm = ' in line:
            initial = float(line.split('Spatial residual norm = ')[1])
            break
    absolute = None if initial is None else relative * initial
    return int(rows[-1][0]), relative, absolute


def point_metrics(settings, frg, raw_nodes, count, index):
    """Compare one PROM against its truth HDM on the airfoil surface and in forces."""
    hdm = hdm_dir(index)
    rom_dir, _ = prom_paths(settings, count, index)
    try:
        campaign.final_residual(hdm, settings.HDMtol2)
    except RuntimeError as error:
        return {'status': 'hdm-unavailable', 'detail': str(error)}
    if not (rom_dir / 'postpro/Residual.out').is_file():
        return {'status': 'prom-unavailable'}
    for directory in (hdm, rom_dir):
        merge_surface_fields(frg, directory)
    hdm_pos, hdm_cp, hdm_sf = surface_curves(settings, hdm, raw_nodes)
    rom_pos, rom_cp, rom_sf = surface_curves(settings, rom_dir, raw_nodes)
    if hdm_pos.shape != rom_pos.shape or not np.allclose(hdm_pos, rom_pos):
        return {'status': 'surface-mismatch'}
    hdm_lift, hdm_drag = lift_drag(hdm)
    rom_lift, rom_drag = lift_drag(rom_dir)
    iterations, relative, absolute = prom_residual(rom_dir)
    return {
        'status': 'ok',
        'cp_l2': relative_error(rom_cp, hdm_cp, 2),
        'cp_l1': relative_error(rom_cp, hdm_cp, 1),
        'sf_l2': relative_error(rom_sf, hdm_sf, 2),
        'sf_l1': relative_error(rom_sf, hdm_sf, 1),
        'drag': 100.0 * abs(rom_drag - hdm_drag) / abs(hdm_drag),
        'lift': 100.0 * abs(rom_lift - hdm_lift) / abs(hdm_lift),
        'prom_iterations': iterations,
        'prom_relative_residual': relative,
        'prom_absolute_residual': absolute,
    }


def metrics(count):
    """Write per-point and aggregate PROM errors of one batch over the test set."""
    settings = campaign.configure_settings()
    frg = pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
    )
    raw_nodes = np.loadtxt('{}_nodes'.format(settings.TopFilePath), dtype=np.float64)
    points = read_points()
    per_point = {}
    for index, point in enumerate(points, start=1):
        result = point_metrics(settings, frg, raw_nodes, count, index)
        result['point'] = point
        per_point['{:03d}'.format(index)] = result
    valid = [value for value in per_point.values() if value['status'] == 'ok']
    aggregate = {
        name: {
            'mean': statistics.mean(value[name] for value in valid),
            'median': statistics.median(value[name] for value in valid),
            'max': max(value[name] for value in valid),
        }
        for name in METRICS
    } if valid else {}
    summary = {
        'pod': count,
        'unit': 'percent relative error; cp/sf over the z = 0 wall nodes',
        'counted': len(valid),
        'not_counted': {key: value['status'] for key, value in per_point.items()
                        if value['status'] != 'ok'},
        'aggregate': aggregate,
        'points': per_point,
    }
    output = Path(TEST_DIR) / 'metrics_pod{:03d}.json'.format(count)
    campaign.write_json(output, summary)
    print('Wrote {}: {} of {} test points counted.'.format(output, len(valid), len(points)))
    for name, values in aggregate.items():
        print('  {:6s} mean {:8.3f}  median {:8.3f}  max {:8.3f}'.format(
            name, values['mean'], values['median'], values['max']
        ))


def summary():
    """Collect every batch's metrics into one error-versus-samples curve."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    batches = []
    for path in sorted(Path(TEST_DIR).glob('metrics_pod*.json')):
        data = json.loads(path.read_text())
        if data['aggregate']:
            batches.append(data)
    if not batches:
        raise RuntimeError('No batch metrics found in {}.'.format(TEST_DIR))
    counts = [data['pod'] for data in batches]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, name, label in ((axes[0], 'cp_l2', r'$c_p$'), (axes[1], 'sf_l2', r'$S_f$')):
        axis.plot(counts, [data['aggregate'][name]['mean'] for data in batches], 'o-', label='Mean')
        axis.plot(counts, [data['aggregate'][name]['max'] for data in batches], 'o-', label='Max')
        axis.set_xlabel('Number of samples')
        axis.set_ylabel('{} relative L2 error (%)'.format(label))
        axis.grid(True)
        axis.legend()
    figure.tight_layout()
    figure.savefig(Path(TEST_DIR) / 'error_curve.pdf')
    plt.close(figure)
    for data in batches:
        print('N = {:3d}: {} counted; cp_l2 mean {:.3f} max {:.3f}; sf_l2 mean {:.3f} max {:.3f}'.format(
            data['pod'], data['counted'],
            data['aggregate']['cp_l2']['mean'], data['aggregate']['cp_l2']['max'],
            data['aggregate']['sf_l2']['mean'], data['aggregate']['sf_l2']['max'],
        ))


def main():
    """Run one explicit stage of the 5D test set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('init', 'prepare-hdm', 'run-hdm', 'prom', 'metrics', 'summary'))
    parser.add_argument('--test-index', type=int)
    parser.add_argument('--pod', type=int)
    args = parser.parse_args()

    if args.mode == 'init':
        initialize()
        return
    if args.mode == 'summary':
        summary()
        return
    if args.mode in ('prepare-hdm', 'run-hdm', 'prom'):
        if args.test_index is None or not 1 <= args.test_index <= TEST_COUNT:
            raise ValueError('--test-index must be in [1, {}].'.format(TEST_COUNT))
    if args.mode in ('prom', 'metrics') and args.pod is None:
        raise ValueError('--pod is required for {}.'.format(args.mode))
    if args.mode == 'prepare-hdm':
        prepare_hdm(args.test_index)
    elif args.mode == 'run-hdm':
        run_hdm(args.test_index)
    elif args.mode == 'prom':
        run_prom(args.pod, args.test_index)
    else:
        metrics(args.pod)


if __name__ == '__main__':
    main()
