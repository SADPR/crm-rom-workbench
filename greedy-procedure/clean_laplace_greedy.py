#!/usr/bin/env python3
"""Extend the clean 32-state CRM PROM with isolated greedy HDMs."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyaeroopt

from runs import create_deformed_top_file, getRuns, solveCurrentLaplace
from setup import Settings
from sobolGenerator import sobolGenerator


SOURCE_DIR = Path('CleanLaplaceRuns')
MASTER_DIR = Path('CleanLaplaceGreedy40')
INITIAL_COUNT = 32
FINAL_COUNT = 40
HOLDOUT_SOBOL_INDEX = 33
DEFAULT_CANDIDATE_COUNT = 24


def configure_settings():
    """Use a new writable root and the validated full-PROM configuration."""
    settings = Settings()
    settings.MasterDir = '{}/'.format(MASTER_DIR)
    settings.InitHDMPreCompDir = None
    settings.RunGreedy = False
    settings.HyperReduced = False
    settings.PODMethod = 'ScalapackSVD'
    return settings


def write_json(path, value):
    """Write compact metadata without exposing a partial file to a later job."""
    temporary = path.with_name('{}.tmp'.format(path.name))
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def sha256(path):
    """Return a stable fingerprint for an immutable source input."""
    digest = hashlib.sha256()
    with open(path, 'rb') as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_file(path):
    """Fail early with the file that makes the current stage unsafe."""
    if not path.is_file():
        raise RuntimeError('Missing required file: {}.'.format(path))


def manifest_path():
    """Return the campaign-wide manifest path."""
    return MASTER_DIR / 'campaign.json'


def read_manifest():
    """Read the immutable description of this enrichment campaign."""
    require_file(manifest_path())
    return json.loads(manifest_path().read_text())


def catalog_paths(root):
    """Return the state and parameter catalog paths under one campaign root."""
    return root / 'statesnapdata.txt', root / 'parsoldata.txt'


def catalog_count(path):
    """Read the leading snapshot count from either AERO-F catalog."""
    require_file(path)
    lines = path.read_text().splitlines()
    if not lines:
        raise RuntimeError('{} is empty.'.format(path))
    try:
        return int(lines[0])
    except ValueError as error:
        raise RuntimeError('Cannot read the snapshot count from {}.'.format(path)) from error


def parse_parameter_catalog(path):
    """Read points from AERO-F's MultipleSolutionsData catalog."""
    require_file(path)
    lines = path.read_text().splitlines()
    try:
        count = int(lines[0])
        dimension = int(lines[1])
    except (IndexError, ValueError) as error:
        raise RuntimeError('Cannot parse {}.'.format(path)) from error
    expected_lines = 2 + count * (dimension + 1)
    if len(lines) != expected_lines:
        raise RuntimeError('{} has {} lines; expected {}.'.format(
            path, len(lines), expected_lines
        ))
    points = []
    offset = 2
    for _ in range(count):
        offset += 1
        try:
            points.append([float(value) for value in lines[offset:offset + dimension]])
        except ValueError as error:
            raise RuntimeError('Cannot parse parameter values in {}.'.format(path)) from error
        offset += dimension
    return points


def points_match(left, right):
    """Compare parameter vectors without relying on a text representation."""
    return len(left) == len(right) and all(
        math.isclose(float(a), float(b), abs_tol=1.0e-12) for a, b in zip(left, right)
    )


def current_count():
    """Return and cross-check the number of snapshots in the enrichment root."""
    state_path, parameter_path = catalog_paths(MASTER_DIR)
    state_count = catalog_count(state_path)
    parameter_count = catalog_count(parameter_path)
    if state_count != parameter_count:
        raise RuntimeError('The state and parameter catalogs have different counts.')
    return state_count


def static_files(settings):
    """List shared full-order mesh files copied into the isolated root."""
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


def verify_source(settings):
    """Require the completed clean 32-state PROM without modifying it."""
    source_state, source_parameter = catalog_paths(SOURCE_DIR)
    for path in (source_state, source_parameter):
        if catalog_count(path) != INITIAL_COUNT:
            raise RuntimeError('{} is not a {}-state source catalog.'.format(
                path, INITIAL_COUNT
            ))
    source_pod = SOURCE_DIR / 'reductionrun{:03d}'.format(INITIAL_COUNT)
    required = [
        source_pod / 'nonlinearrom/cluster0/state.rob001',
        source_pod / 'nonlinearrom/cluster0/state.rob{:03d}'.format(settings.HDMnclust),
        source_pod / 'nonlinearrom/cluster0/state.ref001',
        source_pod / 'nonlinearrom/cluster0/state.ref{:03d}'.format(settings.HDMnclust),
        source_pod / 'nonlinearrom/cluster0/state.svals',
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError('The clean 32-state POD is incomplete: {}.'.format(', '.join(missing)))
    return source_state, source_parameter, source_pod


def initialize(settings):
    """Create a v2 work root while leaving the v1 root and holdout immutable."""
    if MASTER_DIR.exists():
        raise RuntimeError('{} exists; refusing to overwrite an enrichment campaign.'.format(
            MASTER_DIR
        ))
    source_state, source_parameter, source_pod = verify_source(settings)
    source_data = SOURCE_DIR / 'data'
    if not source_data.is_dir():
        raise RuntimeError('Missing {}.'.format(source_data))
    MASTER_DIR.mkdir()
    shutil.copytree(source_data, MASTER_DIR / 'data')
    shutil.copytree(source_pod, MASTER_DIR / source_pod.name)
    shutil.copy2(source_state, MASTER_DIR / source_state.name)
    shutil.copy2(source_parameter, MASTER_DIR / source_parameter.name)
    source_settings = SOURCE_DIR / 'settings.readonly'
    if source_settings.is_file():
        shutil.copy2(source_settings, MASTER_DIR / source_settings.name)
    missing = [path for path in static_files(settings) if not Path(path).is_file()]
    if missing:
        raise RuntimeError('Copied static data is incomplete: {}.'.format(', '.join(missing)))
    write_json(manifest_path(), {
        'name': 'clean-laplace-steady-crm-residual-greedy-enrichment',
        'source_root': str(SOURCE_DIR),
        'source_snapshot_catalog_sha256': sha256(source_state),
        'source_parameter_catalog_sha256': sha256(source_parameter),
        'source_pod': str(source_pod),
        'master_root': str(MASTER_DIR),
        'initial_snapshot_count': INITIAL_COUNT,
        'target_snapshot_count': FINAL_COUNT,
        'excluded_holdout_sobol_index': HOLDOUT_SOBOL_INDEX,
        'shift_type': settings.ShiftType,
        'form': settings.Form,
        'pod_method': settings.PODMethod,
        'hyper_reduced': settings.HyperReduced,
        'selected_points': [],
    })
    print('Initialized isolated enrichment root {}.'.format(MASTER_DIR))


def verify_catalog_prefix(manifest):
    """Protect original catalog entries before appending any new snapshot."""
    source_state, source_parameter = catalog_paths(SOURCE_DIR)
    target_state, target_parameter = catalog_paths(MASTER_DIR)
    if sha256(source_state) != manifest['source_snapshot_catalog_sha256']:
        raise RuntimeError('The source state catalog changed after enrichment initialization.')
    if sha256(source_parameter) != manifest['source_parameter_catalog_sha256']:
        raise RuntimeError('The source parameter catalog changed after enrichment initialization.')
    source_state_lines = source_state.read_text().splitlines()
    target_state_lines = target_state.read_text().splitlines()
    if target_state_lines[1:1 + INITIAL_COUNT] != source_state_lines[1:]:
        raise RuntimeError('The original state catalog entries were modified.')
    source_parameter_lines = source_parameter.read_text().splitlines()
    target_parameter_lines = target_parameter.read_text().splitlines()
    base_lines = 2 + INITIAL_COUNT * 6
    if target_parameter_lines[1:base_lines] != source_parameter_lines[1:]:
        raise RuntimeError('The original parameter catalog entries were modified.')


def sobol_point(settings, index):
    """Return a one-based Sobol continuation point in the fixed CRM space."""
    ranges = list(zip(settings.ParamsLowerBound, settings.ParamsUpperBound))
    points = sobolGenerator(ranges, index, include_corners=True, make_plot=False)
    return [round(float(value), 6) for value in points[index - 1]]


def iteration_dir(iteration):
    """Return the dedicated metadata directory for one greedy selection."""
    return MASTER_DIR / 'screening/iteration{:03d}'.format(iteration)


def candidates_path(iteration):
    """Return the candidate catalog for one greedy selection."""
    return iteration_dir(iteration) / 'candidates.json'


def selection_path(iteration):
    """Return the immutable selected point record for one iteration."""
    return iteration_dir(iteration) / 'selection.json'


def screen_point_index(iteration, candidate_number):
    """Create a collision-free online-ROM point index for a screen run."""
    return iteration * 1000 + candidate_number


def screen_paths(settings, pod_index, point_index):
    """Return full PROM and Laplace output locations for a candidate."""
    root = Path(settings.MasterDir)
    rom_dir = root / 'evaluate/romruns{:03d}/point{:03d}'.format(pod_index, point_index)
    laplace_dir = root / 'evaluate/hromruns{:03d}/point{:03d}/Laplace-bin'.format(
        pod_index, point_index
    )
    return rom_dir, laplace_dir


def screen_initialize(settings, iteration, candidate_count):
    """Freeze a deterministic nontraining Sobol pool for one greedy iteration."""
    manifest = read_manifest()
    verify_catalog_prefix(manifest)
    if candidate_count < 1:
        raise ValueError('--candidate-count must be positive.')
    expected_iteration = current_count() + 1
    if iteration != expected_iteration or iteration > FINAL_COUNT:
        raise RuntimeError('Expected the next iteration to be {}, received {}.'.format(
            expected_iteration, iteration
        ))
    directory = iteration_dir(iteration)
    if directory.exists():
        raise RuntimeError('{} exists; refusing to replace a candidate catalog.'.format(directory))
    sampled = parse_parameter_catalog(MASTER_DIR / 'parsoldata.txt')
    candidates = []
    sobol_index = HOLDOUT_SOBOL_INDEX + 1
    while len(candidates) < candidate_count:
        point = sobol_point(settings, sobol_index)
        if not any(points_match(point, known) for known in sampled):
            number = len(candidates) + 1
            candidates.append({
                'candidate_number': number,
                'sobol_index': sobol_index,
                'point': point,
                'prom_point_index': screen_point_index(iteration, number),
            })
        sobol_index += 1
    directory.mkdir(parents=True)
    write_json(candidates_path(iteration), {
        'iteration': iteration,
        'pod_index': iteration - 1,
        'candidate_count': candidate_count,
        'indicator': 'final_relative_full_residual',
        'indicator_note': 'This ranks full PROM residuals only; it is not a certified physical-state error.',
        'candidates': candidates,
    })
    print('Prepared {} candidates for iteration {:03d}.'.format(candidate_count, iteration))


def read_candidates(iteration):
    """Read and validate a candidate catalog before an array task consumes it."""
    path = candidates_path(iteration)
    require_file(path)
    data = json.loads(path.read_text())
    if data.get('iteration') != iteration or data.get('pod_index') != iteration - 1:
        raise RuntimeError('{} does not match iteration {:03d}.'.format(path, iteration))
    candidates = data.get('candidates')
    if not isinstance(candidates, list) or len(candidates) != data.get('candidate_count'):
        raise RuntimeError('{} has an invalid candidate list.'.format(path))
    return data


def read_final_relative_residual(path):
    """Read AERO-F's final reported relative full residual from a history file."""
    require_file(path)
    rows = [line.split() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')]
    if not rows:
        raise RuntimeError('{} has no numeric rows.'.format(path))
    try:
        value = float(rows[-1][2])
    except (IndexError, ValueError) as error:
        raise RuntimeError('Cannot read the final residual from {}.'.format(path)) from error
    if not math.isfinite(value):
        raise RuntimeError('{} has a non-finite final residual.'.format(path))
    return value


def run_aerof(input_path, log_path, settings):
    """Run AERO-F using all ranks assigned to the current Slurm job."""
    aerof = os.environ.get('AEROF')
    if not aerof or not Path(aerof).is_file():
        raise RuntimeError('AEROF must name the built AERO-F executable.')
    command = [os.environ.get('MPI', 'srun'), '-n', str(settings.HDMnproc), aerof, str(input_path)]
    print(' '.join(command), flush=True)
    with open(log_path, 'w') as log_file:
        result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError('AERO-F failed; inspect {}.'.format(log_path))


def prepare_screen(settings, iteration, candidate_number):
    """Prepare one full PROM indicator evaluation with a correct Laplace shift."""
    candidates = read_candidates(iteration)
    if not 1 <= candidate_number <= len(candidates['candidates']):
        raise ValueError('--candidate-number must be in [1, {}].'.format(
            len(candidates['candidates'])
        ))
    candidate = candidates['candidates'][candidate_number - 1]
    pod_index = candidates['pod_index']
    if current_count() != pod_index:
        raise RuntimeError('The POD catalog changed while screening iteration {:03d}.'.format(iteration))
    rom_dir, laplace_dir = screen_paths(settings, pod_index, candidate['prom_point_index'])
    if rom_dir.exists() or laplace_dir.parent.exists():
        raise RuntimeError('{} or {} exists; refusing to overwrite a screen run.'.format(
            rom_dir, laplace_dir.parent
        ))
    frg = pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
    )
    rom = getRuns(settings)['ROM'](
        frg, p=candidate['point'], HDMind=pod_index,
        pind=candidate['prom_point_index'], hyper=False, hrtest=False, evaluate=True,
    )
    ref_nodes = np.loadtxt('{}_nodes'.format(settings.TopFilePath), dtype=np.float64)[:, 1:]
    ref_stick = np.loadtxt('{}_stick'.format(settings.TopFilePath), dtype=np.int32)[:, 2:]
    rom.prep(ref_nodes, ref_stick)
    deformed_top = create_deformed_top_file('{}/'.format(rom_dir), settings.TopFilePath)
    solveCurrentLaplace(deformed_top, '{}/'.format(laplace_dir.parent), frg, settings,
                        settings.LaplaceNumProc)
    rom.create_input_file()
    rom.writeInputFile()
    for suffix in ('001', '{:03d}'.format(settings.HDMnclust)):
        require_file(laplace_dir / 'ushift.bin{}'.format(suffix))
    write_json(iteration_dir(iteration) / 'candidate{:03d}.json'.format(candidate_number), {
        'candidate': candidate,
        'rom_directory': str(rom_dir),
        'laplace_directory': str(laplace_dir),
        'pod_index': pod_index,
        'shift_type': settings.ShiftType,
        'form': settings.Form,
        'hyper_reduced': False,
    })
    return rom_dir


def screen(settings, iteration, candidate_number):
    """Run and record one full-PROM residual indicator evaluation."""
    rom_dir = prepare_screen(settings, iteration, candidate_number)
    run_aerof(rom_dir / 'input', rom_dir / 'log', settings)
    residual = read_final_relative_residual(rom_dir / 'postpro/Residual.out')
    record_path = iteration_dir(iteration) / 'candidate{:03d}.json'.format(candidate_number)
    record = json.loads(record_path.read_text())
    record['final_relative_full_residual'] = residual
    write_json(record_path, record)
    print('Candidate {:03d} final relative full residual: {:.6e}.'.format(
        candidate_number, residual
    ))


def select(settings, iteration):
    """Choose the largest finite full-PROM residual from a completed screen."""
    candidates = read_candidates(iteration)
    if current_count() != candidates['pod_index']:
        raise RuntimeError('The POD catalog changed before selecting iteration {:03d}.'.format(iteration))
    selection = selection_path(iteration)
    if selection.exists():
        raise RuntimeError('{} exists; refusing to change an earlier selection.'.format(selection))
    scored = []
    for candidate in candidates['candidates']:
        number = candidate['candidate_number']
        record_path = iteration_dir(iteration) / 'candidate{:03d}.json'.format(number)
        require_file(record_path)
        record = json.loads(record_path.read_text())
        residual = record.get('final_relative_full_residual')
        if not isinstance(residual, (int, float)) or not math.isfinite(residual):
            raise RuntimeError('{} has no finite residual score.'.format(record_path))
        scored.append((float(residual), candidate))
    residual, candidate = max(scored, key=lambda entry: entry[0])
    write_json(selection, {
        'iteration': iteration,
        'pod_index': candidates['pod_index'],
        'selection_indicator': candidates['indicator'],
        'selected_candidate': candidate,
        'selected_final_relative_full_residual': residual,
        'candidate_scores': [
            {
                'candidate_number': item[1]['candidate_number'],
                'sobol_index': item[1]['sobol_index'],
                'point': item[1]['point'],
                'final_relative_full_residual': item[0],
            }
            for item in sorted(scored, key=lambda entry: entry[0], reverse=True)
        ],
    })
    print('Selected iteration {:03d}: {} with full residual {:.6e}.'.format(
        iteration, candidate['point'], residual
    ))


def read_selection(iteration):
    """Read the immutable selected point for an HDM stage."""
    path = selection_path(iteration)
    require_file(path)
    selection = json.loads(path.read_text())
    if selection.get('iteration') != iteration or selection.get('pod_index') != iteration - 1:
        raise RuntimeError('{} does not match iteration {:03d}.'.format(path, iteration))
    selected = selection.get('selected_candidate')
    if not isinstance(selected, dict) or not isinstance(selected.get('point'), list):
        raise RuntimeError('{} has no selected point.'.format(path))
    return selection


def hdm_dir(iteration):
    """Return the added HDM directory inside the isolated enrichment root."""
    return MASTER_DIR / 'HDMrun{:03d}'.format(iteration)


def shift_range(directory):
    """Require one nonconstant geometry-dependent Laplace shift per HDM."""
    shifts = sorted((directory / 'xdmf_files').glob('*-u_shift.xpost'))
    if len(shifts) != 1:
        raise RuntimeError('Expected one Laplace XPOST in {}.'.format(directory / 'xdmf_files'))
    _, values = pyaeroopt.util.frg_util.read_xpost(str(shifts[0]))
    values = np.asarray(values, dtype=np.float64)
    result = float(np.max(values) - np.min(values))
    if not math.isfinite(result) or result <= 1.0e-12:
        raise RuntimeError('Laplace shift {} is constant or non-finite.'.format(shifts[0]))
    return shifts[0], result


def prepare_hdm(settings, iteration):
    """Prepare the selected clean HDM without changing either v1 catalog."""
    manifest = read_manifest()
    verify_catalog_prefix(manifest)
    selection = read_selection(iteration)
    if current_count() != iteration - 1:
        raise RuntimeError('Expected {} snapshots before HDM {:03d}.'.format(iteration - 1, iteration))
    directory = hdm_dir(iteration)
    if directory.exists():
        raise RuntimeError('{} exists; refusing to overwrite a selected HDM.'.format(directory))
    point = [float(value) for value in selection['selected_candidate']['point']]
    frg = pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
    )
    runs = getRuns(settings)
    hdm1 = runs['HDM'](frg, p=point, HDMind=iteration, Pind=iteration, step=1)
    hdm2 = runs['HDM'](frg, p=point, HDMind=iteration, Pind=iteration, step=2)
    ref_nodes = np.loadtxt('{}_nodes'.format(settings.TopFilePath), dtype=np.float64)[:, 1:]
    ref_stick = np.loadtxt('{}_stick'.format(settings.TopFilePath), dtype=np.int32)[:, 2:]
    hdm1.prep(ref_nodes, ref_stick)
    for hdm in (hdm1, hdm2):
        hdm.create_input_file()
        hdm.writeInputFile()
    xpost, laplace_range = shift_range(directory)
    write_json(directory / 'greedy_sample.json', {
        'iteration': iteration,
        'point': point,
        'selection': str(selection_path(iteration)),
        'selection_indicator': selection['selection_indicator'],
        'selection_residual': selection['selected_final_relative_full_residual'],
        'laplace_xpost': xpost.name,
        'laplace_shift_range': laplace_range,
        'included_in_pod': False,
    })
    print('Prepared selected HDM {:03d} for {}.'.format(iteration, point))


def run_hdm(settings, iteration):
    """Run both HDM stages and require the existing clean convergence tolerance."""
    directory = hdm_dir(iteration)
    for input_name in ('input1', 'input2'):
        require_file(directory / input_name)
    if (directory / 'postpro/Residual.out').exists():
        raise RuntimeError('{} already has results; refusing to overwrite it.'.format(directory))
    run_aerof(directory / 'input1', directory / 'log1', settings)
    run_aerof(directory / 'input2', directory / 'log2', settings)
    residual = read_final_relative_residual(directory / 'postpro/Residual.out')
    if residual > settings.HDMtol2:
        raise RuntimeError('HDM {:03d} residual {:.6e} exceeds {:.6e}.'.format(
            iteration, residual, settings.HDMtol2
        ))
    metadata_path = directory / 'greedy_sample.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['hdm_final_reported_residual'] = residual
    write_json(metadata_path, metadata)
    print('HDM {:03d} completed with residual {:.6e}.'.format(iteration, residual))


def snapshot_line(iteration, settings):
    """Format one StateSnapshotData entry for the isolated campaign root."""
    return '{}/HDMrun{:03d}/snapshots/State.bin {} {} 1 1'.format(
        MASTER_DIR, iteration, settings.SnapIndex, settings.SnapIndex
    )


def parameter_lines(iteration, point, settings):
    """Format one MultipleSolutionsData entry for the isolated campaign root."""
    return [snapshot_line(iteration, settings)] + ['{}'.format(value) for value in point]


def append_catalogs(settings, iteration):
    """Append one audited HDM only after it has fully converged."""
    manifest = read_manifest()
    verify_catalog_prefix(manifest)
    selection = read_selection(iteration)
    directory = hdm_dir(iteration)
    metadata_path = directory / 'greedy_sample.json'
    require_file(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    point = [float(value) for value in selection['selected_candidate']['point']]
    if not points_match(point, metadata.get('point', [])):
        raise RuntimeError('Selected point and HDM metadata disagree for HDM {:03d}.'.format(iteration))
    for suffix in ('001', '{:03d}'.format(settings.HDMnclust)):
        require_file(directory / 'snapshots/State.bin{}'.format(suffix))
    residual = read_final_relative_residual(directory / 'postpro/Residual.out')
    if residual > settings.HDMtol2:
        raise RuntimeError('HDM {:03d} does not meet the clean residual tolerance.'.format(iteration))
    state_path, parameter_path = catalog_paths(MASTER_DIR)
    count = current_count()
    if count == iteration:
        points = parse_parameter_catalog(parameter_path)
        if not points_match(points[-1], point):
            raise RuntimeError('The existing catalog entry for HDM {:03d} differs from selection.'.format(iteration))
        print('Catalogs already include HDM {:03d}; reusing them.'.format(iteration))
        return
    if count != iteration - 1:
        raise RuntimeError('Expected {} catalog entries before adding HDM {:03d}.'.format(
            iteration - 1, iteration
        ))
    state_lines = state_path.read_text().splitlines()
    parameter_catalog_lines = parameter_path.read_text().splitlines()
    state_lines[0] = str(iteration)
    parameter_catalog_lines[0] = str(iteration)
    state_lines.append(snapshot_line(iteration, settings))
    parameter_catalog_lines.extend(parameter_lines(iteration, point, settings))
    temporary_state = state_path.with_name('{}.tmp'.format(state_path.name))
    temporary_parameter = parameter_path.with_name('{}.tmp'.format(parameter_path.name))
    temporary_state.write_text('\n'.join(state_lines) + '\n')
    temporary_parameter.write_text('\n'.join(parameter_catalog_lines) + '\n')
    os.replace(temporary_state, state_path)
    os.replace(temporary_parameter, parameter_path)
    metadata['included_in_pod'] = True
    metadata['catalog_index'] = iteration
    write_json(metadata_path, metadata)
    manifest['selected_points'].append({
        'iteration': iteration,
        'point': point,
        'selection_file': str(selection_path(iteration)),
        'hdm_directory': str(directory),
        'hdm_final_reported_residual': residual,
    })
    write_json(manifest_path(), manifest)
    print('Appended HDM {:03d} to the clean enrichment catalogs.'.format(iteration))


def build_pod(settings, iteration):
    """Build the next ScaLAPACK POD after one audited catalog append."""
    if current_count() != iteration:
        raise RuntimeError('Catalog count is not {} before building its POD.'.format(iteration))
    pod_dir = MASTER_DIR / 'reductionrun{:03d}'.format(iteration)
    if pod_dir.exists():
        raise RuntimeError('{} exists; refusing to overwrite a POD.'.format(pod_dir))
    points = parse_parameter_catalog(MASTER_DIR / 'parsoldata.txt')
    centroid = [sum(point[column] for point in points) / len(points)
                for column in range(len(points[0]))]
    frg = pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
    )
    pod = getRuns(settings)['POD'](frg, p=centroid, HDMind=iteration)
    hpc = pyaeroopt.interface.Hpc(
        machine='independence', batch=False, bg=False, nproc=settings.HDMnproc
    )
    hpc.mpi = os.environ.get('MPI', 'srun')
    pod.create_input_file()
    pod.writeInputFile()
    command = hpc.execute_str(pod.bin, pod.infile.fname)
    print(command, flush=True)
    with open(pod.infile.log, 'w') as log_file:
        result = subprocess.run(command, shell=True, stdout=log_file, stderr=subprocess.STDOUT,
                                check=False)
    if result.returncode != 0:
        raise RuntimeError('POD {:03d} failed; inspect {}.'.format(iteration, pod.infile.log))
    required = [
        pod_dir / 'nonlinearrom/cluster0/state.rob001',
        pod_dir / 'nonlinearrom/cluster0/state.rob{:03d}'.format(settings.HDMnclust),
        pod_dir / 'nonlinearrom/cluster0/state.ref001',
        pod_dir / 'nonlinearrom/cluster0/state.ref{:03d}'.format(settings.HDMnclust),
        pod_dir / 'nonlinearrom/cluster0/state.svals',
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError('POD {:03d} is incomplete: {}.'.format(iteration, ', '.join(missing)))
    print('Completed ScaLAPACK POD {:03d}.'.format(iteration))


def status(_settings):
    """Report the immutable source and current enrichment checkpoint."""
    manifest = read_manifest()
    count = current_count()
    print('Source root: {}'.format(manifest['source_root']))
    print('Enrichment root: {}'.format(MASTER_DIR))
    print('Snapshots in current POD catalog: {}'.format(count))
    print('Target snapshots: {}'.format(manifest['target_snapshot_count']))
    print('Next iteration: {}'.format(count + 1 if count < FINAL_COUNT else 'complete'))


def main():
    """Run one explicit restart-safe stage of the clean enrichment campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'mode', choices=('init', 'screen-init', 'screen', 'select', 'prepare-hdm',
                         'run-hdm', 'append', 'pod', 'status')
    )
    parser.add_argument('--iteration', type=int)
    parser.add_argument('--candidate-count', type=int, default=DEFAULT_CANDIDATE_COUNT)
    parser.add_argument('--candidate-number', type=int)
    args = parser.parse_args()
    settings = configure_settings()
    if args.mode == 'init':
        if args.iteration is not None or args.candidate_number is not None:
            raise ValueError('The init action does not take an iteration or candidate number.')
        initialize(settings)
        return
    if args.mode == 'status':
        status(settings)
        return
    if args.iteration is None:
        raise ValueError('--iteration is required for {}.'.format(args.mode))
    if args.mode == 'screen-init':
        screen_initialize(settings, args.iteration, args.candidate_count)
    elif args.mode == 'screen':
        if args.candidate_number is None:
            raise ValueError('--candidate-number is required for screen.')
        screen(settings, args.iteration, args.candidate_number)
    elif args.mode == 'select':
        select(settings, args.iteration)
    elif args.mode == 'prepare-hdm':
        prepare_hdm(settings, args.iteration)
    elif args.mode == 'run-hdm':
        run_hdm(settings, args.iteration)
    elif args.mode == 'append':
        append_catalogs(settings, args.iteration)
    else:
        build_pod(settings, args.iteration)


if __name__ == '__main__':
    main()
