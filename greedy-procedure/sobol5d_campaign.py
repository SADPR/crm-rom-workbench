#!/usr/bin/env python3
"""Run the 5D transonic Sobol HDM campaign and build its POD batches."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess

import numpy as np
import pyaeroopt

import initial_hdm_campaign
import initial_pod
from runs import getRuns
from setup import Settings
from sobolGenerator import sobolGenerator


MASTER_DIR = 'Sobol5DRuns/'
PRECOMPUTE_DIR = 'Sobol5DPrecompute/'
LOG_DIR = 'Sobol5DRuns/logs/'
# Yihong Zhu's final box: [Mach, AoA, max-camber location, max camber, thickness].
LOWER_BOUNDS = [0.6, -2.0, 0.2, 0.0, 0.09]
UPPER_BOUNDS = [0.8, 2.0, 0.4, 0.03, 0.12]
# Reconstruction beta of the flow and turbulence Space blocks, not the inlet angle.
RECONSTRUCTION_BETA = 0.5
MAX_COUNT = 512
BATCH_COUNTS = (128, 256, 512)
# Restarts of unconverged HDMs: stage 3 continues stage 2, and stage 4 continues stage 3. Each
# writes a new snapshot folder. AERO-F counts iterations from the first stage, so MaxIts is a total.
EXTENDED_MAX_ITS = 15000
STAGE4_MAX_ITS = 35000
STAGE3_SNAPSHOTS = 'snapshots3'
RESTART_STAGES = (3, 4)
# An HDM that misses HDMtol2 after its restarts can still be accepted explicitly when its lift and
# drag moved less than ACCEPT_DRIFT (relative) over its last ACCEPT_WINDOW iterations.
ACCEPT_WINDOW = 1000
ACCEPT_DRIFT = 1.0e-3


def configure_settings():
    """Pin the 5D box, beta = 0.5, and the campaign roots in one place."""
    settings = Settings()
    settings.MasterDir = MASTER_DIR
    settings.InitHDMPreCompDir = PRECOMPUTE_DIR
    settings.RunGreedy = False
    settings.PODMethod = 'ScalapackSVD'
    settings.ParamsLowerBound = list(LOWER_BOUNDS)
    settings.ParamsUpperBound = list(UPPER_BOUNDS)
    settings.Beta = RECONSTRUCTION_BETA
    settings.numrand = MAX_COUNT
    settings.numSkip = 0
    return settings


def write_json(path, data):
    """Write small campaign metadata without exposing a partial file."""
    path = Path(path)
    temporary = path.with_name('{}.tmp'.format(path.name))
    temporary.write_text(json.dumps(data, indent=2, default=str) + '\n')
    os.replace(temporary, path)


def manifest_path():
    """Return the campaign manifest that freezes the design and batches."""
    return Path(MASTER_DIR) / 'campaign.json'


def read_manifest():
    """Read the frozen campaign manifest."""
    if not manifest_path().is_file():
        raise RuntimeError('Run the init action first.')
    return json.loads(manifest_path().read_text())


def design_points(settings):
    """Return the nested training design: 32 corners, then unscrambled Sobol points."""
    ranges = list(zip(settings.ParamsLowerBound, settings.ParamsUpperBound))
    points = sobolGenerator(ranges, settings.numrand, include_corners=True,
                            numToSkip=settings.numSkip, make_plot=False)
    return [[round(float(value), 6) for value in point] for point in points]


def run_dir(index, root=PRECOMPUTE_DIR):
    """Return the directory of one one-based design point below a campaign root."""
    return Path(root) / 'HDMrun{:03d}'.format(index)


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


def initialize(settings):
    """Create the campaign roots and freeze the training design before any job runs."""
    master_dir = Path(settings.MasterDir)
    precompute_dir = Path(settings.InitHDMPreCompDir)
    if master_dir.exists() or precompute_dir.exists():
        raise RuntimeError('{} or {} already exists; refusing to reuse a campaign root.'.format(
            master_dir, precompute_dir
        ))
    master_dir.mkdir()
    precompute_dir.mkdir()
    Path(LOG_DIR).mkdir()
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    write_json(master_dir / 'settings.effective.json', vars(settings))
    write_json(manifest_path(), {
        'name': 'sobol5d-transonic-steady-crm',
        'source_commit': source_commit,
        'lower_bounds': settings.ParamsLowerBound,
        'upper_bounds': settings.ParamsUpperBound,
        'reconstruction_beta': settings.Beta,
        'shift_type': settings.ShiftType,
        'laplace_shift_each': settings.LaplaceShiftEach,
        'laplace_boundaries': {'InletFixed_2': 1, 'StickMoving_3': 0, 'Symmetry_1': 'natural'},
        'form': settings.Form,
        'pod_method': settings.PODMethod,
        'hdm_tolerance': settings.HDMtol2,
        'design': 'sobolGenerator(..., include_corners=True): 32 corners, then unscrambled Sobol',
        'batch_counts': list(BATCH_COUNTS),
        'points': design_points(settings),
        'batches': {},
    })
    print('Initialized {} and {} with {} design points.'.format(
        master_dir, precompute_dir, MAX_COUNT
    ))


def prepare_static_data(settings):
    """Build a fresh shared mesh decomposition for the campaign."""
    data_dir = Path(settings.MasterDir) / 'data'
    read_manifest()
    if data_dir.exists():
        missing = [path for path in static_files(settings) if not Path(path).is_file()]
        if missing:
            raise RuntimeError('Incomplete static data: {}.'.format(', '.join(missing)))
        return

    data_dir.mkdir()
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
        raise RuntimeError('Missing static data: {}.'.format(', '.join(missing)))
    print('Prepared static CRM data in {}.'.format(data_dir))


def record_shift_stats(run_directory):
    """Require the geometry-specific Laplace shift to be nonconstant."""
    shift_files = sorted((run_directory / 'xdmf_files').glob('*-u_shift.xpost'))
    if len(shift_files) != 1:
        raise RuntimeError('Expected one Laplace shift XPOST in {}, found {}.'.format(
            run_directory / 'xdmf_files', len(shift_files)
        ))
    _, values = pyaeroopt.util.frg_util.read_xpost(str(shift_files[0]))
    values = np.asarray(values, dtype=np.float64)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum - minimum <= 1.0e-12:
        raise RuntimeError('Laplace shift in {} is constant or non-finite.'.format(shift_files[0]))
    write_json(run_directory / 'laplace_shift.json', {
        'xpost': shift_files[0].name,
        'minimum': minimum,
        'maximum': maximum,
        'range': maximum - minimum,
    })
    print('Verified nonconstant Laplace shift: [{:.6e}, {:.6e}].'.format(minimum, maximum))


def final_residual(run_directory, tolerance):
    """Read the final HDM residual and require the campaign tolerance."""
    residual = initial_pod.read_final_residual(run_directory)
    if residual > tolerance:
        raise RuntimeError('Final residual {:.6e} exceeds {:.6e} in {}.'.format(
            residual, tolerance, run_directory
        ))
    return residual


def prepare(settings, run_index):
    """Prepare one frozen design point and its geometry-specific Laplace shift."""
    points = read_manifest()['points']
    if not 1 <= run_index <= len(points):
        raise ValueError('--run-index must be in [1, {}].'.format(len(points)))
    if run_index == 1:
        prepare_static_data(settings)
    point, _ = initial_hdm_campaign.get_point(settings, run_index)
    if point != points[run_index - 1]:
        raise RuntimeError('Design point {} differs from the frozen manifest.'.format(run_index))
    initial_hdm_campaign.prepare(settings, run_index)
    record_shift_stats(run_dir(run_index, settings.InitHDMPreCompDir))


def run(settings, run_index):
    """Run both HDM phases and require the campaign tolerance."""
    initial_hdm_campaign.run(settings, run_index)
    directory = run_dir(run_index, settings.InitHDMPreCompDir)
    residual = final_residual(directory, settings.HDMtol2)
    print('Validated {} with final residual {:.6e}.'.format(directory, residual))


def extend(settings, run_index, stage=3, max_its=None, cfl_max=None):
    """Restart one unconverged HDM from its last state with more iterations.

    Stage 3 continues stage 2 up to EXTENDED_MAX_ITS iterations in total; stage 4 continues
    stage 3 up to STAGE4_MAX_ITS. `cfl_max` lowers the CFL ceiling, which changes the path to
    the steady state but not the converged state itself.
    """
    if stage not in RESTART_STAGES:
        raise ValueError('Restart stages are {}.'.format(RESTART_STAGES))
    directory = run_dir(run_index, settings.InitHDMPreCompDir)
    source = directory / 'input{}'.format(stage - 1)
    target = directory / 'input{}'.format(stage)
    if target.exists():
        raise RuntimeError('{} exists; refusing to repeat a stage-{} restart.'.format(target, stage))
    if stage == 4 and not (directory / 'stage3.json').is_file():
        raise RuntimeError('HDM {:03d} has no stage 3 to continue.'.format(run_index))
    previous = initial_pod.read_final_residual(directory)
    if previous <= settings.HDMtol2:
        raise RuntimeError('HDM {:03d} already meets the tolerance.'.format(run_index))
    aerof = os.environ.get('AEROF')
    if not aerof or not Path(aerof).is_file():
        raise RuntimeError('AEROF must name the built AERO-F executable.')

    prefix = '{}HDMrun{:03d}/'.format(settings.InitHDMPreCompDir, run_index)
    folders = {2: 'snapshots', 3: STAGE3_SNAPSHOTS, 4: 'snapshots4'}
    limits = {2: settings.MaxItsHDM2, 3: EXTENDED_MAX_ITS, 4: max_its or STAGE4_MAX_ITS}
    replacements = [
        ('MaxIts = {};'.format(limits[stage - 1]), 'MaxIts = {};'.format(limits[stage])),
        ('Prefix = "{}{}/";'.format(prefix, folders[stage - 1]),
         'Prefix = "{}{}/";'.format(prefix, folders[stage])),
    ]
    if cfl_max is not None:
        replacements.append(('CflMax = 100;', 'CflMax = {};'.format(cfl_max)))
    text = source.read_text()
    for old, new in replacements:
        if text.count(old) != 1:
            raise RuntimeError('Cannot find {!r} exactly once in {}.'.format(old, source))
        text = text.replace(old, new)
    (directory / folders[stage]).mkdir()
    target.write_text(text)

    hpc = pyaeroopt.interface.Hpc(machine='independence', batch=False, bg=False,
                                  nproc=settings.HDMnproc)
    hpc.mpi = os.environ.get('MPI', 'srun')
    command = hpc.execute_str(aerof, str(target))
    print(command, flush=True)
    log = directory / 'log{}'.format(stage)
    with open(log, 'w') as log_file:
        result = subprocess.run(command, shell=True, stdout=log_file, stderr=subprocess.STDOUT,
                                check=False)
    if result.returncode != 0:
        raise RuntimeError('AERO-F stage {} failed; inspect {}.'.format(stage, log))
    for suffix in ('001', '{:03d}'.format(settings.HDMnclust)):
        state = directory / folders[stage] / 'State.bin{}'.format(suffix)
        if not state.is_file():
            raise RuntimeError('Missing {}'.format(state))
    residual = initial_pod.read_final_residual(directory)
    write_json(directory / 'stage{}.json'.format(stage), {
        'max_its': limits[stage],
        'cfl_max': cfl_max,
        'snapshot': '{}/State.bin'.format(folders[stage]),
        'snap_index': repair_stage3_files(directory, folders[stage]),
        'header_repaired': True,
        'previous_residual': previous,
        'final_residual': residual,
    })
    final_residual(directory, settings.HDMtol2)
    print('Stage {} of {} reached {:.6e}.'.format(stage, directory, residual))


def revert_restart(settings, run_index, stage):
    """Undo a cancelled restart: restore the postpro histories and set its files aside.

    AERO-F copies each postpro history to NAME.back when a restart starts and then appends to
    NAME, so a restart cancelled before it wrote stage{N}.json is undone by restoring the copies.
    Run it only after the restart's job has stopped.
    """
    directory = run_dir(run_index, settings.InitHDMPreCompDir)
    if (directory / 'stage{}.json'.format(stage)).exists():
        raise RuntimeError('Stage {} of HDM {:03d} finished; there is nothing to revert.'.format(
            stage, run_index))
    folder = {3: STAGE3_SNAPSHOTS, 4: 'snapshots4'}[stage]
    moved = [directory / name for name in ('input{}'.format(stage), 'log{}'.format(stage), folder)]
    if not moved[0].exists():
        raise RuntimeError('HDM {:03d} has no stage-{} input.'.format(run_index, stage))
    histories = sorted((directory / 'postpro').glob('*.back'))
    if not histories:
        raise RuntimeError('No postpro backups in {}.'.format(directory / 'postpro'))
    for backup in histories:
        current = backup.with_suffix('')
        if not current.read_text().startswith(backup.read_text()):
            raise RuntimeError('{} does not extend {}; refusing to restore.'.format(current, backup))
    aside = directory / 'stage{}-cancelled'.format(stage)
    aside.mkdir()
    for path in moved:
        if path.exists():
            path.rename(aside / path.name)
    for backup in histories:
        current = backup.with_suffix('')
        current.rename(aside / current.name)
        shutil.copy2(backup, current)
    print('Reverted stage {} of {}: residual back to {:.6e}; files set aside in {}.'.format(
        stage, directory, initial_pod.read_final_residual(directory), aside))


def latest_restart(directory):
    """Return the record of the latest restart of an HDM, or None."""
    for stage in reversed(RESTART_STAGES):
        record = directory / 'stage{}.json'.format(stage)
        if record.is_file():
            return json.loads(record.read_text())
    return None


# A restart opens its snapshot files for appending, so a new file never gets the header's
# byte-order marker, and the frames before the restart step stay zero. Only the last frame
# holds the stage-3 state.
STAGE3_FILES = re.compile(r'State\.bin(\.shift|\.dt)?\d{3}$')


def repair_stage3_files(directory, folder=STAGE3_SNAPSHOTS):
    """Write the missing byte-order marker of every restart snapshot file; return its last frame."""
    frames = set()
    for path in sorted((directory / folder).iterdir()):
        if not STAGE3_FILES.match(path.name):
            continue
        with open(path, 'r+b') as handle:
            header = handle.read(24)
            marker, _, nodes, dim, count = struct.unpack('<id3i', header)
            size = 24 + count * (8 + 8 * nodes * dim)
            if marker not in (0, 1) or nodes <= 0 or dim <= 0 or count <= 0 or path.stat().st_size != size:
                raise RuntimeError('Unexpected stage-3 file layout in {}.'.format(path))
            if path.name.startswith('State.bin') and path.name[9:].isdigit():
                handle.seek(24 + (count - 1) * (8 + 8 * nodes * dim) + 8)
                last = np.frombuffer(handle.read(8 * nodes * dim), dtype='<f8')
                if not np.isfinite(last).all() or not last.any():
                    raise RuntimeError('The last frame of {} is empty or invalid.'.format(path))
            if marker == 0:
                handle.seek(0)
                handle.write(struct.pack('<i', 1))
        frames.add(count)
    if len(frames) != 1:
        raise RuntimeError('Stage-3 files of {} disagree on the frame count: {}.'.format(
            directory, sorted(frames)))
    return frames.pop()


def repair_stage3(settings, count):
    """Repair the snapshot headers and frame index of every restarted HDM up to count."""
    for index in range(1, count + 1):
        for root in (settings.MasterDir, settings.InitHDMPreCompDir):
            directory = run_dir(index, root)
            for stage in RESTART_STAGES:
                record = directory / 'stage{}.json'.format(stage)
                if not record.is_file():
                    continue
                restart = json.loads(record.read_text())
                frame = repair_stage3_files(directory, restart['snapshot'].split('/')[0])
                if restart.get('header_repaired') and restart['snap_index'] == frame:
                    continue
                restart.update(snap_index=frame, header_repaired=True)
                write_json(record, restart)
                print('Repaired stage {} of {}: snapshot frame {}.'.format(stage, directory, frame))


def classify(settings, index, point):
    """Return the state, final residual, and directory of one design point."""
    locations = [run_dir(index, root) for root in (settings.MasterDir, settings.InitHDMPreCompDir)]
    existing = [location for location in locations if location.exists()]
    if len(existing) > 1:
        raise RuntimeError('HDM {:03d} exists in both campaign roots.'.format(index))
    if not existing:
        return 'missing', None, None
    directory = existing[0]
    required = [
        directory / 'parameters.json',
        directory / 'laplace_shift.json',
        directory / 'references/Restart.data',
        directory / 'snapshots/State.bin001',
        directory / 'snapshots/State.bin{:03d}'.format(settings.HDMnclust),
        directory / 'postpro/Residual.out',
    ]
    restart = latest_restart(directory)
    if restart is not None:
        required.append(directory / restart['snapshot'].split('/')[0]
                        / 'State.bin{:03d}'.format(settings.HDMnclust))
    if not all(path.is_file() for path in required):
        return 'incomplete', None, directory
    if restart is not None and not restart.get('header_repaired'):
        return 'restart-unrepaired', None, directory
    if json.loads(required[0].read_text()).get('point') != point:
        raise RuntimeError('HDM {:03d} does not match the frozen design.'.format(index))
    if float(json.loads(required[1].read_text()).get('range', 0.0)) <= 1.0e-12:
        return 'constant-shift', None, directory
    residual = initial_pod.read_final_residual(directory)
    if residual <= settings.HDMtol2:
        return 'converged', residual, directory
    accepted = directory / 'accepted.json'
    if accepted.is_file() and json.loads(accepted.read_text()).get('residual') == residual:
        return 'accepted', residual, directory
    return 'unconverged', residual, directory


def force_drift(directory, window=ACCEPT_WINDOW):
    """Return the relative change of lift and drag over the last `window` iterations."""
    rows = np.array([[float(value) for value in line.split()[:6]]
                     for line in (directory / 'postpro/liftdrag.out').read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith('#')])
    iterations, drag, lift = rows[:, 0], rows[:, 4], rows[:, 5]
    start = np.searchsorted(iterations, iterations[-1] - window)
    return (float(abs(lift[-1] - lift[start]) / abs(lift[-1])),
            float(abs(drag[-1] - drag[start]) / abs(drag[-1])))


def accept(settings, run_index, reason):
    """Accept one restarted HDM that misses HDMtol2 but whose forces are stationary."""
    point = read_manifest()['points'][run_index - 1]
    state, residual, directory = classify(settings, run_index, point)
    if state != 'unconverged':
        raise RuntimeError('HDM {:03d} is {}; only unconverged HDMs can be accepted.'.format(
            run_index, state))
    if latest_restart(directory) is None:
        raise RuntimeError('HDM {:03d} has had no restart; run stage 3 first.'.format(run_index))
    lift, drag = force_drift(directory)
    if max(lift, drag) >= ACCEPT_DRIFT:
        raise RuntimeError('HDM {:03d}: lift/drag drift {:.2e}/{:.2e} over the last {} iterations '
                           'exceeds {:.0e}.'.format(run_index, lift, drag, ACCEPT_WINDOW, ACCEPT_DRIFT))
    # The residual pins the acceptance to this state; a later restart invalidates it.
    write_json(directory / 'accepted.json', {
        'reason': reason,
        'residual': residual,
        'tolerance': settings.HDMtol2,
        'window': ACCEPT_WINDOW,
        'lift_drift': lift,
        'drag_drift': drag,
        'snapshot': latest_restart(directory)['snapshot'],
    })
    print('Accepted HDM {:03d} at residual {:.3e}: lift/drag drift {:.2e}/{:.2e}.'.format(
        run_index, residual, lift, drag))


def geometry_aliases(points):
    """Map each design point to the earlier point with the same airfoil and flow, if any.

    With zero maximum camber the camber location has no effect, so the box corners that differ
    only in it are one HDM: aliases carry no new information and do not count toward a batch.
    """
    first, aliases = {}, {}
    for index, (mach, aoa, location, camber, thickness) in enumerate(points, start=1):
        key = (mach, aoa, None if camber == 0.0 else location, camber, thickness)
        if key in first:
            aliases[index] = first[key]
        else:
            first[key] = index
    return aliases


def batch_range(points, count):
    """Return the last design index of a batch of `count` distinct HDMs, and the aliases before it."""
    aliases = geometry_aliases(points)
    distinct = 0
    for index in range(1, len(points) + 1):
        if index not in aliases:
            distinct += 1
            if distinct == count:
                return index, {alias: original for alias, original in aliases.items() if alias <= index}
    raise RuntimeError('The design has fewer than {} distinct points; extend it.'.format(count))


def audit(settings, count):
    """Report every design point of a batch of `count` distinct HDMs without changing anything."""
    points = read_manifest()['points']
    last, aliases = batch_range(points, count)
    # Aliases need no HDM of their own; assemble gives them their original's state.
    rows = [(index,) + classify(settings, index, point)
            for index, point in enumerate(points[:last], start=1) if index not in aliases]
    print('Batch {}: design points 1-{}, {} of them aliases ({}).'.format(
        count, last, len(aliases), ', '.join('{}->{}'.format(a, o) for a, o in sorted(aliases.items()))))
    tally = Counter(row[1] for row in rows)
    print('Audited {} design points: {}.'.format(
        count, ', '.join('{} {}'.format(value, key) for key, value in sorted(tally.items()))
    ))
    for index, state, residual, _ in rows:
        if state != 'converged':
            suffix = '' if residual is None else ' ({:.6e})'.format(residual)
            print('  HDM {:03d}: {}{}'.format(index, state, suffix))
    return rows


def extend_candidates(settings, count):
    """Return the unconverged HDMs up to count that have not had a stage-3 restart yet."""
    points = read_manifest()['points']
    last, aliases = batch_range(points, count)
    candidates = []
    for index, point in enumerate(points[:last], start=1):
        if index in aliases:
            continue
        state, _, directory = classify(settings, index, point)
        if state == 'unconverged' and not (directory / 'stage3.json').is_file():
            candidates.append(index)
    return candidates


def assemble(settings, count):
    """Move converged HDMs into the training root and write the batch catalogs."""
    manifest = read_manifest()
    rows = audit(settings, count)
    pending = [index for index, state, _, _ in rows
               if state in ('missing', 'incomplete', 'restart-unrepaired')]
    if pending:
        raise RuntimeError('HDMs still missing or incomplete: {}.'.format(pending))
    last, aliases = batch_range(manifest['points'], count)
    entries = {}
    for index, state, _, directory in rows:
        if state not in ('converged', 'accepted'):
            continue
        target = run_dir(index, settings.MasterDir)
        if directory != target:
            directory.rename(target)
        entry = {'index': index, 'point': manifest['points'][index - 1], 'target': target}
        restart = latest_restart(target)
        if restart is not None:
            entry['snapshot'], entry['snap_index'] = restart['snapshot'], restart['snap_index']
        entries[index] = entry
    # The POD sees each distinct state once. The IC catalog also lists every alias with its
    # original's state, which is exact and keeps the whole box inside the interpolation hull.
    snapshots = [entry for _, entry in sorted(entries.items())]
    alias_entries = {alias: dict(entries[original], index=alias, point=manifest['points'][alias - 1])
                     for alias, original in aliases.items() if original in entries}
    parameters = [entry for _, entry in sorted({**entries, **alias_entries}.items())]
    state_path, parameter_path = initial_pod.catalog_paths(settings)
    initial_pod.write_atomically(state_path, initial_pod.snapshot_catalog(settings, snapshots))
    initial_pod.write_atomically(parameter_path, initial_pod.parameter_catalog(settings, parameters))
    # Unconverged or invalid HDMs stay in the precompute root and are listed, never dropped silently.
    manifest['batches'][str(count)] = {
        'last_index': last,
        'included': [entry['index'] for entry in snapshots],
        'aliases': {str(alias): aliases[alias] for alias in sorted(alias_entries)},
        'accepted': {str(index): residual for index, state, residual, _ in rows if state == 'accepted'},
        'excluded': {str(index): state for index, state, _, _ in rows
                     if state not in ('converged', 'accepted')},
    }
    write_json(manifest_path(), manifest)
    print('Wrote catalogs for batch {}: {} distinct snapshots, {} IC entries with aliases.'.format(
        count, len(snapshots), len(parameters)))


def pod_dir(settings, count):
    """Return the reduction directory of one batch."""
    return Path(settings.MasterDir) / 'reductionrun{:03d}'.format(count)


def build_pod(settings, count):
    """Build the ScaLAPACK POD of one assembled batch and keep a copy of its catalogs."""
    manifest = read_manifest()
    batch = manifest['batches'].get(str(count))
    if batch is None:
        raise RuntimeError('Assemble batch {} before building its POD.'.format(count))
    state_path, parameter_path = initial_pod.catalog_paths(settings)
    if int(state_path.read_text().splitlines()[0]) != len(batch['included']):
        raise RuntimeError('The root catalogs do not describe batch {}.'.format(count))
    directory = pod_dir(settings, count)
    if directory.exists():
        raise RuntimeError('{} exists; refusing to overwrite a POD.'.format(directory))
    aerof = os.environ.get('AEROF')
    if not aerof or not Path(aerof).is_file():
        raise RuntimeError('AEROF must name the built AERO-F executable.')

    points = [manifest['points'][index - 1] for index in batch['included']]
    centroid = [sum(point[column] for point in points) / len(points)
                for column in range(len(points[0]))]
    frg = pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
    )
    pod = getRuns(settings)['POD'](frg, p=centroid, HDMind=count)
    hpc = pyaeroopt.interface.Hpc(machine='independence', batch=False, bg=False,
                                  nproc=settings.HDMnproc)
    hpc.mpi = os.environ.get('MPI', 'srun')
    pod.create_input_file()
    pod.writeInputFile()
    command = hpc.execute_str(pod.bin, pod.infile.fname)
    print(command, flush=True)
    with open(pod.infile.log, 'w') as log_file:
        result = subprocess.run(command, shell=True, stdout=log_file, stderr=subprocess.STDOUT,
                                check=False)
    if result.returncode != 0:
        raise RuntimeError('POD {} failed; inspect {}.'.format(count, pod.infile.log))
    required = [
        directory / 'nonlinearrom/cluster0/state.rob001',
        directory / 'nonlinearrom/cluster0/state.rob{:03d}'.format(settings.HDMnclust),
        directory / 'nonlinearrom/cluster0/state.ref001',
        directory / 'nonlinearrom/cluster0/state.ref{:03d}'.format(settings.HDMnclust),
        directory / 'nonlinearrom/cluster0/state.svals',
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError('POD {} is incomplete: {}.'.format(count, ', '.join(missing)))
    # Later batches rewrite the root catalogs; PROMs of this batch read these copies.
    shutil.copy2(state_path, directory / state_path.name)
    shutil.copy2(parameter_path, directory / parameter_path.name)
    print('Completed ScaLAPACK POD {} with {} snapshots.'.format(count, len(points)))


def main():
    """Run one explicit stage of the 5D Sobol campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('init', 'point', 'prepare', 'run', 'extend', 'audit',
                                         'extend-list', 'repair-stage3', 'accept', 'revert-restart',
                                         'assemble', 'pod'))
    parser.add_argument('--run-index', type=int)
    parser.add_argument('--count', type=int)
    parser.add_argument('--stage', type=int, default=3, choices=RESTART_STAGES, help='for extend')
    parser.add_argument('--max-its', type=int, help='for extend, total iterations of the restart')
    parser.add_argument('--cfl-max', type=float, help='for extend, the CFL ceiling (inputs use 100)')
    parser.add_argument('--reason', help='for accept, why the HDM is accepted')
    args = parser.parse_args()

    settings = configure_settings()
    if args.mode == 'init':
        initialize(settings)
    elif args.mode in ('point', 'prepare', 'run', 'extend', 'accept', 'revert-restart'):
        if args.run_index is None:
            raise ValueError('--run-index is required for {}.'.format(args.mode))
        if args.mode == 'point':
            print('{}/{} {}'.format(args.run_index, MAX_COUNT, read_manifest()['points'][args.run_index - 1]))
        elif args.mode == 'prepare':
            prepare(settings, args.run_index)
        elif args.mode == 'revert-restart':
            revert_restart(settings, args.run_index, args.stage)
        elif args.mode == 'accept':
            accept(settings, args.run_index, args.reason or 'stationary lift and drag')
        elif args.mode == 'run':
            run(settings, args.run_index)
        else:
            extend(settings, args.run_index, args.stage, args.max_its, args.cfl_max)
    else:
        if args.count is None or not 1 <= args.count <= MAX_COUNT:
            raise ValueError('--count must be in [1, {}].'.format(MAX_COUNT))
        if args.mode == 'audit':
            audit(settings, args.count)
        elif args.mode == 'repair-stage3':
            repair_stage3(settings, args.count)
        elif args.mode == 'extend-list':
            # One marked line, so a job script can read it past the pyaeroopt banner.
            print('EXTEND: {}'.format(','.join(str(index) for index in extend_candidates(settings, args.count))))
        elif args.mode == 'assemble':
            assemble(settings, args.count)
        else:
            build_pod(settings, args.count)


if __name__ == '__main__':
    main()
