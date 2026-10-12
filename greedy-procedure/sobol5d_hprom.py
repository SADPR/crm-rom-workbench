#!/usr/bin/env python3
"""Hyper-reduce the global PROM of one POD batch with ECSW, and validate the HPROM.

One explicit stage per mode:
  prepare  copy the batch's POD (nonlinearrom/ and its catalogs) to a working directory. ECSW
           writes its reduced mesh into the POD prefix, so reductionrunNNN stays frozen.
  rows     generate the ECSW training rows of one training case: runs.py ROWGEN, one AERO-F
           run on the case's own geometry and inlet, from its converged HDM snapshot projected
           onto the basis (LSPG, Jacobian rows, spatial residual only, Laplace shift).
  mesh     solve the ECSW NNLS over all the cases' rows (runs.py CM, all lift/drag faces kept)
           and decompose the reduced mesh, its wall distance and reduced basis for the online
           ranks.
  run      run the HPROM at one test point: the reference PROM's input (IC weights and
           iterations included) on the reduced mesh, with the truth HDM's deformed geometry and
           Laplace field cut to the reduced nodes.
  postpro  rebuild one HPROM's state on the full mesh from its reduced coordinates (a check of
           the wall Cp read on the reduced mesh, and the full-order residual).
  metrics  wall-Cp errors against the HDM and the reference PROM, reduced-coordinate distance to
           the PROM, lift and drag, and wall time and core-hours against the HDM.

The training cases are the first CASES Sobol points of the design after the 2^5 box corners
(design 33, 34, ...): a space-filling set by construction. A maximin choice over the training
states would take the corners first. For POD 384 the first 32 hold 16 shocked states and span
Mach 0.606-0.794 and alpha -1.88 to 1.88 deg.

Each case's snapshot and frame come from the batch's statesnapdata.txt, which already names the
stage-3 restarts (snapshots3, frame 3). ProjectInitialCondition stays off: the working copy keeps
the batch's IC products, which the online HPROM reads unchanged.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pyaeroopt

import sobol5d_campaign as campaign
import sobol5d_projection as projection
import sobol5d_test as test
from runs import getRuns
from ushift_inject import get_ushift_file

# The design starts with the 2^5 box corners; the Sobol points follow.
FIRST_SOBOL_INDEX = 33
DEFAULT_CASES = 32
# Rows per case for LSPG with Jacobian data: k(k + 1)/2 with k residual components (HGV2: 30).
RESIDUAL_COMPONENTS = 30
# The online HPROM runs on one 24-core node; the reduced mesh is decomposed for it.
HPROM_RANKS = 24
# The PROM the HPROM reproduces: shock IC, 5 x 5, NonDescriptor (sobol5d_sweep.py name, start).
REFERENCE = ('shock5x5_', 'shock')
HDM_TIME_S, HDM_RANKS = 2331.0, 120


def hprom_dir(settings, count, name='hprom'):
    """Return the working directory of one batch's HPROM."""
    return Path(settings.MasterDir) / '{}{:03d}'.format(name, count)


def snapshot_lines(pod):
    """Map each training HDM directory to its snapshot line in the batch's statesnapdata.txt."""
    lines = (pod / 'statesnapdata.txt').read_text().splitlines()
    snapshots = {}
    for line in lines[1:]:
        if line.strip():
            path = line.split()[0]
            snapshots[path.split('/snapshots')[0]] = line.strip()
    if len(snapshots) != int(lines[0]):
        raise RuntimeError('{} lists {} snapshots for {} HDMs.'.format(pod / 'statesnapdata.txt', lines[0],
                                                                     len(snapshots)))
    return snapshots


def training_cases(settings, pod, cases):
    """Return the first `cases` Sobol design points of the batch, with their snapshots."""
    points = campaign.read_manifest()['points']
    snapshots = snapshot_lines(pod)
    selected = []
    for index in range(FIRST_SOBOL_INDEX, len(points) + 1):
        run = '{}HDMrun{:03d}'.format(settings.MasterDir, index)
        if run not in snapshots:
            continue
        selected.append({'case': len(selected) + 1, 'index': index, 'point': points[index - 1],
                         'run': run, 'snapshot': snapshots[run]})
        if len(selected) == cases:
            return selected
    raise RuntimeError('{} has fewer than {} Sobol training points.'.format(pod, cases))


def read_record(target):
    """Read the working directory's record."""
    record = target / 'hprom.json'
    if not record.is_file():
        raise RuntimeError('Run the prepare action first; missing {}.'.format(record))
    return json.loads(record.read_text())


def prepare(count, cases=DEFAULT_CASES, name='hprom', rows_from=None):
    """Copy the batch's POD and catalogs, and freeze the training cases.

    With `rows_from` (another working directory's name), the training rows are that directory's:
    they do not depend on the ECSW tolerance, and the NNLS only reads them.
    """
    settings = campaign.configure_settings()
    pod = campaign.pod_dir(settings, count)
    target = hprom_dir(settings, count, name)
    staging = target.with_name(target.name + '.partial')
    for path in (target, staging):
        if path.exists():
            raise RuntimeError('{} exists; refusing to overwrite it.'.format(path))
    for path in (pod / 'parsoldata.txt', pod / 'statesnapdata.txt', pod / 'nonlinearrom/state.basisUicProducts'):
        if not path.is_file():
            raise RuntimeError('Missing {}; build POD {} first.'.format(path, count))
    selected = training_cases(settings, pod, cases)
    donor = None
    if rows_from is not None:
        donor = hprom_dir(settings, count, rows_from)
        record = read_record(donor)
        if ([case['index'] for case in record['cases']] != [case['index'] for case in selected]
                or record['residual_components'] != RESIDUAL_COMPONENTS):
            raise RuntimeError('{} was trained on other cases or components.'.format(donor))
        missing = [case['index'] for case in selected
                   if not list((donor / 'trainingmatrix/tr').glob('training.rows{}.*'.format(case['index'])))]
        if missing:
            raise RuntimeError('{} has no training rows for design points {}.'.format(donor, missing))
    staging.mkdir(parents=True)
    shutil.copytree(pod / 'nonlinearrom', staging / 'nonlinearrom')
    for name_ in ('parsoldata.txt', 'statesnapdata.txt'):
        shutil.copy2(pod / name_, staging / name_)
    if donor is None:
        (staging / 'trainingmatrix/tr').mkdir(parents=True)
    else:
        (staging / 'trainingmatrix').mkdir(parents=True)
        (staging / 'trainingmatrix/tr').symlink_to((donor / 'trainingmatrix/tr').resolve())
    campaign.write_json(staging / 'hprom.json', {
        'pod': str(pod),
        'directory': str(target),
        'cases': selected,
        'residual_components': RESIDUAL_COMPONENTS,
        'rows_per_case': RESIDUAL_COMPONENTS * (RESIDUAL_COMPONENTS + 1) // 2,
        'selection': 'first {} Sobol design points after the box corners (design {}-{})'.format(
            cases, selected[0]['index'], selected[-1]['index']),
        'rows_from': None if donor is None else str(donor),
    })
    staging.rename(target)
    print('Prepared {} from {} with {} training cases (design {}-{}).'.format(
        target, pod, len(selected), selected[0]['index'], selected[-1]['index']))


def training_snapshot(target, case):
    """Return the snapshot line ROWGEN reads for a case, staging a stage-3 restart's files first.

    Training reads the snapshot's State.bin.dtdata, one line per frame, and aborts when the frame
    has no line. A stage-3 restart appended its state as frame 3 of State.bin, .dt and .shift
    (repaired by repair-stage3), but wrote one line of dtdata. For such a case the working copy
    links the HDM's files read-only and writes a dtdata with one line per frame, repeating that
    line: the spatial-only training does not use the time step.
    """
    path, start = case['snapshot'].split()[:2]
    frame = int(start)
    dtdata = Path(path + '.dtdata')
    lines = [line for line in dtdata.read_text().splitlines() if line.strip()]
    if len(lines) >= frame:
        return case['snapshot']
    if len(lines) != 1:
        raise RuntimeError('{} has {} lines for frame {}; expected one.'.format(dtdata, len(lines), frame))
    staged = target / 'trainingmatrix/snapshots{}'.format(case['index'])
    if not staged.exists():
        staged.mkdir()
        source = Path(path).parent
        for item in sorted(source.glob(Path(path).name + '*')):
            if re.fullmatch(r'State\.bin(\.dt|\.shift)?\d{3}', item.name):
                (staged / item.name).symlink_to(item.resolve())
        (staged / dtdata.name).write_text((lines[0] + '\n') * frame)
    return ' '.join([(staged / Path(path).name).as_posix()] + case['snapshot'].split()[1:])


def rows_input(settings, target, case):
    """Write and return the ROWGEN input of one training case, patched for the working copy."""
    frg = pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
    )
    rowgen_class = getRuns(settings)['ROWGEN']
    # ROWGEN.__init__ writes its snapshot lists into reductionrunNNN/trainingmatrix; build the object
    # without it and write them here instead.
    row = rowgen_class.__new__(rowgen_class)
    pyaeroopt.interface.Aerof.__init__(row)
    row.frg, row.HDMind, row.Gind, row.p = frg, 0, case['index'], case['point']
    row.SuperDir = '{}/trainingmatrix/'.format(target.as_posix())
    row.SuperDirr = '{}/'.format(target.as_posix())
    (target / 'trainingmatrix/samplingsnapshots_rows{}.txt'.format(case['index'])).write_text(
        '1\n{}\n'.format(training_snapshot(target, case)))
    row.create_input_file()
    row.writeInputFile()
    path = target / 'trainingmatrix/input{}'.format(case['index'])
    text = projection.patch(path.read_text(), [
        ('MultipleSolutionsData = "{}parsoldata.txt";'.format(settings.MasterDir),
         'MultipleSolutionsData = "{}/parsoldata.txt";'.format(target.as_posix()), 1),
        ('ProjectInitialCondition = True;', 'ProjectInitialCondition = False;', 1),
        ('MaxNumResidualComponents = {};'.format(settings.MaxNumResComps),
         'MaxNumResidualComponents = {};'.format(RESIDUAL_COMPONENTS), 1),
    ], path)
    for key in ('Position', 'WallDistance'):
        expected = '{} = "{}/deform/{}.bin";'.format(key, case['run'], key)
        if text.count(expected) != 1:
            raise RuntimeError('{} does not read the case geometry {}.'.format(path, expected))
    path.write_text(text)
    return path


def rows(count, case_number, name='hprom'):
    """Generate the ECSW training rows of one training case."""
    settings = campaign.configure_settings()
    target = hprom_dir(settings, count, name)
    record = read_record(target)
    if not 1 <= case_number <= len(record['cases']):
        raise ValueError('--case must be in [1, {}].'.format(len(record['cases'])))
    case = record['cases'][case_number - 1]
    if record.get('rows_from'):
        raise RuntimeError('{} borrows the rows of {}; generate them there.'.format(target, record['rows_from']))
    output = target / 'trainingmatrix/tr'
    stem = 'training.rows{}.'.format(case['index'])
    if list(output.glob(stem + '*')):
        raise RuntimeError('{} already holds {}*; refusing to repeat a case.'.format(output, stem))
    path = rows_input(settings, target, case)
    log = target / 'trainingmatrix/log{}'.format(case['index'])
    projection.run_aerof(settings, path, log)
    written = sorted(output.glob(stem + '*'))
    if not written:
        raise RuntimeError('AERO-F wrote no {}* in {}; inspect {}.'.format(stem, output, log))
    print('Case {} (design {}): wrote {} row files, {:.1f} MB.'.format(
        case_number, case['index'], len(written), sum(p.stat().st_size for p in written) / 1e6))


def run_aerof(settings, input_file, log, ranks):
    """Run AERO-F on `ranks` MPI ranks and fail on a nonzero exit code."""
    aerof = os.environ.get('AEROF')
    if not aerof or not Path(aerof).is_file():
        raise RuntimeError('AEROF must name the built AERO-F executable.')
    hpc = pyaeroopt.interface.Hpc(machine='independence', batch=False, bg=False, nproc=ranks)
    hpc.mpi = os.environ.get('MPI', 'srun')
    command = hpc.execute_str(aerof, str(input_file))
    print(command, flush=True)
    with open(log, 'w') as log_file:
        result = subprocess.run(command, shell=True, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError('AERO-F failed; inspect {}.'.format(log))


def total_time(log):
    """Return AERO-F's Total Simulation wall time (max over ranks) from a log, or None."""
    found = re.findall(r'Total Simulation\s*:\s*[\d.]+\s+([\d.]+)', Path(log).read_text(errors='ignore'))
    return float(found[-1]) if found else None


def full_frg(settings):
    """Return the full-mesh mesh interface (120 clusters)."""
    return pyaeroopt.interface.Frg(top='{}.top'.format(settings.TopFilePath),
                                   geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix))


def reduced_frg(target):
    """Return the reduced-mesh mesh interface of a working directory."""
    return pyaeroopt.interface.Frg(top='{}/nonlinearrom/hrmesh.top'.format(target.as_posix()),
                                   geom_pre='{}/nonlinearrom/hrmeshbinary/fluidmodel'.format(target.as_posix()))


def reduced_nodes(target):
    """Return the 0-based full-mesh index of every reduced-mesh node, in reduced-mesh order."""
    table = np.loadtxt(target / 'nonlinearrom/hrmesh.top.nodes', dtype=np.int64, skiprows=1)
    # Every reduced vector is cut and split in this order, so it must be the reduced numbering.
    if not np.array_equal(table[:, 0], np.arange(1, len(table) + 1)):
        raise RuntimeError('{} is not in reduced-node order.'.format(target / 'nonlinearrom/hrmesh.top.nodes'))
    return table[:, 1] - 1


def count_lines(path):
    """Return the entries of a list file: its leading count when it has one, else its lines."""
    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    first = lines[0].split()
    return int(first[0]) if len(first) == 1 and len(lines) > 1 else len(lines)


def mesh(count, name='hprom', tolerance=None):
    """Solve the ECSW NNLS over every case's rows, then decompose the reduced mesh."""
    settings = campaign.configure_settings()
    target = hprom_dir(settings, count, name)
    record = read_record(target)
    if record.get('mesh') or (target / 'nonlinearrom/hrmesh.top').exists():
        raise RuntimeError('{} already holds a reduced mesh; refusing to rebuild it.'.format(target))
    output = target / 'trainingmatrix/tr'
    missing = [case['index'] for case in record['cases'] if not list(output.glob('training.rows{}.*'.format(case['index'])))]
    if missing:
        raise RuntimeError('No training rows yet for design points {}.'.format(missing))
    lines = ['{}\n'.format(len(record['cases']))]
    lines += ['{}/training.rows{}. 0 0 1 1\n'.format(output.as_posix(), case['index']) for case in record['cases']]
    (target / 'trainingmatrix/samplingsnapshots_allrows.txt').write_text(''.join(lines))
    cm_class = getRuns(settings)['CM']
    # CM.__init__ creates directories inside reductionrunNNN; build the object without it.
    cm = cm_class.__new__(cm_class)
    pyaeroopt.interface.Aerof.__init__(cm)
    cm.frg, cm.HDMind, cm.p = full_frg(settings), 0, record['cases'][0]['point']
    cm.SuperDirr = '{}/'.format(target.as_posix())
    cm.SuperDirrm1 = cm.SuperDirr
    cm.final, cm.NumInit, cm.rebuild = True, 0, True
    cm.create_input_file()
    cm.writeInputFile()
    path = target / 'input.cm.new'
    tolerance = settings.SamplingTol if tolerance is None else tolerance
    text = projection.patch(path.read_text(), [
        ('MultipleSolutionsData = "{}parsoldata.txt";'.format(settings.MasterDir),
         'MultipleSolutionsData = "{}/parsoldata.txt";'.format(target.as_posix()), 1),
        ('ProjectInitialCondition = True;', 'ProjectInitialCondition = False;', 1),
        ('SamplingTolerance = {};'.format(settings.SamplingTol), 'SamplingTolerance = {};'.format(tolerance), 1),
        ('IncludeAllLiftDragFaces = True;', 'IncludeAllLiftDragFaces = True;', 1),
    ], path)
    path.write_text(text)
    log = target / 'log.cm.new'
    run_aerof(settings, path, log, settings.HDMnproc)
    top = target / 'nonlinearrom'
    produced = [top / 'hrmesh.top', top / 'hrmesh.top.nodes', top / 'hrmesh.sampleweights',
                top / 'hrmesh.dwall.reduced.xpost', top / 'cluster0/hrmesh.rob.reduced.xpost',
                top / 'cluster0/hrmesh.ref.reduced.xpost']
    absent = [str(path_) for path_ in produced if not path_.is_file()]
    if absent:
        raise RuntimeError('The NNLS run did not write {}; inspect {}.'.format(', '.join(absent), log))
    hfrg = reduced_frg(target)
    (top / 'hrmeshbinary').mkdir(exist_ok=True)
    hfrg.part_mesh(HPROM_RANKS, log='/dev/null')
    hfrg.sower_fluid_top([HPROM_RANKS], HPROM_RANKS, log='/dev/null')
    hfrg.sower_fluid_split(file2split=str(top / 'hrmesh.dwall.reduced.xpost'),
                           out='{}/fluidmodel.dwall'.format((top / 'hrmeshbinary').as_posix()),
                           nclust=HPROM_RANKS, log='/dev/null')
    for kind in ('rob', 'ref'):
        hfrg.sower_fluid_split(file2split=str(top / 'cluster0/hrmesh.{}.reduced.xpost'.format(kind)),
                               out=str(top / 'cluster0/hrmesh.{}.reduced'.format(kind)),
                               nclust=HPROM_RANKS, log='/dev/null')
    nodes = reduced_nodes(target)
    with open(top / 'hrmesh.top') as handle:
        elements = sum(1 for line in handle if len(line.split()) == 6)
    record['mesh'] = {
        'tolerance': tolerance,
        'training_rows': len(record['cases']) * record['rows_per_case'],
        'sample_nodes': count_lines(top / 'hrmesh.samplenodes'),
        'reduced_nodes': int(len(nodes)),
        'reduced_elements': int(elements),
        'nnls_time_s': total_time(log),
        'ranks': HPROM_RANKS,
    }
    campaign.write_json(target / 'hprom.json', record)
    print('Reduced mesh of {}: {sample_nodes} sample nodes, {reduced_nodes} nodes, {reduced_elements} '
          'elements; NNLS run {nnls_time_s} s.'.format(target, **record['mesh']))


def hprom_run_dir(settings, count, name, index):
    """Return the directory of the HPROM at one test point."""
    return Path(settings.MasterDir) / 'evaluate/{}{:03d}/point{:03d}'.format(name, count, index)


def reference_dir(settings, count, index):
    """Return the reference PROM's run directory at one test point."""
    sweep_name, start = REFERENCE
    return Path(settings.MasterDir) / 'evaluate/{}{:03d}/{}-nondescriptor/point{:03d}'.format(sweep_name, count, start, index)


def cut_xpost(source, destination, rows, kind):
    """Write the given rows of a full-mesh XPOST file (header plus values) as a reduced one."""
    with open(source) as handle:
        header = handle.readline().strip()
        handle.readline()
        handle.readline()
        values = np.loadtxt(handle)
    values = values[rows]
    fmt = '%.10e' if values.ndim == 1 else ' '.join(['%.10e'] * values.shape[1])
    np.savetxt(destination, values, header='{}\n{}\n0'.format(header, len(values)), comments='', fmt=fmt)


def run(count, index, name='hprom'):
    """Run the HPROM at one test point, from the reference PROM's input."""
    settings = campaign.configure_settings()
    target = hprom_dir(settings, count, name)
    record = read_record(target)
    if not record.get('mesh'):
        raise RuntimeError('Run the mesh action first.')
    directory = hprom_run_dir(settings, count, name, index)
    if directory.exists():
        raise RuntimeError('{} exists; refusing to overwrite an HPROM.'.format(directory))
    base = reference_dir(settings, count, index)
    if not (base / 'input').is_file():
        raise RuntimeError('Missing {}; run the reference PROM first.'.format(base / 'input'))
    hdm = test.hdm_dir(index)
    for folder in ('results', 'postpro', 'references', 'deform', 'Laplace-bin'):
        (directory / folder).mkdir(parents=True)
    shutil.copy2(base / 'parameters.txt', directory / 'parameters.txt')
    nodes = reduced_nodes(target)
    hfrg = reduced_frg(target)
    # The truth HDM's deformed geometry and Laplace field, cut to the reduced nodes.
    for key in ('Position', 'WallDistance'):
        cut_xpost(hdm / 'deform/{}.xpost'.format(key), directory / 'deform/{}.xpost'.format(key), nodes, key)
        hfrg.sower_fluid_split(file2split=str(directory / 'deform/{}.xpost'.format(key)),
                               out=str(directory / 'deform/{}.bin'.format(key)), nclust=HPROM_RANKS, log='/dev/null')
    shift = get_ushift_file(str(hdm / 'naca0012_Re1p5_deformed.top'), '{}/'.format((directory / 'Laplace-bin').as_posix()),
                            nodes, u_star_path=str(hdm / 'xdmf_files'))
    hfrg.sower_fluid_split(file2split=shift, out=str(directory / 'Laplace-bin/ushift.bin'),
                           nclust=HPROM_RANKS, log='/dev/null')
    # The POD path exactly as the reference PROM input names it.
    pod = campaign.pod_dir(settings, count).as_posix()
    text = projection.patch((base / 'input').read_text(), [
        (base.as_posix(), directory.as_posix(), None),
        ('GeometryPrefix = "{}data/{}";'.format(settings.MasterDir, settings.GeometryPrefix),
         'GeometryPrefix = "{}/nonlinearrom/hrmeshbinary/fluidmodel";'.format(target.as_posix()), 1),
        ('MultipleSolutionsData = "{}/parsoldata.txt";'.format(pod),
         'MultipleSolutionsData = "{}/parsoldata.txt";'.format(target.as_posix()), 1),
        ('Position = "{}/deform/Position.bin";'.format(hdm.as_posix()),
         'Position = "{}/deform/Position.bin";'.format(directory.as_posix()), 1),
        ('WallDistance = "{}/deform/WallDistance.bin";'.format(hdm.as_posix()),
         'WallDistance = "{}/deform/WallDistance.bin";'.format(directory.as_posix()), 1),
        ('LaplaceSnapshotData = "{}/Laplace-bin/ushift.bin";'.format(hdm.as_posix()),
         'LaplaceSnapshotData = "{}/Laplace-bin/ushift.bin";'.format(directory.as_posix()), 1),
        ('Prefix = "{}/";'.format(pod), 'Prefix = "{}/";'.format(target.as_posix()), 1),
        ('      StatePrefix = "state";\n',
         '      StatePrefix = "state";\n      StateBasis = "hrmesh.rob.reduced";\n'
         '      StateBasisReference = "hrmesh.ref.reduced";\n      ReducedMeshPrefix = "hrmesh";\n', 1),
        ('SystemApproximation = None;', 'SystemApproximation = ECSW;', 1),
    ], base / 'input')
    (directory / 'input').write_text(text)
    campaign.write_json(directory / 'hprom.json', {
        'reference_prom': str(base), 'working_directory': str(target), 'test_index': index,
        'geometry_and_laplace_from': str(hdm), 'ranks': HPROM_RANKS, 'mesh': record['mesh'],
    })
    run_aerof(settings, directory / 'input', directory / 'log', HPROM_RANKS)
    if not (directory / 'postpro/Residual.out').is_file():
        raise RuntimeError('HPROM failed; inspect {}.'.format(directory / 'log'))
    test.merge_surface_fields(hfrg, directory, fields=('PressureCoefficient',))
    print('Completed the HPROM at test point {:03d} in {} s.'.format(index, total_time(directory / 'log')))


def postpro(count, index, name='hprom'):
    """Rebuild one HPROM's state on the full mesh from its reduced coordinates."""
    settings = campaign.configure_settings()
    target = hprom_dir(settings, count, name)
    record = read_record(target)
    directory = hprom_run_dir(settings, count, name, index)
    coordinates = directory / 'postpro/ReducedCoords.out'
    if not coordinates.is_file():
        raise RuntimeError('Missing {}; run the HPROM first.'.format(coordinates))
    full = directory / 'fullmesh'
    if full.exists():
        raise RuntimeError('{} exists; refusing to repeat a postprocessing.'.format(full))
    for folder in ('results', 'postpro', 'references'):
        (full / folder).mkdir(parents=True)
    base = reference_dir(settings, count, index)
    # The POD path exactly as the reference PROM input names it.
    pod = campaign.pod_dir(settings, count).as_posix()
    text = projection.patch((base / 'input').read_text(), [
        ('Type = SteadyNonlinearRom;', 'Type = SteadyNonlinearRomPostprocessing;', 1),
        ('{}/results/'.format(base.as_posix()), '{}/results/'.format(full.as_posix()), 1),
        ('{}/references/'.format(base.as_posix()), '{}/references/'.format(full.as_posix()), 1),
        ('{}/postpro/'.format(base.as_posix()), '{}/postpro/'.format(full.as_posix()), 1),
        ('ReducedCoordinates = "ReducedCoords.out";', 'ReducedCoordinates = "";', 1),
        ('   InterpICWeights = ', '   ReducedCoordinates = "{}";\n   InterpICWeights = '.format(coordinates.as_posix()), 1),
        ('Prefix = "{}/";'.format(pod), 'Prefix = "{}/";'.format(target.as_posix()), 1),
    ], base / 'input')
    (full / 'input').write_text(text)
    run_aerof(settings, full / 'input', full / 'log', settings.HDMnproc)
    test.merge_surface_fields(full_frg(settings), full, fields=('PressureCoefficient',))
    print('Rebuilt the HPROM at test point {:03d} on the full mesh.'.format(index))


def last_row(path):
    """Return the last row of a numeric AERO-F history file."""
    return projection.table(path)[-1]


def metrics(count, name='hprom'):
    """Compare every HPROM with the HDM and the reference PROM."""
    settings = campaign.configure_settings()
    target = hprom_dir(settings, count, name)
    record = read_record(target)
    raw_nodes = np.loadtxt('{}_nodes'.format(settings.TopFilePath), dtype=np.float64)
    wall = test.wall_nodes('{}.top'.format(settings.TopFilePath), raw_nodes)
    nodes = reduced_nodes(target)
    where = {int(node): k for k, node in enumerate(nodes)}
    absent = [int(node) for node in wall if int(node) not in where]
    if absent:
        raise RuntimeError('{} wall nodes are not on the reduced mesh.'.format(len(absent)))
    rows = np.array([where[int(node)] for node in wall])
    points = {}
    for index in range(1, test.TEST_COUNT + 1):
        directory = hprom_run_dir(settings, count, name, index)
        if not directory.exists():
            continue
        entry = {'status': 'ok'}
        history = directory / 'postpro/Residual.out'
        xpost = directory / 'postpro/PressureCoefficient.xpost'
        if not history.is_file() or not xpost.is_file():
            points['{:03d}'.format(index)] = {'status': 'crashed'}
            continue
        hdm = test.hdm_dir(index)
        base = reference_dir(settings, count, index)
        truth = test.wall_pressure(hdm, wall)
        cp = projection.xpost_frames(xpost, rows)[-1]
        reference = projection.xpost_frames(base / 'postpro/PressureCoefficient.xpost', wall)[-1]
        q = projection.table(directory / 'postpro/ReducedCoords.out')[-1, 3:]
        q_ref = projection.table(base / 'postpro/ReducedCoords.out')[-1, 3:]
        lift, drag = last_row(directory / 'postpro/liftdrag.out')[[5, 4]]
        lift_hdm, drag_hdm = last_row(hdm / 'postpro/liftdrag.out')[[5, 4]]
        seconds = total_time(directory / 'log')
        entry.update(
            cp_l2=test.relative_error(cp, truth, 2),
            cp_l2_reference_prom=test.relative_error(reference, truth, 2),
            cp_vs_prom=test.relative_error(cp, reference, 2),
            q_vs_prom=100.0 * float(np.linalg.norm(q - q_ref) / np.linalg.norm(q_ref)),
            lift=100.0 * abs(lift - lift_hdm) / abs(lift_hdm),
            drag=100.0 * abs(drag - drag_hdm) / abs(drag_hdm),
            time_s=seconds,
            speedup_wall=HDM_TIME_S / seconds if seconds else None,
            speedup_core_hours=HDM_TIME_S * HDM_RANKS / (seconds * HPROM_RANKS) if seconds else None,
        )
        full = directory / 'fullmesh/postpro/PressureCoefficient.xpost'
        if full.is_file():
            entry['cp_reduced_vs_full_mesh'] = test.relative_error(cp, projection.xpost_frames(full, wall)[-1], 2)
        points['{:03d}'.format(index)] = entry
    ok = [entry for entry in points.values() if entry['status'] == 'ok']
    summary = {}
    for key in ('cp_l2', 'cp_l2_reference_prom', 'cp_vs_prom', 'q_vs_prom', 'lift', 'drag', 'time_s',
                'speedup_wall', 'speedup_core_hours'):
        values = np.array([entry[key] for entry in ok if entry.get(key) is not None])
        if len(values):
            summary[key] = {'mean': float(values.mean()), 'median': float(np.median(values)), 'max': float(values.max())}
    campaign.write_json(Path(settings.MasterDir) / 'evaluate/{}{:03d}/metrics.json'.format(name, count), {
        'pod': record['pod'], 'mesh': record['mesh'], 'reference': list(REFERENCE),
        'hdm_time_s': HDM_TIME_S, 'hdm_ranks': HDM_RANKS, 'hprom_ranks': HPROM_RANKS,
        'unit': 'percent relative L2 on the z = 0 wall nodes; times in seconds',
        'counted': len(ok), 'crashed': sorted(k for k, entry in points.items() if entry['status'] != 'ok'),
        'summary': summary, 'points': points,
    })
    for key, values in summary.items():
        print('{:24s} mean {mean:9.3f}  median {median:9.3f}  max {max:9.3f}'.format(key, **values))
    print('{} HPROMs counted, crashed: {}.'.format(len(ok), sorted(k for k, entry in points.items() if entry['status'] != 'ok')))


def main():
    """Run one explicit stage of the HPROM construction."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('prepare', 'rows', 'mesh', 'run', 'postpro', 'metrics'))
    parser.add_argument('--pod', type=int, required=True)
    parser.add_argument('--name', default='hprom',
                        help='working directory Sobol5DRuns/NAMEnnn, runs in evaluate/NAMEnnn')
    parser.add_argument('--cases', type=int, default=DEFAULT_CASES, help='for prepare')
    parser.add_argument('--rows-from', help='for prepare, reuse the rows of another working directory NAME')
    parser.add_argument('--case', type=int, help='for rows, 1-based')
    parser.add_argument('--tolerance', type=float, help='for mesh, the ECSW tolerance (setup.py: 1e-3)')
    parser.add_argument('--index', type=int, help='for run and postpro, the test point')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.pod, args.cases, args.name, args.rows_from)
    elif args.mode == 'rows':
        if args.case is None:
            raise ValueError('--case is required for rows.')
        rows(args.pod, args.case, args.name)
    elif args.mode == 'mesh':
        mesh(args.pod, args.name, args.tolerance)
    elif args.mode == 'metrics':
        metrics(args.pod, args.name)
    else:
        if args.index is None or not 1 <= args.index <= test.TEST_COUNT:
            raise ValueError('--index must be in [1, {}].'.format(test.TEST_COUNT))
        (run if args.mode == 'run' else postpro)(args.pod, args.index, args.name)


if __name__ == '__main__':
    main()
