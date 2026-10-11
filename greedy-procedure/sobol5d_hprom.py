#!/usr/bin/env python3
"""Hyper-reduce the global PROM of one POD batch with ECSW, and validate the HPROM.

One explicit stage per mode:
  prepare  copy the batch's POD (nonlinearrom/ and its catalogs) to a working directory. ECSW
           writes its reduced mesh into the POD prefix, so reductionrunNNN stays frozen.
  rows     generate the ECSW training rows of one training case: runs.py ROWGEN, one AERO-F
           run on the case's own geometry and inlet, from its converged HDM snapshot projected
           onto the basis (LSPG, Jacobian rows, spatial residual only, Laplace shift).

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
from pathlib import Path
import shutil

import pyaeroopt

import sobol5d_campaign as campaign
import sobol5d_projection as projection
from runs import getRuns

# The design starts with the 2^5 box corners; the Sobol points follow.
FIRST_SOBOL_INDEX = 33
DEFAULT_CASES = 32
# Rows per case for LSPG with Jacobian data: k(k + 1)/2 with k residual components (HGV2: 30).
RESIDUAL_COMPONENTS = 30


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


def prepare(count, cases=DEFAULT_CASES, name='hprom'):
    """Copy the batch's POD and catalogs, and freeze the training cases."""
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
    staging.mkdir(parents=True)
    shutil.copytree(pod / 'nonlinearrom', staging / 'nonlinearrom')
    for name_ in ('parsoldata.txt', 'statesnapdata.txt'):
        shutil.copy2(pod / name_, staging / name_)
    (staging / 'trainingmatrix/tr').mkdir(parents=True)
    campaign.write_json(staging / 'hprom.json', {
        'pod': str(pod),
        'directory': str(target),
        'cases': selected,
        'residual_components': RESIDUAL_COMPONENTS,
        'rows_per_case': RESIDUAL_COMPONENTS * (RESIDUAL_COMPONENTS + 1) // 2,
        'selection': 'first {} Sobol design points after the box corners (design {}-{})'.format(
            cases, selected[0]['index'], selected[-1]['index']),
    })
    staging.rename(target)
    print('Prepared {} from {} with {} training cases (design {}-{}).'.format(
        target, pod, len(selected), selected[0]['index'], selected[-1]['index']))


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
        '1\n{}\n'.format(case['snapshot']))
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


def main():
    """Run one explicit stage of the HPROM construction."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('prepare', 'rows'))
    parser.add_argument('--pod', type=int, required=True)
    parser.add_argument('--name', default='hprom', help='working directory Sobol5DRuns/NAMEnnn')
    parser.add_argument('--cases', type=int, default=DEFAULT_CASES, help='for prepare')
    parser.add_argument('--case', type=int, help='for rows, 1-based')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.pod, args.cases, args.name)
    else:
        if args.case is None:
            raise ValueError('--case is required for rows.')
        rows(args.pod, args.case, args.name)


if __name__ == '__main__':
    main()
