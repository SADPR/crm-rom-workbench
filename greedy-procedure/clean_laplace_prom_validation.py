#!/usr/bin/env python3
"""Validate the clean global CRM PROM at its first training point."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np
import pyaeroopt

from runs import create_deformed_top_file, getRuns, solveCurrentLaplace
from setup import Settings


MASTER_DIR = 'CleanLaplaceRuns/'
POD_INDEX = 32
POINT_INDEX = 1
PROM_RESIDUAL_TOLERANCE = 1.0e-8


def configure_settings():
    """Use the completed clean POD without starting a greedy iteration."""
    settings = Settings()
    settings.MasterDir = MASTER_DIR
    settings.RunGreedy = False
    settings.HyperReduced = False
    return settings


def paths(settings):
    """Return separate online-ROM and Laplace directories for this check."""
    master = Path(settings.MasterDir)
    run_dir = master / 'evaluate/romruns{:03d}/point{:03d}'.format(
        POD_INDEX, POINT_INDEX
    )
    laplace_dir = master / 'evaluate/hromruns{:03d}/point{:03d}/Laplace-bin'.format(
        POD_INDEX, POINT_INDEX
    )
    return run_dir, laplace_dir


def training_point(settings):
    """Read the exact first Sobol point recorded by the clean HDM campaign."""
    metadata_path = Path(settings.MasterDir) / 'HDMrun001/pilot.json'
    if not metadata_path.is_file():
        raise RuntimeError('Missing {}.'.format(metadata_path))
    metadata = json.loads(metadata_path.read_text())
    point = metadata.get('point')
    if metadata.get('index') != POINT_INDEX or not isinstance(point, list) or len(point) != 5:
        raise RuntimeError('{} does not describe HDMrun001.'.format(metadata_path))
    return [float(value) for value in point]


def check_clean_pod(settings):
    """Require all partitions of the clean ScaLAPACK POD before an online run."""
    cluster_dir = (
        Path(settings.MasterDir)
        / 'reductionrun{:03d}/nonlinearrom/cluster0'.format(POD_INDEX)
    )
    required = [
        Path(settings.MasterDir) / 'parsoldata.txt',
        cluster_dir / 'state.rob001',
        cluster_dir / 'state.rob{:03d}'.format(settings.HDMnclust),
        cluster_dir / 'state.ref001',
        cluster_dir / 'state.ref{:03d}'.format(settings.HDMnclust),
        cluster_dir / 'state.svals',
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError('Clean POD is incomplete: {}.'.format(', '.join(missing)))


def block_bounds(text, block_name):
    """Return the character range occupied by one AERO-F input block."""
    match = re.search(r'under\s+{}\s*\{{'.format(re.escape(block_name)), text)
    if match is None:
        raise RuntimeError('Missing under {} block.'.format(block_name))
    depth = 0
    for index in range(match.start(), len(text)):
        if text[index] == '{':
            depth += 1
        elif text[index] == '}':
            depth -= 1
            if depth == 0:
                return match.start(), index + 1
    raise RuntimeError('Unterminated under {} block.'.format(block_name))


def block_indent(text, start):
    """Return the indentation preceding one AERO-F input block."""
    line_start = text.rfind('\n', 0, start) + 1
    return text[line_start:start]


def add_physical_state_output(input_path):
    """Write the physical PROM state for comparison without changing online shifts."""
    text = input_path.read_text()

    # Remove the three fields from Output if this input came from the original
    # failed attempt, where they were accidentally placed one level too high.
    output_start, output_end = block_bounds(text, 'Output')
    output_block = text[output_start:output_end]
    output_field_indent = block_indent(text, output_start) + '   '
    misplaced = re.compile(
        r'^{}(?:StateVector|Frequency|OutputShiftVectorType)\s*=\s*[^;]*;\n?'.format(
            re.escape(output_field_indent)
        ),
        re.MULTILINE,
    )
    output_block = misplaced.sub('', output_block)
    text = text[:output_start] + output_block + text[output_end:]

    start, end = block_bounds(text, 'NonlinearROM')
    block = text[start:end]
    field_indent = block_indent(text, start) + '   '
    existing = re.compile(
        r'^{}(?:StateVector|Frequency|OutputShiftVectorType)\s*=\s*[^;]*;\n?'.format(
            re.escape(field_indent)
        ),
        re.MULTILINE,
    )
    block = existing.sub('', block)
    additions = (
        '{}StateVector = "State.bin";\n'
        '{}Frequency = 0;\n'
        '{}OutputShiftVectorType = None;\n'
    ).format(field_indent, field_indent, field_indent)
    block = block[:-1].rstrip() + '\n' + additions + block_indent(text, start) + '}'
    input_path.write_text(text[:start] + block + text[end:])


def prepare(settings):
    """Generate the online geometry, its Laplace shift, and one PROM input."""
    check_clean_pod(settings)
    point = training_point(settings)
    run_dir, laplace_dir = paths(settings)
    if run_dir.exists() or laplace_dir.exists():
        raise RuntimeError(
            '{} or {} exists; refusing to overwrite a PROM validation.'.format(
                run_dir, laplace_dir
            )
        )

    frg = pyaeroopt.interface.Frg(
        top='{}.top'.format(settings.TopFilePath),
        geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix),
    )
    rom = getRuns(settings)['ROM'](
        frg,
        p=point,
        HDMind=POD_INDEX,
        pind=POINT_INDEX,
        hyper=False,
        hrtest=False,
        evaluate=True,
    )
    ref_nodes = np.loadtxt(
        '{}_nodes'.format(settings.TopFilePath), dtype=np.float64
    )[:, 1:]
    ref_stick = np.loadtxt(
        '{}_stick'.format(settings.TopFilePath), dtype=np.int32
    )[:, 2:]
    rom.prep(ref_nodes, ref_stick)

    # This is a new online geometry preparation, so make its shift independently.
    deformed_top = create_deformed_top_file('{}/'.format(run_dir), settings.TopFilePath)
    solveCurrentLaplace(
        deformed_top,
        '{}/'.format(laplace_dir.parent),
        frg,
        settings,
        settings.LaplaceNumProc,
    )

    rom.create_input_file()
    rom.writeInputFile()
    input_path = run_dir / 'input'
    add_physical_state_output(input_path)

    required = [
        input_path,
        run_dir / 'deform/Position.bin001',
        run_dir / 'deform/Position.bin{:03d}'.format(settings.HDMnclust),
        laplace_dir / 'ushift.bin001',
        laplace_dir / 'ushift.bin{:03d}'.format(settings.HDMnclust),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError('PROM preparation is incomplete: {}.'.format(', '.join(missing)))

    manifest = {
        'pod_directory': '{}/reductionrun{:03d}'.format(settings.MasterDir.rstrip('/'), POD_INDEX),
        'point_index': POINT_INDEX,
        'point': point,
        'reference_hdm': '{}/HDMrun001'.format(settings.MasterDir.rstrip('/')),
        'form': settings.Form,
        'online_shift_type': settings.ShiftType,
        'physical_state_output': True,
        'hyper_reduced': False,
    }
    (run_dir / 'validation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Prepared clean PROM validation at {}.'.format(run_dir))


def numeric_rows(path):
    """Read numeric rows from an AERO-F history file."""
    if not path.is_file():
        raise RuntimeError('Missing {}.'.format(path))
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        try:
            rows.append([float(value) for value in line.split()])
        except ValueError as error:
            raise RuntimeError('Cannot parse {}.'.format(path)) from error
    if not rows:
        raise RuntimeError('{} has no numeric rows.'.format(path))
    return rows


def initial_residual(log_path):
    """Read AERO-F's unnormalized residual at the online initial condition."""
    pattern = re.compile(r'Spatial residual norm =\s*([+\-0-9.eE]+)')
    for line in log_path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            return float(match.group(1))
    raise RuntimeError('Cannot find the initial spatial residual in {}.'.format(log_path))


def merge_state(settings, result, output_prefix):
    """Merge distributed physical states for an apples-to-apples HDM comparison."""
    sower = os.environ.get('SOWER')
    if not sower or not Path(sower).is_file():
        raise RuntimeError('SOWER must name the sower executable.')
    geometry = '{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix)
    command = [
        sower,
        '-fluid',
        '-merge',
        '-con', '{}.con'.format(geometry),
        '-mesh', '{}.msh'.format(geometry),
        '-result', str(result),
        '-name', 'State',
        '-out', str(output_prefix),
        '-width', '16',
        '-precision', '16',
    ]
    subprocess.run(command, check=True)
    merged = Path('{}.xpost'.format(output_prefix))
    if not merged.is_file():
        raise RuntimeError('Sower did not produce {}.'.format(merged))
    return merged


def relative_error(value, reference):
    """Return a relative error, retaining an absolute fallback near zero."""
    difference = abs(value - reference)
    return difference / abs(reference) if abs(reference) > 1.0e-14 else difference


def state_error(settings, run_dir):
    """Compare the final physical PROM state with the corresponding clean HDM."""
    prom_xpost = merge_state(settings, run_dir / 'postpro/State.bin', run_dir / 'postpro/prom_state')
    hdm_xpost = merge_state(
        settings,
        Path(settings.MasterDir) / 'HDMrun001/references/Solution.bin',
        run_dir / 'postpro/hdm_state',
    )
    _, prom = pyaeroopt.util.frg_util.read_xpost(str(prom_xpost))
    _, hdm = pyaeroopt.util.frg_util.read_xpost(str(hdm_xpost))
    if prom.shape[:2] != hdm.shape[:2]:
        raise RuntimeError('PROM and HDM state dimensions do not match.')
    difference = prom[:, :, -1] - hdm[:, :, -1]
    reference_norm = np.linalg.norm(hdm[:, :, -1])
    if reference_norm == 0.0:
        raise RuntimeError('HDM reference state has zero norm.')
    return {
        'physical_state_relative_l2_error': float(np.linalg.norm(difference) / reference_norm),
        'physical_state_max_absolute_error': float(np.max(np.abs(difference))),
        'prom_state_frames': int(prom.shape[2]),
        'hdm_state_frames': int(hdm.shape[2]),
    }


def summarize(settings):
    """Report convergence, forces, and physical-state agreement with HDMrun001."""
    run_dir, _ = paths(settings)
    prom_residual = numeric_rows(run_dir / 'postpro/Residual.out')[-1]
    prom_liftdrag = numeric_rows(run_dir / 'postpro/liftdrag.out')[-1]
    hdm_liftdrag = numeric_rows(
        Path(settings.MasterDir) / 'HDMrun001/postpro/liftdrag.out'
    )[-1]
    summary = {
        'point_index': POINT_INDEX,
        'iterations': int(prom_residual[0]),
        'initial_absolute_residual': initial_residual(run_dir / 'log'),
        'final_relative_residual': prom_residual[2],
        'prom_drag': prom_liftdrag[4],
        'hdm_drag': hdm_liftdrag[4],
        'drag_relative_error': relative_error(prom_liftdrag[4], hdm_liftdrag[4]),
        'prom_lift': prom_liftdrag[5],
        'hdm_lift': hdm_liftdrag[5],
        'lift_relative_error': relative_error(prom_liftdrag[5], hdm_liftdrag[5]),
    }
    summary.update(state_error(settings, run_dir))
    (run_dir / 'validation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    print('PROM iterations: {}'.format(summary['iterations']))
    print('PROM final relative residual: {:.6e}'.format(summary['final_relative_residual']))
    print('Drag relative error: {:.6e}'.format(summary['drag_relative_error']))
    print('Lift relative error: {:.6e}'.format(summary['lift_relative_error']))
    print('Physical-state relative L2 error: {:.6e}'.format(
        summary['physical_state_relative_l2_error']
    ))
    if summary['final_relative_residual'] > PROM_RESIDUAL_TOLERANCE:
        raise RuntimeError(
            'PROM residual {:.6e} exceeds {:.6e}.'.format(
                summary['final_relative_residual'], PROM_RESIDUAL_TOLERANCE
            )
        )


def run(settings):
    """Execute the prepared clean PROM and compare it with its training HDM."""
    run_dir, laplace_dir = paths(settings)
    required = [
        run_dir / 'input',
        laplace_dir / 'ushift.bin001',
        laplace_dir / 'ushift.bin{:03d}'.format(settings.HDMnclust),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError('Missing prepared PROM files: {}.'.format(', '.join(missing)))
    if (run_dir / 'postpro/Residual.out').exists():
        raise RuntimeError('PROM result exists; refusing to overwrite {}.'.format(run_dir))

    # Make resume safe after a parser-level failure in an earlier validation input.
    add_physical_state_output(run_dir / 'input')

    aerof = os.environ.get('AEROF')
    if not aerof or not Path(aerof).is_file():
        raise RuntimeError('AEROF must name the built AERO-F executable.')
    command = shlex.split(os.environ.get('MPI', 'srun')) + [
        '-n', str(settings.HDMnproc), aerof, str(run_dir / 'input')
    ]
    print(' '.join(command), flush=True)
    log_path = run_dir / 'log'
    previous_log = run_dir / 'log.failed-before-resume'
    if log_path.is_file() and not previous_log.exists():
        log_path.rename(previous_log)
    with open(log_path, 'w') as log_file:
        result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError('PROM validation failed; inspect {}/log.'.format(run_dir))
    summarize(settings)
    print('Clean PROM validation completed: {}.'.format(run_dir))


def main():
    """Run either preparation or execution for the clean training-point test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    args = parser.parse_args()
    settings = configure_settings()
    if args.mode == 'prepare':
        prepare(settings)
    else:
        run(settings)


if __name__ == '__main__':
    main()
