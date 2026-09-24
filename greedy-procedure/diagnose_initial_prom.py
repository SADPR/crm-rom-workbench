#!/usr/bin/env python3
"""Run short, isolated solver diagnostics for the initial global PROM."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np

from pyaeroopt.util.frg_util import read_xpost


POD_INDEX = 32
POINT_INDEX = 1
SOURCE_RUN = Path(
    'GreedyRuns/evaluate/romruns{:03d}/point{:03d}'.format(POD_INDEX, POINT_INDEX)
)
HDM_RUN = Path('GreedyRuns/HDMrun001')
WORKSPACE = Path('CRMDiagnostics/initial-prom')


@dataclass(frozen=True)
class Variant:
    """Describe one solver-only change from the completed PROM validation."""

    name: str
    component_scaling: bool
    newton_max_its: int
    time_max_its: int
    line_search_max_its: int = 0
    capture_initial_state: bool = False


VARIANTS = (
    Variant('one_newton', True, 1, 12, capture_initial_state=True),
    Variant('unscaled', False, 30, 4),
    Variant('line_search', True, 30, 4, line_search_max_its=8),
)


def variant_by_name(name):
    """Return the requested diagnostic variant."""
    for variant in VARIANTS:
        if variant.name == name:
            return variant
    raise ValueError('Unknown variant {}.'.format(name))


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
    """Return the indentation preceding the selected AERO-F block."""
    line_start = text.rfind('\n', 0, start) + 1
    return text[line_start:start]


def update_assignment(text, block_name, key, value):
    """Replace one direct assignment in an AERO-F input block."""
    start, end = block_bounds(text, block_name)
    block = text[start:end]
    field_indent = block_indent(text, start) + '   '
    pattern = re.compile(
        r'^{}{}\s*=\s*[^;]*;'.format(re.escape(field_indent), re.escape(key)),
        re.MULTILINE,
    )
    matches = list(pattern.finditer(block))
    if len(matches) != 1:
        raise RuntimeError(
            'Expected one {} assignment in under {}, found {}.'.format(
                key, block_name, len(matches)
            )
        )
    match = matches[0]
    replacement = '{}{} = {};'.format(field_indent, key, value)
    block = block[:match.start()] + replacement + block[match.end():]
    return text[:start] + block + text[end:]


def add_assignment(text, block_name, key, value):
    """Append one assignment to an AERO-F input block."""
    start, end = block_bounds(text, block_name)
    block = text[start:end]
    parent_indent = block_indent(text, start)
    field_indent = parent_indent + '   '
    insertion = '\n{}{} = {};'.format(field_indent, key, value)
    block = block[:-1] + insertion + '\n' + parent_indent + '}'
    return text[:start] + block + text[end:]


def add_block(text, block_name, name, body):
    """Append a nested AERO-F block without modifying the template source."""
    start, end = block_bounds(text, block_name)
    block = text[start:end]
    parent_indent = block_indent(text, start)
    child_indent = parent_indent + '   '
    lines = ['\n{}under {} {{'.format(child_indent, name)]
    lines.extend('{}   {} = {};'.format(child_indent, key, value) for key, value in body)
    lines.append('{}}}'.format(child_indent))
    block = block[:-1] + '\n'.join(lines) + '\n' + parent_indent + '}'
    return text[:start] + block + text[end:]


def case_dir(variant):
    """Return the private output directory for one diagnostic variant."""
    return WORKSPACE / variant.name


def quoted(path):
    """Format one AERO-F path token."""
    return '"{}"'.format(path.as_posix())


def source_files():
    """Validate the completed PROM run that supplies geometry and Laplace data."""
    required = [
        SOURCE_RUN / 'input',
        SOURCE_RUN / 'parameters.txt',
        SOURCE_RUN / 'deform/Position.bin001',
        SOURCE_RUN / 'deform/Position.bin120',
        Path('GreedyRuns/evaluate/hromruns032/point001/Laplace-bin/ushift.bin001'),
        Path('GreedyRuns/evaluate/hromruns032/point001/Laplace-bin/ushift.bin120'),
        HDM_RUN / 'references/Solution.bin001',
        HDM_RUN / 'references/Solution.bin120',
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError('Missing completed-validation files: {}.'.format(', '.join(missing)))
    return required


def render_input(template, variant, directory):
    """Redirect outputs and apply only the solver setting under examination."""
    result_dir = directory / 'results'
    postpro_dir = directory / 'postpro'
    reference_dir = directory / 'references'

    text = template
    text = update_assignment(text, 'Postpro', 'Prefix', quoted(result_dir))
    text = update_assignment(text, 'Postpro', 'PressureCoefficient', '""')
    text = update_assignment(text, 'Postpro', 'SkinFrictionCoefficient', '""')
    text = update_assignment(text, 'Postpro', 'Mach', '""')
    text = update_assignment(text, 'Postpro', 'Displacement', '""')
    text = update_assignment(text, 'Postpro', 'Velocity', '""')
    text = update_assignment(text, 'Postpro', 'FluxResidual', '""')
    text = update_assignment(text, 'Postpro', 'ControlVolume', '""')
    text = update_assignment(text, 'Restart', 'Prefix', quoted(reference_dir))
    text = update_assignment(text, 'NonlinearROM', 'Prefix', quoted(postpro_dir))
    text = update_assignment(text, 'NonlinearROM', 'ReducedResidual', '"ReducedResidual.out"')
    text = update_assignment(
        text,
        'NonlinearRomOnline',
        'ComponentScaling',
        'True' if variant.component_scaling else 'False',
    )
    text = update_assignment(text, 'Time', 'MaxIts', str(variant.time_max_its))
    text = update_assignment(text, 'Newton', 'MaxIts', str(variant.newton_max_its))

    if variant.line_search_max_its:
        text = add_block(
            text,
            'Newton',
            'LineSearch',
            (
                ('MaxIts', variant.line_search_max_its),
                ('SufficientDecreaseFactor', '0.0001'),
                ('ContractionFactor', '0.5'),
            ),
        )

    if variant.capture_initial_state:
        text = add_assignment(text, 'NonlinearROM', 'StateVector', '"InitialState.bin"')
        text = add_assignment(text, 'NonlinearROM', 'Frequency', '1')
        text = add_assignment(text, 'NonlinearROM', 'OutputShiftVectorType', 'None')
    return text


def prepare():
    """Create fresh, solver-only inputs without altering the completed validation."""
    source_files()
    if WORKSPACE.exists():
        raise RuntimeError('{} exists; refusing to overwrite diagnostics.'.format(WORKSPACE))

    template_path = SOURCE_RUN / 'input'
    template = template_path.read_text()
    digest = hashlib.sha256(template.encode()).hexdigest()
    WORKSPACE.mkdir(parents=True)

    manifest = {
        'source_run': str(SOURCE_RUN),
        'source_input_sha256': digest,
        'reference_hdm': str(HDM_RUN),
        'variants': [],
    }
    for variant in VARIANTS:
        directory = case_dir(variant)
        for subdirectory in ('results', 'postpro', 'references'):
            (directory / subdirectory).mkdir(parents=True)
        (directory / 'input').write_text(render_input(template, variant, directory))
        (directory / 'configuration.json').write_text(
            json.dumps(asdict(variant), indent=2) + '\n'
        )
        manifest['variants'].append(asdict(variant))
        print('Prepared {}.'.format(directory))
    (WORKSPACE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def numeric_rows(path):
    """Read the numeric body of a whitespace-separated AERO-F history file."""
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
    """Read AERO-F's unnormalized residual at the reconstructed initial state."""
    pattern = re.compile(r'Spatial residual norm =\s*([+\-0-9.eE]+)')
    for line in log_path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            return float(match.group(1))
    raise RuntimeError('Cannot find initial spatial residual in {}.'.format(log_path))


def merge_state(result, output_prefix):
    """Merge 120 distributed solution files into one XPOST field."""
    sower = os.environ.get('SOWER')
    if not sower or not Path(sower).is_file():
        raise RuntimeError('SOWER must name the sower executable.')
    command = [
        sower,
        '-fluid',
        '-merge',
        '-con',
        'GreedyRuns/data/fluidmodel.con',
        '-mesh',
        'GreedyRuns/data/fluidmodel.msh',
        '-result',
        str(result),
        '-name',
        'State',
        '-out',
        str(output_prefix),
        '-width',
        '16',
        '-precision',
        '16',
    ]
    subprocess.run(command, check=True)
    merged = Path('{}.xpost'.format(output_prefix))
    if not merged.is_file():
        raise RuntimeError('Sower did not produce {}.'.format(merged))
    return merged


def state_projection_error(directory):
    """Measure the iteration-zero reconstructed state against the HDM solution."""
    postpro_dir = directory / 'postpro'
    prom_xpost = merge_state(postpro_dir / 'InitialState.bin', postpro_dir / 'initial_state')
    hdm_xpost = merge_state(HDM_RUN / 'references/Solution.bin', postpro_dir / 'hdm_state')
    _, prom = read_xpost(prom_xpost)
    _, hdm = read_xpost(hdm_xpost)
    if prom.shape[:2] != hdm.shape[:2]:
        raise RuntimeError('PROM and HDM state dimensions do not match.')
    difference = prom[:, :, 0] - hdm[:, :, -1]
    reference_norm = np.linalg.norm(hdm[:, :, -1])
    return {
        'state_relative_error': float(np.linalg.norm(difference) / reference_norm),
        'state_max_absolute_error': float(np.max(np.abs(difference))),
        'initial_state_frames': int(prom.shape[2]),
    }


def summarize_case(variant):
    """Write a compact, comparable summary for one completed diagnostic."""
    directory = case_dir(variant)
    residual_rows = numeric_rows(directory / 'postpro/Residual.out')
    liftdrag_rows = numeric_rows(directory / 'postpro/liftdrag.out')
    residual_0 = initial_residual(directory / 'log')
    hdm_liftdrag = numeric_rows(HDM_RUN / 'postpro/liftdrag.out')[-1]
    first_force = liftdrag_rows[0]
    final_force = liftdrag_rows[-1]
    final_relative_residual = residual_rows[-1][2]

    summary = {
        'configuration': asdict(variant),
        'iterations': int(residual_rows[-1][0]),
        'initial_relative_residual': residual_rows[0][2],
        'final_relative_residual': final_relative_residual,
        'initial_absolute_residual': residual_0,
        'final_absolute_residual': residual_0 * final_relative_residual,
        'initial_drag': first_force[4],
        'final_drag': final_force[4],
        'hdm_drag': hdm_liftdrag[4],
        'initial_drag_relative_error': abs(first_force[4] - hdm_liftdrag[4]) / abs(hdm_liftdrag[4]),
        'final_drag_relative_error': abs(final_force[4] - hdm_liftdrag[4]) / abs(hdm_liftdrag[4]),
        'initial_lift': first_force[5],
        'final_lift': final_force[5],
        'hdm_lift': hdm_liftdrag[5],
        'initial_lift_relative_error': abs(first_force[5] - hdm_liftdrag[5]) / abs(hdm_liftdrag[5]),
        'final_lift_relative_error': abs(final_force[5] - hdm_liftdrag[5]) / abs(hdm_liftdrag[5]),
    }
    if variant.capture_initial_state:
        summary.update(state_projection_error(directory))
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def write_report(summaries):
    """Write one report that makes the solver variants directly comparable."""
    (WORKSPACE / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    for name, summary in summaries.items():
        print(
            '{}: residual {:.6e} -> {:.6e}; drag error {:.6e} -> {:.6e}'.format(
                name,
                summary['initial_absolute_residual'],
                summary['final_absolute_residual'],
                summary['initial_drag_relative_error'],
                summary['final_drag_relative_error'],
            )
        )


def run(selected):
    """Run selected prepared cases with the batch allocation's 120 MPI ranks."""
    source_files()
    aerof = os.environ.get('AEROF')
    if not aerof or not Path(aerof).is_file():
        raise RuntimeError('AEROF must name the built AERO-F executable.')
    variants = VARIANTS if selected == 'all' else (variant_by_name(selected),)
    for variant in variants:
        directory = case_dir(variant)
        input_file = directory / 'input'
        if not input_file.is_file():
            raise RuntimeError('Missing {}; run prepare first.'.format(input_file))
        residual_file = directory / 'postpro/Residual.out'
        if residual_file.exists():
            raise RuntimeError('{} exists; refusing to overwrite diagnostics.'.format(residual_file))
        command = shlex.split(os.environ.get('MPI', 'srun')) + [
            '-n', '120', aerof, str(input_file)
        ]
        print(' '.join(command), flush=True)
        with open(directory / 'log', 'w') as log_file:
            result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise RuntimeError('{} failed; inspect {}/log.'.format(variant.name, directory))
        summaries[variant.name] = summarize_case(variant)
        summarize_case(variant)


def report():
    """Regenerate the combined report from completed diagnostic case summaries."""
    summaries = {}
    for variant in VARIANTS:
        summary_path = case_dir(variant) / 'summary.json'
        if summary_path.is_file():
            summaries[variant.name] = json.loads(summary_path.read_text())
    if not summaries:
        raise RuntimeError('No completed diagnostic summaries found.')
    write_report(summaries)


def main():
    """Parse a diagnostic action and, optionally, one named solver variant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'report'))
    parser.add_argument('--case', choices=('all',) + tuple(v.name for v in VARIANTS), default='all')
    args = parser.parse_args()
    if args.mode == 'prepare':
        if args.case != 'all':
            parser.error('--case applies only to run.')
        prepare()
    elif args.mode == 'run':
        run(args.case)
    else:
        if args.case != 'all':
            parser.error('--case applies only to run.')
        report()


if __name__ == '__main__':
    main()
