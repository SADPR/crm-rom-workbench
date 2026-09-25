#!/usr/bin/env python3
"""Create ParaView-ready HDM/PROM field comparisons for the clean CRM case."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from pyaeroopt.util.frg_util import read_xpost, write_xpost


DEFAULT_HDM_RESULTS = 'CleanLaplaceRuns/HDMrun001/results'
DEFAULT_PROM_RESULTS = (
    'CleanLaplaceRuns/evaluate/romruns032/point001/results'
)
DEFAULT_OUTPUT = (
    'CleanLaplaceRuns/evaluate/romruns032/point001/postpro/field_comparison'
)
DEFAULT_MESH_PREFIX = 'CleanLaplaceRuns/data/fluidmodel'
DEFAULT_TOP = 'mesh/naca0012_Re1p5.top'

FLOW_FIELDS = (
    ('Mach', 'Mach.bin'),
    ('PressureCoefficient', 'PressureCoefficient.bin'),
    ('SkinFriction', 'SkinFriction.bin'),
    ('Velocity', 'Velocity.bin'),
    ('Displacement', 'Displacement.bin'),
)
FLUX_RESIDUAL = ('FluxResidual', 'FluxRes.bin')


def parse_args():
    """Parse paths so the same postprocessor can later validate a holdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdm-results', default=DEFAULT_HDM_RESULTS)
    parser.add_argument('--prom-results', default=DEFAULT_PROM_RESULTS)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT)
    parser.add_argument('--mesh-prefix', default=DEFAULT_MESH_PREFIX)
    parser.add_argument('--top-file', default=DEFAULT_TOP)
    parser.add_argument('--parts', type=int, default=120)
    parser.add_argument('--sower', default=os.environ.get('SOWER'))
    parser.add_argument('--xp2exo', default=os.environ.get('XP2EXO'))
    parser.add_argument(
        '--force', action='store_true',
        help='replace only the requested comparison output directory',
    )
    return parser.parse_args()


def require_file(path):
    """Fail before launching a tool with an incomplete distributed result."""
    if not path.is_file():
        raise RuntimeError('Missing required file: {}.'.format(path))


def require_result(result_dir, filename, parts):
    """Require every partition that SOWER must merge."""
    for part in range(1, parts + 1):
        require_file(result_dir / '{}{:03d}'.format(filename, part))


def merge_field(sower, mesh_prefix, result_dir, field, filename, output):
    """Merge one distributed AERO-F result into an XPOST field."""
    command = [
        sower,
        '-fluid', '-merge',
        '-con', '{}.con'.format(mesh_prefix),
        '-mesh', '{}.msh'.format(mesh_prefix),
        '-result', str(result_dir / filename),
        '-name', field,
        '-out', str(output),
        '-width', '16',
        '-precision', '16',
    ]
    print('Merging {} from {}.'.format(field, result_dir))
    subprocess.run(command, check=True)
    merged = Path('{}.xpost'.format(output))
    require_file(merged)
    return merged


def load_final_frame(path):
    """Return the final XPOST tag and nodal values from one merged result."""
    tags, values = read_xpost(str(path))
    if values.shape[2] == 0:
        raise RuntimeError('{} contains no frames.'.format(path))
    return float(tags[-1]), values[:, :, -1]


def write_final_frame(path, field, tag, values):
    """Write one single-frame field with a ParaView-safe descriptive name."""
    write_xpost(str(path), 'FluidNodes', [tag], values, field)


def field_summary(hdm, prom):
    """Describe PROM-minus-HDM error without dividing nodal fields by zero."""
    difference = prom - hdm
    hdm_norm = np.linalg.norm(hdm)
    return {
        'relative_l2_difference': float(
            np.linalg.norm(difference) / hdm_norm if hdm_norm else np.linalg.norm(difference)
        ),
        'max_absolute_difference': float(np.max(np.abs(difference))),
        'hdm_minimum': float(np.min(hdm)),
        'hdm_maximum': float(np.max(hdm)),
        'prom_minimum': float(np.min(prom)),
        'prom_maximum': float(np.max(prom)),
    }


def write_exodus(xp2exo, top_file, decomposition, output, fields):
    """Convert a small related group of XPOST fields into one Exodus file."""
    command = [xp2exo, str(top_file), str(output), str(decomposition)]
    command.extend(str(field) for field in fields)
    print('Writing {}.'.format(output))
    subprocess.run(command, check=True)
    outputs = list(output.parent.glob('{}*'.format(output.name)))
    if not outputs:
        raise RuntimeError('xp2exo did not create {}.'.format(output))


def prepare_output(output_dir, force):
    """Create a fresh, explicitly named comparison directory."""
    if output_dir.exists():
        if not force:
            raise RuntimeError(
                '{} exists; use --force to replace only this comparison.'.format(output_dir)
            )
        shutil.rmtree(output_dir)
    for name in ('hdm', 'prom', 'difference'):
        (output_dir / name).mkdir(parents=True)


def main():
    """Merge, compare, and export all requested HDM and PROM fields."""
    args = parse_args()
    hdm_results = Path(args.hdm_results)
    prom_results = Path(args.prom_results)
    output_dir = Path(args.output_dir)
    mesh_prefix = Path(args.mesh_prefix)
    top_file = Path(args.top_file)
    decomposition = Path('{}.dec.{}'.format(top_file, args.parts))

    if not args.sower or not Path(args.sower).is_file():
        raise RuntimeError('Set SOWER or pass --sower with the SOWER executable.')
    if not args.xp2exo or not Path(args.xp2exo).is_file():
        raise RuntimeError('Set XP2EXO or pass --xp2exo with the xp2exo executable.')
    require_file(top_file)
    require_file(decomposition)
    require_file(Path('{}.con'.format(mesh_prefix)))
    require_file(Path('{}001'.format(mesh_prefix.with_suffix('.msh'))))

    for _, filename in FLOW_FIELDS + (FLUX_RESIDUAL,):
        require_result(hdm_results, filename, args.parts)
        require_result(prom_results, filename, args.parts)

    prepare_output(output_dir, args.force)
    values = {'hdm': {}, 'prom': {}, 'difference': {}}
    tag = None

    for field, filename in FLOW_FIELDS + (FLUX_RESIDUAL,):
        for source, result_dir in (('hdm', hdm_results), ('prom', prom_results)):
            merged = merge_field(
                args.sower, mesh_prefix, result_dir, field, filename,
                output_dir / source / field,
            )
            current_tag, current_values = load_final_frame(merged)
            if tag is None:
                tag = current_tag
            values[source][field] = current_values

        hdm = values['hdm'][field]
        prom = values['prom'][field]
        if hdm.shape != prom.shape:
            raise RuntimeError('{} has incompatible HDM/PROM shapes.'.format(field))
        values['difference'][field] = prom - hdm
        write_final_frame(
            output_dir / 'difference/{}_PROM_minus_HDM.xpost'.format(field),
            '{}_PROM_minus_HDM'.format(field), tag, values['difference'][field],
        )

    flux_component_count = values['hdm']['FluxResidual'].shape[1]
    if flux_component_count == 0:
        raise RuntimeError('FluxResidual contains no components.')
    for source in ('hdm', 'prom', 'difference'):
        flux = values[source]['FluxResidual']
        magnitude = np.linalg.norm(flux, axis=1, keepdims=True)
        values[source]['FluxResidualMagnitude'] = magnitude
        write_final_frame(
            output_dir / source / 'FluxResidualMagnitude.xpost',
            '{}_FluxResidualMagnitude'.format(source.upper()), tag, magnitude,
        )
        for component in range(flux_component_count):
            component_values = flux[:, component:component + 1]
            component_name = '{}_FluxResidualComponent{:02d}'.format(
                source.upper(), component
            )
            write_final_frame(
                output_dir / source / '{}.xpost'.format(component_name),
                component_name, tag, component_values,
            )

    flow_fields = {}
    for source in ('hdm', 'prom', 'difference'):
        prefix = 'HDM' if source == 'hdm' else (
            'PROM' if source == 'prom' else 'PROM_minus_HDM'
        )
        flow_fields[source] = []
        for field, _ in FLOW_FIELDS:
            output = output_dir / source / '{}_{}.xpost'.format(prefix, field)
            write_final_frame(output, '{}_{}'.format(prefix, field), tag, values[source][field])
            flow_fields[source].append(output)

    flux_fields = {}
    for source in ('hdm', 'prom', 'difference'):
        prefix = 'HDM' if source == 'hdm' else (
            'PROM' if source == 'prom' else 'PROM_minus_HDM'
        )
        flux_fields[source] = [output_dir / source / 'FluxResidualMagnitude.xpost']
        flux_fields[source].extend(
            output_dir / source / '{}_FluxResidualComponent{:02d}.xpost'.format(
                prefix, component
            )
            for component in range(flux_component_count)
        )

    for source in ('hdm', 'prom', 'difference'):
        write_exodus(
            args.xp2exo, top_file, decomposition,
            output_dir / '{}_flow_fields.exo'.format(source), flow_fields[source],
        )
        write_exodus(
            args.xp2exo, top_file, decomposition,
            output_dir / '{}_flux_residual.exo'.format(source), flux_fields[source],
        )

    summary = {
        'hdm_results': str(hdm_results),
        'prom_results': str(prom_results),
        'comparison': {
            field: field_summary(values['hdm'][field], values['prom'][field])
            for field, _ in FLOW_FIELDS + (FLUX_RESIDUAL,)
        },
        'flux_residual_components': flux_component_count,
        'note': (
            'FluxResidual is AERO-F\'s nodal flux-residual output. It is a spatial '
            'diagnostic and is distinct from the scalar full residual history in Residual.out.'
        ),
    }
    summary['comparison']['FluxResidualMagnitude'] = field_summary(
        values['hdm']['FluxResidualMagnitude'],
        values['prom']['FluxResidualMagnitude'],
    )
    (output_dir / 'comparison_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n'
    )
    print('Comparison postprocessing completed: {}.'.format(output_dir))


if __name__ == '__main__':
    main()
