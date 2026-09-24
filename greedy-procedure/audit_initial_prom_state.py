#!/usr/bin/env python3
"""Compare the HDM POD snapshot, final HDM solution, and PROM reconstruction."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np

from pyaeroopt.util.frg_util import read_xpost


HDM_RUN = Path('GreedyRuns/HDMrun001')
PROM_DIAGNOSTIC = Path('CRMDiagnostics/initial-prom/one_newton/postpro/InitialState.bin')
OUTPUT_DIRECTORY = Path('CRMDiagnostics/initial-prom/state-audit')
VARIABLES = ('density', 'x_momentum', 'y_momentum', 'z_momentum', 'energy', 'turbulence')


def merge_state(result, output_prefix):
    """Merge a distributed AERO-F state into one XPOST file."""
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



def read_frame(path, frame):
    """Read the selected frame of a merged state field."""
    _, state = read_xpost(path)
    if state.ndim != 3 or state.shape[1] != len(VARIABLES):
        raise RuntimeError('Expected a six-component state in {}.'.format(path))
    return state[:, :, frame]


def difference_metrics(left, right):
    """Report global and per-component difference metrics."""
    difference = left - right
    reference_norm = np.linalg.norm(right)
    by_component = {}
    for index, name in enumerate(VARIABLES):
        component = difference[:, index]
        reference = right[:, index]
        max_index = int(np.argmax(np.abs(component)))
        by_component[name] = {
            'relative_l2_error': float(np.linalg.norm(component) / np.linalg.norm(reference)),
            'max_absolute_error': float(np.max(np.abs(component))),
            'max_error_node_index': max_index,
        }
    return {
        'relative_l2_error': float(np.linalg.norm(difference) / reference_norm),
        'max_absolute_error': float(np.max(np.abs(difference))),
        'by_component': by_component,
    }


def main():
    """Merge the existing states once and compare every pair."""

    required = {
        'pod_snapshot': HDM_RUN / 'snapshots/State.bin001',
        'hdm_solution': HDM_RUN / 'references/Solution.bin001',
        'prom_reconstruction': Path('{}001'.format(PROM_DIAGNOSTIC)),
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError('Missing distributed state files: {}.'.format(', '.join(missing)))
    if OUTPUT_DIRECTORY.exists():
        raise RuntimeError('{} exists; refusing to overwrite it.'.format(OUTPUT_DIRECTORY))

    OUTPUT_DIRECTORY.mkdir(parents=True)
    merged = {
        name: merge_state(path.with_name(path.name[:-3]), OUTPUT_DIRECTORY / name)
        for name, path in required.items()
    }
    states = {
        'pod_snapshot': read_frame(merged['pod_snapshot'], -1),
        'hdm_solution': read_frame(merged['hdm_solution'], -1),
        'prom_reconstruction': read_frame(merged['prom_reconstruction'], 0),
    }
    report = {
        'inputs': {name: str(path) for name, path in required.items()},
        'comparisons': {
            'pod_snapshot_vs_hdm_solution': difference_metrics(
                states['pod_snapshot'], states['hdm_solution']
            ),
            'prom_reconstruction_vs_pod_snapshot': difference_metrics(
                states['prom_reconstruction'], states['pod_snapshot']
            ),
            'prom_reconstruction_vs_hdm_solution': difference_metrics(
                states['prom_reconstruction'], states['hdm_solution']
            ),
        },
    }
    (OUTPUT_DIRECTORY / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    for name, values in report['comparisons'].items():
        print('{}: relative L2 {:.6e}, max abs {:.6e}'.format(
            name, values['relative_l2_error'], values['max_absolute_error']
        ))


if __name__ == '__main__':
    main()
