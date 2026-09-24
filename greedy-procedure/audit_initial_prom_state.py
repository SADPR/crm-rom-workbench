#!/usr/bin/env python3
"""Compare the HDM POD snapshot, final HDM solution, and PROM reconstruction."""

import argparse
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



def read_state(path):
    """Read every frame of a merged six-component state field."""
    _, state = read_xpost(path)
    if state.ndim != 3 or state.shape[1] != len(VARIABLES):
        raise RuntimeError('Expected a six-component state in {}.'.format(path))
    return state


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


def frame_summary(state, reference):
    """Locate the state frame closest to the final HDM solution."""
    relative_errors = [
        difference_metrics(state[:, :, frame], reference)['relative_l2_error']
        for frame in range(state.shape[2])
    ]
    closest = int(np.argmin(relative_errors))
    return {
        'frame_count': int(state.shape[2]),
        'first_frame_relative_l2_error': relative_errors[0],
        'last_frame_relative_l2_error': relative_errors[-1],
        'closest_frame_index': closest,
        'closest_frame_relative_l2_error': relative_errors[closest],
    }

def main():
    """Merge the existing states once and compare every pair."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--analyze-existing', action='store_true',
        help='reuse the merged XPOST fields from a completed audit',
    )
    args = parser.parse_args()

    required = {
        'pod_snapshot': HDM_RUN / 'snapshots/State.bin001',
        'hdm_solution': HDM_RUN / 'references/Solution.bin001',
        'prom_reconstruction': Path('{}001'.format(PROM_DIAGNOSTIC)),
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError('Missing distributed state files: {}.'.format(', '.join(missing)))
    if OUTPUT_DIRECTORY.exists() and not args.analyze_existing:
        raise RuntimeError('{} exists; refusing to overwrite it.'.format(OUTPUT_DIRECTORY))

    if args.analyze_existing:
        merged = {
            name: OUTPUT_DIRECTORY / '{}.xpost'.format(name)
            for name in required
        }
        missing = [str(path) for path in merged.values() if not path.is_file()]
        if missing:
            raise RuntimeError('Missing prior merged fields: {}.'.format(', '.join(missing)))
    else:
        OUTPUT_DIRECTORY.mkdir(parents=True)
        merged = {
            name: merge_state(path.with_name(path.name[:-3]), OUTPUT_DIRECTORY / name)
            for name, path in required.items()
        }
    fields = {name: read_state(path) for name, path in merged.items()}
    states = {
        'pod_snapshot': fields['pod_snapshot'][:, :, -1],
        'hdm_solution': fields['hdm_solution'][:, :, -1],
        'prom_reconstruction': fields['prom_reconstruction'][:, :, 0],
    }
    report = {
        'inputs': {name: str(path) for name, path in required.items()},
        'frames_against_final_hdm_solution': {
            'pod_snapshot': frame_summary(fields['pod_snapshot'], states['hdm_solution']),
            'prom_reconstruction': frame_summary(
                fields['prom_reconstruction'], states['hdm_solution']
            ),
        },
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
    summary_name = 'frame-summary.json' if args.analyze_existing else 'summary.json'
    (OUTPUT_DIRECTORY / summary_name).write_text(json.dumps(report, indent=2) + '\n')
    for name, values in report['comparisons'].items():
        print('{}: relative L2 {:.6e}, max abs {:.6e}'.format(
            name, values['relative_l2_error'], values['max_absolute_error']
        ))
    for name, values in report['frames_against_final_hdm_solution'].items():
        print('{}: {} frames; closest frame {} has relative L2 {:.6e}'.format(
            name,
            values['frame_count'],
            values['closest_frame_index'],
            values['closest_frame_relative_l2_error'],
        ))


if __name__ == '__main__':
    main()
