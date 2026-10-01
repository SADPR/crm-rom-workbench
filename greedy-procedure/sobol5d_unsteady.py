#!/usr/bin/env python3
"""Run an unsteady (URANS) HDM from the converged steady state of a 5D training HDM.

The steady input of the HDM's last stage (input2, or input3 after a restart) is the template,
so the physics and discretization stay those of the steady HDM. Only these change:
- the problem type, to Unsteady;
- the Time block, to implicit BDF2 with a global physical time step and Newton subiterations;
- the outputs, which go to HDMrunNNN/unsteady/;
- the start, which is the steady references/Solution.bin at time zero (no RestartData).
"""

import argparse
import json
import os
from pathlib import Path
import re

import numpy as np

import sobol5d_campaign as campaign
import sobol5d_projection as projection


DEFAULT_DT = 1.0e-4
DEFAULT_MAX_TIME = 0.3
DEFAULT_NEWTON = 5
SNAPSHOT_EVERY = 50
FIELD_EVERY = 100
SPECTRUM_PADDING = 16
# Output fields the unsteady test does not need; blanking them does not change the solve.
UNUSED_OUTPUTS = ('FluxResidual = "FluxRes.bin";', 'Displacement = "Displacement.bin";')
SPEED_OF_SOUND = (1.4 * 22632.0 / 0.3639) ** 0.5


def hdm_directory(settings, index):
    """Return the existing directory of one training HDM."""
    for root in (settings.MasterDir, settings.InitHDMPreCompDir):
        directory = campaign.run_dir(index, root)
        if directory.is_dir():
            return directory
    raise RuntimeError('HDM {:03d} does not exist.'.format(index))


def last_stage(directory):
    """Return the number of the last steady stage whose input exists (2, 3, or 4)."""
    stages = [stage for stage in (2, 3, 4) if (directory / 'input{}'.format(stage)).is_file()]
    if not stages:
        raise RuntimeError('{} has no steady stage-2 input.'.format(directory))
    return stages[-1]


def unsteady_input(directory, stage, dt, max_time, newton):
    """Return the unsteady input built from the steady input of the given stage."""
    source = directory / 'input{}'.format(stage)
    text = source.read_text()
    ran_in = re.findall(r'Prefix = "([^"]*/HDMrun\d{3}/)references/";', text)
    if len(ran_in) != 1:
        raise RuntimeError('Cannot find the restart prefix in {}.'.format(source))
    old, here = ran_in[0], '{}/'.format(directory.as_posix())
    out = '{}unsteady/'.format(here)
    # The Newton block's linear solver, up to its own closing brace (nine spaces in).
    linear = re.search(r'\n(         under LinearSolver \{.*?\n         \}\n)', text, re.S)
    time_block = re.search(r'\nunder Time \{.*?\n\}\n', text, re.S)
    snapshots = re.findall(r'Prefix = "{}(snapshots\d?)/";'.format(re.escape(old)), text)
    if not linear or not time_block or len(snapshots) != 1:
        raise RuntimeError('Unexpected Time or NonlinearROM block in {}.'.format(source))
    new_time = ('\nunder Time {{\n   Form = NonDescriptor;\n   Type = Implicit;\n   TypeTimeStep = Global;\n'
                '   TimeStep = {:g};\n   MaxTime = {:g};\n   MaxIts = {};\n   Eps = 1e-14;\n'
                '   under Implicit {{\n      Type = ThreePointBackwardDifference;\n'
                '      MatrixVectorProduct = FiniteDifference;\n      under Newton {{\n'
                '         MaxIts = {};\n         FailSafe = AlwaysOn;\n         Eps = 0.001;\n'
                .format(dt, max_time, int(round(max_time / dt)) + 10, newton)
                + linear.group(1) + '      }\n   }\n}\n')
    replacements = [
        ('Type = Steady;', 'Type = Unsteady;', 1),
        ('RestartData = "{}references/Restart.data";'.format(old), 'RestartData = "";', 1),
        ('Solution = "{}references/Solution.bin";'.format(old),
         'Solution = "{}references/Solution.bin";'.format(here), 1),
        ('Prefix = "{}results/";'.format(old), 'Prefix = "{}results/";'.format(out), 1),
        ('Prefix = "{}references/";'.format(old), 'Prefix = "{}references/";'.format(out), 1),
        ('Prefix = "{}{}/";'.format(old, snapshots[0]), 'Prefix = "{}snapshots/";'.format(out), 1),
        # The remaining references to the run directory are inputs that moved with it.
        (old, here, None),
        (time_block.group(0), new_time, 1),
        ('   under Postpro {\n      Frequency = 0;', '   under Postpro {{\n      Frequency = {};'
         .format(FIELD_EVERY), 1),
        ('      StateVector = "State.bin";\n      Frequency = 0;',
         '      StateVector = "State.bin";\n      Frequency = {};'.format(SNAPSHOT_EVERY), 1),
        ('OutputResidualSnapshotData = True;', 'OutputResidualSnapshotData = False;', 1),
    ]
    replacements += [(line, '{} = "";'.format(line.split(' = ')[0]), 1) for line in UNUSED_OUTPUTS]
    text = projection.patch(text, replacements, source)
    if 'Type = Steady;' in text or 'references/Restart.data' in text or 'CflLaw' in text:
        raise RuntimeError('The unsteady input from {} still holds steady settings.'.format(source))
    return text


def prepare(index, dt=DEFAULT_DT, max_time=DEFAULT_MAX_TIME, newton=DEFAULT_NEWTON):
    """Write the unsteady input of one converged or accepted training HDM."""
    settings = campaign.configure_settings()
    point = campaign.read_manifest()['points'][index - 1]
    state, residual, directory = campaign.classify(settings, index, point)
    if state not in ('converged', 'accepted'):
        raise RuntimeError('HDM {:03d} is {}; it has no steady state to start from.'.format(index, state))
    stage = last_stage(directory)
    # The start must be the final state of that stage: AERO-F writes Solution.bin at its end.
    solution = directory / 'references/Solution.bin001'
    log = directory / 'log{}'.format(stage)
    if abs(solution.stat().st_mtime - log.stat().st_mtime) > 600:
        raise RuntimeError('{} was not written at the end of stage {}.'.format(solution, stage))
    target = directory / 'unsteady'
    if target.exists():
        raise RuntimeError('{} exists; refusing to overwrite an unsteady run.'.format(target))
    text = unsteady_input(directory, stage, dt, max_time, newton)
    for name in ('results', 'postpro', 'references', 'snapshots'):
        (target / name).mkdir(parents=True)
    (target / 'input').write_text(text)
    campaign.write_json(target / 'unsteady.json', {
        'index': index, 'point': point, 'steady_state': state, 'steady_residual': residual,
        'start': str(solution.parent / 'Solution.bin'), 'template': 'input{}'.format(stage),
        'time_step': dt, 'max_time': max_time, 'newton_iterations': newton,
        'convective_time': 1.0 / (point[0] * SPEED_OF_SOUND),
    })
    print('Prepared {} from input{} ({} steady state).'.format(target / 'input', stage, state))


def run(index):
    """Run one prepared unsteady HDM."""
    settings = campaign.configure_settings()
    target = hdm_directory(settings, index) / 'unsteady'
    if not (target / 'input').is_file():
        raise RuntimeError('Run the prepare action first.')
    if (target / 'log').exists():
        raise RuntimeError('{} exists; refusing to rerun.'.format(target / 'log'))
    projection.run_aerof(settings, target / 'input', target / 'log')
    print('Completed the unsteady run of HDM {:03d}.'.format(index))


def summary(index):
    """Report how lift and drag evolve from the steady state, and any oscillation frequency."""
    settings = campaign.configure_settings()
    directory = hdm_directory(settings, index)
    meta = json.loads((directory / 'unsteady/unsteady.json').read_text())
    rows = projection.table(directory / 'unsteady/postpro/liftdrag.out')
    steady = projection.table(directory / 'postpro/liftdrag.out')[-1]
    time, drag, lift = rows[:, 1], rows[:, 4], rows[:, 5]
    late = time >= 0.5 * time[-1]
    values = {}
    for name, series, reference in (('lift', lift, steady[5]), ('drag', drag, steady[4])):
        tail = series[late]
        # The late window holds only a few buffet periods: a Hann window and zero padding
        # interpolate the spectral peak between the coarse FFT bins.
        detrended = tail - np.polyval(np.polyfit(time[late], tail, 1), time[late])
        padded = SPECTRUM_PADDING * len(tail)
        spectrum = np.abs(np.fft.rfft(detrended * np.hanning(len(tail)), n=padded))
        frequencies = np.fft.rfftfreq(padded, d=float(np.median(np.diff(time[late]))))
        # A period longer than the window is a drift, not an oscillation.
        lowest = SPECTRUM_PADDING
        peak = int(np.argmax(spectrum[lowest:]) + lowest) if len(spectrum) > lowest else 0
        values[name] = {
            'steady': float(reference),
            'final': float(series[-1]),
            'change_from_steady': float((series[-1] - reference) / abs(reference)),
            'late_amplitude': float(np.ptp(tail) / abs(np.mean(tail))),
            'late_slope_per_convective_time': float(np.polyfit(time[late], tail, 1)[0]
                                                    * meta['convective_time'] / abs(np.mean(tail))),
            'peak_frequency_hz': float(frequencies[peak]) if peak else None,
            'peak_strouhal': float(frequencies[peak] * meta['convective_time']) if peak else None,
        }
    result = {'index': index, 'time': float(time[-1]),
              'convective_times': float(time[-1] / meta['convective_time']), **values}
    campaign.write_json(directory / 'unsteady/summary.json', result)
    print(json.dumps(result, indent=1))


def main():
    """Run one explicit stage of an unsteady HDM."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('prepare', 'run', 'summary'))
    parser.add_argument('--run-index', type=int, required=True)
    parser.add_argument('--dt', type=float, default=DEFAULT_DT, help='physical time step (s)')
    parser.add_argument('--max-time', type=float, default=DEFAULT_MAX_TIME, help='physical time (s)')
    parser.add_argument('--newton', type=int, default=DEFAULT_NEWTON, help='Newton iterations per step')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.run_index, args.dt, args.max_time, args.newton)
    elif args.mode == 'run':
        run(args.run_index)
    else:
        summary(args.run_index)


if __name__ == '__main__':
    main()
