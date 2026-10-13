#!/usr/bin/env python3
"""Run an unsteady (URANS) HDM from the converged steady state of a 5D training HDM.

The steady input of the HDM's last stage (input2, or input3 after a restart) is the template,
so the physics and discretization stay those of the steady HDM. Only these change:
- the problem type, to Unsteady;
- the Time block, to implicit BDF2 with one fixed, global physical time step. Each step is
  solved with Newton iterations under dual time stepping: a local pseudo-time term added to the
  Newton matrix only, so GMRES converges while the converged step is unchanged (TimeState.C,
  MatVecProd.C). The term is for the HDM only; a PROM or HPROM must run with it off, since there
  it would change the LSPG test basis J V;
- the outputs, which go to HDMrunNNN/unsteady/ (or another unsteady* folder, e.g. for a dt study);
- the start: the steady references/Solution.bin at time zero (no RestartData), or, with
  --start freestream, the uniform inlet state, as stage 1 of the HDM starts (Solution = ""):
  an impulsive start, to follow the shock from its formation to its steady position.

The impulsive start needs a first step of about 4e-8 s. AERO-F halves a step that leaves a
non-physical state, at most 9 times (DistTimeState::updateDtCoeff), and doubles it back every
3 steps, so only dt <= 2e-5 s starts on its own. The continue mode then carries a run on from its
restart files, in the same folder, with another dt: BDF2 takes its coefficients from the ratio
of the new step to the previous one (TimeData::computeCoefficients), and the restart holds both
states and that step. Field frames are written every --field-interval of physical time
(TimeInterval), so their numbering does not depend on dt or on the shortened steps.

The transient mode follows the upper-surface shock frame by frame and compares the runs with
the steady HDM.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np
import pyaeroopt

import sobol5d_campaign as campaign
import sobol5d_projection as projection
import sobol5d_test as test


DEFAULT_DT = 1.0e-4
DEFAULT_MAX_TIME = 0.3
# The first runs (jobs 46203007) used 5 plain Newton iterations: GMRES stalled, and the step
# residual fell by only 2-10 of the required 1000.
DEFAULT_NEWTON = 30
DEFAULT_DUAL_CFL = 100.0
# Output intervals in physical time, so runs with different dt write at the same instants
# (every 50 and 100 steps at the default dt).
SNAPSHOT_INTERVAL = 0.005
FIELD_INTERVAL = 0.01
DEFAULT_NAME = 'unsteady'
# Restart files every this much physical time (s), on top of the final ones, so a run that fails
# can be continued from close to where it stopped.
RESTART_INTERVAL = 0.01
# Iterations allowed beyond the nominal count, for the steps AERO-F shortens; MaxTime ends a run.
EXTRA_STEPS = 100
SPECTRUM_PADDING = 16
# Output fields the unsteady test does not need; blanking them does not change the solve.
UNUSED_OUTPUTS = ('FluxResidual = "FluxRes.bin";', 'Displacement = "Displacement.bin";')
SPEED_OF_SOUND = (1.4 * 22632.0 / 0.3639) ** 0.5
EXO_FIELDS = ('Mach', 'PressureCoefficient', 'SkinFriction', 'Velocity')
# AERO-F appends a field frame every FIELD_INTERVAL (about 100 min) in a few seconds; merge
# only when the last write is older than this, so no frame is read half written.
WRITE_QUIET = 120.0
# Steady restart from the final unsteady state: enough iterations to see whether the steady
# solver keeps that state or drifts away; the tolerance is never met on purpose.
RESTEADY_ITS = 5000
GAMMA = 1.4
STARTS = ('steady', 'freestream')
# A transient has settled once the shock stays within this of the steady one, and the lift within
# this fraction of the steady lift, for the rest of the run.
SETTLED_SHOCK = 0.005
SETTLED_LIFT = 0.005


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


def steps(interval, dt):
    """Return the number of time steps in one output interval."""
    return max(1, int(round(interval / dt)))


def run_folder(name):
    """Check an unsteady run's folder name: one level, starting with 'unsteady'."""
    if not re.fullmatch(r'unsteady[\w.+-]*', name):
        raise ValueError('The run name must be one folder starting with "unsteady", not {!r}.'.format(name))
    return name


def unsteady_input(directory, stage, dt, max_time, newton, dual_cfl, name=DEFAULT_NAME, start='steady',
                   snapshot_interval=SNAPSHOT_INTERVAL, field_interval=FIELD_INTERVAL):
    """Return the unsteady input built from the steady input of the given stage."""
    if start not in STARTS:
        raise ValueError('The start is one of {}.'.format(STARTS))
    for interval in (snapshot_interval, field_interval):
        # Runs with different time steps must write at the same physical instants.
        if abs(interval / dt - round(interval / dt)) > 1e-9:
            raise ValueError('The output interval {:g} s is not a multiple of dt = {:g} s.'.format(interval, dt))
    source = directory / 'input{}'.format(stage)
    text = source.read_text()
    ran_in = re.findall(r'Prefix = "([^"]*/HDMrun\d{3}/)references/";', text)
    if len(ran_in) != 1:
        raise RuntimeError('Cannot find the restart prefix in {}.'.format(source))
    old, here = ran_in[0], '{}/'.format(directory.as_posix())
    out = '{}{}/'.format(here, run_folder(name))
    # The Newton block's linear solver, up to its own closing brace (nine spaces in).
    linear = re.search(r'\n(         under LinearSolver \{.*?\n         \}\n)', text, re.S)
    time_block = re.search(r'\nunder Time \{.*?\n\}\n', text, re.S)
    snapshots = re.findall(r'Prefix = "{}(snapshots\d?)/";'.format(re.escape(old)), text)
    if not linear or not time_block or len(snapshots) != 1:
        raise RuntimeError('Unexpected Time or NonlinearROM block in {}.'.format(source))
    dual = ('   DualTimeStepping = On;\n   DualTimeCfl = {:g};\n'.format(dual_cfl)) if dual_cfl > 0 else ''
    new_time = ('\nunder Time {{\n   Form = NonDescriptor;\n   Type = Implicit;\n   TypeTimeStep = Global;\n'
                '   TimeStep = {:g};\n   MaxTime = {:g};\n   MaxIts = {};\n   Eps = 1e-14;\n'
                .format(dt, max_time, int(round(max_time / dt)) + EXTRA_STEPS) + dual
                + '   under Implicit {{\n      Type = ThreePointBackwardDifference;\n'
                '      MatrixVectorProduct = FiniteDifference;\n      under Newton {{\n'
                '         MaxIts = {};\n         FailSafe = AlwaysOn;\n         Eps = 0.001;\n'
                .format(newton)
                + linear.group(1) + '      }\n   }\n}\n')
    replacements = [
        ('Type = Steady;', 'Type = Unsteady;', 1),
        ('RestartData = "{}references/Restart.data";'.format(old), 'RestartData = "";', 1),
        ('Solution = "{}references/Solution.bin";'.format(old),
         'Solution = "{}references/Solution.bin";'.format(here) if start == 'steady' else 'Solution = "";', 1),
        ('Prefix = "{}results/";'.format(old), 'Prefix = "{}results/";'.format(out), 1),
        ('Prefix = "{}references/";'.format(old),
         'Prefix = "{}references/";\n      TimeInterval = {:g};'.format(out, RESTART_INTERVAL), 1),
        ('Prefix = "{}{}/";'.format(old, snapshots[0]), 'Prefix = "{}snapshots/";'.format(out), 1),
        # The remaining references to the run directory are inputs that moved with it.
        (old, here, None),
        (time_block.group(0), new_time, 1),
        ('   under Postpro {\n      Frequency = 0;',
         '   under Postpro {{\n      Frequency = 0;\n      TimeInterval = {:g};'.format(field_interval), 1),
        ('      StateVector = "State.bin";\n      Frequency = 0;',
         '      StateVector = "State.bin";\n      Frequency = {};'.format(steps(snapshot_interval, dt)), 1),
        ('OutputResidualSnapshotData = True;', 'OutputResidualSnapshotData = False;', 1),
    ]
    replacements += [(line, '{} = "";'.format(line.split(' = ')[0]), 1) for line in UNUSED_OUTPUTS]
    text = projection.patch(text, replacements, source)
    if 'Type = Steady;' in text or 'references/Restart.data' in text or 'CflLaw' in text:
        raise RuntimeError('The unsteady input from {} still holds steady settings.'.format(source))
    return text


def prepare(index, dt=DEFAULT_DT, max_time=DEFAULT_MAX_TIME, newton=DEFAULT_NEWTON,
            dual_cfl=DEFAULT_DUAL_CFL, name=DEFAULT_NAME, start='steady',
            snapshot_interval=SNAPSHOT_INTERVAL, field_interval=FIELD_INTERVAL):
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
    target = directory / run_folder(name)
    if target.exists():
        raise RuntimeError('{} exists; refusing to overwrite an unsteady run.'.format(target))
    text = unsteady_input(directory, stage, dt, max_time, newton, dual_cfl, name, start,
                          snapshot_interval, field_interval)
    for folder in ('results', 'postpro', 'references', 'snapshots'):
        (target / folder).mkdir(parents=True)
    (target / 'input').write_text(text)
    campaign.write_json(target / 'unsteady.json', {
        'index': index, 'point': point, 'steady_state': state, 'steady_residual': residual,
        'start': str(solution.parent / 'Solution.bin') if start == 'steady' else 'freestream (uniform inlet state)',
        'template': 'input{}'.format(stage),
        'snapshot_interval': snapshot_interval, 'field_interval': field_interval,
        'name': name, 'time_step': dt, 'max_time': max_time, 'newton_iterations': newton,
        'dual_time_cfl': dual_cfl if dual_cfl > 0 else None,
        'convective_time': 1.0 / (point[0] * SPEED_OF_SOUND),
        'phases': [{'input': 'input', 'log': 'log', 'start_iteration': 0, 'start_time': 0.0,
                    'time_step': dt, 'max_time': max_time, 'newton_iterations': newton}],
    })
    print('Prepared {} from input{} ({} steady state), starting from {}.'.format(
        target / 'input', stage, state, 'that state' if start == 'steady' else 'the uniform inlet state'))


def run_phases(meta):
    """Return the phases of a run; runs prepared before the continue mode have one."""
    return meta.get('phases') or [{'input': 'input', 'log': 'log', 'start_iteration': 0, 'start_time': 0.0,
                                   'time_step': meta['time_step'], 'max_time': meta['max_time']}]


def restart_parameters(path):
    """Return the iteration, time and last time step of an AERO-F Restart.data file."""
    text = path.read_text()
    values = [re.findall(r'\n\s*{} = (\S+);'.format(key), text) for key in ('Iteration', 'Time', 'TimeStep1')]
    if any(len(value) != 1 for value in values):
        raise RuntimeError('Unexpected restart parameters in {}.'.format(path))
    return int(values[0][0]), float(values[1][0]), float(values[2][0])


def continue_input(text, out, iteration, start_time, dt, max_time, newton, snapshot_interval, source):
    """Return the input of a phase that carries a run on from its restart files with another dt."""
    inputs = re.search(r'\nunder Input \{\n.*?\n\}\n', text, re.S)
    if not inputs or 'TimeInterval' not in re.search(r'under Postpro \{.*?\}', text, re.S).group(0):
        # Step-numbered field frames would not line up once dt changes.
        raise RuntimeError('{} writes fields by step count; it cannot be continued.'.format(source))
    block, restarts = re.subn(r'\n   RestartData = "[^"]*";',
                              '\n   RestartData = "{}references/Restart.data";'.format(out), inputs.group(0))
    block, solutions = re.subn(r'\n   Solution = "[^"]*";', '\n   Solution = "{}references/Solution.bin";'.format(out),
                               block)
    if (restarts, solutions) != (1, 1):
        raise RuntimeError('Unexpected Input block in {}.'.format(source))
    max_its = iteration + int(round((max_time - start_time) / dt)) + EXTRA_STEPS
    return projection.patch(text, [
        (inputs.group(0), block, 1),
        (re.search(r'\n   TimeStep = [^;]+;\n   MaxTime = [^;]+;\n   MaxIts = \d+;\n', text).group(0),
         '\n   TimeStep = {:g};\n   MaxTime = {:g};\n   MaxIts = {};\n'.format(dt, max_time, max_its), 1),
        (re.search(r'under Newton \{\n         MaxIts = \d+;', text).group(0),
         'under Newton {{\n         MaxIts = {};'.format(newton), 1),
        (re.search(r'StateVector = "State.bin";\n      Frequency = \d+;', text).group(0),
         'StateVector = "State.bin";\n      Frequency = {};'.format(steps(snapshot_interval, dt)), 1),
    ], source)


def continue_run(index, name, dt, max_time, newton=None, snapshot_interval=None):
    """Write the input of a new phase of an unsteady run, from where its last phase stopped.

    The Newton iterations and the snapshot interval stay those of the last phase unless given."""
    settings = campaign.configure_settings()
    target = hdm_directory(settings, index) / run_folder(name)
    meta = json.loads((target / 'unsteady.json').read_text())
    phases = run_phases(meta)
    newton = newton or phases[-1].get('newton_iterations', meta['newton_iterations'])
    snapshot_interval = snapshot_interval or phases[-1].get('snapshot_interval', meta['snapshot_interval'])
    log, restart = target / phases[-1]['log'], target / 'references/Restart.data'
    if not log.is_file():
        raise RuntimeError('Phase {} of {} has not run yet.'.format(len(phases) - 1, target))
    # AERO-F writes the restart files when a run ends, at MaxTime or on SIGUSR1; a run that
    # failed leaves the last periodic ones, which may be far behind.
    if not restart.is_file() or restart.stat().st_mtime < log.stat().st_mtime - 600:
        raise RuntimeError('{} was not written at the end of {}; inspect that log.'.format(restart, log))
    iteration, start_time, last_dt = restart_parameters(restart)
    if start_time >= max_time:
        raise RuntimeError('{} already reached t = {:g} s.'.format(target, start_time))
    for interval in (snapshot_interval, meta['field_interval']):
        if abs(interval / dt - round(interval / dt)) > 1e-9:
            raise ValueError('The output interval {:g} s is not a multiple of dt = {:g} s.'.format(interval, dt))
    phase = len(phases)
    # The restart files are overwritten by the new phase; keep the ones it starts from.
    backup = target / 'references/phase{}'.format(phase)
    backup.mkdir()
    for path in [restart] + sorted((target / 'references').glob('Solution.bin[0-9]*')):
        shutil.copy2(path, backup / path.name)
    source = target / phases[-1]['input']
    text = continue_input(source.read_text(), '{}/'.format(target.as_posix()), iteration, start_time, dt,
                          max_time, newton, snapshot_interval, source)
    phases.append({'input': 'input.{}'.format(phase), 'log': 'log.{}'.format(phase),
                   'start_iteration': iteration, 'start_time': start_time, 'time_step': dt,
                   'max_time': max_time, 'newton_iterations': newton, 'snapshot_interval': snapshot_interval})
    (target / phases[-1]['input']).write_text(text)
    meta['phases'] = phases
    campaign.write_json(target / 'unsteady.json', meta)
    print('Prepared {}: from iteration {} at t = {:g} s (last step {:g} s) with dt = {:g} s to t = {:g} s; '
          'the first step is {:.2f} times the last one.'.format(
              target / phases[-1]['input'], iteration, start_time, last_dt, dt, max_time, dt / last_dt))


def run(index, name=DEFAULT_NAME):
    """Run the last prepared phase of one unsteady HDM."""
    settings = campaign.configure_settings()
    target = hdm_directory(settings, index) / run_folder(name)
    if not (target / 'unsteady.json').is_file():
        raise RuntimeError('Run the prepare action first.')
    phase = run_phases(json.loads((target / 'unsteady.json').read_text()))[-1]
    if (target / phase['log']).exists():
        raise RuntimeError('{} exists; refusing to rerun.'.format(target / phase['log']))
    projection.run_aerof(settings, target / phase['input'], target / phase['log'])
    print('Completed {} of the unsteady run of HDM {:03d}.'.format(phase['input'], index))


def forces_history(path):
    """Return the lift-and-drag rows of a run, one per time: a restart rewrites its first row."""
    rows = projection.table(path)
    keep = np.append(rows[1:, 1] > rows[:-1, 1], True)
    return rows[keep]


def step_convergence(logs, tolerance=1.0e-3):
    """Return the Newton iterations per time step and how far the steps that missed the tolerance fell."""
    texts = [log.read_text() for log in logs if log.is_file()]
    # Each log opens with the line of its starting state, which takes no Newton iteration.
    newton = [int(value) for text in texts for value in re.findall(r'^It \d+ \(\d+,(\d+)\):', text, re.M)[1:]]
    text = ''.join(texts)
    missed = [float(reached) / float(initial) for initial, reached in re.findall(
        r'Newton solver reached \d+ its \(Residual: initial=(\S+), reached=(\S+),', text)]
    return {
        'steps': len(newton),
        'mean_newton_iterations': float(np.mean(newton)) if newton else None,
        'tolerance': tolerance,
        'steps_missing_tolerance': len(missed),
        'worst_reduction': float(max(missed)) if missed else None,
        'median_missed_reduction': float(np.median(missed)) if missed else None,
    }


def summary(index, name=DEFAULT_NAME):
    """Report how lift and drag evolve from the steady state, and any oscillation frequency."""
    settings = campaign.configure_settings()
    directory = hdm_directory(settings, index)
    target = directory / run_folder(name)
    meta = json.loads((target / 'unsteady.json').read_text())
    rows = forces_history(target / 'postpro/liftdrag.out')
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
    # Steps AERO-F shortened after a non-physical state, per phase; the history prints seven
    # significant digits, so a step within 1% of the imposed one is that step, and a phase owns
    # the steps that end 1e-3 of its dt after its start (its shortest step is dt / 512).
    phases, steps_taken = run_phases(meta), np.diff(time)
    starts = [phase['start_time'] + 1e-3 * phase['time_step'] for phase in phases]
    owner = np.searchsorted(starts, time[1:], side='left') - 1
    time_steps = []
    for number, phase in enumerate(phases):
        mine = steps_taken[owner == number]
        short = mine < 0.99 * phase['time_step']
        time_steps.append({'time_step': phase['time_step'], 'steps': int(len(mine)),
                           'shortened_steps': int(short.sum()), 'time_in_shortened_steps': float(mine[short].sum())})
    result = {'index': index, 'time': float(time[-1]),
              'convective_times': float(time[-1] / meta['convective_time']),
              'time_steps': time_steps,
              'convergence': step_convergence([target / phase['log'] for phase in phases]), **values}
    campaign.write_json(target / 'summary.json', result)
    print(json.dumps(result, indent=1))


def exo(index, name=DEFAULT_NAME):
    """Merge the field frames written so far into one Exodus file; the run keeps going."""
    settings = campaign.configure_settings()
    directory = hdm_directory(settings, index)
    results, out = directory / run_folder(name) / 'results', directory / run_folder(name) / 'exo'
    written = [path for field in EXO_FIELDS for path in results.glob('{}.bin[0-9]*'.format(field))]
    if not written:
        raise RuntimeError('{} holds no field frames yet.'.format(results))
    age = time.time() - max(path.stat().st_mtime for path in written)
    if age < WRITE_QUIET:
        raise RuntimeError('AERO-F wrote fields {:.0f} s ago; retry in a few minutes.'.format(age))
    # The deformed mesh of this HDM, so the airfoil is the real one and not the NACA 0012.
    top = directory / '{}_deformed.top'.format(Path(settings.TopFilePath).name)
    frg = pyaeroopt.interface.Frg(top=str(top),
                                  geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix))
    out.mkdir(exist_ok=True)
    xposts = []
    for field in EXO_FIELDS:
        frg.sower_fluid_merge(str(results / '{}.bin'.format(field)), str(out / field), field,
                              log=str(out / 'sower.log'))
        xposts.append(out / '{}.xpost'.format(field))
        if not xposts[-1].is_file():
            raise RuntimeError('sower did not write {}; see {}.'.format(xposts[-1], out / 'sower.log'))
    exo_file = out / '{}_{:03d}.exo'.format(name, index)
    if exo_file.exists():
        exo_file.unlink()
    decomposition = '{}.top.dec.{}'.format(settings.TopFilePath, settings.HDMnproc)
    frg.run_xp2exo(str(exo_file), [decomposition] + [str(path) for path in xposts], log=str(out / 'xp2exo.log'))
    if not exo_file.is_file():
        raise RuntimeError('xp2exo did not write {}; see {}.'.format(exo_file, out / 'xp2exo.log'))
    print('Wrote {} with the frames written so far.'.format(exo_file))


def resteady_input(directory, stage, max_its):
    """Return the steady input of the given stage restarted from the final unsteady state."""
    source = directory / 'input{}'.format(stage)
    text = source.read_text()
    ran_in = re.findall(r'Prefix = "([^"]*/HDMrun\d{3}/)references/";', text)
    snapshots = re.findall(r'Prefix = "[^"]*/HDMrun\d{3}/(snapshots\d?)/";', text)
    criteria = re.findall(r'\nunder Time \{\n   Form = NonDescriptor;\n   MaxIts = \d+;\n   Eps = [^;]+;', text)
    if len(ran_in) != 1 or len(snapshots) != 1 or len(criteria) != 1:
        raise RuntimeError('Unexpected restart, snapshot or Time block in {}.'.format(source))
    old, here = ran_in[0], '{}/'.format(directory.as_posix())
    out = '{}resteady/'.format(here)
    replacements = [
        # A fresh run (iteration 0, CFL law from Cfl0) whose state is the final unsteady one.
        ('RestartData = "{}references/Restart.data";'.format(old), 'RestartData = "";', 1),
        ('Solution = "{}references/Solution.bin";'.format(old),
         'Solution = "{}unsteady/references/Solution.bin";'.format(here), 1),
        ('Prefix = "{}results/";'.format(old), 'Prefix = "{}results/";'.format(out), 1),
        ('Prefix = "{}references/";'.format(old), 'Prefix = "{}references/";'.format(out), 1),
        ('Prefix = "{}{}/";'.format(old, snapshots[0]), 'Prefix = "{}snapshots/";'.format(out), 1),
        (old, here, None),
        (criteria[0], '\nunder Time {{\n   Form = NonDescriptor;\n   MaxIts = {};\n   Eps = 1e-14;'.format(max_its), 1),
        ('OutputResidualSnapshotData = True;', 'OutputResidualSnapshotData = False;', 1),
    ]
    replacements += [(line, '{} = "";'.format(line.split(' = ')[0]), 1) for line in UNUSED_OUTPUTS]
    text = projection.patch(text, replacements, source)
    if 'references/Restart.data' in text or 'Type = Unsteady;' in text:
        raise RuntimeError('The steady restart input from {} still restarts the old run.'.format(source))
    return text


def cp_star(mach):
    """Critical pressure coefficient: Cp at which the local flow becomes sonic."""
    ratio = (2.0 + (GAMMA - 1.0) * mach ** 2) / (GAMMA + 1.0)
    return 2.0 / (GAMMA * mach ** 2) * (ratio ** (GAMMA / (GAMMA - 1.0)) - 1.0)


def upper_shock(settings, directory, cp_file, mach):
    """Return x/c of the steepest Cp rise behind a supersonic region on the upper wall, or None."""
    raw_nodes = np.loadtxt('{}_nodes'.format(settings.TopFilePath), dtype=np.float64)
    wall = test.wall_nodes('{}.top'.format(settings.TopFilePath), raw_nodes)
    with open(directory / 'deform/Position.xpost', 'rb') as handle:
        handle.readline()
        count = int(handle.readline())
        handle.readline()
        position = np.array(handle.read().split(), dtype=float).reshape(count, 3)
    cp = np.loadtxt(cp_file, skiprows=3)
    upper = wall[raw_nodes[wall, 2] >= 0]
    x = position[upper, 0]
    x = (x - position[wall, 0].min()) / (position[wall, 0].max() - position[wall, 0].min())
    order = np.argsort(x)
    x, cp = x[order], cp[upper][order]
    keep = (x > 0.1) & (x < 0.95)
    x, cp = x[keep], cp[keep]
    if cp.min() >= cp_star(mach):
        return None
    gradient = np.gradient(cp, x)
    gradient[~np.maximum.accumulate(cp < cp_star(mach))] = 0.0
    return float(x[int(np.argmax(gradient))])


def resteady(index, max_its=RESTEADY_ITS):
    """Restart the steady solver from the final unsteady state and report where it goes."""
    settings = campaign.configure_settings()
    directory = hdm_directory(settings, index)
    unsteady = directory / 'unsteady'
    solution, log = unsteady / 'references/Solution.bin001', unsteady / 'log'
    # AERO-F writes the final state when the run ends, at MaxTime or on SIGUSR1.
    if not solution.is_file() or solution.stat().st_mtime < log.stat().st_mtime - 600:
        raise RuntimeError('{} holds no final unsteady state yet.'.format(solution))
    target = directory / 'resteady'
    if target.exists():
        raise RuntimeError('{} exists; refusing to overwrite a steady restart.'.format(target))
    stage = last_stage(directory)
    text = resteady_input(directory, stage, max_its)
    for name in ('results', 'postpro', 'references', 'snapshots'):
        (target / name).mkdir(parents=True)
    (target / 'input').write_text(text)
    projection.run_aerof(settings, target / 'input', target / 'log')
    frg = pyaeroopt.interface.Frg(top='{}.top'.format(settings.TopFilePath),
                                  geom_pre='{}data/{}'.format(settings.MasterDir, settings.GeometryPrefix))
    test.merge_surface_fields(frg, target, fields=('PressureCoefficient',))
    test.merge_surface_fields(frg, directory, fields=('PressureCoefficient',))
    initial = float(re.search(r'Spatial residual norm = (\S+)', (target / 'log').read_text()).group(1))
    original = float(re.search(r'Spatial residual norm = (\S+)', (directory / 'log1').read_text()).group(1))
    rows = projection.table(target / 'postpro/Residual.out')
    steady_forces = projection.table(directory / 'postpro/liftdrag.out')[-1]
    unsteady_forces = projection.table(unsteady / 'postpro/liftdrag.out')[-1]
    forces = projection.table(target / 'postpro/liftdrag.out')
    point = campaign.read_manifest()['points'][index - 1]
    change = lambda value, reference: float((value - reference) / abs(reference))
    result = {
        'index': index,
        'template': 'input{}'.format(stage),
        'iterations': int(rows[-1, 0]),
        'steady_final_absolute_residual': float(original * projection.table(directory / 'postpro/Residual.out')[-1, 2]),
        'start_absolute_residual': initial,
        'final_absolute_residual': float(initial * rows[-1, 2]),
        'lift_vs_steady': {'start': change(forces[0, 5], steady_forces[5]), 'final': change(forces[-1, 5], steady_forces[5]),
                           'unsteady_final': change(unsteady_forces[5], steady_forces[5])},
        'drag_vs_steady': {'start': change(forces[0, 4], steady_forces[4]), 'final': change(forces[-1, 4], steady_forces[4]),
                           'unsteady_final': change(unsteady_forces[4], steady_forces[4])},
        'upper_shock_xc': {'steady': upper_shock(settings, directory, directory / 'postpro/PressureCoefficient.xpost', point[0]),
                           'restart_final': upper_shock(settings, directory, target / 'postpro/PressureCoefficient.xpost', point[0])},
    }
    campaign.write_json(target / 'summary.json', result)
    print(json.dumps(result, indent=1))


def wall_geometry(settings, directory):
    """Return the upper-surface wall rows (into the wall-node list) sorted by x/c, and their x/c."""
    raw_nodes = np.loadtxt('{}_nodes'.format(settings.TopFilePath), dtype=np.float64)
    wall = test.wall_nodes('{}.top'.format(settings.TopFilePath), raw_nodes)
    with open(directory / 'deform/Position.xpost', 'rb') as handle:
        handle.readline()
        count = int(handle.readline())
        handle.readline()
        position = np.array(handle.read().split(), dtype=float).reshape(count, 3)
    x = (position[wall, 0] - position[wall, 0].min()) / (position[wall, 0].max() - position[wall, 0].min())
    upper = np.nonzero(raw_nodes[wall, 2] >= 0)[0]
    upper = upper[np.argsort(x[upper])]
    return wall, upper, x[upper]


def shock_position(cp, x, mach):
    """Return x/c of the steepest Cp rise behind a supersonic region on 0.1 < x/c < 0.95, or None."""
    keep = (x > 0.1) & (x < 0.95)
    x, cp = x[keep], cp[keep]
    if cp.min() >= cp_star(mach):
        return None
    gradient = np.gradient(cp, x)
    gradient[~np.maximum.accumulate(cp < cp_star(mach))] = 0.0
    return float(x[int(np.argmax(gradient))])


def xpost_history(path, rows):
    """Return the times and the given rows of every frame of a merged scalar XPOST file."""
    times, frames = [], []
    with open(path) as handle:
        handle.readline()
        count = int(handle.readline())
        while True:
            line = handle.readline()
            if not line.strip():
                break
            times.append(float(line))
            values = np.array([float(handle.readline()) for _ in range(count)])
            frames.append(values[rows])
    return np.array(times), np.array(frames)


def settling_time(times, values, reference, tolerance, relative):
    """Return the first time after which every value stays within the tolerance, or None."""
    error = np.abs(values - reference) / (abs(reference) if relative else 1.0)
    outside = np.nonzero(~(error <= tolerance))[0]
    if len(outside) == 0:
        return float(times[0])
    return float(times[outside[-1] + 1]) if outside[-1] + 1 < len(times) else None


def transient(index, names):
    """Follow the upper shock and the forces of one or more unsteady runs against the steady HDM."""
    settings = campaign.configure_settings()
    directory = hdm_directory(settings, index)
    point = campaign.read_manifest()['points'][index - 1]
    mach = point[0]
    wall, upper, x = wall_geometry(settings, directory)
    steady_cp = test.wall_pressure(directory, wall)
    steady_forces = projection.table(directory / 'postpro/liftdrag.out')[-1]
    steady_shock = shock_position(steady_cp[upper], x, mach)
    runs, histories = {}, {}
    for name in names:
        target = directory / run_folder(name)
        meta = json.loads((target / 'unsteady.json').read_text())
        tc = meta['convective_time']
        times, frames = xpost_history(target / 'exo/PressureCoefficient.xpost', wall)
        shocks = np.array([shock_position(frame[upper], x, mach) or np.nan for frame in frames])
        forces = forces_history(target / 'postpro/liftdrag.out')
        lift_time, lift, drag = forces[:, 1], forces[:, 5], forces[:, 4]
        histories[name] = (times, shocks, lift_time, lift)
        runs[name] = {
            'start': meta['start'], 'time_steps': [phase['time_step'] for phase in run_phases(meta)],
            'convective_time': tc, 'frames': len(times), 'final_time': float(lift_time[-1]),
            'shock_history': [[float(t), float(t / tc), None if np.isnan(s) else float(s)] for t, s in zip(times, shocks)],
            'settled_shock_time': settling_time(times, shocks, steady_shock, SETTLED_SHOCK, False),
            'settled_lift_time': settling_time(lift_time, lift, steady_forces[5], SETTLED_LIFT, True),
            'final': {
                'wall_cp_vs_steady': test.relative_error(frames[-1], steady_cp, 2),
                'shock': None if np.isnan(shocks[-1]) else float(shocks[-1]),
                'lift_vs_steady': float((lift[-1] - steady_forces[5]) / abs(steady_forces[5])),
                'drag_vs_steady': float((drag[-1] - steady_forces[4]) / abs(steady_forces[4])),
            },
        }
        for key in ('settled_shock_time', 'settled_lift_time'):
            value = runs[name][key]
            runs[name][key.replace('_time', '_convective_times')] = None if value is None else value / tc
    comparison = None
    if len(names) == 2:
        (t0, s0, lt0, l0), (t1, s1, lt1, l1) = histories[names[0]], histories[names[1]]
        common = np.intersect1d(np.round(t0, 9), np.round(t1, 9))
        pick = lambda times, values: values[np.isin(np.round(times, 9), common)]
        lift1 = np.interp(lt0, lt1, l1)
        overlap = lt0 <= lt1[-1]
        comparison = {
            'common_frames': len(common),
            'max_shock_difference': float(np.nanmax(np.abs(pick(t0, s0) - pick(t1, s1)))) if len(common) else None,
            'max_lift_difference': float(np.max(np.abs(l0[overlap] - lift1[overlap])) / abs(steady_forces[5])),
        }
    result = {'index': index, 'point': point, 'steady': {'shock': steady_shock, 'lift': float(steady_forces[5]),
                                                          'drag': float(steady_forces[4])},
              'settling': {'shock_within': SETTLED_SHOCK, 'lift_within': SETTLED_LIFT},
              'runs': runs, 'comparison': comparison}
    campaign.write_json(directory / run_folder(names[0]) / 'transient.json', result)
    print('Steady HDM {:03d}: upper shock x/c = {}, lift {:.4f}.'.format(index, steady_shock, steady_forces[5]))
    for name, run_ in runs.items():
        print('{}: {} frames to t = {:.3f} s ({:.1f} c/U); shock settled at {}, lift at {} c/U; final shock {}, '
              'wall Cp {:.3f}% from the steady, lift {:+.3%}, drag {:+.3%}.'.format(
                  name, run_['frames'], run_['final_time'], run_['final_time'] / run_['convective_time'],
                  run_['settled_shock_convective_times'], run_['settled_lift_convective_times'],
                  run_['final']['shock'], run_['final']['wall_cp_vs_steady'], run_['final']['lift_vs_steady'],
                  run_['final']['drag_vs_steady']))
    if comparison:
        print('Between the two runs: max shock difference {}, max lift difference {:.3%} of the steady lift '
              '({} common frames).'.format(comparison['max_shock_difference'], comparison['max_lift_difference'],
                                           comparison['common_frames']))


def main():
    """Run one explicit stage of an unsteady HDM."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('prepare', 'continue', 'run', 'summary', 'exo', 'resteady', 'transient'))
    parser.add_argument('--run-index', type=int, required=True)
    parser.add_argument('--dt', type=float, default=DEFAULT_DT, help='physical time step (s)')
    parser.add_argument('--max-time', type=float, default=DEFAULT_MAX_TIME, help='physical time (s)')
    parser.add_argument('--newton', type=int,
                        help='Newton iterations per step (default {}; continue keeps the run\'s)'.format(DEFAULT_NEWTON))
    parser.add_argument('--name', default=DEFAULT_NAME,
                        help='run folder below HDMrunNNN/, starting with "unsteady" (e.g. unsteady-dt5e-5)')
    parser.add_argument('--dual-cfl', type=float, default=DEFAULT_DUAL_CFL,
                        help='pseudo-time CFL of dual time stepping; 0 turns it off')
    parser.add_argument('--start', choices=STARTS, default='steady',
                        help='steady state at t = 0, or the uniform inlet state (an impulsive start)')
    parser.add_argument('--snapshot-interval', type=float,
                        help='physical time between state snapshots (s), a multiple of dt '
                             '(default {:g}; continue keeps the run\'s)'.format(SNAPSHOT_INTERVAL))
    parser.add_argument('--field-interval', type=float, default=FIELD_INTERVAL,
                        help='physical time between field outputs (s), a multiple of dt')
    parser.add_argument('--names', help='for transient, comma-separated run folders (default: --name)')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.run_index, args.dt, args.max_time, args.newton or DEFAULT_NEWTON, args.dual_cfl, args.name,
                args.start, args.snapshot_interval or SNAPSHOT_INTERVAL, args.field_interval)
    elif args.mode == 'continue':
        continue_run(args.run_index, args.name, args.dt, args.max_time, args.newton, args.snapshot_interval)
    elif args.mode == 'transient':
        transient(args.run_index, (args.names or args.name).split(','))
    elif args.mode == 'run':
        run(args.run_index, args.name)
    elif args.mode == 'exo':
        exo(args.run_index, args.name)
    elif args.mode == 'resteady':
        resteady(args.run_index)
    else:
        summary(args.run_index, args.name)


if __name__ == '__main__':
    main()
