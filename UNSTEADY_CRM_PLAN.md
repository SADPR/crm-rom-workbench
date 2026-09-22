# CRM Unsteady ROM Work Plan

## Objective

Convert the current steady CRM HDM/ROM/HROM workflow into an unsteady
workflow while preserving the existing structure and changing as little of
the supplied code as practical.

The conversion will be performed and validated in stages. The complete
greedy pipeline will not be modified until a single unsteady HDM case has
been demonstrated to run correctly.

## Working locations

- Edit and version the code in `/home/sares/crm-rom-workbench`.
- Use `/home/sares/Sherlock_CRM` only to inspect the read-only SSHFS view of
  the Sherlock execution clone and its results.
- Compile AERO-F and submit jobs from a regular SSH session on Sherlock, not
  through SSHFS.
- Keep generated meshes, binaries, snapshots, and simulation results out of
  Git.

## Guiding principles

- Use the current upstream AERO-F input style as the reference.
- Preserve the steady workflow until the unsteady path is independently
  validated.
- Prefer small, local changes over reorganizing or broadly refactoring the
  existing scripts.
- Validate every stage before modifying the next one.
- Keep HDM, ROM, and HROM physical-time settings consistent.
- Preserve backward compatibility for users of the existing steady workflow.

## Decision required before implementation

The intended source of unsteadiness must be defined. Possible interpretations
include:

1. A startup transient from freestream to the developed flow.
2. A perturbation of an existing steady solution.
3. A forced response such as a gust, prescribed pitch, or moving geometry.
4. An intrinsically unsteady flow at fixed geometry and boundary conditions.

Starting from a converged steady solution with unchanged boundary conditions
may produce an almost constant trajectory. The expected physical transient
and the quantities of interest should therefore be confirmed before selecting
the final time horizon and training data.

## Stage 0: Record the steady baseline

Before changing the workflow:

- Record the AERO-F revision and build configuration used on Sherlock.
- Identify one nominal CRM parameter point.
- Preserve its generated steady input file, convergence history, force
  history, and final solution.
- Record the current processor layout and wall-clock cost.
- Confirm that the current workflow can reproduce the baseline.

This baseline will distinguish pre-existing behavior from errors introduced
by the unsteady conversion.

## Stage 1: Run one short unsteady HDM pilot

Modify only the input generation needed for one nominal full-order case:

- Change the AERO-F problem type from `Steady` to `Unsteady`.
- Add explicit physical-time controls:
  - time step,
  - final time,
  - maximum number of time steps,
  - time integration scheme,
  - nonlinear tolerance and iterations per time step.
- Select and document the initial condition.
- Write force histories, restart files, and state snapshots at explicit
  frequencies.
- Begin with a short time horizon so input and runtime failures are cheap to
  diagnose.

The first pilot should use a single parameter point and the existing CRM
partitioning. It should not invoke POD, ROM, HROM, or the greedy algorithm.

### Stage 1 acceptance checks

- AERO-F accepts the generated input without warnings caused by unsupported
  or inconsistent options.
- The time integration advances for the requested physical time.
- Residual and force histories are finite and physically plausible.
- Restart output can be read successfully.
- `State.bin` contains a time trajectory rather than one final state.
- Snapshot indices and physical times can be mapped unambiguously.

## Stage 2: Generalize unsteady HDM generation

After the pilot is stable:

- Place the unsteady controls in `greedy-procedure/setup.py`.
- Make the smallest necessary changes to the HDM input generator in
  `greedy-procedure/runs.py`.
- Retain the steady path for regression testing.
- Define snapshot start, end, frequency, and any discarded startup interval.
- Replace the current single-state snapshot entry with a temporal range.
- Ensure repeated or restarted runs cannot silently duplicate snapshot
  metadata.

HDM output frequency and snapshot frequency should be chosen independently:
human-readable postprocessing does not need to be written as frequently as
ROM training states.

## Stage 3: Build the unsteady state basis

- Assemble state trajectories from one or more parameter points.
- Decide whether initial transients belong in the training set.
- Verify temporal and parametric weighting so long trajectories do not
  dominate merely because they contain more snapshots.
- Run the existing POD preprocessing with the new snapshot catalog.
- Check singular-value decay, retained energy, basis dimension, and
  reconstruction error.

The POD implementation should be reused unless the unsteady data reveals a
specific incompatibility.

## Stage 4: Validate an unsteady ROM

- Generate an `UnsteadyNonlinearRom` input.
- Use the same initial condition, time step, final time, integrator, and
  boundary conditions as the corresponding HDM.
- Initially test a training parameter point.
- Compare state error and relevant quantities of interest over the complete
  trajectory.
- Then test at least one parameter point not used to build the basis.

ROM validation should precede all unsteady HROM changes.

## Stage 5: Validate an unsteady HROM

- Reuse the existing hyperreduction workflow where compatible.
- Verify that residual snapshots represent the intended temporal window.
- Confirm the spatial training/stacking assumptions used by the current
  ECSW configuration.
- Generate an `UnsteadyNonlinearRom` input with reduced geometry.
- Compare HDM, ROM, and HROM trajectories and quantities of interest.
- Measure both accuracy and runtime reduction.

## Stage 6: Define the unsteady greedy indicator

The current steady indicator cannot be adopted blindly because it represents
one solution state. Define a trajectory-level scalar for ranking candidate
parameters. Candidates include:

- maximum residual indicator over time,
- time-averaged residual indicator,
- time-integrated residual indicator,
- maximum or integrated error in a selected quantity of interest.

The chosen indicator should state explicitly:

- the temporal window,
- any transient discarded from the comparison,
- normalization,
- aggregation across time,
- behavior when a simulation fails or terminates early.

Only after this definition is validated should
`greedy-procedure/GreedyAlgorithm.py` be changed.

## Stage 7: Regression and compatibility checks

Before considering the conversion complete:

- Re-run the preserved steady reference case.
- Run the unsteady HDM, ROM, and HROM cases from clean directories.
- Test restart behavior.
- Check all generated input files for unintended differences.
- Confirm that default steady users do not read unsteady-only files or
  require new settings.
- Confirm that generated data and binaries remain ignored by Git.
- Review the final diff against the original repository for unnecessary
  renaming, movement, comments, or formatting changes.

## First implementation milestone

The first milestone is intentionally narrow:

> Generate and run one short, physically defined, unsteady CRM HDM trajectory
> on Sherlock and verify its time history and snapshots.
No ROM/HROM or greedy change is part of this milestone.
