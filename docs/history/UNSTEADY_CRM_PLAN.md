# CRM ROM Work Plan

> **Historical record (superseded 2026-09-25).** The active plan is the 5D
> transonic Sobol campaign in `CRM_WORKFLOW_HANDOFF.md`. Scripts named here
> were removed from HEAD and remain in git history.

## Objective

Build and validate the reduced-order workflow incrementally. The immediate
objective is a steady global PROM for the current three-dimensional active
parameter space. Unsteady URANS, a second temporal PROM, local bases, nonlinear
manifolds, and hyperreduction remain later stages and will only be introduced
after the simpler model is understood.

## Working locations

- Edit and version code in `/home/sares/crm-rom-workbench`.
- Use `/home/sares/Sherlock_CRM` only to inspect the Sherlock execution clone
  and its results through SSHFS.
- Compile AERO-F, postprocess large files, and submit simulations from a
  regular Sherlock session, not through SSHFS.
- Keep meshes, binaries, snapshots, Exodus files, and simulation outputs out
  of Git.
- Use `/home/sares/hgv2-rom-workbench` as a technical reference for proven
  AERO-F, SOWER, xp2exo, PROM, and HROM procedures. Adapt those procedures to
  this case without coupling the two repositories.

## Guiding principles

- Use current upstream AERO-F behavior and naming as the primary reference.
- Preserve the supplied steady workflow and change as little code as practical.
- Add isolated drivers before modifying the production greedy workflow.
- Validate each stage before adding the next layer of complexity.
- Begin with a global linear model. Introduce local, piecewise, or nonlinear
  models only when measured errors justify them.
- Separate accuracy validation from speed validation: PROM first, HROM second.
- Preserve backward compatibility for current steady AERO-F users.

## Current verified state

- The nominal parameter point is
  `[0.5, 0.0, 0.45, 0.03, 0.12]`.
- The private mesh and 120-way decomposition are installed on Sherlock.
- AERO-F was built from commit
  `069ef9d8e904746652d94d062a5958e77e0686df`.
- Sherlock job `44737107` completed the isolated two-step steady HDM.
- The restarted solve reached residual `4.983595e-07` at iteration 2245.
- The distributed flow fields, restart data, state snapshots, and force
  histories are preserved under `greedy-procedure/BaselineRuns/`.

## Stage 0: Close and inspect the steady baseline

The existing solution must be examined physically before choosing training
points.

1. Merge the distributed AERO-F fields with SOWER:
   - Mach,
   - pressure coefficient,
   - skin-friction coefficient,
   - velocity,
   - displacement.
2. Convert the merged XPOST fields to an Exodus file with xp2exo and the
   120-way mesh decomposition.
3. Open the Exodus result in ParaView and inspect:
   - the deformed airfoil geometry,
   - freestream orientation,
   - Mach and pressure-coefficient fields,
   - boundary-layer and skin-friction behavior,
   - whether a shock is present and, if so, its location,
   - symmetry and far-field behavior,
   - finite values and obvious discontinuities caused by setup errors.
4. Review the residual, lift/drag, and force histories together with the flow
   field.
5. Record screenshots or observations needed to identify the physical regime.

### Stage 0 acceptance checks

- The Exodus output opens without mesh or variable errors.
- The expected fields are present on the complete 812,098-node mesh.
- The displayed geometry agrees with
  `[maximum-camber location, maximum camber, thickness] =
  [0.45, 0.03, 0.12]`.
- The solution is finite and physically plausible.
- The converged field, inputs, logs, and runtime metadata remain reproducible
  and isolated from the production greedy directories.

No training design begins until these checks are complete.

## Stage 1: Define steady training and validation points

After inspecting the nominal field:

- Confirm the three active parameters:
  Mach number, angle of attack, and maximum-camber location.
- Keep maximum camber and thickness fixed for the first ROM.
- Normalize the three active coordinates before computing parameter-space
  distances.
- Choose a space-filling training design, with Sobol sampling as the initial
  candidate.
- Reserve independent validation points that are never used to build the
  basis.
- Include enough boundary and physically difficult cases to exercise moving
  shocks or strong gradients observed in Stage 0.
- Run one non-nominal point as an end-to-end smoke test before launching the
  full campaign.

The final sample count will be selected after considering the measured cost
and physics. A count such as 27 is a pilot, not an accuracy guarantee; powers
of two are natural when retaining the balance properties of a Sobol sequence.

## Stage 2: Build the steady HDM database

For every training parameter:

- Deform the mesh without changing its topology or node correspondence.
- Execute the same reviewed two-step steady solve used by the baseline.
- Require the final solve to meet the selected convergence tolerance.
- Store exactly one converged steady state for the initial steady POD.
- Record parameters, convergence, forces, allocation, runtime, and output
  paths.
- Keep failed attempts separate and exclude them from snapshot catalogs.

Run the independent validation HDMs with the same settings, but do not include
their states in the training snapshot matrix.

## Stage 3: Diagnose the global steady POD space

Assemble

```text
S = [U(mu_1), U(mu_2), ..., U(mu_N)]
```

and then:

- choose and document the reference or centering convention,
- compute the singular-value decay and retained energy,
- measure reconstruction error for every training and validation state,
- inspect field errors near shocks, walls, and geometric deformations,
- determine whether a modest global linear basis is adequate.

Slow singular-value decay or localized shock-position errors are evidence for
a later local or nonlinear model; they are not reasons to skip the global
baseline.

## Stage 4: Build and validate the steady global PROM

Construct the steady PROM by projecting the steady residual:

```text
V_s^T R(U_ref + V_s q_s; mu) = 0.
```

Validation order:

1. reproduce training parameters,
2. solve at independent validation parameters,
3. compare HDM and PROM residuals,
4. compare state and projection errors,
5. compare pressure coefficient, Mach, lift, drag, and shock location,
6. record convergence behavior and cost.

At this stage the PROM may still evaluate the full HDM residual. The objective
is accuracy and robustness, not yet maximum speedup.

## Stage 5: Build and validate the steady HROM

Only after the steady PROM is accurate:

- generate the required residual and training snapshots,
- apply the existing hyperreduction procedure,
- compare HDM, PROM, and HROM at the same validation points,
- measure both error and wall-clock speedup,
- verify that sampled meshes and weights remain valid across geometry changes.

## Stage 6: Introduce local, piecewise, or nonlinear models if justified

Use the global-model evidence to decide whether to introduce:

- clustered/local POD bases,
- piecewise PROMs,
- quadratic or other nonlinear manifolds,
- GPR, ANN, or RBF mappings,
- local HROM construction.

Each addition must be compared against the global steady PROM with identical
training and validation data. Complexity is justified only by a measurable
gain in accuracy, robustness, or cost.

## Stage 7: Define the unsteady physical problem

Unsteady work begins only after the steady pipeline is reliable.

The parameters remain fixed during each trajectory unless the scientific
problem is explicitly changed. For fixed `mu`, determine:

- whether the intended model is autonomous URANS,
- the initial steady state and controlled perturbation,
- whether perturbations decay or lead to persistent dynamics,
- physical time step, horizon, integration scheme, and output frequency,
- the quantities of interest and temporal window.

A short URANS stability pilot at a few parameter points must precede any
unsteady training campaign.

## Stage 8: Build a hierarchical steady/unsteady ROM

The preferred long-term architecture is:

```text
mu -> steady PROM -> steady base flow -> unsteady fluctuation PROM
```

For a fixed online parameter, the steady PROM supplies
`U_steady_ROM(mu)`. The temporal model then represents

```text
U(t; mu) = U_steady_ROM(mu) + V_u q_u(t; mu).
```

This separates parametric variation of the equilibrium from temporal
fluctuations. IDW may be retained as an inexpensive initial guess for the
steady reduced coordinates, but it need not replace the physics-based steady
PROM.

## Stage 9: Unsteady HROM and trajectory-level greedy logic

After validating the unsteady HDM and PROM:

- construct and validate the unsteady HROM,
- preserve identical physical-time settings across HDM, PROM, and HROM,
- define temporal snapshot weighting,
- define a trajectory-level greedy indicator,
- test restarts and reproducibility,
- confirm that all steady paths remain unchanged.

## Immediate milestone

> Generate and inspect the Exodus postprocessing of the completed nominal
> steady HDM. Then define the steady training and validation design.

No unsteady input or greedy-algorithm change belongs to this milestone.
