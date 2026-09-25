# Clean Laplace Campaign

## Decision

The initial steady CRM HDM/POD/PROM campaign is retained only as an
exploratory diagnostic. Its Laplace boundary configuration assigned the unit
value to `Symmetry_1`. In the one-layer extrusion, that boundary touches every
vertex and produced a constant field instead of a geometry-specific Laplace
shift.

The official steady ROM database will therefore be regenerated from scratch.
No existing `ushift.bin`, `State.bin`, `Solution.bin`, POD, or PROM result is
reused as an input to the clean campaign.

## Preserved exploratory campaign

Before the clean campaign, the complete generated-data workspace was copied
on Sherlock to:

```text
/scratch/users/sadpr/Code3Aug/crm-rom-workbench-backups/
2026-09-24_pre-clean-laplace
```

It contains `GreedyRuns`, `InitialHDMruns`, `BaselineRuns`, and
`CRMDiagnostics`, together with the source Git revision and mesh checksum.
The original workspace is also left unchanged.

## Clean campaign contract

The clean workflow writes only to these ignored directories:

```text
greedy-procedure/CleanLaplaceRuns/
greedy-procedure/CleanLaplacePrecompute/
greedy-procedure/CleanLaplaceBaseline/
```

For every run it uses the named physical boundaries:

| Boundary | Unit Laplace value |
| --- | ---: |
| `InletFixed_2` | 1 |
| `StickMoving_3` | 0 |
| `Symmetry_1` | natural |

The clean driver rejects a constant or non-finite generated shift and rejects
an HDM whose second stage misses `HDMtol2`.

## Execution order

1. Run `clean_laplace_initial.py init` once to create empty output roots and
   record the exact 32-point Sobol catalog.
2. Submit `submit_clean_laplace_pilot.sbatch` for point 1. It creates a fresh
   120-way static decomposition, computes the first corrected shift, and runs
   both HDM stages.
3. Submit `submit_clean_laplace_campaign.sbatch` after the pilot succeeds.
   It runs points 2--32 as a Slurm array throttled to three simultaneous,
   five-node HDMs.
4. Submit `submit_clean_laplace_pod.sbatch` after the array succeeds. It
   audits all 32 runs, moves only audited data into `CleanLaplaceRuns`, and
   builds a new `ScalapackSVD` POD with the ScaLAPACK-enabled AERO-F binary.
5. Submit `submit_clean_laplace_baseline.sbatch` to regenerate the nominal
   physical inspection case in an independent directory.
6. Submit `submit_clean_laplace_prom_validation.sbatch` to validate the global
   PROM at `HDMrun001`, using only `CleanLaplaceRuns/reductionrun032`. It
   regenerates that online geometry's Laplace field, runs a full PROM, and
   compares its physical state, lift, drag, and convergence with the clean
   HDM. The PROM state is output with `OutputShiftVectorType = None` solely
   for this comparison; the online formulation still uses
   `ShiftVectorType = Laplace`.
7. Submit `submit_clean_laplace_prom_comparison.sbatch` after the validation
   has completed. It exports six ParaView-ready Exodus files at
   `CleanLaplaceRuns/evaluate/romruns032/point001/postpro/field_comparison/`:
   HDM, PROM, and PROM-minus-HDM fields for the flow and for the nodal flux
   residual. `FluxResidual` is a spatial AERO-F diagnostic; it is not the
   scalar full-residual history written to `Residual.out`.

No job in this sequence overwrites the exploratory campaign or its backup.

## Independent holdout

`submit_clean_laplace_holdout.sh` submits a three-job dependency chain for
Sobol continuation point 33, which is not in the 32-state POD:

```text
[0.53125, 1.5625, 0.309375, 0.03, 0.12]
```

Its HDM is written only to `CleanLaplaceHoldout/HDMrun001`. The frozen POD,
the two 32-sample catalogs, and the holdout geometry/Laplace field are checked
before the PROM runs. The PROM is written separately at
`CleanLaplaceRuns/evaluate/romruns032/point901`; it is an online result only
and is never added to the POD. The final dependent job writes field comparison
Exodus files to `CleanLaplaceHoldout/field_comparison/`.
