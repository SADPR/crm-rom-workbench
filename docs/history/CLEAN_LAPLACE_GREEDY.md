# Clean Laplace Greedy Enrichment

> **Historical record (superseded 2026-09-25).** The active plan is the 5D
> transonic Sobol campaign in `CRM_WORKFLOW_HANDOFF.md`. Scripts named here
> were removed from HEAD and remain in git history.

## Purpose

`CleanLaplaceRuns` and its Sobol point-33 holdout are the immutable, audited
32-state baseline. The next eight training samples are built in a separate
directory:

```text
greedy-procedure/CleanLaplaceGreedy40/
```

This keeps all prior results available for comparison. It also avoids copying
the approximately 31 GB database of original snapshots: the new catalogs keep
the original 32 paths and append only HDMs 33 through 40 under their new root.

## What one greedy iteration does

For iteration `k`, where `k` starts at 33:

1. Create a deterministic Sobol continuation candidate pool.
2. Run a full Laplace-shifted, `NonDescriptor` PROM at each candidate.
3. Select the candidate with the largest final *absolute* full residual: the
   relative `Residual.out` value times the initial `Spatial residual norm` in
   that PROM log, as the original greedy does in `runs.py` `getRes`.
4. Run the corresponding two-stage HDM with the corrected named-boundary
   Laplace shift.
5. Append that HDM only after it reaches `HDMtol2`.
6. Rebuild a ScaLAPACK POD with the enlarged catalog.

The residual is used only as a greedy ranking indicator; it is not a certified
physical-state error. The screen is full PROM only: it creates neither ECSW
weights nor a reduced mesh, so it is not HPROM work.

The relative value alone is not used because each candidate normalizes it by
its own IDW initial residual. That can invert the ranking: training point 001
reports 0.969 (absolute 1.86e4), while holdout point 33 reports 0.168
(absolute 4.47e4). Iteration 033 was screened before this change, so its
`candidates.json` still names the relative residual; its `selection.json`
records the indicator actually used and all three quantities per candidate.

## Safety contract

- `CleanLaplaceRuns` is read-only to this workflow.
- `CleanLaplaceHoldout` is not consumed or modified.
- The source state and parameter catalogs are SHA-256 fingerprinted.
- A selection record is frozen before its HDM is prepared.
- Each new Laplace shift must be finite and nonconstant.
- A new catalog entry is allowed only after both HDM stages complete and the
  final HDM residual meets the existing tolerance.
- Every stage refuses to overwrite an existing result.

## Sherlock commands

From `crm-rom-workbench/greedy-procedure`, after loading the normal compiler,
MPI, and `GreedyAEROF` environment, initialize once:

```bash
python3 -B clean_laplace_greedy.py init
```

Then submit exactly one dependency chain at a time:

```bash
./submit_clean_laplace_greedy_iteration.sh 33
```

The default pool contains 24 candidates; the screen Slurm array runs at most
three five-node PROMs at once. Therefore it can use at most 15 nodes during
the screen. A smaller pool is explicit, for example:

```bash
./submit_clean_laplace_greedy_iteration.sh 33 16
```

Only after iteration 33 succeeds, submit 34; continue sequentially through
40. This preserves a clear checkpoint and a one-to-one mapping between each
selected candidate, HDM, and POD.
