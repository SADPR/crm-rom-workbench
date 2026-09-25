# CRM steady-ROM workflow: history, operating procedure, and current plan

This document is the operational memory for the CRM workbench. It records what
exists, why particular decisions were made, how a source-code change reaches
Sherlock, and how to distinguish an exploratory result from a result that may
be used by the current steady-ROM campaign.

It is intentionally more detailed than a normal README. The goal is that a
future version of us can restart safely without reconstructing decisions from
terminal history.

## 1. Current status at a glance

The current objective is a **steady, global, nonlinear PROM** for a parametric
CRM/NACA airfoil problem. It is not yet an HPROM and it is not yet unsteady.

The stable, audited starting point is:

- 32 clean, converged steady HDMs;
- a 32-state distributed `ScalapackSVD` POD;
- a full PROM using `ShiftVectorType = Laplace` and `Form = NonDescriptor`;
- one independent, out-of-training Sobol holdout at point 33;
- a field/force comparison for both a training point and the independent
  holdout.

The independent holdout is promising, but it is not a universal certification
of the PROM. The next construction step is residual-greedy enrichment from 32
to 40 HDMs in a **new, isolated campaign root**. After the 40-state POD is
validated on several independent points, then and only then should we begin
HPROM work. Unsteady work comes after a trusted steady baseline.

At the time this file was added, the source-tree checkpoint is:

```text
830e79e Add isolated clean Laplace greedy enrichment
```

The eight extra HDMs have not been launched merely by committing this code.
Submitting jobs on Sherlock remains an explicit human action.

## 2. Non-negotiable operating rules

These rules prevent accidental loss, mixing incompatible databases, or turning
an otherwise reproducible campaign into an undocumented experiment.

1. **Edit source code locally; execute simulations only on Sherlock.**
   Local machines are for code review, Git, visualization, and reading the
   mounted remote results. AERO-F, FEniCS/Laplace, POD, PROM, and Slurm jobs run
   on Sherlock.

2. **Source changes travel through Git, never through a manual source copy.**
   The normal route is local edit -> local test/review -> commit -> push ->
   Sherlock fetch/reset. This gives every remote result an identifiable source
   revision.

3. **Do not use `rsync` for this project.**
   Do not rsync source files to Sherlock and do not rsync generated results
   back into the repository. The SSHFS mount is the normal way to inspect
   remote files. If an individual Exodus file must be visualized locally, make
   an explicit, disposable copy into `/tmp`; do not make it a second canonical
   results tree and never copy it back into Sherlock as input.

4. **Generated data are not Git data.**
   HDMs, PODs, ROM outputs, FEniCS products, mesh data, logs, conda
   installations, and AERO-F build directories stay ignored on Sherlock. Git
   contains source, submission scripts, documentation, and small manifests.

5. **Never put `mesh/` in ordinary Git.**
   The supplied mesh is hundreds of MB and has files larger than GitHub's
   normal 100 MB limit. Public redistribution has not been authorized. Do not
   use Git LFS unless both authorization and an explicit data-management plan
   exist.

6. **Do not run broad cleanup commands.**
   In particular, do not run `clean.sh`, `git clean -fdx`, `rm -r GreedyRuns*`,
   or an unscoped recursive removal from this workbench. Some older scripts
   contain broad cleanups appropriate only for a disposable experiment. Every
   generated directory is evidence until intentionally archived.

7. **Never overwrite a campaign root.**
   The safe drivers refuse to overwrite a pre-existing result. If a fresh
   campaign is genuinely required, create a new named root or archive the old
   one first. Do not solve this with an unreviewed `rm -rf`.

8. **Do not modify tracked source directly on Sherlock.**
   Remote source changes make the Git revision and results disagree. Generated
   ignored output is expected remotely; tracked edits are not.

9. **Do not silently change physical or ROM conventions mid-campaign.**
   A different shift, form, parameter range, POD method, solver build, or
   reference method defines a different database. Record it and start a
   separate campaign.

10. **Do not infer that a PROM is bad from `Residual.out` alone.**
    It records a full-order residual diagnostic. The LSPG PROM minimizes its
    projected residual, not necessarily that full residual. Physical field and
    force comparisons against independent HDMs remain necessary.

## 3. Canonical locations

### Local workstation

```text
/home/sares/crm-rom-workbench
    Canonical local Git worktree. Edit source and documentation here.

/home/sares/Sherlock_CRM
    SSHFS mount of Sherlock's execution clone. Inspect-only in normal use.

/home/sares/CRM_tmp
    Local staging copy of the originally supplied material. It is provenance,
    not the active source worktree.
```

### Sherlock

```text
/scratch/users/sadpr/Code3Aug/crm-rom-workbench
    Canonical execution clone. Git-tracked source arrives here from GitHub.
    All current generated CRM results are written below greedy-procedure/.

/scratch/users/sadpr/Code3Aug/crm-rom-workbench/mesh
    Private execution mesh directory. It is intentionally ignored by Git.
    greedy-procedure/mesh -> ../mesh is a tracked relative link.

/scratch/users/sadpr/Code3Aug/crm-rom-workbench-backups/
    Explicit archival area.
    2026-09-24_pre-clean-laplace preserves the old exploratory outputs.

/scratch/users/sadpr/Code3Aug/aero-f
    Sherlock AERO-F source clone.

/scratch/users/sadpr/Code3Aug/aero-f/build-scalapack/bin/aerof.opt
    Current executable used by the clean POD/PROM workflow.

/scratch/users/sadpr/Code3Aug/miniconda3
    Private Sherlock Miniconda installation and conda environments.
```

### Original supplied delivery

```text
/scratch/users/sadpr/Code3Aug/CRM_tmp
```

Treat the original delivery as read-only provenance. Do not write experiments
there and do not replace it with workbench results.

## 4. Repository and Git model

The public repository is:

```text
https://github.com/SADPR/crm-rom-workbench
```

The local workstation uses SSH for the GitHub remote. Sherlock cannot
authenticate through the user's GitHub SSH key, so its execution clone uses
public HTTPS. This is normal: Sherlock only needs to fetch published commits.

### Local change procedure

Work from the local repository:

```bash
cd /home/sares/crm-rom-workbench
git status --short
```

Before changing anything, note existing modifications. In particular, the
local `.gitignore` can contain user-owned work. It must not be staged, reverted,
or reformatted merely because we are committing another task.

The normal sequence is:

```bash
# 1. Edit only the relevant tracked source/doc files locally.
# 2. Inspect the diff and run proportionate local static checks.
git diff --check
git status --short

# 3. Stage explicitly named files. Never use git add -A here.
git add path/to/changed_source.py path/to/document.md
git commit -m "Concise description"
git push origin main
```

Why explicit `git add` matters:

- it prevents accidentally adding ignored results after a local copy;
- it prevents including a user's unrelated `.gitignore` change;
- it makes a checkpoint explain one logical change.

### Sherlock update procedure

Only after the local push succeeds, update the execution clone:

```bash
set -euo pipefail

cd /scratch/users/sadpr/Code3Aug/crm-rom-workbench
git fetch origin
git reset --hard origin/main
```

`git reset --hard origin/main` updates tracked source to the exact published
revision. It does **not** delete ignored simulation results, ignored conda
environments, or the private ignored `mesh/` directory. It is appropriate on
the disposable execution clone, not on the local working tree.

Afterward, verify the revision if it matters for a report:

```bash
git rev-parse --short HEAD
git status --short --branch
```

If Sherlock reports tracked modifications before an update, stop and inspect
them. Do not erase unrecognized tracked work automatically.

## 5. SSHFS mount: what it is and is not

The mount gives a convenient local view of the remote execution clone:

```bash
~/mount_sherlock_crm.sh
```

Its intended mapping is:

```text
sadpr@dtn.sherlock.stanford.edu:
/scratch/users/sadpr/Code3Aug/crm-rom-workbench
    -> /home/sares/Sherlock_CRM
```

The user completes the password and two-factor authentication interactively.
That is expected; it is not something an automated agent can or should try to
store.

Use the mount for:

- browsing logs and manifests;
- opening an Exodus file in ParaView;
- checking whether a Slurm job produced expected files;
- copying one explicitly chosen visualization artifact to `/tmp` if SSHFS
  performance makes direct visualization impractical.

Do **not** use it for:

- launching AERO-F or FEniCS through the mounted path;
- bulk copying campaign directories;
- editing source on the mount;
- treating it as a second primary source tree;
- rsync in either direction.

SSHFS can be slower and less reliable for large random-access files. If
ParaView struggles, copy an individual file to a local temporary directory,
for example:

```bash
mkdir -p /tmp/crm-view
cp /home/sares/Sherlock_CRM/greedy-procedure/CleanLaplaceHoldout/field_comparison/prom_flow_fields.exo \
   /tmp/crm-view/
```

That copy is a disposable visualization cache. It is not an input to the
workflow and is not synchronized elsewhere.

## 6. Sherlock execution environment

The clean workflow was built and validated with:

```text
cmake/3.24.2
gcc/10.1.0
openmpi/4.1.2
imkl/2019
```

The standard setup inside an interactive allocation or a Slurm batch script is:

```bash
module purge
module load cmake/3.24.2 gcc/10.1.0 openmpi/4.1.2 imkl/2019

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

export SOWER=/home/groups/cfarhat/bin/sower
export PARTMESH=/home/groups/cfarhat/bin/partnmesh
export CD2TET=/home/groups/cfarhat/bin/cd2tet

export CRM_CONDA_BASE=/scratch/users/sadpr/Code3Aug/miniconda3
export CRM_LAPLACE_ENV=CRM_Laplace

export AEROF=/scratch/users/sadpr/Code3Aug/aero-f/build-scalapack/bin/aerof.opt
export MPI=srun
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

`GreedyAEROF` provides the workflow dependencies, including NumPy, SciPy,
joblib, numba, matplotlib, and scikit-learn. `CRM_Laplace` is a separate
environment used by the corrected FEniCS Laplace-shift runner.

The conda environment names are not decorative. Activating the wrong one can
make FEniCS fail to find DOLFIN or can make a conda-provided MPI conflict with
Sherlock OpenMPI. The batch scripts load the compiler/MPI modules first and
then activate `GreedyAEROF`; the Laplace runner switches to the dedicated
environment as needed.

### AERO-F build that the clean workflow uses

The AERO-F source revision is:

```text
069ef9d8e904746652d94d062a5958e77e0686df
```

The ScaLAPACK executable is built with:

```text
Eigen: 3.4.0 at 3147391d946bb4b6c68edd901f2add6ac1f31f8c
Boost: 1.74.0
WITH_SCALAPACK=ON
WITH_BLACS=ON
WITH_TORCH=OFF
WITH_ARPACK=OFF
WITH_PARMETIS=OFF
USE_MARCH_NATIVE=OFF
```

`ldd` verified that it links MKL's OpenMPI BLACS and ScaLAPACK libraries. This
matters because the clean POD uses `ScalapackSVD`, not the earlier randomized
SVD fallback.

Do not silently change the AERO-F binary halfway through a campaign. A new
binary build should be recorded, smoke-tested, and treated as a new numerical
provenance point.

### Interactive allocations versus batch jobs

An interactive allocation is useful for preparing inputs, reading logs,
testing a short command, or submitting a batch chain. The production HDMs,
PODs, and PROM screens are Slurm batch jobs. A typical interactive allocation
used for preparation has one node and 24 tasks:

```bash
salloc -N 1 -n 24 -p cfarhat --time=2600
```

This does not imply that a 120-rank HDM can run inside that allocation. The
standard HDM/POD scripts request five nodes with 120 MPI ranks. The user
decides when to keep or release an interactive node; scripts and documentation
must never assume it can be released automatically.

## 7. Physical and numerical problem being solved

The current problem is a steady, compressible RANS/Navier-Stokes airfoil flow,
not Euler:

- governing equations: compressible Navier-Stokes;
- turbulence closure: Spalart-Allmaras;
- viscosity: Sutherland model;
- thermal conductivity: constant Prandtl number;
- wall: adiabatic;
- angle of attack is passed as AERO-F `Beta`, with `Alpha = 0`, because of the
  mesh orientation.

The current full five-component parameter vector is:

```text
[Mach, angle of attack, maximum camber location, maximum camber, thickness]
```

The active three-dimensional box is currently:

```text
Mach                  [0.4, 0.6]
angle of attack       [-5, 5] degrees
maximum camber loc.   [0.3, 0.6] chord fractions
```

The remaining shape parameters are intentionally fixed for this first model:

```text
maximum camber = 0.03
thickness     = 0.12
```

They still appear in every parameter vector because the deformation routine
expects all five entries. They do not currently add parametric variation.

The initial HDM catalog has eight corners of the active three-dimensional box
and Sobol continuation points to reach 32 total samples. The Sobol generator
prints a warning when its requested sample count is not a power of two. That
warning describes Sobol balance quality; it does not invalidate the points or
mean the runs failed.

## 8. Why each HDM has two stages

Every steady HDM uses two sequential AERO-F inputs:

1. `input1` starts from no restart, uses the approximate matrix-vector product,
   a looser tolerance (`1e-4`), and up to 1500 nonlinear iterations.
2. `input2` restarts from stage 1, uses the finite-difference matrix-vector
   product, a tighter tolerance (`5e-7`), and up to 9000 nonlinear iterations.

The first stage is a robust approach phase; the second is the accuracy phase.
They are not two unrelated physical simulations and they are not Euler followed
by RANS. The second stage refines the same RANS problem from the saved restart.

A completed clean HDM is accepted only if its final stage-2 reported residual
is no greater than `HDMtol2 = 5e-7`. The 32 clean initial HDMs satisfied this,
with final residuals from approximately `4.983595e-07` to `4.999285e-07`.

## 9. The Laplace shift: the critical correction

### Purpose

The geometry changes from one airfoil parameter point to another. A Laplace
shift provides a geometry-dependent field used to align state representations
across those deformed geometries. It is the component intended to reduce
geometric snapping. The `NonDescriptor` form is the compatible ROM
formulation used with this workflow; the two choices should not be described as
two independent anti-snapping mechanisms.

The intended online configuration is:

```text
ShiftVectorType = Laplace
Form = NonDescriptor
```

For physical-state comparison only, an output field can be requested with
`OutputShiftVectorType = None`. This does **not** turn off the online Laplace
shift. It only asks AERO-F to write the reconstructed physical state rather
than the shifted representation.

### What was wrong in the exploratory campaign

The initial exploratory Laplace configuration imposed the unit value on
`Symmetry_1`. The mesh uses a one-layer extrusion, and that boundary reaches
every vertex. The resulting Laplace field was constant rather than
geometry-specific. A constant shift cannot do the alignment that this method
needs.

This is a workflow/configuration issue in the exploratory CRM pipeline. It is
not a statement that another person's CFD solution is physically wrong.

### Correct named-boundary condition

Every clean Laplace solve uses:

| Boundary | Laplace value |
| --- | ---: |
| `InletFixed_2` | 1 |
| `StickMoving_3` | 0 |
| `Symmetry_1` | natural |

The clean driver rejects a non-finite or constant shift. It writes
`laplace_shift.json` alongside each clean initial HDM with the minimum,
maximum, and range. The corrected clean shifts have a nonzero range of order
one, as expected.

Do not change the named tags, replace them with a broad geometric selection, or
reuse an old `ushift.bin` from the exploratory campaign.

## 10. Historical generated-data campaigns

The directory name tells us which results may be used for what.

### A. Exploratory / historical directories: do not use as current training data

```text
greedy-procedure/GreedyRuns/
greedy-procedure/InitialHDMruns/
greedy-procedure/BaselineRuns/
greedy-procedure/CRMDiagnostics/
```

These were useful for discovering the Laplace boundary error, installing the
environment, building AERO-F, exercising the two HDM stages, and learning the
postprocessing conventions. They are not the official clean ROM database.

Before regenerating clean data, these directories were preserved at:

```text
/scratch/users/sadpr/Code3Aug/crm-rom-workbench-backups/
2026-09-24_pre-clean-laplace
```

The backup is about 36 GB. It should not be deleted casually, copied into Git,
or mixed into the clean state catalog.

### B. Clean 32-state campaign: official current v1 training database

```text
greedy-procedure/CleanLaplaceRuns/
greedy-procedure/CleanLaplacePrecompute/
greedy-procedure/CleanLaplaceBaseline/
```

Meaning:

- `CleanLaplacePrecompute/` held the separate precompute HDM directories
  before audit/assembly.
- `CleanLaplaceRuns/HDMrun001` through `HDMrun032` are the audited training
  HDMs used by the v1 POD.
- `CleanLaplaceRuns/statesnapdata.txt` and `parsoldata.txt` describe exactly
  those 32 state snapshots and parameter points.
- `CleanLaplaceRuns/reductionrun032/` is the full distributed ScaLAPACK POD.
- `CleanLaplaceBaseline/` is a nominal independent physical-inspection case;
  it is not a training state unless its catalog says so.

The initial POD assembly deliberately moves only audited HDM runs into the
training root. It checks each metadata point, required binary partition,
nonconstant Laplace field, and final HDM residual before building the catalog.

### C. Clean 32-state training-point PROM validation

```text
CleanLaplaceRuns/evaluate/romruns032/point001/
CleanLaplaceRuns/evaluate/hromruns032/point001/
```

This checks that the online Laplace solve, AERO-F files, POD, and full PROM
pipeline reproduce a point that is already represented in the training
database. It is a useful wiring test, but it is not a generalization test.

At that point, fields and forces agreed closely with the HDM. Typical force
errors were approximately 0.049% drag and 0.073% lift. The full residual was
about 0.969; that fact alone does not contradict the good physical comparison
for the LSPG PROM.

### D. Independent v1 holdout: the actual generalization check already done

```text
greedy-procedure/CleanLaplaceHoldout/
```

The held-out point is Sobol continuation point 33:

```text
[0.53125, 1.5625, 0.309375, 0.03, 0.12]
```

It is deliberately excluded from the 32-state POD. Before its PROM ran, the
workflow fingerprinted the 32-state state catalog, parameter catalog, and
singular values. It also required exact matching geometry positions and a
Laplace relative L2 difference no larger than `1e-10` between HDM and PROM.

The resulting v1 holdout reported:

```text
HDM final residual       4.989276e-07
PROM outer iterations    30
PROM full residual       1.682292e-01
drag relative error      3.868541e-03  (about 0.387%)
lift relative error      4.232737e-04  (about 0.0423%)
Mach relative L2 error   3.68349e-03   (about 0.368%)
velocity relative L2     3.69914e-03   (about 0.370%)
Cp relative L2           1.78521e-02   (about 1.785%)
skin-friction relative   1.65951e-02   (about 1.660%)
```

These are encouraging results for a first global 32-state PROM. They do not
prove every region of the parameter domain is equally accurate. The full
residual stagnation and inner Newton warnings are diagnostics to retain, not a
reason to silently redefine the method or declare success everywhere.

### E. Clean residual-greedy enrichment: official current v2 construction

```text
greedy-procedure/CleanLaplaceGreedy40/
```

This directory is created only when the v2 initialization command is run. It
is intentionally separate from `CleanLaplaceRuns`.

It contains:

- a copied static mesh/decomposition directory;
- a copy of `reductionrun032` used as the initial POD;
- new state/parameter catalogs whose first 32 entries still reference the
  original clean training snapshots;
- screen records, candidate scores, frozen point selections, added HDMs 33--40,
  and `reductionrun033` through `reductionrun040`.

It does not use symbolic links. It does not duplicate the 32 large HDM
directories. It never writes to `CleanLaplaceRuns` or `CleanLaplaceHoldout`.

## 11. PROM residuals, flux residuals, and what to compare

There are several quantities called "residual" in the workflow. They must not
be conflated.

### `postpro/Residual.out`

This is the history of the full-order residual norm evaluated at the PROM
state. For a converged HDM, it should meet the HDM tolerance. For an LSPG PROM,
it does not have to vanish, because the online method minimizes a projected
least-squares residual in its reduced space.

Use it to monitor convergence behavior and to rank candidates during the
residual-greedy screen. Do not use it by itself as a physical-error metric.

### `FluxRes.bin` / `FluxResidual`

This is an AERO-F nodal spatial diagnostic. It can be merged, visualized, and
compared field by field. It is not the scalar global residual history above.

### What constitutes a meaningful PROM validation

For each independent holdout, compare:

1. HDM and PROM final forces: drag and lift;
2. physical fields: Mach, velocity, pressure coefficient, skin friction;
3. field-error location: leading edge, trailing edge, wake, and wall;
4. PROM history: iterations and full residual behavior;
5. HDM final residual and successful completion;
6. exact equality of the HDM and PROM geometry and online Laplace shift;
7. invariance of the frozen training catalog and POD inputs.

The quality of an HPROM later must be compared separately against both the
full PROM and the HDM. Otherwise POD/ROM error and hyper-reduction error are
mixed together.

## 12. Postprocessing and ParaView

The comparison scripts create ParaView-ready Exodus files for HDM, PROM, their
difference, and their nodal flux-residual fields. For the initial clean
training validation, these are under:

```text
CleanLaplaceRuns/evaluate/romruns032/point001/postpro/field_comparison/
```

For the independent holdout, they are under:

```text
CleanLaplaceHoldout/field_comparison/
```

Typical files are:

```text
hdm_flow_fields.exo
prom_flow_fields.exo
difference_flow_fields.exo
hdm_flux_residual.exo
prom_flux_residual.exo
difference_flux_residual.exo
comparison_summary.json
```

The standardized flow-field comparisons include Mach, pressure coefficient,
skin friction, velocity, and displacement. The difference file is
`PROM_minus_HDM`.

Recommended ParaView practice:

- use the same camera and manually matched color range for HDM versus PROM;
- use a sequential scale for a magnitude, such as velocity-difference
  magnitude;
- use a divergent scale centered at zero for a signed scalar difference, such
  as `PROM_minus_HDM_PressureCoefficient`;
- select the `stickmoving_3` block to inspect wall skin friction rather than
  interpreting the whole far-field extrusion as a wall quantity;
- interpret leading-edge/trailing-edge/wake localized differences differently
  from a diffuse discrepancy over the whole domain.

The apparent large disk or wide exterior region is a consequence of the
computational far-field/one-layer extrusion and visualization scale. It is not
by itself evidence of a physical anomaly.

For the validated subsonic examples, maximum Mach remained below one, so the
absence of a shock is expected.

## 13. How to run the 32 -> 40 residual-greedy enrichment

The new code is:

```text
greedy-procedure/clean_laplace_greedy.py
greedy-procedure/submit_clean_laplace_greedy_screen.sbatch
greedy-procedure/submit_clean_laplace_greedy_select.sbatch
greedy-procedure/submit_clean_laplace_greedy_hdm.sbatch
greedy-procedure/submit_clean_laplace_greedy_pod.sbatch
greedy-procedure/submit_clean_laplace_greedy_iteration.sh
```

Read `CLEAN_LAPLACE_GREEDY.md` before the first submission. The short version
is below.

### Step 0: update Sherlock to the committed source revision

```bash
set -euo pipefail

cd /scratch/users/sadpr/Code3Aug/crm-rom-workbench
git fetch origin
git reset --hard origin/main

cd greedy-procedure
module purge
module load cmake/3.24.2 gcc/10.1.0 openmpi/4.1.2 imkl/2019

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

export SOWER=/home/groups/cfarhat/bin/sower
export PARTMESH=/home/groups/cfarhat/bin/partnmesh
export CD2TET=/home/groups/cfarhat/bin/cd2tet
export CRM_CONDA_BASE=/scratch/users/sadpr/Code3Aug/miniconda3
export CRM_LAPLACE_ENV=CRM_Laplace
export AEROF=/scratch/users/sadpr/Code3Aug/aero-f/build-scalapack/bin/aerof.opt
export MPI=srun
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

### Step 1: initialize the isolated v2 root once

```bash
python3 -B clean_laplace_greedy.py init
python3 -B clean_laplace_greedy.py status
```

Expected status before the first selection:

```text
Snapshots in current POD catalog: 32
Target snapshots: 40
Next iteration: 33
```

If `CleanLaplaceGreedy40` already exists, the initialization refuses to
overwrite it. Do not delete it just to make the command pass. First inspect
its `campaign.json`, `screening/`, current catalog counts, and `status`.

### Step 2: submit one complete iteration

```bash
bash submit_clean_laplace_greedy_iteration.sh 33
```

The default candidate pool has 24 deterministic Sobol continuation points. It
submits this dependency chain:

```text
screen array -> selection -> selected HDM -> POD
```

The screen array is throttled to three concurrent jobs. Each screen PROM asks
for five nodes / 120 ranks, so no more than 15 nodes are requested by that
array at a time. The selected HDM and POD each request five nodes afterward.

Why the stages are sequential:

- the selected HDM must be chosen from actual PROM screen results;
- the HDM must converge before it becomes a catalog entry;
- POD 33 must use the new catalog before the next greedy iteration ranks points
  with a 33-state basis.

Do not submit 34 through 40 at once. Each must use the POD built by the prior
iteration. After a successful iteration 33, repeat:

```bash
python3 -B clean_laplace_greedy.py status
bash submit_clean_laplace_greedy_iteration.sh 34
```

Continue sequentially through 40.

The candidate count can be reduced explicitly if cluster availability makes 24
full PROM evaluations inappropriate, for example:

```bash
bash submit_clean_laplace_greedy_iteration.sh 33 16
```

This parameter changes the size of the ranking pool, not the number of HDMs
added. One successful iteration always adds exactly one HDM and one POD.

### Monitoring one iteration

```bash
squeue -u sadpr
squeue -p cfarhat
```

The submitted job IDs printed by the wrapper identify the four stages. A
pending dependent job is normal. For example, `Reason=Dependency` means it is
waiting for the preceding stage, while `JobArrayTaskLimit` means three screen
tasks are already active.

After a stage completes, inspect the specific files rather than guessing from
the queue alone:

```bash
python3 -B clean_laplace_greedy.py status
find CleanLaplaceGreedy40/screening/iteration033 -maxdepth 1 -type f -printf '%f\n' | sort
cat CleanLaplaceGreedy40/screening/iteration033/selection.json
cat CleanLaplaceGreedy40/HDMrun033/greedy_sample.json
```

The selection file records all candidate scores and the chosen point. The HDM
metadata records the Laplace range and final HDM residual. The POD completion
is represented by `CleanLaplaceGreedy40/reductionrun033/` and its partitioned
ROB/reference files.

### What the enrichment driver guarantees

- Source catalogs from `CleanLaplaceRuns` are SHA-256 fingerprinted.
- The first 32 entries of the new catalogs must remain text-identical to the
  v1 source catalogs.
- Each selected point is recorded before its HDM starts.
- The selected HDM gets a new deformed geometry and a new named-boundary
  Laplace solve; it does not reuse a v1 shift.
- The Laplace field must be nonconstant and finite.
- Both HDM stages must finish and the final residual must meet `HDMtol2`.
- Only then is a 33rd--40th entry appended to the new catalog.
- The new POD is `ScalapackSVD` and refuses to overwrite an existing
  `reductionrunNNN`.
- It uses full PROM screens (`HyperReduced = False`); it cannot accidentally
  become HPROM/ECSW training.

## 14. Validation plan after the 40-state POD

After `reductionrun040` exists, do not immediately start HPROM. First create
and run **new 40-state holdout drivers**. The existing
`clean_laplace_holdout.py` is deliberately frozen to POD 32 and point 33. Do
not edit or repurpose it in place, because that would destroy the meaning of
the completed v1 holdout.

The correct next implementation is a sibling 40-state validation workflow
that:

1. chooses points outside the final 40-state catalog;
2. creates HDM directories outside `CleanLaplaceGreedy40`;
3. fingerprints the 40-state catalogs and POD inputs;
4. runs independent HDM and full PROM with separately generated but identical
   geometry/Laplace fields;
5. outputs field and force comparisons;
6. never appends holdout snapshots to the training catalog.

Use at least two or three validation points:

- an interior point;
- a more challenging point near an active-domain boundary or a higher observed
  PROM-residual region;
- optionally a second region that is physically distinct in its angle of
  attack or Mach response.

The former point-33 result stays useful as a v1 record. It should not be called
a clean 40-state holdout if a selected greedy HDM happens to use it; the v2
candidate generator intentionally starts after point 33 to preserve it.

## 15. HPROM comes after the 40-state PROM is trusted

The planned HPROM order is:

1. freeze a validated full PROM/POD and its exact catalogs;
2. generate ECSW sampling/training data and weights from that frozen full ROM;
3. run HPROM at the same independent holdouts;
4. compare HDM versus PROM and PROM versus HPROM separately;
5. tune sampling tolerance, training data, and reduced mesh only with those
   separable errors visible.

Starting HPROM before the full PROM is accepted mixes basis/parametric error
with hyper-reduction error. That makes negative results hard to interpret and
positive results less defensible.

## 16. Unsteady work is later and has a different data model

The requested eventual direction is unsteady RANS, not an implicit instruction
to make parameters vary in time. The natural staged plan is:

1. establish the steady PROM first;
2. decide the final parametric space and sample plan;
3. for each parameter point, converge a steady RANS HDM;
4. use that steady state as the initial condition for an unsteady RANS HDM;
5. save time snapshots from the unsteady solution;
6. build an unsteady POD/PROM from those snapshots;
7. define online initial-condition interpolation from the steady database.

An unsteady Euler calculation can show transient adjustment or shocks in other
regimes, but it is not the present configuration. The current problem includes
viscosity and Spalart-Allmaras turbulence. Do not change to Euler merely to
make a transient experiment simpler unless the scientific request explicitly
changes.

For unsteady ROM design, distinguish carefully between:

- a physical time step, which advances from \(t^n\) to \(t^{n+1}\); and
- nonlinear/Newton iterations, which solve the implicit equations within one
  physical time step.

## 17. Questions to resolve with the prior CRM user/colleague

Before committing major additional compute, confirm the intended reference
configuration with the person who created the original CRM example:

1. Did the final study vary only Mach, angle of attack, and maximum camber
   location, with maximum camber and thickness fixed? Or did it vary all five
   parameters? If all five, what ranges were used for maximum camber and
   thickness?
2. Did the final PROM use the Laplace shift with the `NonDescriptor`
   formulation? The Laplace component is the part intended to prevent
   geometry-driven snapping.
3. How many HDM simulations/snapshots were ultimately used to build the PROM?
   We currently have 32 initial Sobol points and are preparing controlled
   enrichment while avoiding using Sherlock more aggressively than necessary.

This information improves the sampling plan; it does not retroactively justify
mixing old exploratory snapshots into the corrected clean campaign.

## 18. Common failures and how to reason about them

### GitHub authentication from Sherlock

Sherlock's clone is public HTTPS because SSH key authentication failed there.
That is sufficient for `git fetch origin`. Do not paste a personal GitHub token
into batch scripts or repository files.

### Bitbucket API-token history

The AERO-F source was pushed to Bitbucket master through an API token because
the user signs in with Google and ordinary password authentication did not
work. That is separate from the public CRM GitHub repository. Do not print or
commit tokens; remote URLs containing a token should be treated as sensitive.

### Missing Eigen or Boost during an AERO-F build

The first build correctly discovered missing dependencies. Eigen 3.4.0 and
Boost 1.74.0 were provisioned under the private Sherlock dependencies area.
Do not confuse an initial CMake dependency failure with a solver failure.

### Boost extraction appears stalled

The `tar -xjf` process can show `D` state and low CPU while scratch I/O is
busy. Inspect process state and elapsed time before killing it. A long
extraction is not necessarily a dead build.

### Conda Terms of Service

Creating the Miniconda environment initially required accepting Anaconda
channel Terms of Service. That is a one-time environment setup issue, not a
ROM algorithm error.

### `rg` is not available on Sherlock

Use `grep` or `find` in commands intended for Sherlock. Do not assume ripgrep
is installed just because it is available locally.

### FEniCS says it cannot find DOLFIN/pkg-config

This usually means the FEniCS/Laplace environment was invoked outside the
correct module/conda setup. Re-establish the standard module environment and
the `CRM_Laplace` activation path used by the Laplace runner. Do not alter the
Laplace weak form or boundary tags as a first response.

### MPI mismatch between OpenMPI and conda MPI

`CRM_Laplace` may expose `mpiexec.hydra`, while the AERO-F workflow is built
against Sherlock OpenMPI. The intended AERO-F MPI command is `srun`; the
batch scripts explicitly export `MPI=srun`. Keep that choice.

### A pending Slurm job is not a failed job

`PD (None)` means it awaits resources. `PD (Dependency)` means the predecessor
has not completed. `PD (JobArrayTaskLimit)` means the array throttle is working.
Inspect `squeue`, then the designated output/error file after completion.

### A full PROM residual is not an HDM convergence tolerance

Do not require a PROM to reach `5e-7` just because an HDM does. The correct
question is whether the reduced solve behaves consistently and whether fields
and quantities of interest agree with independent HDM truth.

### Do not confuse a training check with a holdout

Running the PROM at `HDMrun001` tests plumbing and reconstruction at a sampled
point. It is not evidence of interpolation/generalization. A holdout must be
excluded from the state and parameter catalogs before its HDM is run.

### Do not confuse a display scale with a physical discrepancy

ParaView's automatic color range can make tiny error fields look dramatic or
hide meaningful wall/wake errors. Set meaningful shared ranges, inspect
individual components when necessary, and report numerical norms alongside
images.

## 19. Compact restart checklist

When returning after a break, follow this order:

```text
1. Read this document and CLEAN_LAPLACE_GREEDY.md.
2. Inspect local git status; preserve unrelated user modifications.
3. Make source/doc changes only locally.
4. Commit explicitly named files and push main.
5. On Sherlock, fetch and reset only the execution clone to origin/main.
6. Load the recorded compiler/MPI/conda environment.
7. Inspect the campaign manifest and status before submitting anything.
8. Submit one dependency chain only when its inputs and output root are clear.
9. Monitor Slurm and inspect specific artifacts on completion.
10. Validate physically against independent HDMs before escalating to HPROM
    or unsteady work.
```

The question to ask before every expensive command is simple: *which exact
campaign root will this write, which exact POD/catalog will it consume, and can
it overwrite or contaminate an existing result?* If the answer is not explicit,
stop and inspect before submitting.
