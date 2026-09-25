# CRM steady-ROM workflow: history, operating procedure, and current plan

This document is the operational memory for the CRM workbench. It records what
exists, why particular decisions were made, how a source-code change reaches
Sherlock, and how to distinguish an exploratory result from a result that may
be used by the current steady-ROM campaign.

It is intentionally more detailed than a normal README. The goal is that a
future version of us can restart safely without reconstructing decisions from
terminal history.

## 1. Current status at a glance

The objective is a **steady, global, nonlinear PROM** for a parametric NACA
airfoil problem, later an HPROM, local/nonlinear manifolds, and unsteady URANS.

On 2026-09-25 the target changed. Yihong Zhu, who built the original case,
confirmed that her final study, and the regime Farhat asked for, is a
**5D transonic box** (Section 7). The 3D subsonic box we had been running came
from the delivered `setup.py` and was never her final configuration.

What exists now:

- A validated pipeline: named-boundary Laplace shift, two-stage HDM,
  ScaLAPACK POD, Laplace/`NonDescriptor` LSPG PROM, and field comparison. It
  was exercised on 32 clean 3D subsonic HDMs plus one independent holdout
  (drag error 0.39%). That campaign is archived as pipeline validation
  (Section 10).
- A clean Sherlock tree: `greedy-procedure/` holds only source; all earlier
  generated data are in the backups area.

What comes next (approved plan, executed one phase at a time):

1. Confirm details with Yihong and compare her `xdmf-files` Laplace setup.
2. Write the 5D Sobol campaign driver and the test-set driver.
3. Pilot of 4 HDMs: the easiest, the two hardest, and the central point.
4. Training batches of 128 → 256 → 512 HDMs plus 32 independent test HDMs,
   with a go/no-go decision after each batch.

The residual greedy was dropped (Section 13). Submitting jobs on Sherlock
remains an explicit human action.

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
   generated directory is evidence until intentionally archived. On Sherlock,
   `git clean -fd` and `git stash -u` are just as dangerous for any result
   directory missing from `.gitignore`. Deleting is acceptable only as an
   explicit, listed step of an agreed plan, with a check before each removal
   (see the 2026-09-25 cleanup in Section 10).

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
    It records the full-order residual relative to the PROM's own initial
    (IDW) residual. LSPG minimizes the full residual norm over its subspace,
    but that minimum is not zero (Section 11). Physical field and force
    comparisons against independent HDMs remain necessary.

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
    Explicit archival area (not visible through the SSHFS mount).
    2026-09-24_pre-clean-laplace: the exploratory outputs (their only copy).
    2026-09-25_clean-laplace-3d-subsonic: the clean 3D campaign and the
    metadata of the cancelled greedy.
    Both live on scratch, which Sherlock purges after 90 days without
    modification; copy anything that must last to Oak.

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

### Yihong Zhu's reference data (read-only)

```text
/home/users/zyh03/xdmf-files
    Laplace inputs her runs used; she asked us to copy it into the project.
/oak/stanford/groups/cfarhat/zyh03/
/scratch/users/zyh03/FinalSnappingPaper
    Her final 5D runs and error reports, where readable.
```

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
cp /home/sares/Sherlock_CRM/greedy-procedure/<campaign-root>/<run>/field_comparison/prom_flow_fields.exo \
   /tmp/crm-view/
```

The backups area is outside the mounted directory, so archived campaigns are
only reachable from a Sherlock shell.

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

The five-component parameter vector is:

```text
[Mach, angle of attack, maximum camber location, maximum camber, thickness]
```

The target box is Yihong's final study, in which all five parameters vary:

```text
Mach                  [0.6, 0.8]      transonic: shocks expected
angle of attack       [-2, 2] degrees
maximum camber loc.   [0.2, 0.4] chord fractions
maximum camber        [0, 0.03]
thickness             [0.09, 0.12]
```

Yihong also recommends `Beta = 0.5` instead of the delivered 1/3 so that
transonic HDMs converge cleanly. `settings.Beta` is the reconstruction
parameter of both the flow and turbulence `Space` blocks, not the inlet
`Beta` angle, so it defines the HDM and PROM operators alike. Set it once in
the campaign settings.

With `include_corners=True`, `sobolGenerator` places all 2^5 = 32 corners
first, followed by unscrambled Sobol points, so growing N keeps every earlier
point. With maximum camber 0 the camber location has no geometric effect
(`deform_naca.py`), so 8 corners duplicate geometry; we accept them to keep
Yihong's generator unchanged. AERO-F normalizes each parameter to [0, 1]
before computing IDW distances, so the different parameter scales are safe.

The earlier 3D validation box was Mach [0.4, 0.6], angle [-5, 5] degrees,
camber location [0.3, 0.6], with camber 0.03 and thickness 0.12 fixed.

The Sobol generator prints a warning when its requested sample count is not a
power of two. That warning describes Sobol balance quality; it does not
invalidate the points or mean the runs failed.

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

An independent check on the holdout geometry confirmed the node mapping: in
AERO-F node order, u* is exactly 1 on all 2698 `InletFixed_2` nodes and exactly
0 on all 2698 `StickMoving_3` nodes. One solve takes about 45–55 s on 24 ranks.

Yihong's runs used a folder `/home/users/zyh03/xdmf-files`. Before the 5D
campaign, compare its facet tags and Laplace field with this named-boundary
setup on the same mesh, and keep ours unless they differ.

## 10. Campaign history and where the data are now

The directory name tells us which results may be used for what. On 2026-09-25
the Sherlock tree was cleaned as an explicit plan step: each generated
directory below was archived, or deleted only after its file list matched a
backup.

### A. Exploratory campaign (Sept 22–24): never training data

`GreedyRuns/`, `InitialHDMruns/`, `BaselineRuns/`, `CRMDiagnostics/`. They
served to build the environment and AERO-F, exercise the two HDM stages, and
discover the constant-Laplace error. Their only copy is now
`crm-rom-workbench-backups/2026-09-24_pre-clean-laplace` (36 GB, with the
source commit and mesh checksum); the in-tree duplicates were deleted.

### B. Clean 3D subsonic campaign (Sept 24–25): pipeline validation

Archived by `mv` to
`crm-rom-workbench-backups/2026-09-25_clean-laplace-3d-subsonic/`:

- `CleanLaplaceRuns/`: 32 audited HDMs (final residuals 4.9836e-07 to
  4.9993e-07, nonconstant shifts), their catalogs, the 32-state
  `ScalapackSVD` POD (`reductionrun032/`), and the PROM runs;
- `CleanLaplacePrecompute/` and `CleanLaplaceBaseline/` (nominal inspection
  case);
- `CleanLaplaceHoldout/`: the independent holdout HDM and its comparison.

Catalog paths inside the archive are relative to that directory.

Training-point check (`HDMrun001`): drag 0.049% and lift 0.073% error. It
tests plumbing, not generalization.

Independent holdout at Sobol point 33, `[0.53125, 1.5625, 0.309375, 0.03,
0.12]`, excluded from the POD, with identical geometry and a Laplace
difference of 7.9e-12:

```text
HDM final residual       4.989276e-07
PROM outer iterations    30
PROM full residual       1.682292e-01 (relative)
drag relative error      0.387%
lift relative error      0.0423%
Mach / velocity L2       0.368% / 0.370%
Cp / skin friction L2    1.785% / 1.660%
```

The Cp and skin-friction numbers are L2 over all mesh nodes. They are not
Yihong's surface-curve metric (Section 14), so they cannot be compared with
her ~10% Cp error.

The drivers for this campaign were removed from HEAD on 2026-09-25 and remain
in git history. `clean_laplace_initial.py` stays until the 5D driver absorbs
its helpers.

### C. Cancelled greedy 32→40 (Sept 25)

Iteration 33 was cancelled during its screen. Its small metadata (catalogs,
candidates, screen and PROM logs) are in the 3D archive under
`CleanLaplaceGreedy40-metadata/`; the root itself was deleted. Commit 0a72f33
(absolute-residual selection) is the last version of that code.

## 11. PROM residuals, flux residuals, and what to compare

There are several quantities called "residual" in the workflow. They must not
be conflated.

### `postpro/Residual.out`

This is the full-order residual norm at each iteration divided by its value at
iteration 0, which the log prints as `Spatial residual norm`.

- **HDM:** iteration 0 is the freestream start, so the ratio is the
  convergence measure behind `HDMtol2`.
- **PROM:** iteration 0 is the IDW initial guess. LSPG minimizes the full
  residual norm over its subspace, but that minimum is not zero. At a training
  point the IDW guess is already the training state, so the ratio stays near 1
  (0.969 at training point 001).
- **Across parameter points:** only the absolute value (ratio × initial norm)
  is comparable. Even at a training point it has a floor of about 2×10^4,
  whereas converged HDMs sit at 35–61 absolute. The floor is unexplained; one
  candidate is `PreComputeLimiter = On` in HDM stage 2 versus `Off` in the
  PROM.
- **Stagnation:** in the v1 PROMs the ratio barely moved after the first outer
  iteration, while the CFL fell from 5 to between 1e-2 and 2e-4. The 30
  iterations are the `MaxIts` cap, not convergence.

Do not use it by itself as a physical-error metric.

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
difference, and their nodal flux-residual fields. For the 3D validation they
are in the archive, under:

```text
crm-rom-workbench-backups/2026-09-25_clean-laplace-3d-subsonic/
    CleanLaplaceRuns/evaluate/romruns032/point001/postpro/field_comparison/
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
absence of a shock is expected. In the 5D transonic box, check for a region
with Mach above one and a jump in the surface Cp.

## 13. Why the residual greedy was dropped

The clean greedy screened 24 candidate points with full PROMs before each
added HDM. With the measured v1 costs:

```text
one screen PROM     ~17 min on 5 nodes  ≈ 1.4 node-h
24 screens                              ≈ 34 node-h
selected HDM        ~35 min on 5 nodes  ≈ 2.9 node-h
per added snapshot                      ≈ 37 node-h
```

A Sobol HDM costs about 2.9 node-h, so each greedy snapshot costs about 13
Sobol snapshots, and the greedy is sequential. Yihong's greedy reached only
200 samples for the same reason. Sobol campaigns run as Slurm arrays, and the
design is nested, so batches can grow without recomputing earlier HDMs.

The first greedy version also ranked candidates by the relative `Residual.out`
value. That value is normalized by each candidate's own IDW initial residual,
which inverts the ranking: training point 001 reports 0.969 against 0.168 at
holdout 33, while their absolute residuals are 1.86e4 and 4.47e4. The
delivered greedy (`runs.py` `getRes`) uses the absolute value. Any future
greedy should rank by the absolute residual and use a cheap (HPROM)
indicator.

## 14. 5D validation plan

- **Test set:** 32 points from an independent low-discrepancy sequence
  (scrambled Sobol, fixed seed), disjoint from the training points. Each gets
  its own HDM outside the training root and is never added to a catalog.
- **Evaluation:** the full PROM of every training batch (128, 256, 512) at
  every test point.
- **Metrics:**
  - Yihong's surface Cp error, `ROMerror.getCp` + `cpL2error`: L2 over the
    z = 0 wall nodes, top and bottom sorted by x, in percent;
  - the same L2 for skin friction;
  - drag and lift relative error;
  - field L2 from `postprocess_clean_prom_comparison.py`;
  - a snapping flag as in `checkSnapping.py`.
- **Report:** mean, median and maximum per batch, and the number of snapped or
  failed points. They are never excluded silently.
- **Reference:** Yihong's Laplace-affine 5D greedy (her Fig. 3) gave a mean
  surface-Cp error of 21% at 32 samples and 9–10% at 50–200, with maxima of
  43–55%.

## 15. HPROM comes after the 5D global PROM is trusted

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

## 17. Yihong's answers (2026-09-25) and open questions

Her answers:

1. She varied all five parameters, with the ranges of Section 7. Farhat
   wanted the transonic regime, which she considers a bit aggressive.
2. Use `Beta = 0.5` instead of 1/3 for clean HDM convergence.
3. She used the Laplace shift (`ShiftVectorType = Laplace`, also during the
   HDM runs) and asked us to copy `/home/users/zyh03/xdmf-files`.
4. Her greedy reached about 200 samples with ~10% surface-Cp error. She
   recommends Sobol with 500–1000 points, at about 20 min per HDM on 5 nodes.

Open questions for her:

1. The path to her final 5D run, with `settings.readonly` and
   `SampledPointsOutput.txt`.
2. Whether her curves come from `ROMerror.py` (L2) or
   `GreedyAlgorithm.saveResults` (L1), how many evaluation points she used,
   and whether she excluded snapped points.
3. What `xdmf-files` contains and how her code uses it.

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

### Pasting `set -euo pipefail` into an interactive shell

It stays active in that shell, and the next failing command closes the
session, including an interactive `salloc` node. Wrap multi-line blocks in
`bash <<'EOF' ... EOF`, or run `set +euo pipefail` afterwards.

## 19. Compact restart checklist

When returning after a break, follow this order:

```text
1. Read this document.
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
