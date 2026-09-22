# Steady Baseline Record

Status: in progress. No AERO-F simulation has been submitted from this
repository yet.

This record describes the supplied steady workflow before any unsteady
scientific changes are made.

## Repository state

- Local repository: `/home/sares/crm-rom-workbench`
- Sherlock execution clone:
  `/scratch/users/sadpr/Code3Aug/crm-rom-workbench`
- Imported baseline commit: `6ddfd91`
- Local mesh: a private, unversioned copy in the repository-root `mesh/`
  directory
- `greedy-procedure/mesh`: tracked relative link to `../mesh`

The delivered mesh under `/home/sares/CRM_tmp/mesh` is treated as read-only
source material. Runs and preprocessing must use the private copy so generated
partitions cannot alter the delivery.

Sherlock also contains a private, unversioned copy in the execution clone's
repository-root `mesh/` directory. The copy was verified through SSHFS using
the main topology checksum below. Neither working copy resolves into the
delivered `CRM_tmp/mesh` directory.

## Active case identity

Despite the repository's CRM working name, the active workflow is configured
for:

```text
mesh/naca0012_Re1p5.top
```

The deformation implementation is based on a four-digit NACA airfoil. The
intended meaning of "CRM" and whether this is the correct physical benchmark
must be confirmed before an expensive run.

### Mesh inventory

- Private mesh size: approximately 534 MiB
- Nodes file: 812,098 rows
- Fluid connectivity file: 2,428,200 rows
- Moving-wall connectivity file: 2,698 rows
- Main `.top` SHA-256:
  `3a409c6e4fa1f23521bb192609b3e8c3eab89f43d814beda15d1047d6cd1d8e5`
- Main topology sections: `FluidMesh_0`, `Symmetry_1`, `InletFixed_2`, and
  `StickMoving_3`

## Current parameter space

The five-component parameter vector is:

```text
[Mach, angle of attack, maximum-camber location, maximum camber, thickness]
```

The current bounds are:

| Parameter | Lower | Upper |
| --- | ---: | ---: |
| Mach | 0.4 | 0.6 |
| Angle of attack | -5 deg | 5 deg |
| Maximum-camber location | 0.3 | 0.6 |
| Maximum camber | 0.03 | 0.03 |
| Thickness | 0.12 | 0.12 |

The proposed nominal baseline point is the center of the active parameter
space:

```text
[0.5, 0.0, 0.45, 0.03, 0.12]
```

This point is proposed for a controlled one-case comparison. It is not the
first point selected by the current Sobol initialization. The supplied greedy
configuration requests 32 initial points, beginning with the corners of the
three active dimensions.

## Current steady HDM configuration

- Problem type: `Steady`
- Form: `NonDescriptor`
- First solve:
  - maximum iterations: 1,500
  - tolerance: `1e-4`
  - matrix-vector product: `Approximate`
- Restarted solve:
  - maximum iterations: 9,000
  - tolerance: `5e-7`
  - matrix-vector product: `FiniteDifference`
- Newton iterations per pseudo-time step: 1
- Snapshot frequency: 0
- Selected steady snapshot index: 2
- HDM mesh partitions: 120
- HDM MPI processes: 120

The current workflow runs the two steady solves sequentially and records one
state from the resulting snapshot file.

## Current CFD configuration

- Dimensional formulation
- Inlet pressure: 22,632
- Inlet density: 0.3639
- Flux: Roe
- Limiter: Venkatakrishnan
- Spatial operator: finite volume
- Turbulence closure: Spalart-Allmaras
- The mesh is oriented so the parameter called angle of attack is written to
  AERO-F as inlet `Beta`; inlet `Alpha` is zero.

## Sherlock execution configuration requiring replacement

The delivered `greedy-procedure/run.sh` must not be submitted unchanged. It
contains another user's:

- AERO-F executable:
  `/home/users/zyh03/codes/aero-f/build/bin/aerof.opt`
- email address,
- Conda environment assumptions.

It requests 10 nodes with 24 tasks per node, while an individual HDM uses 120
MPI processes. The isolated baseline instead requests exactly 120 tasks as
five nodes with 24 tasks per node.

## Sherlock AERO-F core build

The HDM baseline executable was built successfully on Sherlock:

```text
/scratch/users/sadpr/Code3Aug/aero-f/build-core/bin/aerof.opt
```

Build record:

- AERO-F commit: `069ef9d8e904746652d94d062a5958e77e0686df`
- Build type: `Release`
- GCC: 10.1.0
- OpenMPI: 4.1.2
- CMake: 3.24.2
- Eigen: 3.4.0, commit `3147391d946bb4b6c68edd901f2add6ac1f31f8c`
- Boost: 1.74.0, archive SHA-256
  `83bfc1507731a0906e387fc28b7ef5417d591429e51e788417fe9ff025e116b1`
- `USE_MARCH_NATIVE=OFF`
- `WITH_TORCH=OFF`
- `WITH_SCALAPACK=OFF`
- `WITH_BLACS=OFF`
- `WITH_ARPACK=OFF`
- `WITH_PARMETIS=OFF`
- Executable size: approximately 26 MiB
- Missing dynamic libraries under the loaded build modules: none

This deliberately minimal build is sufficient for the steady and unsteady
HDM validation stages. A separate ROM-capable build with ScaLAPACK and a
separate Torch-enabled build will be configured only after the HDM baseline
works.

The delivered general-manifold build was also inspected. Its reliable
Sherlock configuration used GCC 10.3.0, OpenMPI 4.1.2, Eigen 3.4.0, Boost
1.64.0, and enabled ScaLAPACK, BLACS, ARPACK, and ParMETIS. Builds whose
caches mixed OpenMPI 3.1.2 and 4.1.2 will not be used as references.

## Baseline execution rule

Do not run `python3 main.py` or submit the delivered `run.sh` merely to obtain
the baseline. With the current settings, that starts the full greedy campaign,
including many initial HDMs.

The repository includes `greedy-procedure/baseline_steady.py`, an isolated
one-point driver with two explicit modes:

- `prepare` performs the mesh preprocessing and generates the two HDM inputs
  without invoking AERO-F.
- `run` executes only those already-reviewed inputs.

It uses a separate `BaselineRuns/` directory and never updates the production
greedy snapshot catalogs.

Submit `greedy-procedure/submit_baseline_steady.sbatch` from the
`greedy-procedure/` directory only after preparation succeeds. It records the
Slurm allocation, loaded modules, Python version, executable path, and Git
revision in `BaselineRuns/job_<job-id>.env`, then uses `srun` for the two HDM
steps. Its logs remain with the baseline outputs and are ignored by Git.

The controlled point is:

```text
[0.5, 0.0, 0.45, 0.03, 0.12]
```

That case must use its own output directory and must not update the production
greedy snapshot catalogs.

## Baseline acceptance checks

- [x] Preserve the imported Git revision.
- [x] Create a private local mesh copy.
- [x] Verify the main topology checksum against the delivered copy.
- [x] Create and verify a private mesh copy on Sherlock.
- [ ] Confirm that `naca0012_Re1p5` is the intended case.
- [x] Build and record our core AERO-F executable on Sherlock.
- [ ] Generate and review one isolated nominal steady HDM input.
- [ ] Run the nominal case to the requested steady tolerance.
- [ ] Record the generated input, residual history, force history, restart,
      final solution, allocation, and wall time.
- [ ] Repeat or restart the run to confirm reproducibility.
