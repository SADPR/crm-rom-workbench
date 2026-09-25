CRM geometry-specific Laplace shift
===================================

The legacy FEniCS solver is included as AirfoilPoisson3D-clean.py.  The CRM
workflow uses ShiftType = 'Laplace' and LaplaceShiftEach = True, so each
deformed airfoil requires its own unit Laplace field.

The solver sets boundary values by group name: InletFixed_2 = 1,
StickMoving_3 = 0, and Symmetry_1 natural.  Symmetry_1 touches every vertex of
the one-layer extrusion, so a Dirichlet value there makes the field constant.
One solve takes about 45-55 s on 24 ranks.

On Sherlock, create a separate environment once.  This intentionally stays
separate from GreedyAEROF, which runs the workflow and AERO-F:

  export CRM_CONDA_BASE=/scratch/users/sadpr/Code3Aug/miniconda3
  source "${CRM_CONDA_BASE}/etc/profile.d/conda.sh"
  conda create -y -n CRM_Laplace -c conda-forge \
      python=3.12 fenics=2019.1.0 meshio h5py mpi4py

run_laplace_shift.sh is the launcher used by the HDM and PROM drivers.  It receives DEFORMED_TOP OUTPUT_DIR MPI_RANKS and requires
CRM_CONDA_BASE.  Override CRM_LAPLACE_ENV or CRM_LAPLACE_SOLVER only when a
different environment or solver file is deliberate.
