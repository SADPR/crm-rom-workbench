CRM geometry-specific Laplace shift
===================================

The legacy FEniCS solver is included as AirfoilPoisson3D-clean.py.  The CRM
workflow uses ShiftType = 'Laplace' and LaplaceShiftEach = True, so each
deformed airfoil requires its own unit Laplace field.

On Sherlock, create a separate environment once.  This intentionally stays
separate from GreedyAEROF, which runs the workflow and AERO-F:

  export CRM_CONDA_BASE=/scratch/users/sadpr/Code3Aug/miniconda3
  source "${CRM_CONDA_BASE}/etc/profile.d/conda.sh"
  conda create -y -n CRM_Laplace -c conda-forge \
      python=3.12 fenics=2019.1.0 meshio h5py mpi4py

To measure the nominal shift without touching its converged HDM outputs:

  cd /scratch/users/sadpr/Code3Aug/crm-rom-workbench/greedy-procedure
  conda activate GreedyAEROF
  export CRM_CONDA_BASE=/scratch/users/sadpr/Code3Aug/miniconda3
  export CRM_LAPLACE_ENV=CRM_Laplace
  python3 -B benchmark_laplace_shift.py prepare
  python3 -B benchmark_laplace_shift.py run

The benchmark writes only below BaselineRuns/HDMrun001/laplace/.  It creates
the deformed .top file, solves the FEniCS problem on 24 ranks, converts the
result to a six-component SA shift, and partitions it as ushift.bin001 ...
ushift.bin120.  It refuses to overwrite an existing benchmark.

After the benchmark succeeds, prepare a one-iteration AERO-F read check:

  python3 -B benchmark_laplace_shift.py prepare-aerof-check
  srun -N 5 -n 120 "${AEROF}" \
      BaselineRuns/HDMrun001/laplace/aerof-check/input_laplace \
      > BaselineRuns/HDMrun001/laplace/aerof-check/log 2>&1

The check starts from the converged nominal solution, writes only below
laplace/aerof-check/, and must report that it loaded the Laplace unit solution.

run_laplace_shift.sh is the common launcher used by the benchmark and the
HDM workflow.  It receives DEFORMED_TOP OUTPUT_DIR MPI_RANKS and requires
CRM_CONDA_BASE.  Override CRM_LAPLACE_ENV or CRM_LAPLACE_SOLVER only when a
different environment or solver file is deliberate.
