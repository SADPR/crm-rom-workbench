#!/usr/bin/env bash
# Run the legacy FEniCS Laplace solver for one deformed CRM mesh.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 DEFORMED_TOP OUTPUT_DIR MPI_RANKS" >&2
  exit 2
fi

top_file=$1
output_dir=$2
mpi_ranks=$3
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
solver=${CRM_LAPLACE_SOLVER:-"${script_dir}/AirfoilPoisson3D-clean.py"}
conda_base=${CRM_CONDA_BASE:-}
conda_env=${CRM_LAPLACE_ENV:-CRM_Laplace}

if [[ ! -f "${top_file}" ]]; then
  echo "Missing deformed mesh: ${top_file}" >&2
  exit 1
fi
if [[ ! "${mpi_ranks}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MPI_RANKS must be a positive integer: ${mpi_ranks}" >&2
  exit 2
fi
if [[ -z "${conda_base}" || ! -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
  echo "Set CRM_CONDA_BASE to the scratch Miniconda installation." >&2
  exit 1
fi
if [[ ! -f "${solver}" ]]; then
  echo "Missing Laplace solver: ${solver}" >&2
  exit 1
fi

mesh_name=$(basename -- "${top_file}")
mesh_stem=${mesh_name%.top}
expected_xpost="${output_dir}/${mesh_stem}-u_star.xpost"
if [[ -e "${expected_xpost}" ]]; then
  echo "Refusing to overwrite existing Laplace output: ${expected_xpost}" >&2
  exit 1
fi

mkdir -p "${output_dir}"
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${conda_env}"
trap 'conda deactivate >/dev/null 2>&1 || true' EXIT

python - <<'PY'
import fenics
import h5py
import meshio
try:
    import dolfin
    fenics_version = getattr(dolfin, "__version__", "available")
except ImportError:
    fenics_version = getattr(fenics, "__version__", "available")
print("FEniCS:", fenics_version)
print("h5py:", h5py.__version__)
print("meshio:", meshio.__version__)
PY

export OMPI_MCA_opal_cuda_support=0
start_time=$(date +%s)
mpirun -np "${mpi_ranks}" python "${solver}" "${top_file}" --output-dir "${output_dir}"
elapsed=$(( $(date +%s) - start_time ))

if [[ ! -s "${expected_xpost}" ]]; then
  echo "Laplace solver finished without ${expected_xpost}" >&2
  exit 1
fi

printf 'Laplace solve completed in %s s: %s\n' "${elapsed}" "${expected_xpost}"
