#!/usr/bin/env bash
# Build an AERO-F variant (a git worktree carrying patches) with the production configuration.
# The production build's CMakeCache.txt is copied and only the entries naming its source or
# build directory are remapped, so compilers, options and dependency paths stay identical.
# The production tree and binary are only read.

#SBATCH --job-name=naca5d_aerof_build
#SBATCH --partition=cfarhat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=03:00:00
#SBATCH --output=Sobol5DRuns/logs/aerof_build_%j.out
#SBATCH --error=Sobol5DRuns/logs/aerof_build_%j.err

set -euo pipefail

source_dir=${1:?Usage: sbatch build_aerof_variant.sbatch SOURCE_DIR [PRODUCTION_DIR]}
production=${2:-/scratch/users/sadpr/Code3Aug/aero-f}
production_build=${production}/build-scalapack
build=${source_dir}/build-scalapack

[[ -f "${production_build}/CMakeCache.txt" ]] || {
    echo "Missing ${production_build}/CMakeCache.txt." >&2
    exit 1
}
[[ "$(realpath "${source_dir}")" != "$(realpath "${production}")" ]] || {
    echo "Refusing to build inside the production tree." >&2
    exit 1
}
[[ -d "${source_dir}/.git" ]] || {
    echo "${source_dir} is not a git clone." >&2
    exit 1
}
[[ ! -e "${build}" ]] || {
    echo "${build} exists; refusing to overwrite a build." >&2
    exit 1
}

# The modules of the production jobs, which provide the cached compilers and MKL.
module purge
module load cmake/3.24.2 gcc/10.1.0 openmpi/4.1.2 imkl/2019

mkdir -p "${build}"
# Every reference into the production build directory moves to the new one; the source
# directory changes only in its own entries, so dependency paths are left untouched.
awk -v ps="${production}" -v pb="${production_build}" -v ns="${source_dir}" -v nb="${build}" '
    function literal(s, a, b,    out, i) {
        out = ""
        while ((i = index(s, a)) > 0) { out = out substr(s, 1, i - 1) b; s = substr(s, i + length(a)) }
        return out s
    }
    { line = literal($0, pb, nb) }
    line ~ /^(CMAKE_HOME_DIRECTORY|[A-Za-z0-9_.+-]+_SOURCE_DIR)(:|=)/ { line = literal(line, ps, ns) }
    { print line }' "${production_build}/CMakeCache.txt" > "${build}/CMakeCache.txt"
echo "Remapped cache entries:"
diff "${production_build}/CMakeCache.txt" "${build}/CMakeCache.txt" || true

cmake -S "${source_dir}" -B "${build}"
cmake --build "${build}" -j "${SLURM_CPUS_PER_TASK}"

new=$(find "${build}/bin" -name aerof.opt -type f)
old=${production_build}/bin/aerof.opt
echo "Built ${new}"
# Sherlock's git predates "git -C", so change directory instead.
echo "Source: $(cd "${source_dir}" && git log --oneline -1)"
echo "Base:   $(cd "${source_dir}" && git log --oneline -1 HEAD~1)"
# Same configuration, same linked BLAS/ScaLAPACK: both must list the same libraries.
echo "Linked libraries that differ from production (expected: none):"
diff <(ldd "${old}" | awk '{print $1}' | sort) <(ldd "${new}" | awk '{print $1}' | sort) && echo none
