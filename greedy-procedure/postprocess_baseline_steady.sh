#!/usr/bin/env bash
# Merge the isolated steady-baseline fields and write an Exodus file.
# Run on a Sherlock compute node from the greedy-procedure directory.

set -euo pipefail

work_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${work_dir}"

sower_executable=${SOWER_EXECUTABLE:-/home/groups/cfarhat/bin/sower}
xp2exo_executable=${XP2EXO_EXECUTABLE:-/home/groups/cfarhat/bin/xp2exo}

top_file=mesh/naca0012_Re1p5.top
decomposition_file=${top_file}.dec.120
mesh_prefix=BaselineRuns/data/fluidmodel
results_dir=BaselineRuns/HDMrun001/results
postpro_dir=BaselineRuns/HDMrun001/postpro
exo_file=${postpro_dir}/fluid_solution.exo

required_files=(
    "${top_file}"
    "${decomposition_file}"
    "${mesh_prefix}.con"
    "${mesh_prefix}.msh001"
    "${results_dir}/Mach.bin001"
    "${results_dir}/PressureCoefficient.bin001"
    "${results_dir}/SkinFriction.bin001"
    "${results_dir}/Velocity.bin001"
    "${results_dir}/Displacement.bin001"
)

for required_file in "${required_files[@]}"; do
    [[ -f "${required_file}" ]] || {
        echo "Missing required file: ${required_file}" >&2
        exit 1
    }
done

[[ -x "${sower_executable}" ]] || {
    echo "Missing SOWER executable: ${sower_executable}" >&2
    exit 1
}
[[ -x "${xp2exo_executable}" ]] || {
    echo "Missing xp2exo executable: ${xp2exo_executable}" >&2
    exit 1
}

module purge
module load gcc/9.1.0 netcdf/4.4.1.1

mkdir -p "${postpro_dir}"

merge_field() {
    local field=$1
    local output_prefix=${postpro_dir}/${field}
    local output_file=${output_prefix}.xpost

    if [[ -s "${output_file}" ]]; then
        echo "Using existing ${output_file}"
        return
    fi

    echo "Merging ${field}..."
    "${sower_executable}" -fluid -merge \
        -con "${mesh_prefix}.con" \
        -mesh "${mesh_prefix}.msh" \
        -result "${results_dir}/${field}.bin" \
        -name "${field}" \
        -out "${output_prefix}" \
        -width 16 \
        -precision 16

    [[ -s "${output_file}" ]] || {
        echo "SOWER did not create ${output_file}" >&2
        exit 1
    }
}

merge_field Mach
merge_field PressureCoefficient
merge_field SkinFriction
merge_field Velocity
merge_field Displacement

shopt -s nullglob
existing_exodus=("${exo_file}"*)
if (( ${#existing_exodus[@]} > 0 )); then
    echo "Using existing Exodus output:"
    ls -lh "${existing_exodus[@]}"
    exit 0
fi

echo "Writing ${exo_file}..."
"${xp2exo_executable}" \
    "${top_file}" \
    "${exo_file}" \
    "${decomposition_file}" \
    "${postpro_dir}/Mach.xpost" \
    "${postpro_dir}/PressureCoefficient.xpost" \
    "${postpro_dir}/SkinFriction.xpost" \
    "${postpro_dir}/Velocity.xpost" \
    "${postpro_dir}/Displacement.xpost"

exodus_outputs=("${exo_file}"*)
if (( ${#exodus_outputs[@]} == 0 )); then
    echo "xp2exo completed but created no ${exo_file} output." >&2
    exit 1
fi

echo "Baseline postprocessing completed:"
ls -lh "${postpro_dir}"/*.xpost "${exodus_outputs[@]}"
