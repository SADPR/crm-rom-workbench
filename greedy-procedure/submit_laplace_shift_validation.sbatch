#!/bin/bash
#SBATCH --job-name=crm_lap_check
#SBATCH --partition=cfarhat
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --time=00:15:00
#SBATCH --output=GreedyRuns/laplace_shift_validation_%j.out
#SBATCH --error=GreedyRuns/laplace_shift_validation_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this script from greedy-procedure}"

top_file=GreedyRuns/HDMrun001/naca0012_Re1p5_deformed.top
output_dir=CRMDiagnostics/laplace-boundary-validation/xdmf_files
u_star="${output_dir}/naca0012_Re1p5_deformed-u_star.xpost"

test -f "${top_file}"
test ! -e "${u_star}"

export CRM_CONDA_BASE=/scratch/users/sadpr/Code3Aug/miniconda3
export CRM_LAPLACE_ENV=CRM_Laplace

./run_laplace_shift.sh "${top_file}" "${output_dir}" 24

awk 'NR == 4 { min = max = $1 } NR >= 4 { min = ($1 < min ? $1 : min); max = ($1 > max ? $1 : max); sum += $1; count += 1 } END { printf "u_star: nodes=%d min=%.16e max=%.16e mean=%.16e\\n", count, min, max, sum / count }' "${u_star}"
