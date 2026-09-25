#!/usr/bin/env bash
# Export field-by-field HDM/PROM comparisons for ParaView.

#SBATCH --job-name=crm_prom_fields
#SBATCH --partition=cfarhat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=CleanLaplaceRuns/prom_field_comparison_%j.out
#SBATCH --error=CleanLaplaceRuns/prom_field_comparison_%j.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"

module purge
module load gcc/9.1.0 netcdf/4.4.1.1

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

export SOWER=/home/groups/cfarhat/bin/sower
export XP2EXO=/home/groups/cfarhat/bin/xp2exo

python3 -B postprocess_clean_prom_comparison.py "$@"
