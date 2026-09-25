#!/usr/bin/env bash
# Export the independent holdout HDM/PROM field comparison for ParaView.

#SBATCH --job-name=crm_holdout_fields
#SBATCH --partition=cfarhat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=clean_holdout_fields_%j.out
#SBATCH --error=clean_holdout_fields_%j.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"

module purge
module load gcc/9.1.0 netcdf/4.4.1.1

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

export SOWER=/home/groups/cfarhat/bin/sower
export XP2EXO=/home/groups/cfarhat/bin/xp2exo

python3 -B postprocess_clean_prom_comparison.py \
    --hdm-results CleanLaplaceHoldout/HDMrun001/results \
    --prom-results CleanLaplaceRuns/evaluate/romruns032/point901/results \
    --output-dir CleanLaplaceHoldout/field_comparison \
    --mesh-prefix CleanLaplaceRuns/data/fluidmodel
