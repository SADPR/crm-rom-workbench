#!/bin/bash
#SBATCH --job-name=crm_state_audit
#SBATCH --partition=cfarhat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --output=GreedyRuns/prom_state_audit_%j.out
#SBATCH --error=GreedyRuns/prom_state_audit_%j.err

set -euo pipefail

module purge
module load gcc/10.1.0

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

export SOWER=/home/groups/cfarhat/bin/sower

python3 -B audit_initial_prom_state.py
