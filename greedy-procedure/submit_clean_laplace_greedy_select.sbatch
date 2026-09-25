#!/usr/bin/env bash
# Select the largest full-PROM residual after one complete screen array.

#SBATCH --job-name=crm_greedy_select
#SBATCH --partition=cfarhat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"
iteration=${GREEDY_ITERATION:?GREEDY_ITERATION is required.}

module purge
module load cmake/3.24.2 gcc/10.1.0 openmpi/4.1.2 imkl/2019

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

python3 -B clean_laplace_greedy.py select --iteration "${iteration}"
