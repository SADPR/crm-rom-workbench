#!/bin/bash
#SBATCH --job-name=crm_lap_tags
#SBATCH --partition=cfarhat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=GreedyRuns/laplace_tag_audit_%j.out
#SBATCH --error=GreedyRuns/laplace_tag_audit_%j.err

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate CRM_Laplace
trap 'conda deactivate >/dev/null 2>&1 || true' EXIT

export OMPI_MCA_opal_cuda_support=0
unset PMI_FD PMI_RANK PMI_SIZE PMIX_NAMESPACE PMIX_RANK

"${CONDA_PREFIX}/bin/mpiexec.hydra" -launcher fork -n 1 \
  python3 -B audit_laplace_tags.py
