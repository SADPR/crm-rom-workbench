#!/usr/bin/env bash
# Export the unsteady fields written so far to one Exodus file per HDM. It only reads the
# running HDMs' result files, so the runs keep going.

#SBATCH --job-name=naca5d_exo
#SBATCH --partition=cfarhat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=Sobol5DRuns/logs/unsteady_exo_%j.out
#SBATCH --error=Sobol5DRuns/logs/unsteady_exo_%j.err

set -euo pipefail

work_dir=${SLURM_SUBMIT_DIR:?Submit this script from greedy-procedure/.}
cd "${work_dir}"
(( $# > 0 )) || {
    echo 'Usage: sbatch submit_sobol5d_unsteady_exo.sbatch HDM...' >&2
    exit 2
}

module purge
module load gcc/9.1.0 netcdf/4.4.1.1

source /scratch/users/sadpr/Code3Aug/miniconda3/etc/profile.d/conda.sh
conda activate GreedyAEROF

export SOWER=/home/groups/cfarhat/bin/sower
export XP2EXO=/home/groups/cfarhat/bin/xp2exo

status=0
for index in "$@"; do
    # One failed export must not hide the others.
    python3 -B sobol5d_unsteady.py exo --run-index "${index}" || {
        echo "HDM ${index}: export failed." >&2
        status=1
    }
done
exit "${status}"
