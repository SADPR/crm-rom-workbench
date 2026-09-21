#!/bin/bash
#
#################
#set a job name
#SBATCH --job-name=Laplace
#SBATCH --partition=cfarhat
#################
#time you think you need hh:mm::ss (On Sherlock cfarhat, max is 7 days)
#SBATCH --time=168:00:00
#################
#quality of service; think of it as job priority
#SBATCH --qos=normal
#################
#number of nodes you are requesting
#SBATCH --nodes=10
#################
#tasks to run per node; a "task" is usually mapped to a MPI processes.
# for local parallelism (OpenMP or threads), use "--ntasks-per-node=1 --cpus-per-task=16" instead
#SBATCH --ntasks-per-node=24
#################
#SBATCH --mail-type=END
#SBATCH --mail-user=zyh03@stanford.edu

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate GreedyAEROF
export MPI=mpirun
export MPIEXEC=mpirun

# export AEROF=/home/users/rtezaur/aero-f/bin/aerof.opt
# module load cmake/3.8.1 gcc/9.1.0 openmpi/4.1.2 imkl/2019

# export AEROF=/home/groups/cfarhat/bin/aerof2
# module load cmake/3.8.1 gcc/9.1.0 openmpi/4.0.3 imkl/2019

# export AEROF=/home/users/faisal3/codes/aero-f_new/bin/aerof.opt
# export AEROF=/home/users/zyh03/codes/aero-f_new/bin/aerof.opt
# module load cmake/3.8.1 gcc/9.1.0 openmpi/4.0.3 imkl/2019

export AEROF=/home/users/zyh03/codes/aero-f/build/bin/aerof.opt
module load cmake/3.8.1 gcc/10.1.0 openmpi/4.1.2 imkl/2019

export PARTMESH=/home/groups/cfarhat/bin/partnmesh
export SOWER=/home/groups/cfarhat/bin/sower
export CD2TET=/home/groups/cfarhat/bin/cd2tet

START="$(date +%s)"

python3 main.py > log.out

conda deactivate

DURATION=$[ $(date +%s) - ${START} ]
