#!/bin/bash
#
#################
#set a job name
#SBATCH --job-name=Laplace3D
#SBATCH --partition=cfarhat
#################
#time you think you need hh:mm::ss (On Sherlock cfarhat, max is 7 days)
#SBATCH --time=10:00:00
#################
#quality of service; think of it as job priority
#SBATCH --qos=normal
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################
#tasks to run per node; a "task" is usually mapped to a MPI processes.
# for local parallelism (OpenMP or threads), use "--ntasks-per-node=1 --cpus-per-task=16" instead
#SBATCH --ntasks-per-node=24
#################
#SBATCH --mail-type=END
#SBATCH --mail-user=zyh03@stanford.edu

#SBATCH --output=log.out
#SBATCH --error=log.out

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate fenics

module load openmpi
# mpirun -np 8 --mca opal_cuda_support 0 python3 AirfoilPoisson3D-clean.py mesh/ascii/fluid.top
export OMPI_MCA_opal_cuda_support=0
srun -n 24 python3 AirfoilPoisson3D-clean.py mesh/ascii/fluid.top

conda deactivate
