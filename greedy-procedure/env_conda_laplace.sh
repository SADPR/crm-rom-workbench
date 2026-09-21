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
