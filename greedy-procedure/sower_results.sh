#!/bin/bash

# Sower executable
SOWER_EXECUTABLE=/home/groups/cfarhat/bin/sower

module load gcc/9.1.0 netcdf/4.4.1.1

CON=GreedyRuns/data/fluidmodel.con
MSH=GreedyRuns/data/fluidmodel.msh
RESULTS_DIR=GreedyRuns/evaluate/hromruns004/point001/results
POSTPRO_DIR=GreedyRuns/evaluate/hromruns004/point001/postpro

# RESULTS_DIR=InitialHDMruns/HDMrun001/results
# POSTPRO_DIR=InitialHDMruns/HDMrun001/postpro

# Postprocess fluid solution
$SOWER_EXECUTABLE -fluid -merge -con $CON -mesh $MSH \
	-result $RESULTS_DIR/Mach.bin -output $POSTPRO_DIR/Mach

$SOWER_EXECUTABLE -fluid -merge -con $CON -mesh $MSH \
        -result $RESULTS_DIR/PressureCoefficient.bin -output $POSTPRO_DIR/PressureCoefficient

$SOWER_EXECUTABLE -fluid -merge -con $CON -mesh $MSH \
        -result $RESULTS_DIR/SkinFriction.bin -output $POSTPRO_DIR/SkinFriction

$SOWER_EXECUTABLE -fluid -merge -con $CON -mesh $MSH \
        -result $RESULTS_DIR/Displacement.bin -output $POSTPRO_DIR/Displacement
