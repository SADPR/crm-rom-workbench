#!/bin/bash
module load netcdf/4.4.1.1
module load gcc/9.1.0

NS=120

XP2EXO_EXECUTABLE=/home/groups/cfarhat/bin/xp2exo

POSTPRO_DIR=GreedyRuns/evaluate/hromruns004/point001/postpro
DISP_DIR=GreedyRuns/evaluate/HDMrun001/postpro
# POSTPRO_DIR=GreedyRuns/evaluate/HDMrun001/postpro
# POSTPRO_DIR=InitialHDMruns/HDMrun001/postpro

# Convert fluid outputs to .exo format
$XP2EXO_EXECUTABLE mesh/naca0012_Re1p5.top $POSTPRO_DIR/fluid_solution.exo \
		    mesh/naca0012_Re1p5.top.dec.$NS \
	        $POSTPRO_DIR/Mach.xpost \
	        $POSTPRO_DIR/PressureCoefficient.xpost \
            $POSTPRO_DIR/SkinFriction.xpost \
			$DISP_DIR/Displacement.xpost \
