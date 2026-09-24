import numpy as np
#*****************************************************************************#
#************************ USER INPUTS *****************************************
#*****************************************************************************#
class Settings:
    def __init__(self):
        ###############################################################################
        # Inputs related to the run type
        ###############################################################################
        
        # The directory where most files will be written
        self.MasterDir = 'GreedyRuns/'
        
        # Select a run to precompute the initial HDM solutions
        # True = I have already computed initial HDM runs
        # False = I have NOT computed initial HDM runs
        self.InitHDMruns = False
        
        # Specify that the HDM has been precomputed at initial seed points in this directory. 
        self.InitHDMPreCompDir = 'InitialHDMruns/'
        
        # Run a greedy procedure
        self.RunGreedy = True
        
        # Restart and continue from a previous greedy run
        self.Restart = False

        # Evaluate the final ROM at random points
        self.EvalROM = False
        
        
        
        ###############################################################################
        # Inputs related to building ROM
        ###############################################################################
        
        # Number of local ROMS
        self.NumClusters = 1
        
        # Percent overlap of local ROMS
        self.PercentOverlap = 10 
        
        # Snapshot referencing method. 'None' or 'Center' or Input'. 'Input' chooses first sample.
        self.RefMethod ='None'
        
        # Read the SnapIndex'th solution from the snapshot files
        self.SnapIndex = 2
        
        # Type of projection
        self.Projection ='LeastSquaresPetrovGalerkin'
        
        # For ROB truncation
        # self.StateBasisMaxEnergy = 1-1e-9
        self.StateBasisMaxEnergy = 1.0
        
        # How many singular vectors to keep
        self.StateBasisMaxDim = 1000
        
        # Use RSVD instead of ScalapackSVD because Sherlock AERO-F lacks ScaLAPACK.
        self.PODMethod = 'RSVD'

        # Whether or not to use RBF
        self.rbf = False
        self.rbf_dimVmax = 50 # maximum dim(V), rest is dim(\bar V)
        self.rbf_pctVdim = 0.95 # percentage of dim(V_total) to use as dim(V), the rest is dim(\bar V)

        # Options: 'None', 'FreeStream', 'FreeStreamBC', 'Laplace'
        self.ShiftType = 'Laplace'
        # Use the Laplace field of each deformed airfoil for its HDM/ROM state.
        self.LaplaceShiftEach = True
        self.LaplaceNumProc = 24 # Number of processors to use for solving each Laplace problem for the shift
        
        ###############################################################################
        # Inputs related to building reduced mesh
        ###############################################################################
        
        # If True, each snapshot is given equal importance
        self.UseRowScaling = True
        
        # Sample each type of BC seperately
        if self.ShiftType != 'None': 
            self.DecompBC = True
        else: 
            self.DecompBC = False
        
        # Use the training tolerance for each type of BC seperately
        if self.DecompBC is True: 
            self.UseGT = True
        else: 
            self.UseGT = False
        
        # Scale columns of training matrix (not recommended)
        self.ColumnScaling = False
        
        # If this is chosen, we simply train the ECSW on the centroid of the parameter space
        # and avoid writing to disk (less accurate)
        self.TrainOnCentroid = False
        
        # ECSW sampling tolerance
        self.SamplingTol = 1e-3
        
        # Leniency factor for retraining mesh
        self.Ell = 10
        
        # Tolerance for skipping retraining
        self.Delta = self.Ell*self.SamplingTol
        
        # Whether or not to hotstart from previous weights
        self.HotStart = False
        
        # Train on reduced jacobian or reduced residual
        self.TrainingData ='Jacobian'
        
        # For the skipping check, either reduced 'Residual' or reduced 'Jacobian' is checked
        self.TestType = 'Jacobian'
        
        # Where to check hyperreduction accuracy. Either 'Fist', 'Latest' or 'Farthest'
        self.MuStarType = 'Latest'
        
        # Test the accuracy of the hyperreduction at each candidate point (only for debugging/testing purposes)
        self.DoComp = False
        
        # Maximum number of samples to use for training matrix
        self.MaxNumTrainSnaps = 1000
       
        # Maximum number of components of the residual to consider for hyperreduction training
        self.MaxNumResComps = 25
        
        ###############################################################################
        # Inputs related to greedy procedure
        ###############################################################################
        
        # Lower bounds of Mach, AoA, MaxCamberLoc, MaxCamber, Thickness
        self.ParamsLowerBound = [0.4, -5, 0.3, 0.03, 0.12]
        #self.ParamsLowerBound = [0.4,-5, 0.2, 0.0 , 0.05] #[0.45, -2.5, 0.25, 0.0, 0.075]
        #self.ParamsLowerBound = [0.45, -2.5, 0.25, 0.0, 0.075]
        #self.ParamsLowerBound = [0.4, -5, 0.3, 0.0, 0.12]

        # Upper bounds of Mach, AoA, MaxCamberLoc, MaxCamber, Thickness
        self.ParamsUpperBound = [0.6, 5, 0.6, 0.03, 0.12]
        #self.ParamsUpperBound = [0.6, 5, 0.4, 0.05, 0.15] #[0.55, 2.5, 0.35, 0.025, 0.125] #[0.6, 5, 0.4, 0.05, 0.15]
        #self.ParamsUpperBound = [0.55, 2.5, 0.35, 0.025, 0.125] #[0.6, 5, 0.4, 0.05, 0.15]
        #self.ParamsUpperBound = [0.6, 5, 0.3, 0.0, 0.12]

        # How many points to discretize the parameter space in each direction (orthogonal grid).
        # Should be odd numbers for a well-defined center
        numpointsperdim = 5
        self.NumPointsPerDim = np.ones(len(self.ParamsLowerBound))
        self.NumPointsPerDim[np.nonzero(np.array(self.ParamsUpperBound) - np.array(self.ParamsLowerBound))[0]] = numpointsperdim

        self.NumPointsPerDim = list(map(int, self.NumPointsPerDim)) 
         
        # How many candidate points to generate and check per greedy iterations
        self.NumCandsPerIter = 125 # 500 was default
        
        # How many points to initialize with. Center, Corners, or number for random, or 'Corners+Random' for both
        self.NumInit = 'Sobol' # 'Corners' was default
        self.numrand = 32 # number of random points in addition to corners
        self.numSkip = 0 # number of points to skip in Sobol sequence to avoid duplicate generation of points from previous runs. Only used if NumInit=='Sobol'
        
        # Terminate when this many sampled collected
        self.MaxNumSamples = 40
        
        # Terminate when max error indicator hits this value
        self.MaxResNormTarget = 1e-8
        
        # Error indicator. Either HROM or ROM
        self.HyperReduced = False
        # temporary directory to store unit Laplace snapshots for error indicator runs. Overwritten for each EI run.
        self.EI_Laplacebin = 'tempEI_Laplace/'  

        # After how many samples to start clustering
        self.ClusteringStart = 50
        
        
        
        ###############################################################################
        # Inputs related to ROM online
        ###############################################################################
        
        # Form
        self.Form = 'NonDescriptor'
        
        # Maximum number of solutions to use for interpolation of IC
        self.MaxIntSols = 100
        
        # Distance exponent for interpolation of IC
        self.DistExp = 2 
       
        # Manually input linear ND interpolant weights
        self.LinearNDInterp = False

        # Prevents ROM simulation from crashing when pressure is negative. Make sure PressureCutOff is set.
        self.CheckPressure = 'On'
        
        # Prevents ROM simulation from crashing when density is negative. Make sure DensityCutOff is set.
        self.CheckDensity = 'On'
        
        # Numbe of newton iterations for HROM
        self.MaxItsHROM = 30
        
        # Number of newton iterations for ROM
        self.MaxItsROM = 30

        # Number of newton iterations for HROM
        self.MaxNewtItsHROM = 30

        # Number of newton iterations for ROM
        self.MaxNewtItsROM = 30

        # Jacobian for ROM
        self.MvpROM = 'Exact'
        
        # Component-wise scaling of residual for error indicator
        self.ComponentScalingIndicator = False

        # Component-wise scaling of residual
        self.ComponentScaling = True
        
        # Line Search for Newton method in ROM
        self.useLineSearch = False
        self.linesearch_maxits = 0
        self.linesearch_decreaseFactor = 0 # sufficient decrease factor
        self.linesearch_contractFactor = 0.5 
        
        
        ###############################################################################
        # Inputs related to the HDM
        ###############################################################################
        
        # Max iteration number for first HDM run
        self.MaxItsHDM1 = 1500 # 1500 was default
        
        # Max iteration number for HDM restart
        self.MaxItsHDM2 = 9000 # 9000 was default
        
        # Convergence tolerance for first HDM step
        self.HDMtol1 = 1e-4
        
        # Convergence tolerance for HDM
        self.HDMtol2 = 5e-7
        
        # Jacobian for first HDM run
        self.MvpHDM1 = 'Approximate'
        
        # Jacobian for HDM restart
        self.MvpHDM2 = 'FiniteDifference'
        
        # How often to sample HDM solution
        self.SamplingFreq = 0
        
        # Form
        self.FormHDM = 'NonDescriptor'
        
        
        
        ###############################################################################
        # Inputs related to the CFD. See AERO-F manual for explanations.
        ###############################################################################
        
        self.DimOrNon = 'Dimensional' 
        self.InletPressure = 22632.   
        self.InletDensity=0.3639    
        self.Prandtl = 0.72
        self.SutherlandConstant = 1.458e-6
        self.SutherlandReferenceTemperature = 110.6
        self.Flux = 'Roe'             
        self.Limiter = 'Venkatakrishnan'    
        self.AdvOp = 'FiniteVolume'   
        self.Beta = 0.33333333333333 # 0.5
        self.Gamma = 1.0                       
        self.Reconstruction = 'Linear'         
        self.PressureCutOff  =  -1e9      
        self.DensityCutOff  =  -1e9       
        self.VerifyClipping  =  'Off'        
        
        
        
        ###############################################################################
        # Inputs related to HPC
        ###############################################################################
        
        # Number of clusters for the HDM mesh
        self.HDMnclust = 120
        
        # Number of processors on the HDM mesh
        self.HDMnproc = 120
        
        # Number of cluster for the reduced mesh
        self.HROMnclust = 8
        
        # Number of processors on the reduced mesh
        self.HROMnproc = 8
        
        # Number of cores for parallel HROM indicator
        self.HROMindproc = 12
        
        # Local scratch space root for the HROM indicator runs; "" to disable
        self.HROMlscratch = ""

        # Local scratch space root for the training matrix rows; "" to disable
        self.TRlscratch = "/lscratch/"
        
        ###############################################################################
        # Input related to the geometry
        ###############################################################################
        
        # Where to write binaries
        self.GeometryPrefix = "fluidmodel"
        
        # Where to find the top file
        self.TopFilePath = "mesh/naca0012_Re1p5"
        
        # For parameters for deform_naca. Don't change unless mesh is changing.
        self.gen_dist_flag = False
        self.plt_flag = False
        self.run_decomp = False
        self.xb = [-0.20, 1.50]
        self.yb = [-0.50, 0.50]
        self.P = 0.4
        self.M = 0.00
        self.T = 0.12
        
        
        
        ###############################################################################
        # Inputs related to post-processing
        ###############################################################################
        
        # Run sower/xp2exo for HDMs
        self.PostproHDM = False
        
        # How many points to generate and evaluate the ROM at in an eval run
        self.NumEvals = 125

        # Select the evaluation points at the begining of the python run so that, along with the seed, they are consistent accross runs
        self.PreSelectEval = True
