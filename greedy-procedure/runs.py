import subprocess

from pyaeroopt.interface import Aerof, AerofInputFile, AerofInputBlock
import pyaeroopt
import os
import re
import shutil
import numpy as np
import math
import scipy.interpolate
import sys
from deform_naca import deform_naca
from ushift_inject import Laplace_ushiftsnapdata, get_ushift_file

def create_deformed_top_file(current_sim_dir, topfilepath):
    meshname = os.path.basename(topfilepath)
    shutil.copy(f'mesh/{meshname}.top', f'{current_sim_dir}/{meshname}_deformed.top')
    print('copied undeformed top file to {}'.format(f'{current_sim_dir}/{meshname}_deformed.top'), flush=True)

    # Now replace only the node coordinates in the deformed top file with the coordinates from Position.xpost
    posFile = '{}deform/Position.xpost'.format(current_sim_dir)
    pos = np.loadtxt(posFile, skiprows=3)
    print('Loading {} for creating deformed top file'.format(posFile), flush=True)
    
    # Read the original top file to find where Elements section starts
    deformed_top_file = f'{current_sim_dir}/{meshname}_deformed.top'
    elements_section = []
    num_nodes = 0
    with open(deformed_top_file, 'r') as f:
        # Read first line (should be "Nodes FluidNodes")
        header = f.readline()
        # Count node lines until we hit Elements section
        for line in f:
            if line.strip().startswith('Elements'):
                # Found Elements section, save it and the rest
                elements_section.append(line)
                elements_section.extend(f.readlines())
                break
            num_nodes += 1
    
    # Write the new .top file with updated node coordinates
    with open(deformed_top_file, 'w') as f:
        # Write the header
        f.write('Nodes FluidNodes\n')
        
        # Write all nodes with new coordinates from Position.xpost
        for i, coords in enumerate(pos):
            # Node number is 1-indexed
            node_num = i + 1
            # Format: node_number x y z (using %.10e precision)
            f.write(f'{node_num:14d}   {coords[0]:.10e}  {coords[1]:.10e}  {coords[2]:.10e}\n')
        
        # Write the Elements section unchanged
        for line in elements_section:
            f.write(line)

    print('Deformed top file created at {}'.format(deformed_top_file), flush=True)
    return deformed_top_file

def solveCurrentLaplace(deformed_top_file, SuperDir, frg, settings, numproc=24):
    Laplace_bin_Dir = "{}/Laplace-bin/".format(SuperDir)
    xdmf_dir = "{}/xdmf_files/".format(SuperDir)

    os.makedirs(xdmf_dir, exist_ok=True)

    laplace_runner = os.environ.get('CRM_LAPLACE_RUNNER', './run_laplace_shift.sh')
    with open('{}log.Laplace'.format(SuperDir), 'w') as log_file:
        subprocess.run(
            [laplace_runner, deformed_top_file, xdmf_dir, str(numproc)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )

    u_shift_xpost = get_ushift_file(deformed_top_file, xdmf_dir, hrmesh_nodes=None, u_star_path=xdmf_dir)
    Laplace_ushiftsnapdata(settings, frg, u_shift_xpost, Laplace_bin_Dir, False)

def getRuns(settings):
    """Template the classes with the values of settings"""
    class common(Aerof):
        """ create common parent class, which will have the common blocks and member functions """
    
        def __init__(self):
            super().__init__()

        def get_laplace_snapshot_data_file(self, hyper=False):
            """Return Laplace snapshot data path when Laplace shifts are enabled."""
            if settings.ShiftType != 'Laplace':
                return ''

            if settings.ShiftType == 'Laplace' and not settings.LaplaceShiftEach:
                if hyper:
                    laplace_bin_dir = '{}hromruns{:03d}/'.format(settings.EI_Laplacebin, self.HDMind)
                else:
                    laplace_bin_dir = settings.MasterDir

                return '{}Laplace-bin/ushift.bin'.format(laplace_bin_dir)

            if settings.ShiftType == 'Laplace' and settings.LaplaceShiftEach:
                if self.hyper:
                    Laplace_bin_Dir = '{}evaluate/hromruns{:03d}/point{:03d}/Laplace-bin/'.format(settings.MasterDir, self.HDMind, self.pind)
                    LaplaceSnapshotdata_file = '{}ushift.bin'.format(Laplace_bin_Dir)
                else:
                    Laplace_bin_Dir = settings.MasterDir
                    # LaplaceSnapshotdata_file = '{}Laplace-bin/HDMrun{:03d}/ushift.bin'.format(Laplace_bin_Dir, self.HDMind)
                    LaplaceSnapshotdata_file = "{}evaluate/hromruns{:03d}/point{:03d}/Laplace-bin/ushift.bin".format(Laplace_bin_Dir, self.HDMind, self.pind)

                return LaplaceSnapshotdata_file
            
        def prep(self, ref_nodes, ref_stick, hrnodes=None):
            """ deform hrnodes and sower the position file """

            if hrnodes is not None:
                nc = settings.HROMnclust
                wdf = True
            else:
                nc = settings.HDMnclust
                wdf = True
            pos,d2w = deform_naca([ref_nodes, None, ref_stick], [np.array(settings.xb), np.array(settings.yb)], [settings.P, settings.M, settings.T], 
                    [self.p[2], self.p[3], self.p[4]], gen_dist_flag = False, wall_dist_flag = wdf, 
                    plt_flag = False,
                              sampled_nodes_ix= hrnodes)
    
            lenpos = pos.shape[0]
            np.savetxt('{}deform/Position.xpost'.format(self.SuperDir),
                       pos,
                       header='Vector Position under load for FluidNodes\n{}\n0'.format(lenpos),
                       comments='',
                       fmt = ['%.10e', '%.10e', '%.10e'])

            self.frg.sower_fluid_split(file2split='{}deform/Position.xpost'.format(self.SuperDir),
                                   out='{}deform/Position.bin'.format(self.SuperDir),
                                   nclust=nc, log='/dev/null')
            if wdf:
                lend2w = d2w.shape[0]
                np.savetxt('{}deform/WallDistance.xpost'.format(self.SuperDir),
                       d2w,
                       header='Scalar WallDistance under load for FluidNodes\n{}\n0'.format(lend2w),
                       comments='',
                       fmt = ['%.10e'])


                self.frg.sower_fluid_split(file2split='{}deform/WallDistance.xpost'.format(self.SuperDir),
                                   out='{}deform/WallDistance.bin'.format(self.SuperDir),
                                   nclust=nc, log='/dev/null')
        def get_common_blocks(self, mach, alpha, clip=False, precomp = False):
            """ These blocks will exist in every input file, so we avoid redundancy """
    
            if clip:
                pc = settings.PressureCutOff
                dc = settings.DensityCutOff
                vc = settings.VerifyClipping 
            else:
                pc = -1e9
                dc = -1e9
                vc = 'Off'

            if precomp:
                pclim = 'On'
            else:
                pclim = 'Off'

            model = AerofInputBlock('FluidModel[0]',
                                    ['Fluid', 'PerfectGas'],
                                    ['PressureCutOff', pc],
                                    ['DensityCutOff', dc],
                                    ['VerifyClipping', vc])
    
            visc = AerofInputBlock('ViscosityModel',
                                    ['Type', 'Sutherland'],
                                    ['SutherlandConstant', settings.SutherlandConstant],
                                    ['SutherlandReferenceTemperature', settings.SutherlandReferenceTemperature])
    
            therm = AerofInputBlock('ThermalConductivityModel',
                                    ['Type', 'ConstantPrandtl'],
                                    ['Prandtl', settings.Prandtl])
            turbm = AerofInputBlock('TurbulenceModel',
                                    ['Type', 'SpalartAllmaras'])
            turbc = AerofInputBlock('TurbulenceClosure',
                                    ['Type', 'TurbulenceModel'],
                                    ['TurbulenceModel', turbm])
    
            equa = AerofInputBlock('Equations',
                                   ['Type', 'NavierStokes'],
                                   ['FluidModel[0]', model],
                                   ['TurbulenceClosure', turbc],
                                   ['ThermalConductivityModel', therm],
                                   ['ViscosityModel', visc])
    
            wall = AerofInputBlock('Wall',
                                   ['Type', 'Adiabatic'],
                                   ['Integration', 'Full'])
   

            sym = AerofInputBlock('Symmetry',
                                   ['Treatment', 'Weak'])


            inlet = AerofInputBlock('Inlet',
                                    ['Mach', mach],
                                    ['Beta', alpha],  # this mesh is oriented so
                                    ['Alpha', 0],     # that AoA is actually Beta
                                    ['Pressure', settings.InletPressure],
                                    ['Density', settings.InletDensity])
    
            bc = AerofInputBlock('BoundaryConditions',
                                 ['Inlet', inlet],
                                 ['Wall', wall],
                                 ['Symmetry',sym])
    
            nav = AerofInputBlock('NavierStokes',
                                  ['Flux', settings.Flux],
                                  ['Reconstruction', settings.Reconstruction],
                                  ['Limiter', settings.Limiter],
                                  ['Gradient', 'LeastSquares'],
                                  ['Beta', settings.Beta],
                                  ['Gamma', settings.Gamma],
                                  ['PreComputeLimiter', pclim])

            tm = AerofInputBlock('TurbulenceModel',
                                  ['Reconstruction', settings.Reconstruction],
                                  ['Limiter', 'VanAlbada'],
                                  ['Gradient', 'LeastSquares'],
                                  ['Beta', settings.Beta],
                                  ['Gamma', settings.Gamma],
                                  ['PreComputeLimiter', pclim])
    
            space = AerofInputBlock('Space',
                                    ['NavierStokes', nav],
                                    ['TurbulenceModel', tm])
    
            return equa, bc, space
    
        def getLD(self):
            """ get lift and drag from a run """
            path = '{}postpro/liftdrag.out'.format(self.SuperDir)
            with open(path) as f:
                lines = f.readlines()
                line = lines[-1].split(' ')
                lift = float(line[5])
                drag = float(line[4])
            return lift, drag
        
        def getFyFx(self):
            """ get Fx and Fy from a run """
            path = '{}postpro/force.out'.format(self.SuperDir)
            with open(path) as f:
                lines = f.readlines()
                line = lines[-1].split(' ')
                Fy = float(line[5])
                Fx = float(line[4])
            return Fy, Fx

        def postPro(self):
            """ sower/xp2exo results """
    
            path = '{}results/'.format(self.SuperDir)
            outpath = '{}postpro/'.format(self.SuperDir)
            self.frg.sower_fluid_merge('{}PressureCoefficient.bin'.format(
                path), '{}PressureCoefficient'.format(outpath), 'PressureCoefficient', log='/dev/null')
            self.frg.sower_fluid_merge('{}SkinFriction.bin'.format(
                path), '{}SkinFriction'.format(outpath), 'SkinFriction', log='/dev/null')
            self.frg.sower_fluid_merge('{}Mach.bin'.format(
                path), '{}Mach'.format(outpath), 'Mach', log='/dev/null')
            self.frg.sower_fluid_merge('{}Displacement.bin'.format(
                path), '{}Displacement'.format(outpath), 'Displacement', log='/dev/null')
            self.frg.run_xp2exo('{}fluidmodel.exo'.format(outpath), 
                                [ '{}Mach.xpost'.format(outpath),
                                  '{}Displacement.xpost'.format(outpath)],
                                log='/dev/null')

        def getSkinForces(self,ref_nodes,hrnodes=None):
           
            if hrnodes is not None:
                ref_pos=ref_nodes[hrnodes,:]
            else:
                ref_pos=ref_nodes

            # Read files, rely on the fact that skinfriction=0.0 for non-surface node
            sfFile = '{}postpro/SkinFriction.xpost'.format(self.SuperDir)
            sf = np.loadtxt(sfFile, skiprows = 3)
            idx = np.nonzero(sf) 
            sf = sf[idx]

            posFile = '{}deform/Position.xpost'.format(self.SuperDir)
            pos = np.loadtxt(posFile, skiprows = 3)
            pos = pos[idx]

            pressFile = '{}postpro/PressureCoefficient.xpost'.format(self.SuperDir)
            press = np.loadtxt(pressFile, skiprows = 3)
            press = press[idx]

            ref_pos=ref_pos[idx]

            # Split into top and bottom surfaces
            top_idx = np.nonzero((ref_pos[:,2]>=0)*(ref_pos[:,3]==0.))
            bot_idx = np.nonzero((ref_pos[:,2]<0)*(ref_pos[:,3]==0.))

            pos_top = pos[top_idx]
            pos_bot = pos[bot_idx]

            sf_top = sf[top_idx]
            sf_bot = sf[bot_idx]

            press_top = press[top_idx]
            press_bot = press[bot_idx]

            # Sort by x location
            top_idx = np.argsort(pos_top[:,0])
            bot_idx = np.argsort(-pos_bot[:,0])

            pos_top = pos_top[top_idx]
            pos_bot = pos_bot[bot_idx]

            sf_top = sf_top[top_idx]
            sf_bot = sf_bot[bot_idx]

            press_top = press_top[top_idx]
            press_bot = press_bot[bot_idx]
            
            #Concatenate arrays
            pos = np.concatenate((pos_top, pos_bot))
            press = np.concatenate((press_top, press_bot))
            sf = np.concatenate((sf_top, sf_bot))

            return pos, press, sf




        def getRes(self):
            """ Return the residual of the rom solution """
            with open(self.resfile, 'r') as f:
                lastline = f.readlines()[-1]
    
            nums = lastline.split(' ')
    
            relres = float(nums[2])
            with open(self.log) as logfile:
                for line in logfile:
                    try:
                        initresst = line.split('Spatial residual norm = ')[1]
                        initres = float(initresst)
                        break
                    except:
                        pass
    
            return relres*initres
    
        def getResVec(self):
            """ Return the reduced residual """
    
            with open(self.resvecfile, 'r') as f:
                lastline = f.readlines()[2]
    
            nums = lastline.split()
            resvecstr = nums[3:]
    
            return np.array([float(x) for x in resvecstr])
    ###### SET UP HDM RUNS################################################
    
    
    class HDM(common):
        """ First step of the HDM. Currently, it has a different MVP than HDM2 """
        def __init__(self, frg, p=None, HDMind=0, Pind=0, step=1, evaluate=False, precomp=False, maindir=False):
            """ 
            If evaluate, then write to evaluate/ directory.
            If precomp, symbolically link to existing solution.
            If maindir, write to init directory rather than MasterDir
            p = [mach, alpha, maxCamberLocation, maxCamber, thickness]
            """
    
            super().__init__()
            self.HDMind = HDMind
            self.frg = frg
            if evaluate:
                prefix = 'evaluate/'
            else:
                prefix = ''
    
            if maindir:
                md = settings.InitHDMPreCompDir
            else:
                md = settings.MasterDir
    
            self.SuperDir = '{}{}HDMrun{:03d}/'.format(md, prefix,HDMind)
            self.SuperDirLink = '{}HDMrun{:03d}/'.format(settings.InitHDMPreCompDir, HDMind)
            self.p = p
            self.step = step

            if self.step==1:
                self.mvp = settings.MvpHDM1
                self.maxits = settings.MaxItsHDM1
                self.tol = settings.HDMtol1
                self.pcl =False
                self.resdat = ''
                self.sol = ''
                neededDIRs = ['references/', 'results/','postpro/','snapshots/','deform/']
    
                if not precomp:
                    if not os.path.exists(self.SuperDir):
                        # Create needed directories and subdirectories
                        os.makedirs(self.SuperDir)
                        for dirr in neededDIRs:
                            os.mkdir('{}{}'.format(self.SuperDir, dirr))
                    else:
                        raise Exception(
                            "***Error: folder {} already exists***".format(self.SuperDir))
                else:
                    # Symbolic link to existing directory
                    os.symlink(os.path.abspath(self.SuperDirLink), os.path.abspath(self.SuperDir), target_is_directory = True)
    
            elif  self.step==2:
                self.mvp = settings.MvpHDM2
                self.maxits = settings.MaxItsHDM2
                self.tol = settings.HDMtol2
                self.pcl = True
                self.resdat = '{}references/Restart.data'.format(self.SuperDir)
                self.sol = '{}references/Solution.bin'.format(self.SuperDir)
    
        def create_input_file(self):
            """Define the input file for this problem."""
            mach = self.p[0]         # this simulation is parameterized by Mach number
            alpha = self.p[1]         # and angle of attack
    
            SuperDir = self.SuperDir

            # Solve laplace problem for each HDM
            if settings.ShiftType == 'Laplace' and settings.LaplaceShiftEach:
                if self.step == 1:
                    deformed_top_file = create_deformed_top_file(SuperDir, settings.TopFilePath)
                    solveCurrentLaplace(deformed_top_file, SuperDir, self.frg, settings, settings.LaplaceNumProc)

                laplace_bin_dir = SuperDir
                LaplaceSnapshotdata_file = '{}Laplace-bin/ushift.bin'.format(laplace_bin_dir)
            elif settings.ShiftType == 'Laplace' and not settings.LaplaceShiftEach:
                LaplaceSnapshotdata_file = self.get_laplace_snapshot_data_file(hyper=False)
            else:
                LaplaceSnapshotdata_file = ''

            prob = AerofInputBlock('Problem',
                                   ['Type', 'Steady'],
                                   ['Mode', settings.DimOrNon])
            
            inp = AerofInputBlock('Input',
                                  ['GeometryPrefix', '{}'.format(self.frg.geom_pre)],
                                  ['Position', '{}deform/Position.bin'.format(SuperDir)],
                                  ['WallDistance','{}deform/WallDistance.bin'.format(SuperDir)],
                                  ['RestartData', self.resdat],
                                  ['LaplaceSnapshotData', LaplaceSnapshotdata_file],
                                  ['Solution', self.sol])
    
            rest = AerofInputBlock('Restart',
                                   ['Prefix', '{}references/'.format(SuperDir)],
                                   ['FilePackage',''],
                                   ['RestartData', 'Restart.data'],
                                   ['Solution', 'Solution.bin'])
    
            post = AerofInputBlock('Postpro',
                                   ['Frequency', 0],
                                   ['Prefix', '{}results/'.format(SuperDir)],
                                   ['LiftandDrag', '../postpro/liftdrag.out'],
                                   ['Force', '../postpro/force.out'],
                                   ['PressureCoefficient', 'PressureCoefficient.bin'],
                                   ['SkinFrictionCoefficient', 'SkinFriction.bin'],
                                   ['Mach', 'Mach.bin'],
                                   ['Velocity', 'Velocity.bin'],
                                   ['FluxResidual', 'FluxRes.bin'],
                                   ['ControlVolume', 'ControlVolume.bin'],
                                   ['Displacement', 'Displacement.bin'],
                                   ['Residual', '../postpro/Residual.out'])
    
            nlr = AerofInputBlock('NonlinearROM',
                                  ['Prefix' , '{}snapshots/'.format(SuperDir)],
                                  ['StateVector', 'State.bin'],
                                  ['Frequency', 0],
                                  ['OutputResidualSnapshotData', True],
                                  ['OutputShiftVectorType', settings.ShiftType],
                                  ['ReducedCoordinates', 'ReducedCoords.bin'])
    
            outp = AerofInputBlock('Output',
                                   ['Postpro', post],
                                   ['Restart', rest],
                                   ['NonlinearROM', nlr])
    
            equa, bc, space = self.get_common_blocks(mach, alpha, precomp = self.pcl)
    
            prec = AerofInputBlock('Preconditioner',
                                   ['Type', 'Ras'],
                                   ['Fill', 0])
    
            navNewton = AerofInputBlock('NavierStokes',
                                        ['MaxIts', 200],
                                        ['KrylovVectors', 200],
                                        ['Eps', 0.001],
                                        ['Preconditioner', prec])
    
            navLin = AerofInputBlock('LinearSolver',
                                     ['NavierStokes', navNewton])
    
            newton = AerofInputBlock('Newton',
                                     ['MaxIts', 1],
                                     ['FailSafe', 'AlwaysOn'],
                                     ['Eps', 0.001],
                                     ['LinearSolver', navLin])
    
            impl = AerofInputBlock('Implicit',
                                   ['MatrixVectorProduct', self.mvp],
                                   ['Newton', newton])
    
            cfl = AerofInputBlock('CflLaw',
                                  ['Strategy', 'Residual'],
                                  ['Cfl0', 5],
                                  ['Cfl1', 5],
                                  ['Cfl2', 0],
                                  ['CflMax', 100])
    
            time = AerofInputBlock('Time',
                                   ['Form', settings.FormHDM],
                                   ['MaxIts', self.maxits],
                                   ['Eps', self.tol],
                                   ['Implicit', impl],
                                   ['CflLaw', cfl])
   
            sd = AerofInputBlock('SurfaceData[1]',
                                   ['Nz',1.0],
                                   ['Nx',0.0],
                                   ['Ny',0.0])

            surf = AerofInputBlock('Surfaces',
                                    ['SurfaceData[1]', sd])

            fname = '{}input{}'.format(SuperDir,self.step)
            log = '{}log{}'.format(SuperDir,self.step)
            self.res = '{}postpro/Residual.out'.format(SuperDir)
            self.infile = AerofInputFile(fname,
                                         [prob, inp, outp, equa, bc, space, time, surf], log)
    
        def post(self):
    
            """ Update snapshot matrix and parametric initial condition files """
            ssname = '{}statesnapdata.txt'.format(settings.MasterDir)
            try:
                snapdata = open(ssname, 'x')
                snapdata.close()
                snapdata = open(ssname, 'r+')
            except FileExistsError:
                snapdata = open(ssname, 'r+')
    
            lines = snapdata.readlines()
    
            L = '{}snapshots/State.bin {} {} 1 1 \n'.format(
                self.SuperDir, settings.SnapIndex, settings.SnapIndex)
    
            if len(lines) > 0:
                current = lines[0]
                lines[0] = "{}\n".format(int(current)+1)
            else:
                lines.append("1\n")
    
            lines.append(L)
            snapdata.close()
            snapdata = open(ssname, 'w')
            snapdata.writelines(lines)
    
            snapdata.close()
    
            # Update Parameterdata text file
    
            psname = '{}parsoldata.txt'.format(settings.MasterDir)
            try:
                pardata = open(psname, 'x')
                pardata.close()
                pardata = open(psname, 'r+')
            except FileExistsError:
                pardata = open(psname, 'r+')
            lines = pardata.readlines()
            nP = self.p
            L = '{}snapshots/State.bin {}\n'.format(self.SuperDir, settings.SnapIndex)
            for mu in nP:
                L += ('{}\n'.format(mu))
    
            if len(lines) > 0:
                current = lines[0]
                lines[0] = "{}\n".format(int(current)+1)
            else:
                lines.append("{}\n".format(1))
                lines.append('{}\n'.format(len(nP)))
    
            lines.append(L)
    
            pardata.close()
    
            pardata = open(psname, 'w')
            pardata.writelines(lines)
    
            pardata.close()
            #######
    
    ####### SET UP POD RUN###########################################################
    
    
    class POD(common):
        """ Generate ROB and reference state """
    
        def __init__(self, frg, p=None, HDMind=0):
            """
            p = [mach, alpha, maxCamberLocation, maxCamber, thickness]
            """
            super().__init__()
            self.HDMind = HDMind
            self.frg = frg
            self.SuperDir = '{}reductionrun{:03d}/'.format(settings.MasterDir, HDMind)
            self.p = p
    
            # make necessary dirs
    
            if not os.path.exists(self.SuperDir):
                os.makedirs(self.SuperDir)
    
            neededDIRs = ['nonlinearrom/data', 'trainingmatrix']
    
            for dirr in neededDIRs:
                os.makedirs('{}{}'.format(self.SuperDir, dirr))
    
            if settings.RefMethod == "Input":
                # If Input, we use the first snapshot as our reference state
                ssname = '{}statesnapdata.txt'.format(settings.MasterDir)
                snapdata = open(ssname, 'r+')
                l = snapdata.readlines()[1]
                self.snapref = l.split(' ')[0]
            else:
                self.snapref = ''
    
        def create_input_file(self):
            """Define the input file for this problem."""
            mach = self.p[0]
            alpha = self.p[1]
    
            if self.HDMind > 1:
                RM = settings.RefMethod
            else:
                RM = 'None'
    
            SuperDir = self.SuperDir
    
            prob = AerofInputBlock('Problem',
                                   ['Type', 'NonlinearRomPreprocessing'],
                                   ['Mode', settings.DimOrNon])
    
            inp = AerofInputBlock('Input',
                                  ['GeometryPrefix', '{}'.format(self.frg.geom_pre)],
                                  ['MultipleSolutionsData', '{}parsoldata.txt'.format(settings.MasterDir)],
                                  ['MaxInterpolatedSolutions', settings.MaxIntSols],
                                  ['ParametricDistanceExponent', settings.DistExp],
                                  ['ProjectionErrorSnapshotData', '{}statesnapdata.txt'.format(settings.MasterDir)],
                                  ['StateSnapshotData', '{}statesnapdata.txt'.format(settings.MasterDir)],
                                  ['SnapshotReferenceSolution', self.snapref],
                                  ['SnapshotReferenceSolutionIndex', settings.SnapIndex])

            if settings.rbf:
                GenManifoldName = '{}/nonlinearrom/cluster0/'.format(self.SuperDir)
            else:
                GenManifoldName = ''

            files = AerofInputBlock('Files',
                                    ['StatePrefix', 'state'],
                                    ['ReducedMeshPrefix', 'hrmesh'],
                                    ['GeneralManifoldRbfName', GenManifoldName])

            if self.HDMind<settings.ClusteringStart:
                NC = 1
            else:
                NC = settings.NumClusters

            nlrfs = AerofInputBlock('NonlinearRomFileSystem',
                                    ['Prefix', '{}'.format(SuperDir)],
                                    ['TopLevelDirectory', 'nonlinearrom'],
                                    ['NumClusters', NC],
                                    ['Files', files])
    
            ss = AerofInputBlock('Snapshots',
                                 ['NormalizeSnaps', True],
                                 ['ReferenceState', RM])
    
            dc = AerofInputBlock('DataCompression',
                                 ['UseGeneralManifold', False],
                                 ['ComputePOD', True],
                                 ['PODMethod', settings.PODMethod],
                                 ['MaxEnergyRetained', settings.StateBasisMaxEnergy],
                                 ['SingularValueTolerance', -1])
    
            srob = AerofInputBlock('StateROB',
                                   ['Snapshots', ss],
                                   ['DataCompression', dc])

            rpe = AerofInputBlock('RelativeProjectionError', 
                                   ['ComputeProjectionError', settings.rbf],
                                   ['MaximumEnergy', settings.StateBasisMaxEnergy])

            clst = AerofInputBlock('Clustering',
                                   ['PercentOverlap', settings.PercentOverlap])
    
            crob = AerofInputBlock('ConstructROB',
                                   ['Clustering', clst],
                                   ['ProjectInitialCondition', True],
                                   ['PreprocessForProjections', True],
                                   ['StateROB', srob],
                                   ['RelativeProjectionError', rpe])
    
            ecsw = AerofInputBlock('SamplingWeighting',
                                   ['SamplingTolerance', settings.SamplingTol],
                                   ['Projection', settings.Projection])
    
            nlrpp = AerofInputBlock('NonlinearRomPreprocessing',
                                    ['ConstructROB', crob])
    
            equa, bc, space = self.get_common_blocks(mach, alpha)
    
            prec = AerofInputBlock('Preconditioner',
                                   ['Type', 'Ras'],
                                   ['Fill', 0])
    
            navNewton = AerofInputBlock('NavierStokes',
                                        ['MaxIts', 100],
                                        ['KrylovVectors', 100], ['Eps', 0.001],
                                        ['Preconditioner', prec])
    
            navLin = AerofInputBlock('LinearSolver',
                                     ['NavierStokes', navNewton])
    
            newton = AerofInputBlock('Newton',
                                     ['MaxIts', 200], ['Eps', 0.001],
                                     ['LinearSolver', navLin])
    
            impl = AerofInputBlock('Implicit',
                                   ['MatrixVectorProduct', settings.MvpROM],
                                   ['Type', 'SpatialOnly'],
                                   ['ExactImplementation', 'New'],
                                   ['Newton', newton])
    
            time = AerofInputBlock('Time',
                                   ['Form', settings.Form],
                                   ['MaxIts', 1000],
                                   ['Eps', 1e-6],
                                   ['Implicit', impl])
    
            fname = '{}input.pod'.format(SuperDir)
    
            log = '{}log.pod'.format(SuperDir)
    
            self.infile = AerofInputFile(fname,
                                         [prob, inp, nlrfs, nlrpp, equa, bc, space, time], log)
    
    ####### SET UP TRAINING MATRIX GENERATION RUN#######################################################
    
    
    class ROWGEN(common):
        """ generate a block of the training matrix """
        def __init__(self, frg, p=None, HDMind=0, Gind=0):
            """
            HDMind is the index of the ROM.
            Gind is the index of the block.
            p = [mach, alpha, maxCamberLocation, maxCamber, thickness]
            """
            super().__init__()
            self.frg = frg
            self.HDMind = HDMind
            self.Gind = Gind
            self.SuperDir = '{}reductionrun{:03d}/trainingmatrix/'.format(settings.MasterDir, HDMind)
            self.SuperDirr = '{}reductionrun{:03d}/'.format(settings.MasterDir, HDMind)
            self.p = p
    
            # Make necessary dirs
    
            if not os.path.exists(self.SuperDir):
                os.makedirs(self.SuperDir)

            if settings.TRlscratch:
                os.system("rm -f %s/tr" % (self.SuperDir))
                os.system("ln -s %s/%s/tr %s/tr"  % 
                  (settings.TRlscratch,
                   os.environ.get('USER'),  self.SuperDir))
            else:
                os.system("mkdir -p %s/tr" % (self.SuperDir))
    
            # Set up local text file
            filename = '{}samplingsnapshots_rows{}.txt'.format(self.SuperDir, self.Gind)
            snapdata = open(filename, 'w+')
    
            L = '{}HDMrun{:03d}/snapshots/State.bin {} {} 1 1 \n'.format(settings.MasterDir, self.Gind, settings.SnapIndex, settings.SnapIndex)
    
            lines = ['1\n']
            lines.append(L)
            snapdata.writelines(lines)
    
            snapdata.close()
    
            # Update global text file
    
            filename = '{}samplingsnapshots_allrows.txt'.format(self.SuperDir)
            try:
                snapdata = open(filename, 'x')
                snapdata.close()
                snapdata = open(filename, 'r+')
            except FileExistsError:
                snapdata = open(filename, 'r+')
    
            lines = snapdata.readlines()
    
            L = '{}/tr/training.rows{}. 0 0 1 1 \n'.format(self.SuperDir, self.Gind)
    
            if len(lines) > 0:
                lines[0] = "{}\n".format(int(lines[0])+1)
            else:
                lines.append("{}\n".format(1))
    
            lines.append(L)
            snapdata.close()
            snapdata = open(filename, 'w')
            snapdata.writelines(lines)
    
            snapdata.close()
    
        def create_input_file(self):
            """Define the input file for this problem."""
            mach = self.p[0]
            alpha = self.p[1]
    
            SuperDir = self.SuperDir
            SuperDirr = self.SuperDirr
    
            prob = AerofInputBlock('Problem',
                                   ['Type', 'NonlinearRomPreprocessing'],
                                   ['Mode', settings.DimOrNon])
    
            inp = AerofInputBlock('Input',
                                  ['GeometryPrefix', '{}'.format(self.frg.geom_pre)],
                                  ['MultipleSolutionsData', '{}parsoldata.txt'.format(settings.MasterDir)],
                                  ['MaxInterpolatedSolutions', settings.MaxIntSols],
                                  ['MeshSamplingSnapshotData', '{}samplingsnapshots_rows{}.txt'.format(SuperDir, self.Gind)],
                                  ['Position', '{}HDMrun{:03d}/deform/Position.bin'.format(settings.MasterDir, self.Gind)],
                                  ['WallDistance', '{}HDMrun{:03d}/deform/WallDistance.bin'.format(settings.MasterDir, self.Gind)])
    
            nlr = AerofInputBlock('NonlinearROM',
                                  ['Prefix', '{}'.format(SuperDir)],
                                  ['OutputResidualSnapshotData', True],
                                  ['ResidualVector', 'tr/training.rows{}.'.format(self.Gind)])
    
            out = AerofInputBlock('Output',
                                  ['NonlinearROM', nlr])
    
            files = AerofInputBlock('Files',
                                    ['DuplicateSnapshots', False],
                                    ['StatePrefix', 'state'],
                                    ['StateBasis', 'state.rob'],
                                    ['StateBasisReference', 'state.ref'],
                                    ['ReducedMeshPrefix', 'hrmesh'])
   
            if self.HDMind<settings.ClusteringStart:
                NC = 1
            else:
                NC = settings.NumClusters

            nlrfs = AerofInputBlock('NonlinearRomFileSystem',
                                    ['Prefix', '{}'.format(SuperDirr)],
                                    ['TopLevelDirectory', 'nonlinearrom'],
                                    ['NumClusters', NC],
                                    ['Files', files])
    
            ss = AerofInputBlock('Snapshots',
                                 ['NormalizeSnaps', True],
                                 ['ReferenceState', settings.RefMethod])
    
            dc = AerofInputBlock('DataCompression',
                                 ['ComputePOD', False],
                                 ['PODMethod', settings.PODMethod],
                                 ['MaxEnergyRetained', settings.StateBasisMaxEnergy])
    
            srob = AerofInputBlock('StateROB',
                                   ['Snapshots', ss],
                                   ['DataCompression', dc])
    
            clst = AerofInputBlock('Clustering',
                                   ['UseExistingClusters', True])
    
            crob = AerofInputBlock('ConstructROB',
                                   ['Clustering', clst],
                                   ['ProjectInitialCondition', True],
                                   ['StateROB', srob])
    
            ecsw = AerofInputBlock('SamplingWeighting',
                                   ['DataType', settings.TrainingData],
                                   ['UseRowScaling', settings.UseRowScaling],
                                   ['UseColumnScaling', settings.ColumnScaling],
                                   ['SamplingTolerance', settings.SamplingTol],
                                   ['Projection', settings.Projection],
                                   ['MaxNumResidualComponents', settings.MaxNumResComps])
    
            crm = AerofInputBlock('ConstructReducedMesh',
                                  ['GenerateTrainingData', True],
                                  ['Type', 'SamplingWeighting'],
                                  ['StackedTraining', 'SpatialOnly'],
                                  ['ShiftVectorType', settings.ShiftType],
                                  ['SamplingWeighting', ecsw],
                                  ['MaximumEnergyStateBasis', settings.StateBasisMaxEnergy],
                                  ['MaximumDimensionStateBasis', settings.StateBasisMaxDim])
    
            nlrpp = AerofInputBlock('NonlinearRomPreprocessing',
                                    ['ConstructReducedMesh', crm],
                                    ['ConstructROB', crob])
    
            equa, bc, space = self.get_common_blocks(mach, alpha)
    
            prec = AerofInputBlock('Preconditioner',
                                   ['Type', 'Ras'],
                                   ['Fill', 0])
    
            navNewton = AerofInputBlock('NavierStokes',
                                        ['MaxIts', 100],
                                        ['KrylovVectors', 100], ['Eps', 0.001],
                ['Preconditioner', prec])
    
            navLin = AerofInputBlock('LinearSolver',
                                     ['NavierStokes', navNewton])
    
            newton = AerofInputBlock('Newton',
                                     ['MaxIts', 1],
                                     ['FailSafe', 'AlwaysOn'],
                                     ['Eps', 0.001],
                                     ['LinearSolver', navLin])
    
            impl = AerofInputBlock('Implicit',
                                   ['Type', 'SpatialOnly'],
                                   ['MatrixVectorProduct', settings.MvpROM],
                                   ['ExactImplementation', 'New'],
                                   ['TurbulenceModelCoupling', 'Strong'],
                                   ['Newton', newton])
    
            time = AerofInputBlock('Time',
                                   ['Form', settings.Form],
                                   ['MaxIts', settings.MaxItsHDM2],
                                   ['Eps', 1e-12],
                                   ['Implicit', impl])
    
    
            fname = '{}input{}'.format(SuperDir, self.Gind)
    
            log = '{}log{}'.format(SuperDir, self.Gind)
    
            self.infile = AerofInputFile(fname,
                                         [prob, inp, out, nlrfs, nlrpp, equa, bc, space, time], log)
    
    ####### CONSTRUCT REDUCED MESH###########################################################
    
    
    class CM(common):
        """ Construct reduced mesh, either by applying previous weights or generating new weights """
    
        def __init__(self, frg, p=None, HDMind=0, rebuild=True, final = False, nc = 1, NumInit=0):
            """
            If rebuild, rerun NNLS. Else, use previous weights.
            If TrainOnCentroid, training matrix generated at p on the fly.
            p = [mach, alpha, maxCamberLocation, maxCamber, thickness]
            """
            super().__init__()
            self.HDMind = HDMind
            self.frg = frg
            self.SuperDirrm1 = '{}reductionrun{:03d}/'.format(settings.MasterDir, HDMind-1)
            self.SuperDirr = '{}reductionrun{:03d}/'.format(settings.MasterDir, HDMind)
            self.p = p
            self.final = final
            self.NumInit = NumInit
    
            self.rebuild = rebuild
    
            # Make necessary dirs
    
            self.topfile = '{}nonlinearrom/hrmesh.top'.format(self.SuperDirr)
            self.dwallfile = '{}nonlinearrom/hrmesh.dwall.reduced.xpost'.format(
                self.SuperDirr)
            self.binarydir = '{}nonlinearrom/hrmeshbinary/'.format(self.SuperDirr)
            if os.path.exists(self.binarydir):
                pass
            else:
                os.makedirs(self.binarydir)
    
            self.hfrg = pyaeroopt.interface.Frg(top=self.topfile,   geom_pre='{}fluidmodel'.format(self.binarydir))
    
        def post(self):
            """ Prepare the results of the RomPreprocessing for online rom """
    
            if settings.HROMnclust > 1:
                # part the mesh
                self.hfrg.part_mesh(settings.HROMnclust, log='/dev/null')
    
            # sower the top file
            self.hfrg.sower_fluid_top([settings.HROMnproc], settings.HROMnclust, log='/dev/null')
    
            self.hfrg.sower_fluid_split(file2split=self.dwallfile,
                                        out='{}/fluidmodel.dwall'.format(self.binarydir), nclust=settings.HROMnclust, log='/dev/null')
    
            if self.HDMind<settings.ClusteringStart:
                NC = 1
            else:
                NC = settings.NumClusters

            for i in range(NC):
                self.hfrg.sower_fluid_split(
                    file2split='{}nonlinearrom/cluster{}/hrmesh.rob.reduced.xpost'.format(self.SuperDirr, i),
                    out='{}nonlinearrom/cluster{}/hrmesh.rob.reduced'.format(self.SuperDirr, i),
                    nclust=settings.HROMnclust, log='/dev/null')
                self.hfrg.sower_fluid_split(
                    file2split='{}nonlinearrom/cluster{}/hrmesh.ref.reduced.xpost'.format(self.SuperDirr, i),
                    out='{}nonlinearrom/cluster{}/hrmesh.ref.reduced'.format(self.SuperDirr, i),
                    nclust=settings.HROMnclust, log='/dev/null')
    
        def create_input_file(self):
            """Define the input file for this problem."""
            mach = self.p[0]
            alpha = self.p[1]
    
            SuperDirrm1 = self.SuperDirrm1
            SuperDirr = self.SuperDirr
    
            prob = AerofInputBlock('Problem',
                                   ['Type', 'NonlinearRomPreprocessing'],
                                   ['Mode', settings.DimOrNon])
    
            if (settings.HotStart and self.HDMind > self.NumInit) or (not self.rebuild):
                inpsampnodes = '{}nonlinearrom/hrmesh.samplenodes.fullmesh'.format(
                    SuperDirrm1)
                inpsampweights = '{}nonlinearrom/hrmesh.sampleweights'.format(
                    SuperDirrm1)
            else:
                inpsampnodes = ''
                inpsampweights = ''
    
            if settings.TrainOnCentroid:
                meshsamp = '{}statesnapdata.txt'.format(settings.MasterDir)
                traindat = ''
            else:
                meshsamp = ''
                traindat = '{}trainingmatrix/samplingsnapshots_allrows.txt'.format(
                    SuperDirr)
    
            inp = AerofInputBlock('Input',
                                  ['GeometryPrefix', '{}'.format(self.frg.geom_pre)],
                                  ['MultipleSolutionsData', '{}parsoldata.txt'.format(settings.MasterDir)],
                                  ['MaxInterpolatedSolutions', settings.MaxIntSols],
                                  ['TrainingData', traindat],
                                  ['MeshSamplingSnapshotData', meshsamp],
                                  ['InputSampleNodes', inpsampnodes],
                                  ['InputSampleWeights', inpsampweights])
    
            nlr = AerofInputBlock('NonlinearROM',
                                  ['Prefix', '{}'.format(SuperDirr)])
    
            out = AerofInputBlock('Output',
                                  ['NonlinearROM', nlr])
            
            if settings.rbf:
                GenManifoldName = '{}nonlinearrom/cluster0/'.format(SuperDirr)
            else:
                GenManifoldName = ''

            files = AerofInputBlock('Files',
                                    ['DuplicateSnapshots', False],
                                    ['StatePrefix', 'state'],
                                    ['ReducedMeshPrefix', 'hrmesh'],
                                    ['GeneralManifoldRbfName', GenManifoldName])
            
            if self.HDMind<settings.ClusteringStart:
                NC = 1
            else:
                NC = settings.NumClusters

            nlrfs = AerofInputBlock('NonlinearRomFileSystem',
                                    ['Prefix', '{}'.format(SuperDirr)],
                                    ['NumClusters', NC],
                                    ['TopLevelDirectory', 'nonlinearrom'],
                                    ['Files', files])
    
            clst = AerofInputBlock('Clustering',
                                   ['UseExistingClusters', True])
    
            ss = AerofInputBlock('Snapshots',
                                 ['NormalizeSnaps', True],
                                 ['ReferenceState', settings.RefMethod])
    
            dc = AerofInputBlock('DataCompression',
                                 ['UseGeneralManifold', settings.rbf],
                                 ['ComputePOD', False],
                                 ['PODMethod', settings.PODMethod],
                                 ['MaxEnergyRetained', settings.StateBasisMaxEnergy])
    
            srob = AerofInputBlock('StateROB',
                                   ['Snapshots', ss],
                                   ['DataCompression', dc])
    
            crob = AerofInputBlock('ConstructROB',
                                   ['Clustering', clst],
                                   ['ProjectInitialCondition', True],
                                   ['StateROB', srob])
    
            ecsw = AerofInputBlock('SamplingWeighting',
                                   ['DataType', settings.TrainingData],
                                   ['SamplingTolerance', settings.SamplingTol],
                                   ['DecomposeBoundaryConditions', settings.DecompBC],
                                   ['UseRowScaling', settings.UseRowScaling],
                                   ['UseColumnScaling', settings.ColumnScaling],
                                   ['UseGlobalTolerance', settings.UseGT],
                                   ['Projection', settings.Projection])
    
            useinp = (not self.rebuild)
   
            
            crm = AerofInputBlock('ConstructReducedMesh',
                                  ['ComputeReducedMesh', True],
                                  ['Type', 'SamplingWeighting'],
                                  ['UseInputSamplingSolution', useinp],
                                  ['IncludeAllLiftDragFaces', self.final],
                                  ['StackedTraining', 'SpatialOnly'],
                                  ['SamplingWeighting', ecsw],
                                  ['MaximumEnergyStateBasis', settings.StateBasisMaxEnergy],
                                  ['MaximumDimensionStateBasis', settings.StateBasisMaxDim])
    
            nlrpp = AerofInputBlock('NonlinearRomPreprocessing',
                                    ['ConstructReducedMesh', crm],
                                    ['ConstructROB', crob])
    
            equa, bc, space = self.get_common_blocks(mach, alpha)
    
            prec = AerofInputBlock('Preconditioner',
                                   ['Type', 'Ras'],
                                   ['Fill', 0])
    
            navNewton = AerofInputBlock('NavierStokes',
                                        ['MaxIts', 100],
                                        ['KrylovVectors', 100], 
                                        ['Eps', 0.001],
                                        ['Preconditioner', prec])
    
            navLin = AerofInputBlock('LinearSolver',
                                     ['NavierStokes', navNewton])
    
            newton = AerofInputBlock('Newton',
                                     ['MaxIts', 200], ['Eps', 0.001],
                                     ['LinearSolver', navLin])
    
            impl = AerofInputBlock('Implicit',
                                   ['Type', 'SpatialOnly'],
                                   ['MatrixVectorProduct', settings.MvpROM],
                                   ['ExactImplementation', 'New'],
                                   ['Newton', newton])
    
            time = AerofInputBlock('Time',
                                   ['Form', settings.Form],
                                   ['MaxIts', 1000],
                                   ['Eps', 1e-6],
                                   ['Implicit', impl])
    
            if self.rebuild:
                suff = 'new'
            else:
                suff = 'old'
            fname = '{}input.cm.{}'.format(SuperDirr, suff)
    
            log = '{}log.cm.{}'.format(SuperDirr, suff)
    
            self.infile = AerofInputFile(fname,
                                         [prob, inp, out, nlrfs, nlrpp, 
                                          equa, bc, space, time], log)
    
        def getHrGeom(self):
            """ get information relevant to the geometry of the new reduced mesh """
            hrnodes = np.loadtxt('{}nonlinearrom/hrmesh.top.nodes'.format(self.SuperDirr), 
                                 dtype=np.int32, skiprows=1)
            hrnodes = hrnodes[:, 1]-1
            return hrnodes, self.hfrg
    ############ SET UP ROM ONLINE###############################################
    
    
    class ROM(common):
        """ HROM run """
        def __init__(self, frg, p=None, HDMind=0, pind=0, hyper=True, hrtest=False, evaluate=False, icweights=None):
            """ 
            pind is the index of the candidate point.
            If hrtest, only run 1 iteration.
            If evaluate, write to evaluate/ dir
            """
    
            super().__init__()
            self.evaluate = evaluate
            if self.evaluate:
                prefix = 'evaluate/'
                self.liftdragname = '../postpro/liftdrag.out'
                self.forcename = '../postpro/force.out'
                self.redcoordname = 'ReducedCoords.out'
                self.presscoeffname = 'PressureCoefficient.bin'
                self.skinfricname = 'SkinFriction.bin'
                self.machname = 'Mach.bin'
                self.dispname = 'Displacement.bin'
                self.velname = 'Velocity.bin'
                self.fluxresname = 'FluxRes.bin'
                self.cvname = 'ControlVolume.bin'
            else:
                prefix = ''
                self.liftdragname = ''
                self.forcename = ''
                self.redcoordname = ''
                self.presscoeffname = ''
                self.skinfricname = ''
                self.machname = ''
                self.dispname = ''
                self.velname = ''
                self.fluxresname = ''
                self.cvname = ''
                
            self.HDMind = HDMind
            self.frg = frg
            self.hrtest = hrtest
            self.pind = pind
            
            self.hyper = hyper

            if self.hrtest or settings.DoComp:
                self.resvecname = 'ResVec.out'
                self.jacvecname = 'JacVec.out'
            else:
                self.resvecname = ''
                self.jacvecname = ''

            if self.hyper:
                name = 'hrom'
            else:
                name = 'rom'

            if settings.HROMlscratch and not self.evaluate and self.hyper:
                self.SuperDir = '{}/{}/{}runs{:03d}/point{:03d}/'.format(
                  settings.HROMlscratch + os.environ.get('USER'),
                  prefix,name,HDMind,pind)
            else:
                self.SuperDir = '{}{}{}runs{:03d}/point{:03d}/'.format(
                  settings.MasterDir, prefix, name, HDMind,pind)
            self.SuperDirr = '{}reductionrun{:03d}/'.format(settings.MasterDir, HDMind)
            self.p = p

            # Create necessary directories
    
            if not os.path.exists(self.SuperDir):
                os.makedirs(self.SuperDir)
    
            neededDIRs = ['results/','postpro/','deform/']
    
            for dirr in neededDIRs:
                os.makedirs('{}{}'.format(self.SuperDir, dirr))
    
            # Write parameter textfile
            nP = p
            pars = open('{}parameters.txt'.format(self.SuperDir), 'w+')
            lines = ['{}\n'.format(len(nP))]
    
            for x in nP:
                lines.append('{}\n'.format(x))
    
            pars.writelines(lines)
            pars.close()
    
            # Write linear ND inerpolation weights
            
            if icweights is not None:
                self.icname = '{}icweights.txt'.format(self.SuperDir)
                icw = open(self.icname, 'w+')
                lines = ['{}\n'.format(len(icweights[0]))]
                
                for x in icweights[0]:
                    lines.append('{}\n'.format(x))

                icw.writelines(lines)
                icw.close()
            else:
                self.icname=''

        def create_input_file(self):
            """Define the input file for this problem."""
            mach = self.p[0]         # this simulation is parameterized by Mach number
            alpha = self.p[1]         # and angle of attack
    
            SuperDir = self.SuperDir
            SuperDirr = self.SuperDirr
    
            prob = AerofInputBlock('Problem',
                                   ['Type', 'SteadyNonlinearRom'],
                                   ['Mode', settings.DimOrNon])

            LaplaceSnapshotdata_file = self.get_laplace_snapshot_data_file(hyper=self.hyper)

            inp = AerofInputBlock('Input',
                                  ['GeometryPrefix', '{}'.format(self.frg.geom_pre)],
                                  ['MultipleSolutionsData', '{}parsoldata.txt'.format(settings.MasterDir)],
                                  ['MaxInterpolatedSolutions', settings.MaxIntSols],
                                  ['ParametricDistanceExponent', settings.DistExp],
                                  ['ParameterData', '{}parameters.txt'.format(SuperDir)],
                                  ['InterpICWeights', self.icname],
                                  ['Position', '{}deform/Position.bin'.format(SuperDir)],
                                  ['WallDistance', '{}deform/WallDistance.bin'.format(SuperDir)],
                                  ['LaplaceSnapshotData', LaplaceSnapshotdata_file]
                                  )
    
            if settings.rbf:
                GenManifoldName = '{}nonlinearrom/cluster0/'.format(SuperDirr)
            else:
                GenManifoldName = ''

            if self.hyper:
                files = AerofInputBlock('Files',
                                    ['StatePrefix', 'state'],
                                    ['StateBasis' , 'hrmesh.rob.reduced'],
                                    ['ReducedMeshPrefix', 'hrmesh'],
                                    ['StateBasisReference', 'hrmesh.ref.reduced'],
                                    ['GeneralManifoldRbfName', GenManifoldName])
            else:
                files = AerofInputBlock('Files',
                                    ['StatePrefix', 'state'],
                                    ['GeneralManifoldRbfName', GenManifoldName])

            if self.HDMind<settings.ClusteringStart:
                NC = 1
            else:
                NC = settings.NumClusters

            nlrfs = AerofInputBlock('NonlinearRomFileSystem',
                                    ['Prefix', '{}'.format(SuperDirr)],
                                    ['NumClusters', NC],
                                    ['TopLevelDirectory', 'nonlinearrom'],
                                    ['Files', files])
   
            if self.hyper:
                SA = 'ECSW'
            else:
                SA = 'None'

            if self.evaluate:
                CS = settings.ComponentScaling
            else:
                CS = settings.ComponentScalingIndicator and not self.hrtest

            nlro = AerofInputBlock('NonlinearRomOnline',
                                   ['Projection', settings.Projection],
                                   ['ComponentScaling', CS],
                                   ['LeastSquaresSolver', 'QR'],
                                   ['ShiftVectorType', settings.ShiftType],
                                   ['SystemApproximation', SA],
                                   ['MaximumEnergy', settings.StateBasisMaxEnergy],
                                   ['MaximumDimension', settings.StateBasisMaxDim],
                                   ['UseGeneralManifold', settings.rbf])

            rest = AerofInputBlock('Restart',
                                   ['Prefix', '{}references/'.format(SuperDir)],
                                   ['FilePackage',''],
                                   ['RestartData', ''],
                                   ['Solution', ''],
                                   ['WallDistance', ''],
                                   ['LimiterPhi', ''],
                                   ['Fixes', ''])


            post = AerofInputBlock('Postpro',
                                   ['Frequency', 0],
                                   ['Prefix', '{}results/'.format(SuperDir)],
                                   ['LiftandDrag', self.liftdragname],
                                   ['Force', self.forcename],
                                   ['PressureCoefficient', self.presscoeffname],
                                   ['SkinFrictionCoefficient', self.skinfricname],
                                   ['Mach', self.machname],
                                   ['Displacement', self.dispname],
                                   ['Velocity', self.velname],
                                   ['FluxResidual', self.fluxresname],
                                   ['ControlVolume', self.cvname],
                                   ['Residual', '../postpro/Residual.out'])
    
            nlr = AerofInputBlock('NonlinearROM',
                                  ['Prefix' , '{}postpro/'.format(SuperDir)],
                                  ['ReducedCoordinates', self.redcoordname],
                                  ['ReducedResidual', self.resvecname],
                                  ['ReducedJacobian', self.jacvecname])
    
            outp = AerofInputBlock('Output',
                                   ['Postpro', post],
                                   ['Restart', rest],
                                   ['NonlinearROM', nlr])
    
            equa, bc, space = self.get_common_blocks(mach, alpha, clip=True)
    
    
            if self.hrtest == False:
                HIts = settings.MaxItsHROM
                NIts = settings.MaxNewtItsHROM
            else:
                HIts = 1
                NIts = 1
  
            linesearch = AerofInputBlock('LineSearch',
                                         ['MaxIts', settings.linesearch_maxits],
                                         ['SufficientDecreaseFactor', settings.linesearch_decreaseFactor],
                                         ['ContractionFactor', settings.linesearch_contractFactor])
            
            if settings.useLineSearch:
                newton = AerofInputBlock('Newton',
                                        ['MaxIts', NIts],
                                        ['FailSafe', 'AlwaysOn'],
                                        ['Eps', 1e-6],
                                        ['LineSearch', linesearch])
            else:
                newton = AerofInputBlock('Newton',
                                        ['MaxIts', NIts],
                                        ['FailSafe', 'AlwaysOn'],
                                        ['Eps', 1e-6])

            impl = AerofInputBlock('Implicit',
                                   ['Type', 'SpatialOnly'],
                                   ['MatrixVectorProduct', settings.MvpROM],
                                   ['ExactImplementation', 'New'],
                                   ['TurbulenceModelCoupling','Strong'],
                                   ['Newton', newton])

            time = AerofInputBlock('Time',
                                   ['CheckPressure', settings.CheckPressure],
                                   ['CheckDensity', settings.CheckDensity],
                                   ['Form', settings.Form],
                                   ['MaxIts', HIts],
                                   ['Eps', 1e-10],
                                   ['Implicit', impl])
    
            fname = '{}input'.format(SuperDir)
            log = '{}log'.format(SuperDir)
    
            self.log = log
            self.resfile = '{}postpro/Residual.out'.format(SuperDir)
            if settings.TestType == 'Jacobian':
                self.resvecfile = '{}postpro/JacVec.out'.format(SuperDir)
            elif settings.TestType == 'Residual':
                self.resvecfile = '{}postpro/ResVec.out'.format(SuperDir)
    
            self.infile = AerofInputFile(fname,
                                         [prob, inp, outp, nlrfs, nlro, equa,
                                     bc, space, time], log)
    
    
    return {'HDM':HDM, 'POD': POD, 'CM': CM, 'ROWGEN': ROWGEN, 'ROM': ROM}
