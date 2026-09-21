import os
import itertools
import shutil
import time
import numpy as np
import scipy.interpolate
from joblib import Parallel, delayed, parallel_backend
from ushift_inject import Laplace_ushiftsnapdata, get_ushift_file
import sys
import rbf_trainer
from sobolGenerator import sobolGenerator

####################################################################################
def errorIndicatorParallel(ROM, point, pind, HDMind, ref_nodes, ref_stick, hrnodes, hfrg, hrhpc2, icweights = None):
    """Error indicator function to enable parallelism"""

    # Run HROM
    hrom = ROM(hfrg, p=point, HDMind=HDMind,
                     pind=pind, hrtest=False, hyper=True, evaluate=False, icweights = icweights)
    hrom.prep(ref_nodes, ref_stick, hrnodes)
    hrom.execute(hpc=hrhpc2)
    err = hrom.getRes()


    return err


class GreedyAlgorithmRun:

    def __init__(self, settings, runs, frg,  hpc, hrhpc, hpc1, hrhpc2):
        """ Initialize greedy algorithm member variables and preprocess the mesh """

        self.settings = settings
        self.frg = frg
        self.hfrg = None
        self.hrnodes = None
        self.hpc = hpc
        self.hrhpc = hrhpc
        self.hpc1 = hpc1
        self.hrhpc2 = hrhpc2
        # For reproducibility
        self.rng = np.random.default_rng(2021)

        # For keeping track of quantities throughout the procedure
        self.numSampledPoints = 0
        self.MuStars = []
        self.SampledPoints = []
        self.SampledInds = []
        self.QueriedPoints = []
        self.QPErrors = []
        self.MaxErrors = []
        self.Rebuild = []
        self.MuStars = []
        self.StarErrors = []

        # User set parameters
        self.tolerance = self.settings.MaxResNormTarget
        self.ei = 0.1*self.tolerance
        self.maxPoints = self.settings.MaxNumSamples
        self.NumInit = self.settings.NumInit
        self.numrand = self.settings.numrand
        self.ParEI = (self.settings.HROMindproc > 1 and not self.settings.DoComp
                        and self.settings.HyperReduced)

        # For evaluating the ROM against the HDM once the procedure is complete
        self.EvalLift = []
        self.EvalDrag = []
        self.EvalLiftROM = []
        self.EvalDragROM = []
        self.ForceErr = []
        self.CpErr = []
        self.SfErr = []
        self.Cps = []
        self.Sfs = []
        self.CpsROM = []
        self.SfsROM = []
        self.Pos = []
        # For timing
        self.timer = {'HDM': 0., 'Indicator': 0., 'POD': 0.,
                      'Training matrix': 0., 'NNLS': 0., 'Misc': 0.,
                      'Test hyperreduction': 0.}

        # Assign runs classes

        self.HDM = runs['HDM']
        self.POD = runs['POD']
        self.ROWGEN = runs['ROWGEN']
        self.CM = runs['CM']
        self.ROM = runs['ROM']


        # For mesh deformation
        ref_nodes = np.loadtxt('{}_nodes'.format(settings.TopFilePath), dtype=np.float64)
        self.ref_nodes = ref_nodes[:, 1:]

        ref_stick = np.loadtxt('{}_stick'.format(settings.TopFilePath), dtype=np.int32)
        self.ref_stick = ref_stick[:,2:]

        # Pre-defined discretization of parameter space
        self.createGridPoints()

        # Pre-select points for evaluation. This makes it consistent across runs if you use the same seed
        if self.settings.PreSelectEval:
            self.cpEval, self.ciEval = self.createCandidatePoints(size=self.settings.NumEvals)

        print('-----Initialized-----',flush=True)

    def prepareLaplaceBin(self, frg, laplace_bin_dir=None, hrnodes=None, hyper=False):
        """Generate Laplace snapshot binaries when Laplace shifts are enabled."""
        if self.settings.ShiftType != 'Laplace':
            return

        if self.settings.ShiftType == 'Laplace' and self.settings.LaplaceShiftEach is False:
            if laplace_bin_dir is None:
                u_shift_xpost = get_ushift_file(self.settings.TopFilePath, hrmesh_nodes=hrnodes)
                Laplace_ushiftsnapdata(self.settings, frg, u_shift_xpost, hyper=hyper)
                return
            else:
                os.makedirs(laplace_bin_dir, exist_ok=True)
                u_shift_xpost = get_ushift_file(self.settings.TopFilePath, hrmesh_nodes=hrnodes)
                Laplace_ushiftsnapdata(self.settings, frg, u_shift_xpost, laplace_bin_dir, hyper=hyper)
                return
                # raise Exception("For LaplaceShiftEach = False, please do not provide laplace_bin_dir. The code will automatically generate the Laplace snapshot binaries in the correct location for error indicator evaluation.")

        if self.settings.ShiftType == 'Laplace' and self.settings.LaplaceShiftEach is True:
            if laplace_bin_dir is not None:
                os.makedirs(laplace_bin_dir, exist_ok=True)
            else:
                return

            meshname = os.path.basename(self.settings.TopFilePath)
            meshpath = "{}HDMrun{:03d}/{}_deformed.top".format(self.settings.InitHDMPreCompDir, self.HDMind, meshname)
            xdmf_folder = '{}HDMrun{:03d}/xdmf_files/'.format(self.settings.MasterDir, self.HDMind)

            u_shift_xpost = get_ushift_file(meshpath, laplace_bin_dir, hrnodes, u_star_path=xdmf_folder)
            Laplace_ushiftsnapdata(self.settings, frg, u_shift_xpost, laplace_bin_dir, hyper)

    def run(self):
        """ Runs the greedy procedure in the MasterDir directory """

        # Sower and create binaries for Laplace solutions
        self.prepareLaplaceBin(self.frg)

        if not self.settings.HyperReduced:
            print("#### Starting Greedy Algorithm with ROM error indicator ####",flush=True)
        else:
            print("#### Starting Greedy Algorithm with HROM error indicator, Ell = {} ####".format(self.settings.Ell),flush=True)

        rst = self.settings.Restart

        if rst: self.restart()

        # Generate inital seeding of procedure
        nextP, ind = self.firstPoints()
        self.InitHDMnum = len(ind)

        if not self.settings.InitHDMruns:
            self.doHDMruns(nextP)

        while True:
            if not rst:
                for i in range(len(ind)):
                    idx = ind[i]
                    P = nextP[i]
                    self.Pind = idx+1
                    self.numSampledPoints += 1
                    self.HDMind = self.numSampledPoints
                    self.addPointToROM(P)
            else:
                self.formInterpolator()
            rst = False


            # Where we will query the ROM
            candP, candI = self.createCandidatePoints(self.settings.NumCandsPerIter)
            self.QueriedPoints.append(candP)

            maxEI = 0.0
            maxIndex = len(candP)+1
            ncp = len(candP)
            self.eI = np.zeros(ncp)

            # Break if we generated enough samples
            if self.numSampledPoints >= self.maxPoints:
                break

            # if rbf = True, run the rbf_trainer first
            if self.settings.rbf is True:
                self.run_RBF_trainer()

            # Check for maximum error indicator over all candidate points
            t = time.time()
            if self.ParEI:
                
                weights_ = [self.getICWeights(x) for x in candP]
                
                # sower_fluid_split using hfrg for HPROM run
                self.prepareLaplaceBin(
                    self.hfrg,
                    "{}hromruns{:03d}/Laplace-bin/".format(self.settings.EI_Laplacebin, self.HDMind),
                    self.hrnodes,
                    True,
                )

                ei_ = Parallel(n_jobs=self.settings.HROMindproc) \
                        (delayed(errorIndicatorParallel) (self.ROM,
                        candP[i], i+1,
                       self.HDMind, self.ref_nodes, self.ref_stick, self.hrnodes,
                        self.hfrg, self.hrhpc2[i % self.settings.HROMindproc], icweights = weights_[i])
                     for i in range(ncp))
            else:
                ei_ = []
                for i in range(ncp):
                    self.pind = i+1
                    ei_.append(self.errorIndicator(candP[i]))
 
            for i in range(ncp):
                self.eI[i] = ei_[i]
                if self.eI[i] > maxEI:
                    if candP[i] in self.SampledPoints:
                        continue
                    else:
                        maxEI = self.eI[i]
                        maxIndex = i
            nextP = [candP[maxIndex]]
            ind = [candI[maxIndex]]
            self.QPErrors.append(self.eI)
            self.MaxErrors.append(maxEI)

            elapsed = time.time() - t

            fmtL = ', '.join(["{:.2e}"]*len(candP[maxIndex])).format(*candP[maxIndex])
            print("----Error ind. max at {:03d}:[{}] with ".format(maxIndex+1,fmtL) +
                    "res = {:.2e},  t = {:.1e} s----".format(maxEI, elapsed),flush=True)
            if self.ParEI:
              self.timer['Indicator'] += elapsed

            # Write information to disk
            self.writeToFileGreedy()

            os.system("rm -r GreedyRuns/hromruns{:03d}".format(self.HDMind)) # Remove the hromruns directory to save space
            # Break if we hit the target tolerance
            if maxEI < self.tolerance:
                break

        self.writeToFileGreedy()
        print("#### Greedy Algorithm terminated with max error " + 
               "of {0} using {1} points #####".format(maxEI, self.numSampledPoints),flush=True)

        return maxEI

    def firstPoints(self):
        """ Generate initial points for greedy procedure. Can either be center, corners, or random. """
        
        cP = self.grid
        if self.NumInit == 'Center':
            # Center
            inds = [len(cP)//2]
        elif self.NumInit == 'Corners':
            # Corners
            inds = []
            corners = [[0, n-1] if n>1 else [0] for n in self.settings.NumPointsPerDim]
            coords = list(itertools.product(*corners))
            for c in coords:
                ind = np.ravel_multi_index(c, self.settings.NumPointsPerDim)
                inds.append(ind)
        elif isinstance(self.NumInit,int):
            # Random
            inds = self.rng.choice(len(cP), size=self.NumInit, replace=False)
        elif self.NumInit == 'Corners+Random':
            # Reset the RNG to ensure reproducibility
            self.rng = np.random.default_rng(42)

            inds = []
            # Corners
            corners = [[0, n-1] if n>1 else [0] for n in self.settings.NumPointsPerDim]
            coords = list(itertools.product(*corners))
            for c in coords:
                ind = np.ravel_multi_index(c, self.settings.NumPointsPerDim)
                inds.append(ind)

            # Append self.numrand Random inds that are not part of corners
            exclude_corners = np.setdiff1d(np.arange(len(cP)), inds)
            randinds = self.rng.choice(exclude_corners, size=self.numrand, replace=False)
            inds.extend([i for i in randinds])
        elif self.NumInit == 'Sobol':
            # Sobol quasi-random points (not on grid). Which of the 5-D
            # parameter vector [Mach, AoA, MaxCamberLoc, MaxCamber, Thickness]
            # actually varies is inferred from the bounds, so collapsing a
            # parameter to a single value in setup.py drops it from the
            # sequence without any change here.
            ranges = list(zip(self.settings.ParamsLowerBound,
                              self.settings.ParamsUpperBound))
            samples = sobolGenerator(ranges, self.numrand, include_corners=True,
                                     numToSkip=self.settings.numSkip)
            nP = [[round(v, 6) for v in p] for p in samples]
            # Assign pseudo-indices beyond grid range so they do not
            # interfere with grid-based candidate selection.
            offset = len(self.grid)
            inds = list(range(offset, offset + len(nP)))
            self.NumInit = len(inds)
            return nP, inds
        elif isinstance(self.NumInit, list):
            # User defined list of indices
            inds = self.NumInit

        nP = [cP[i] for i in inds]
        self.NumInit=len(inds)
        return nP, inds

    def doHDMruns(self, cp):
        """For precomputing HDM solutions at initial points. Active if InitHDMruns = True"""
        
        print("#### Running HDM precomputations ####",flush=True)
        i = 0
        for p in cp:
            self.Pind = i+1
            self.runHDM(self.Pind, p, precomp=False, maindir=True)
            i += 1
        print("#### Completed HDM precomputations ####",flush=True)

    def getSampledPoints(self):
        """Read a file for a previous greedy run and check which points were sampled"""
        
        parsolpath = "{}parsoldata.txt".format(self.settings.MasterDir)
        with open(parsolpath, 'r') as fp:
            lines = fp.readlines()
        num = int(lines[0])
        dim = int(lines[1])
        points = []
        for i in range(num):
            p = []
            j = 2+(dim+1)*i
            for k in range(dim):
                p.append(float(lines[j+1+k]))
            points.append(p)
        return points

    def buildReducedMesh(self, points, final=False):
        """ 
        Build a reduced mesh.
        If TrainOnCentroid = True, training matrix is evaluated at centeroid of snapshots.
        Otherwise, it is generated at each element of points.
        """

        if not self.settings.TrainOnCentroid:
            
            NS = min(len(points),self.settings.MaxNumTrainSnaps)
            TP = self.rng.choice(len(points), size=NS, replace=False)
            for g in TP:
                P = points[g]
                t = time.time()

                # Generate rows of training matrix
                self.gen = self.ROWGEN(self.frg, p=P, HDMind=self.HDMind, Gind=g+1)
                self.gen.execute(hpc=self.hpc)

                elapsed = time.time() - t
                print('----Generated block {:02d} of training matrix, t = {:.1e} s----'.format(g+1, elapsed),flush=True)
                self.timer['Training matrix'] += elapsed
        t = time.time()

        # Run NNLS algorithm to build mesh
        centroid = self.computeSampledCentroid()
        self.cm = self.CM(self.frg, p=centroid, HDMind=self.HDMind, rebuild=True, final=final, NumInit=self.InitHDMnum)
        self.cm.execute(hpc=self.hpc)

        # sower relevant reduced quantites
        self.cm.post()

        # get the hyperreduced nodes and the frg instance for the reduced mesh
        self.hrnodes, self.hfrg = self.cm.getHrGeom()

        elapsed = time.time()-t
        print('----Built reduced mesh, t = {:.1e} s----'.format(elapsed),flush=True)
        self.timer['NNLS'] += elapsed

    def restart(self):
        """ If evaluating the ROM built from a previous greedy run, set the necessary variables """

        try:
            self.SampledPoints = self.getSampledPoints()
            self.numSampledPoints = len(self.SampledPoints)
            self.HDMind = len(self.SampledPoints)
            centroid = self.computeSampledCentroid()
            self.cm = self.CM(self.frg, p=centroid, HDMind=self.HDMind, rebuild=True)
            self.hrnodes, self.hfrg = self.cm.getHrGeom()

            ### Ashok's fix to inform the program of the already-sampled points. ###
            # This is important for the greedy algorithm to not sample the same points again.
            self.SampledInds = []
            for p in self.SampledPoints:
                if p in self.grid:
                    self.SampledInds.append(self.grid.index(p) + 1)
            ### End of Ashok's fix ###
        except:
            print("Error with restarting, exiting...")
            exit()

    def evalROM(self):
        """Evaluate an existing ROM at a random array of points, and compare with HDM"""
        evaldir = "{}evaluate/".format(self.settings.MasterDir)
        if os.path.exists(evaldir):
            shutil.rmtree(evaldir)
        os.makedirs(evaldir)

        print("#### Building ROM from existing samples ####",flush=True)

        # Restart if needed
        if not (self.settings.RunGreedy):
            self.restart()
        centroid = self.computeSampledCentroid()

        # create points for ROM evaluation
        if self.settings.PreSelectEval:
            self.cp = self.cpEval
            self.ci = self.ciEval

            # # Hard code to force evaluation in the desired order
            self.cp = [self.grid[i] for i in range(len(self.grid))]
            self.ci = [i for i in range(len(self.grid))]
        else:
            self.cp, self.ci = self.createCandidatePoints(size=self.settings.NumEvals)

        # sower_fluid_split using hfrg/frg for HPROM/PROM run
        if self.settings.HyperReduced:
            hyper = True
            genfrg = self.hfrg
            hrnodes = self.hrnodes
        else:
            hyper = False
            genfrg = self.frg
            hrnodes = None
        
        if self.settings.ShiftType == 'Laplace' and self.settings.LaplaceShiftEach is False:
            self.prepareLaplaceBin(
                genfrg,
                "{}hromruns{:03d}/Laplace-bin/".format(self.settings.EI_Laplacebin, self.HDMind),
                hrnodes,
                hyper,
            )

        for i in range(len(self.cp)):
            point = self.cp[i]
            self.Pind = self.ci[i]+1
            self.pind = i+1

            # # Run HDM
            # self.runHDM(i+1, point, evaluate=True)

            # self.hdm2.postPro()

            # self.L, self.D = self.hdm2.getLD()
            # Fy, Fx = self.hdm2.getFyFx()
            # self.F = np.array([Fx, Fy])
            # self.pos, self.press, self.sf = self.hdm2.getSkinForces(self.ref_nodes)

            if self.settings.ShiftType == 'Laplace' and self.settings.LaplaceShiftEach is True:
                storeXPOST = "{}hromruns{:03d}/point{:03d}/Laplace-bin/".format(evaldir, self.HDMind, self.pind)

                os.makedirs(storeXPOST, exist_ok=True)
                meshname = os.path.basename(self.settings.TopFilePath)
                meshpath = "{}HDMrun{:03d}/{}_deformed.top".format(self.settings.MasterDir, self.pind, meshname)
                xdmf_folder = '{}HDMrun{:03d}/xdmf_files/'.format(self.settings.MasterDir, self.pind)

                u_shift_xpost = get_ushift_file(meshpath, storeXPOST, hrnodes, u_star_path=xdmf_folder)
                Laplace_ushiftsnapdata(self.settings, genfrg, u_shift_xpost, storeXPOST, hyper)

            # Run ROM
            _ = self.errorIndicator(point, evaluate=True)
            
        #     self.genrom.postPro()

        #     self.Lr, self.Dr = self.genrom.getLD()
        #     Fyr, Fxr = self.genrom.getFyFx()
        #     self.Fr = np.array([Fxr, Fyr])
        #     self.posr, self.pressr, self.sfr = self.genrom.getSkinForces(self.ref_nodes,self.hrnodes)

        #     self.saveResults()
        # # Write results to file
        # self.writeToFileEval()
        print("#### ROM evaluation done ####",flush=True)

    def testHyperreduction(self):
        """ Compare the reduced residual/jacobian on the reduced mesh vs on the full mesh """
        try:
            hResVec = self.genrom.getResVec()
            rResVec = self.testrom.getResVec()
            herr = np.linalg.norm(hResVec-rResVec)/np.linalg.norm(rResVec)
        except IndexError:
            herr = 1.1*self.settings.Delta+1e-6

        return herr

    def errorIndicator(self, point, evaluate=False):
        """Error indicator for greedy. Either runs the HROM or the ROM"""

        if self.settings.HyperReduced:
            hyper=True
            genfrg = self.hfrg
            hpc = self.hrhpc
            hrnodes = self.hrnodes
        else:
            hyper = False
            genfrg = self.frg
            hpc = self.hpc 
            hrnodes = None
        t = time.time()

        # if rbf = True, run the rbf_trainer first
        if self.settings.rbf is True:
            self.run_RBF_trainer()

        # Run HROM
        icweights = self.getICWeights(point)
        self.genrom = self.ROM(genfrg, p=point, HDMind=self.HDMind,
                         pind=self.pind, hyper=hyper, hrtest=False, evaluate=evaluate, icweights=icweights)
        self.genrom.prep(self.ref_nodes, self.ref_stick, hrnodes)
        self.genrom.execute(hpc=hpc)
        err = self.genrom.getRes()
        elapsed = time.time() - t
        fmtL = ', '.join(["{:.6e}"]*len(point)).format(*point)
        print("----{:03d}: ROM executed at [{}] with ".format(self.pind,fmtL) +
                "res = {:.2e},  t = {:.1e} s----".format(err, elapsed),flush=True)
        self.timer['Indicator'] += elapsed

        if self.settings.DoComp and self.settings.HyperReduced:
            # For debugging, might want to test hyperreduction at each point.
            t = time.time()
            self.testrom = self.ROM(self.frg, p=point, HDMind=self.HDMind,
                           pind=self.pind, hyper = False, hrtest=True, evaluate=False, icweights=icweights)
            self.testrom.prep(self.ref_nodes, self.ref_stick)
            self.testrom.execute(hpc=self.hpc)
            herr = self.testHyperreduction()
            elapsed = time.time()-t
            print('----Tested hyperreduction, error = {:.2e}, t = {:.1e} s---'.format(herr, elapsed),flush=True)
            self.timer['Test hyperreduction'] += elapsed

        return err


    def runHDM(self, ind, point, evaluate=False, precomp=False, maindir=False):
        """ 
        Run HDM at point. 
        If evaluate, write to the evaluate/ directory.
        If precomp, then just create a symlink to a precomputed solution.
        If maindir, write to the run directory rather then MasterDir (for precomputations for example).
        """

        t = time.time()

        # Initialize HDM
        self.hdm1 = self.HDM(self.frg, p=point, HDMind=ind, Pind=self.Pind, step=1,
                         evaluate=evaluate, precomp=precomp, maindir=maindir)
        self.hdm2 = self.HDM(self.frg, p=point, HDMind=ind, Pind=self.Pind, step=2,
                         evaluate=evaluate, precomp=precomp, maindir=maindir)
        fmtL = ', '.join(["{:.6e}"]*len(point)).format(*point)
        if not precomp:
            # Run the HDM
            print('----Begin HDM at [{}]----'.format(fmtL),flush=True)
            self.hdm1.prep(self.ref_nodes, self.ref_stick)
            self.hdm1.execute(hpc=self.hpc)
            self.hdm2.execute(hpc=self.hpc)

        # Update snapshot matrix
        if not evaluate and not maindir:
            self.hdm2.post()

        # Sower and xp2exo if desired
        if self.settings.PostproHDM:
            self.hdm2.postPro()
        # fmtL = ', '.join(["{:.2e}"]*len(point)).format(*point)
        elapsed = time.time()-t
        print('----HDM executed at [{}], t = {:.1e} s----'.format(fmtL, elapsed),flush=True)
        self.timer['HDM'] += elapsed

    def run_RBF_trainer(self):
        print('----rbf_trainer() Serial errorIndicator----', flush=True)
        with open('{}reductionrun{:03d}/log.pod'.format(self.settings.MasterDir, self.HDMind), 'r') as f:
            lines = f.readlines()
            # extract keyword "Retaining 27 of 35 vectors in POD basis"
            for line in lines:
                if "Retaining" in line:
                    words = line.split()
                    numPODretained = int(words[1])
                    break

        rbf_dimV = int(min(self.settings.rbf_dimVmax, np.floor(numPODretained * self.settings.rbf_pctVdim)))
        print(r'dim(V)={}, dim(\bar V)={}'.format(rbf_dimV, numPODretained-rbf_dimV), flush=True)

        if not os.path.exists("{}reductionrun{:03d}/nonlinearrom/cluster0/".format(self.settings.MasterDir, self.HDMind)):
            os.makedirs("{}reductionrun{:03d}/nonlinearrom/cluster0/".format(self.settings.MasterDir, self.HDMind))

        sys.argv = [
            "rbf_trainer.py",
            "--data_file", "{}reductionrun{:03d}/nonlinearrom/cluster0/state.coords".format(self.settings.MasterDir, self.HDMind),
            "--dimV", "{}".format(rbf_dimV), 
            "--output_path", "{}reductionrun{:03d}/nonlinearrom/cluster0/".format(self.settings.MasterDir, self.HDMind),
            "--skip_columns", "0",
            "--skip_rows", "1",
        ]
        
        rbf_trainer.main()

    def buildROM(self):
        """ Build ROM from collected snapshots """
        t = time.time()
        centroid = self.computeSampledCentroid()
            
        # Compute POD
        self.pod = self.POD(self.frg, p=centroid, HDMind=self.HDMind)
        self.pod.execute(hpc=self.hpc)

        elapsed = time.time()-t
        print('----POD computed, t = {:.1e} s----'.format(elapsed),flush=True)
        self.timer['POD'] += elapsed
        rebuild = False
        final = self.numSampledPoints >= self.maxPoints

        # if rbf = True, run the rbf_trainer
        if self.settings.rbf is True:
            self.run_RBF_trainer()

        if self.settings.HyperReduced:
            rebuild = True
            if self.HDMind >= self.NumInit+1:
                # Check quality of hyperreducton at mus
                mus = self.computeMuStar()
                self.MuStars.append(mus)
                rebuild = self.checkForRetrain(mus, final = final)

                # Always rebuild on the last iteration
                if final:
                    rebuild = True
        if rebuild:
            # rebuild reduced mesh
            self.Rebuild.append(1)


            self.buildReducedMesh(self.SampledPoints, final=final)
        else:
            # use previous reduced mesh
            self.Rebuild.append(0)

        print('---ROM built with {} samples---'.format(self.HDMind),flush=True)
        for i in range(3):
            print('***********************************************************',flush=True)
        print(' ',flush=True)

    def addPointToROM(self, point):
        """Sample and update ROM"""
        self.SampledPoints.append(point)
        self.SampledInds.append(self.Pind)
        ind = self.HDMind

        if self.HDMind <= self.NumInit and self.settings.InitHDMPreCompDir is not None:
            precomp = True
        else:
            precomp = False
        self.runHDM(ind, point, precomp=precomp)
        centroid = self.computeSampledCentroid()
        if self.HDMind >= self.NumInit:
            self.formInterpolator()
            self.buildROM()

    def checkForRetrain(self, point,final=False):
        """Query the accuracy of the hyperreduction by comparing the ROM residual and HROM residual"""
        t = time.time()

        # Construct reduced mesh from previous weights
        self.cm = self.CM(self.frg, p=point, HDMind=self.HDMind, rebuild=False, final=final)
        self.cm.execute(hpc=self.hpc)
        self.cm.post()
        self.hrnodes, self.hfrg = self.cm.getHrGeom()
        
        # sower_fluid_split using hfrg for HPROM run
        # if self.settings.LaplaceShift is True:
        self.prepareLaplaceBin(
            self.hfrg,
            "{}hromruns{:03d}/Laplace-bin/".format(self.settings.EI_Laplacebin, self.HDMind),
            self.hrnodes,
            True,
        )

        icweights = self.getICWeights(point)

        # Run HROM with new ROB on old mesh
        self.genrom = self.ROM(self.hfrg, p=point, HDMind=self.HDMind, pind=0, hyper=True, hrtest=True, icweights = icweights)
        self.genrom.prep(self.ref_nodes, self.ref_stick, self.hrnodes)
        self.genrom.execute(hpc=self.hrhpc)

        # Run ROM with new ROB on old mesh
        self.testrom = self.ROM(self.frg, p=point, HDMind=self.HDMind, pind=0, hyper=False, hrtest=True, icweights = icweights)
        self.testrom.prep(self.ref_nodes, self.ref_stick)
        self.testrom.execute(hpc=self.hpc)

        herr = self.testHyperreduction()

        elapsed = time.time() - t
        fmtL = ', '.join(["{:.6e}"]*len(point)).format(*point)
        print('----Tested hyperreduction at [{}], error = {:.2e}, t = {:.1e} s----'.format(fmtL, herr, elapsed),flush=True)
        self.timer['Test hyperreduction'] += elapsed
        self.StarErrors.append(herr)
        if herr > self.settings.Delta:
            return True
        else:
            return False

    def createCandidatePoints(self, size=1):
        """ generate size points in the unsampled parameter space"""
        t = time.time()
        unsampled = [i for i in range(
            len(self.grid)) if i not in self.SampledInds]
        
        # Sampled without replacement from the unsampled indices
        cI = self.rng.choice(unsampled, size=size, replace=True).tolist()

        # Candiddate points
        cP = [self.grid[i] for i in cI]

        elapsed = time.time() - t
        print('----Generated new candidate points, t = {:.1e} s----'.format(elapsed),flush=True)
        self.timer['Misc'] += elapsed

        return cP, cI

    def createGridPoints(self):
        """Create orthogonal grid of points tht will serve as parameter soace"""

        t = time.time()
        N = self.settings.NumPointsPerDim
        self.ParamsLowerBound = self.settings.ParamsLowerBound
        self.ParamsUpperBound = self.settings.ParamsUpperBound
        dim = len(self.ParamsLowerBound)
        sampling = [np.linspace(0.0, 1.0, n) for n in N]
        ncp = 1
        for n in N:
            ncp *= n
        cP = []

        for i in range(ncp):
            coords = np.unravel_index(i, N)
            cPP = []
            for j in range(dim):
                cPP.append(round(self.ParamsLowerBound[j] + sampling[j][coords[j]]
                                 * (self.ParamsUpperBound[j]-self.ParamsLowerBound[j]), 6))
            cP.append(cPP)
        elapsed = time.time() - t
        print('----Generated grid of points, t = {:.1e} s----'.format(elapsed))
        self.timer['Misc'] += elapsed
        self.grid = cP

    def computeMuStar(self):
        """
        Compute hyperreduction testing point. 
        Either the first sampled point, the last or the furthest from all points
        """
        scale = np.absolute(np.array(self.ParamsLowerBound) -
                            np.array(self.ParamsUpperBound))
        for j in range(scale.size):
            if scale[j]==0: 
                scale[j]=1
    
        SP = np.array(self.SampledPoints)/scale
        QP = np.array(self.grid)/scale
        dist = 0
        if self.settings.MuStarType == 'Farthest':
            for q in QP:
                qdist = 1000
                for s in SP:
                    testdist = np.linalg.norm(s-q)
                    if testdist < qdist:
                        qdist = testdist
                if qdist > dist:
                    dist = qdist
                    MuStar = q

        elif self.settings.MuStarType == 'Latest':
            MuStar = SP[-1]
        elif self.settings.MuStarType == 'First':
            MuStar = SP[0]
        return MuStar*scale

    def computeSampledCentroid(self):
        """compute centroid (mean) of sampled points"""

        SP = np.array(self.SampledPoints)
        cent = np.mean(SP, axis=0)
        return cent

    def formInterpolator(self):
        if self.settings.LinearNDInterp:
            values = np.eye(len(self.SampledPoints))
            points = np.array(self.SampledPoints)
            self.interpolator = scipy.interpolate.LinearNDInterpolator(points, values, fill_value=np.nan, rescale=True)

    def getICWeights(self,p):

        if self.settings.LinearNDInterp:
            return self.interpolator(p)
        else:
            return None

    def saveResults(self):
        # Save results
        self.EvalLift.append(self.L)
        self.EvalDrag.append(self.D)
        self.EvalLiftROM.append(self.Lr)
        self.EvalDragROM.append(self.Dr)

        self.ForceErr.append( np.linalg.norm(self.F-self.Fr, ord=1)/np.linalg.norm(self.F,ord=1))

        self.CpErr.append( np.linalg.norm(self.press-self.pressr,ord=1)/np.linalg.norm(self.press,ord=1))

        self.SfErr.append( np.linalg.norm(self.sf-self.sfr,ord=1)/np.linalg.norm(self.sf,ord=1))

        self.Cps.append(self.press)
        self.Sfs.append(self.sf)
        self.CpsROM.append(self.pressr)
        self.SfsROM.append(self.sfr)
        self.Pos.append(self.pos)

    def writeToFileGreedy(self):
        """Write info to disk during greedy procedure"""

        with open('{}SampledPointsOutput.txt'.format(self.settings.MasterDir), 'w+') as f:
            for i in range(len(self.SampledPoints)):
                point = self.SampledPoints[i]
                fmtL = ', '.join(["{:.6e}"]*len(point)).format(*point)
                line = "{:03d}: [{}]\n".format(i+1, fmtL)
                f.write(line)

        with open('{}MuStarsOutput.txt'.format(self.settings.MasterDir), 'w+') as f:
            for i in range(len(self.MuStars)):
                point = self.MuStars[i]
                fmtL = ', '.join(["{:.6e}"]*len(point)).format(*point)
                line = "{:03d}: [{}]\n".format(i+1, fmtL)
                f.write(line)

        with open('{}StarErrorsOutput.txt'.format(self.settings.MasterDir), 'w+') as f:
            for i in range(len(self.StarErrors)):
                line = "{:03d}: {:.3e}\n".format(i+2, self.StarErrors[i])
                f.write(line)

        with open('{}MaxErrorsOutput.txt'.format(self.settings.MasterDir), 'w+') as f:
            for i in range(len(self.MaxErrors)):
                line = "{:03d}: {:.3e}\n".format(i+1, self.MaxErrors[i])
                f.write(line)

        with open('{}Timing_{}.txt'.format(self.settings.MasterDir, self.settings.MaxNumSamples), 'w+') as f:
            for key, value in self.timer.items():
                f.write('{}: {:.2e} s\n'.format(key, value))

    def writeToFileEval(self):
        """ Write info to disk for an evaluation run """

        with open('{}evaluate/EvalLiftDrag_{}.txt'.format(self.settings.MasterDir, self.settings.MaxNumSamples), 'w+') as f:
            line = "# Lift, LiftROM, Drag, DragROM\n"
            f.write(line)
            for i in range(len(self.cp)):
                line = "{:.3e}     {:.3e}      {:.3e}      {:.3e}\n".format(self.EvalLift[i], self.EvalLiftROM[i],
                                                                            self.EvalDrag[i], self.EvalDragROM[i])
                f.write(line)

        with open('{}evaluate/EvalErrors_{}.txt'.format(self.settings.MasterDir, self.settings.MaxNumSamples), 'w+') as f:
            line = "# CpErr, SfErr, ForceErr\n"
            f.write(line)
            for i in range(len(self.cp)):
                line = "{:.3e}      {:.3e}   {:.3e}\n".format(self.CpErr[i], 
                                                              self.SfErr[i], 
                                                              self.ForceErr[i])
                f.write(line)


        with open('{}evaluate/EvalPoints_{}.txt'.format(self.settings.MasterDir, self.settings.MaxNumSamples), 'w+') as f:
            for i in range(len(self.cp)):
                point = self.cp[i]
                fmtL = ', '.join(["{:.6e}"]*len(point)).format(*point)
                line = "{:03d}: [{}]\n".format(i+1, fmtL)
                f.write(line)

        for i in range(len(self.Cps)):
            with open('{}evaluate/Cps_{}.txt'.format(self.settings.MasterDir, i+1), 'w+') as f:
                cp = self.Cps[i]
                cpr = self.CpsROM[i]

                f.write('# Cp,      CpROM\n')
                for j in range(len(cp)):
                        f.write('{:.5e}     {:.5e}\n'.format(cp[j], cpr[j]))

            with open('{}evaluate/Sfs_{}.txt'.format(self.settings.MasterDir, i+1), 'w+') as f:
                sf = self.Sfs[i]
                sfr = self.SfsROM[i]

                f.write('# Sf,      SfROM\n')
                for j in range(len(sf)):
                        f.write('{:.5e}     {:.5e}\n'.format(sf[j], sfr[j]))
            with open('{}evaluate/Pos_{}.txt'.format(self.settings.MasterDir,i+1), 'w+') as f:
                pos = self.Pos[i]

                f.write('# X, Y, Z\n')
                for j in range(len(pos)):
                    f.write('{:.5e}     {:.5e}      {:.5e}\n'.format(pos[j,0], pos[j,1], pos[j,2]))
