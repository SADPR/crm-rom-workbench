################################################################################
# Import modules                               
################################################################################
import pyaeroopt
import re
import numpy as np
import socket
import os
import shutil
from setup import Settings
from runs import getRuns
from GreedyAlgorithm import GreedyAlgorithmRun
################################################################################

def getSLURMInfo():
  """Read SLURM information to initialize the list of nodes and number of cores"""
  unodes  = os.getenv('SLURM_JOB_NODELIST')
  if unodes == None:
    return 0, []
  if isinstance(unodes, str):
    nodes = [unodes]
  else:
    nodes = list(unodes)
  nodenames=[]
  for node in nodes:
    if node.find('[')!=-1:
      idx = node.find('[')
      prefix = node[:idx]
      numbers = re.findall(r'\d+',node[idx+1:-1])
      a=re.findall(r'\d+-\d+',node[idx+1:-1])
      if a:
        for ran in a:
          bounds = re.findall(r'\d+', ran)
          interval = np.arange(int(ran[:2]), int(ran[-2:])+1)
          for num in interval:
            numbers+=[str(num).zfill(2)]
          numbers = list(set(numbers))
      for number in numbers:
        nodenames += [(prefix + number)]
    else:
      nodenames+=[node]
  ppn = int(os.getenv('SLURM_CPUS_ON_NODE'))
  return ppn, nodenames

################################################################################
# Create frg and hpc objects
################################################################################
settings = Settings()
runs = getRuns(settings)

frg = pyaeroopt.interface.Frg(top='{}.top'.format(settings.TopFilePath), geom_pre="{}data/{}".format(settings.MasterDir, 
                                                                            settings.GeometryPrefix))

hpc = pyaeroopt.interface.Hpc(machine="independence",
                              batch=False,
                              bg=False,
                              nproc=settings.HDMnproc)

hrhpc = pyaeroopt.interface.Hpc(machine="independence",
                                batch=False,
                                bg=False,
                                nproc=settings.HROMnproc)

hpc1 = pyaeroopt.interface.Hpc(machine="independence",
                               batch=False,
                               bg=False,
                               nproc=1)


################################################################################
# Make directories
################################################################################
if len(settings.MasterDir) > 0 and (settings.RunGreedy):
    if not settings.Restart:
        if os.path.exists(settings.MasterDir):
            raise Exception(("MasterDir exists while RunGreedy is true. "
                        "Consider running clean.sh, chaning MasterDir or activating evaluation mode"))
        else:
            os.makedirs(settings.MasterDir)
    else:
        print("Continuing greedy algorithim in {} from a previous run".format(settings.MasterDir))


# Copy settings file to the directory for future reference
shutil.copy('setup.py','{}settings.readonly'.format(settings.MasterDir))
slurmdir = '{}slurm/'.format(settings.MasterDir)
if os.path.exists(slurmdir):
    shutil.rmtree(slurmdir)
os.makedirs(slurmdir)

bindir = '{}data/'.format(settings.MasterDir)
if os.path.exists(bindir):
    shutil.rmtree(bindir)
os.makedirs(bindir)


################################################################################
# Preprocess mesh
################################################################################
frg.part_mesh(settings.HDMnclust, log='/dev/null')
frg.sower_fluid_top([settings.HDMnclust, settings.HDMnproc, settings.HROMnproc, 1],
                    settings.HDMnclust, log='/dev/null')
pyaeroopt.util.frg_util.run_cd2tet_fromtop('{}.top'.format(settings.TopFilePath), '{}.sinus'.format(frg.geom_pre), log='/dev/null')
frg.sower_fluid_split(file2split='{}.sinus.dwall'.format(frg.geom_pre),
                      out='{}.dwall'.format(frg.geom_pre), nclust=settings.HDMnclust, log='/dev/null')

################################################################################
# Enable Python parallelism and/or local scratch spaces
################################################################################
ppn, cnodes = getSLURMInfo()
username = os.environ.get('USER')
if settings.TRlscratch:
  # Machine file to keep node/cpu assignments between aero-f runs
  f = open('{}mf'.format(slurmdir), 'w')
  for i in range(len(cnodes)):
    f.write('%s slots=%d\n' % (cnodes[i], ppn))
  f.close()
  hpc.mpi = 'mpirun --machinefile {}mf'.format(slurmdir)

  # Create a username subdirectory in the local scratch of the allocated nodes
  os.system("srun -N %d -n %d mkdir -p %s/%s/tr" % (len(cnodes), len(cnodes), 
            settings.TRlscratch, username))

if settings.HROMlscratch:
  # Create a username subdirectory in the local scratch of the current node
  # os.system("mkdir -p %s/%s" % (settings.HROMlscratch, username))
  os.system("srun -N %d -n %d mkdir -p %s/%s/tr" % (len(cnodes), len(cnodes), 
            settings.HROMlscratch, username))
  # cnodes = [socket.gethostname()]
#else:
#  cnodes.remove(socket.gethostname())

hrhpc2 = []
maxi = len(cnodes)*int(int(ppn)/int(settings.HROMnproc))
for i in range(maxi):
  hrhpc2.append(pyaeroopt.interface.Hpc(machine = 'independence', batch = False,
                              bg=False, nproc = settings.HROMnproc))
  f = open('{}mf_{:02d}'.format(slurmdir,i), 'w')
  f.write('{} slots={}\n' .format(cnodes[i%(len(cnodes))], settings.HROMnproc))
  f.close()
  hrhpc2[i].mpi = 'mpirun --machinefile {}mf_{:02d} --bind-to none'.format(slurmdir,i)

#################################################################################
# run code
################################################################################
if __name__ == "__main__":
    gar = GreedyAlgorithmRun(settings, runs, frg, hpc, hrhpc, hpc1, hrhpc2)
    if settings.RunGreedy:
        ei = gar.run()
    if settings.EvalROM:
        gar.evalROM()
