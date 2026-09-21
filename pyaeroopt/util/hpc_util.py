import os, subprocess

def execute_code(exec_str, log=None, make_call=True, bg=False):
    if log is not None:
        exec_str = "{0:s} >& {1:s}".format(exec_str, log)
    if bg:
        exec_str = "{0:s} &".format(exec_str)
    # print(exec_str)

    #RT
    #if make_call: subprocess.call(exec_str, shell=True)
    if make_call:
       done = False
       while not done:
         ret = subprocess.call(exec_str, shell=True)
         if ret == 0:
           # print("success for %s detected" % exec_str, flush = True)
           done = True
         else:
           #RT - search for srun failure
           print("failure detected for %s " % exec_str, flush = True)
           cmd2 = 'grep -i "fatal" %s' % log
           ret2 = subprocess.call(cmd2, shell=True)
           if ret2 != 0:
             #RT - other failure
             done = True
           else:
             print("repeating  %s " % exec_str, flush = True)


def execute_str(bin, infile, bg=False):

    exec_str = "{0:s} {1:s}".format(bin, infile)
    if bg: exec_str = "{0:s} &".format(exec_str)
    return exec_str

def mpi_execute_str(bin, infile, nproc=1, mpi=None, bg=False, machinefile=None):

    if mpi is None: mpi = os.path.expandvars('$MPIEXEC')
    # mpi = 'srun'

    # Allow bin, infile, ncpu to be lists of length 2 for case where MPI
    # communicates between two codes
    if type(bin)    is str: bin    = [bin]
    if type(infile) is str: infile = [infile]
    if type(nproc)  is int: nproc  = [nproc]

    # Execution string
    exec_str = "{0:s} -n {1:d} {2:s} {3:s}".format(mpi, nproc[0], bin[0],
                                                   infile[0])
    if len(bin) == 2 and len(infile) == 2 and len(nproc) == 2:
        exec_str = "{0:s} : -n {1:d} {2:s} {3:s}".format(exec_str, nproc[1],
                                                         bin[1], infile[1])
    if machinefile is not None:
        exec_str = "{0:s} -machinefile {1:s}".format(exec_str, machinefile)
    if bg: exec_str = "{0:s} &".format(exec_str)

    return exec_str

def batch_maui_pbs():
    pass

def batch_slurm_pbs():
    pass

def batch_machine_specific():
    pass
