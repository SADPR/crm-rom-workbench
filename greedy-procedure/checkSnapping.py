import numpy as np
import os
from pathlib import Path
import subprocess
from joblib import Parallel, delayed
import shutil

def count_folders(path):
    # '_Des' directories (e.g. point077_Des) are hand-copied comparison runs kept
    # alongside the real ones; counting them would push the loop past the last run
    return sum(1 for p in Path(path).iterdir() if p.is_dir() and not p.name.endswith('_Des'))

def readXPOST(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines[3:]:  # Skip the header
            values = line.split()
            if len(values) >= 1:
                if isinstance(values, list) and len(values) == 1:
                    data.append(float(values[0]))
                elif isinstance(values, list) and len(values) > 1:
                    data.append([float(v) for v in values])

        return np.array(data)

def checkSnapping(HDM_xpost_data, PROM_xpost_data, HDM_CVdata, PROM_CVdata, HDM_params, ROM_params, snapping_tol, output_file=None):
    numHDM = HDM_xpost_data.shape[-1]
    numROM = PROM_xpost_data.shape[-1]

    # write the snapped indices to another file
    snapped_idx_file = os.path.join(os.path.dirname(output_file), "snapped_indices_{}.txt".format(numHDM))

    print("Comparing HDM and PROM runs for snapping...")
    PROM_idx_snap = [] # to store the indices of snapped runs in PROM
    HDM_idx_snap = [] # to store the corresponding HDM indices for snapped runs

    # Both files are opened once for the whole numHDM x numROM sweep and written
    # through the held handles. Mode 'w' truncates whatever a previous run left
    # behind, so no explicit os.remove is needed.
    with open(output_file, 'w') as f_out, open(snapped_idx_file, 'w') as f_snap:
        for k in range(numHDM):
            hdm_value = HDM_xpost_data[..., k]
            # hdm_cv = HDM_CVdata[:,k] # control volume data for HDM run k
            # hdm_value *= hdm_cv[:,None] # scale by cell volume

            # Norm of the reference run: identical for every PROM run, so it is
            # computed once per HDM instead of once per pair
            if HDM_xpost_data.ndim == 3: # 3d array, each slice along the last dimension is a run
                hdm_norm = np.sum(hdm_value**2)**0.5
            elif HDM_xpost_data.ndim == 2: # 2d array: each column is a run
                hdm_norm = np.linalg.norm(hdm_value)

            for j in range(numROM):
                rom_value = PROM_xpost_data[..., j]
                # rom_cv = PROM_CVdata[:,j] # control volume data for PROM run j
                # rom_value *= rom_cv[:,None] # scale by cell volume

                DOF_wise_diff = hdm_value - rom_value
                if HDM_xpost_data.ndim == 3:
                    node_wise_diff = np.sum(DOF_wise_diff**2)**0.5
                elif HDM_xpost_data.ndim == 2:
                    node_wise_diff = np.linalg.norm(DOF_wise_diff)
                rel_diff = node_wise_diff / hdm_norm # relative difference

                # Write to file or print results
                f_out.write(f"HDM run {k+1}, PROM run {j+1}\n")
                f_out.write(f"HDM Parameters: {np.array2string(HDM_params[k, :], separator=', ')}\n")
                f_out.write(f"PROM Parameters: {np.array2string(ROM_params[j, :], separator=', ')}\n")
                f_out.write(f'Average Relative Difference: {np.mean(rel_diff):.4e}\n')
                f_out.write(f'Maximum Relative Difference: {np.max(rel_diff):.4e}\n')
                f_out.write('\n')

                # if average relative diff is below threshold and if parameters are different
                if np.mean(rel_diff) < snapping_tol and (HDM_params[k, :] != ROM_params[j, :]).any():
                    PROM_idx_snap.append(j+1)
                    HDM_idx_snap.append(k+1)

                    print(f"Snapping HDM {k+1} --- PROM {j+1}")
                    print(f"HDM Parameters: {HDM_params[k, :]}")
                    print(f"PROM Parameters: {ROM_params[j, :]}\n")

                    # Write to file
                    f_snap.write(f"HDM run {k+1}, PROM run {j+1}\n")
                    f_snap.write(f"HDM Parameters: {np.array2string(HDM_params[k, :], separator=', ')}\n")
                    f_snap.write(f"PROM Parameters: {np.array2string(ROM_params[j, :], separator=', ')}\n")
                    f_snap.write(f'Average Relative Difference: {np.mean(rel_diff):.4e}\n')
                    f_snap.write(f'Maximum Relative Difference: {np.max(rel_diff):.4e}\n')
                    f_snap.write('\n')

    # concatenate two lists
    # first column: PROM indices
    # second column: corresponding HDM indices
    idx_snap = np.vstack((PROM_idx_snap, HDM_idx_snap)).T

    return idx_snap

def readLog(log_file, numHDM, numROM):
    HDM_params = np.zeros((numHDM, 5))
    ROM_params = np.zeros((numROM, 5))

    i = 0
    j = 0
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'HDM executed at' in line:
                content = line.split('[')[1].split(']')[0]
                numbers = [float(x) for x in content.split(',')]

                if i < numHDM:
                    HDM_params[i, :] = numbers
                    i += 1

            elif 'ROM executed at' in line:
                content = line.split('[')[1].split(']')[0]
                numbers = [float(x) for x in content.split(',')]

                if j < numROM:
                    ROM_params[j, :] = numbers
                    j += 1

        return HDM_params, ROM_params

def runHROMpostpro(folder, testcase, numHDM, k, ROM_params):
    HROMpostpro_file = os.path.join(folder, testcase, 'HROMpostpro')
    tmp_file = HROMpostpro_file + f".point{k:03d}"

    if os.path.exists(tmp_file): os.remove(tmp_file) # remove existing file in case of previous runs
    shutil.copy(HROMpostpro_file, tmp_file) # One copy for each point for parallelization

    with open(tmp_file, "r") as f:
        original_hrom = f.read()

    if "Laplace" in testcase:
        ShiftVectorType = "Laplace"
        LaplaceSnapshotData = f"{GreedyFolder}/Laplace-bin/ushift.bin"
    else:
        ShiftVectorType = "None"
        LaplaceSnapshotData = ""
    
    if "NonDes" in testcase:
        ROM_form = "NonDescriptor"
    else:
        ROM_form = "Descriptor"

    hromruns_id = f"hromruns{numHDM:03d}"
    point_name = f"point{k  :03d}"
    mach = ROM_params[k-1, 0] # assuming Mach number is the first parameter
    beta = ROM_params[k-1, 1] # assuming beta is the second parameter

    updated_hrom = original_hrom.replace('point001', point_name)
    updated_hrom = updated_hrom.replace('hromruns002', hromruns_id)
    updated_hrom = updated_hrom.replace('Mach = 0.7;', f'Mach = {mach:.3f};')
    updated_hrom = updated_hrom.replace('Beta = 2.0;', f'Beta = {beta:.3f};')
    updated_hrom = updated_hrom.replace('Form = NonDescriptor;', f'Form = {ROM_form};')
    updated_hrom = updated_hrom.replace('ShiftVectorType = None;', f'ShiftVectorType = {ShiftVectorType};')
    updated_hrom = updated_hrom.replace('LaplaceSnapshotData = "";', f'LaplaceSnapshotData = "{LaplaceSnapshotData}";') 

    # Write to the temp file
    with open(tmp_file, "w") as f:
        f.write(updated_hrom)

    # Run the command
    subprocess.run(f"mpirun -n 8 $AEROF {os.path.basename(tmp_file)}", shell=True, check=True, cwd=f"{folder}/{testcase}/")
    print(f"ran HROMpostpro for point{k:03d}")

    # Remove the temporary HROMpostpro file
    os.remove(tmp_file)

def runsower(folder, testcase, numHDM, point_k, type):
    # run sower to get the missing Mach.xpost file
    proj_folder = os.path.join(folder, testcase)

    if type == 'HDM':
        results_dir = (
            f"{proj_folder}/InitialHDMruns/HDMrun{point_k:03d}/results"
        )

        postpro_dir = (
            f"{proj_folder}/InitialHDMruns/HDMrun{point_k:03d}/postpro"
        )
    elif type == 'PROM':
        results_dir = (
        f"{proj_folder}/{GreedyFolder}/evaluate/romruns{numHDM:03d}/"
        f"point{point_k:03d}/results"
        )

        postpro_dir = (
            f"{proj_folder}/{GreedyFolder}/evaluate/romruns{numHDM:03d}/"
            f"point{point_k:03d}/postpro"
        )
    elif type == 'HPROM':
        results_dir = (
        f"{proj_folder}/{GreedyFolder}/evaluate/hromruns{numHDM:03d}/"
        f"point{point_k:03d}/results"
        )

        postpro_dir = (
            f"{proj_folder}/{GreedyFolder}/evaluate/hromruns{numHDM:03d}/"
            f"point{point_k:03d}/postpro"
        )

    subprocess.run(
        [
            "bash",
            "./run_sower.sh",
            proj_folder,
            results_dir,
            postpro_dir,
            GreedyFolder,
        ],
        check=True,
        stdout=subprocess.DEVNULL
    )

def run_xp2exo(folder, testcase, numHDM, point_k):
        # run sower to get the missing Mach.xpost file
    proj_folder = os.path.join(folder, testcase)

    mesh_dir = (
        f"{proj_folder}/mesh"
    )

    postpro_dir = (
        f"{proj_folder}/{GreedyFolder}/evaluate/romruns{numHDM:03d}/"
        f"point{point_k:03d}/postpro"
    )

    subprocess.run(
        [
            "bash",
            "./xp2exo.sh",
            mesh_dir,
            postpro_dir,
        ],
        check=True,
        stdout=subprocess.DEVNULL
    )

def download_exofiles(exo_file, dest_dir, k):
    # download exo
    downloaded_exo_filename = os.path.join(dest_dir, f'point{k:03d}.exo')
    subprocess.run(
        [
            "scp",
            f"zyh03@login.sherlock.stanford.edu:/scratch/users/zyh03/FinalSnappingPaper/{exo_file}",
            f"{downloaded_exo_filename}"
        ],
        check=True,
    )

    # download parameters.txt
    params_file = os.path.join(os.path.dirname(os.path.dirname(exo_file)), "parameters.txt".format(k))
    downloaded_params_filename = os.path.join(dest_dir, f'params_point{k:03d}.txt')
    subprocess.run(
        [
            "scp",
            f"zyh03@login.sherlock.stanford.edu:/scratch/users/zyh03/FinalSnappingPaper/{params_file}",
            f"{downloaded_params_filename}"
        ],
        check=True,
    )

def read_single_PROM_Data(folder, testcase, numHDM, k, whichdata, numDOF, ROM_params, hyper):
    """Read a single PROM run and return the data"""
    if hyper: romfolder = 'hromruns'
    else: romfolder = 'romruns'
    
    # Initialize local arrays for this run
    if whichdata == 'Mach':
        ROMdata_local = np.zeros((numDOF, 1))
    elif whichdata == 'Velocity':
        ROMdata_local = np.zeros((numDOF, 3, 1))
    
    ROM_CVdata_local = np.zeros((numDOF, 1))
    
    romMachXPOST_file = os.path.join(folder, testcase, f'{GreedyFolder}/evaluate', 
                                      f'{romfolder}{numHDM:03d}', f'point{k:03d}/postpro', 'Mach.xpost')
    romDispXPOST_file = os.path.join(folder, testcase, f'{GreedyFolder}/evaluate', 
                                     f'{romfolder}{numHDM:03d}', f'point{k:03d}/postpro', 'Displacement.xpost')
    romVelXPOST_file = os.path.join(folder, testcase, f'{GreedyFolder}/evaluate', 
                                    f'{romfolder}{numHDM:03d}', f'point{k:03d}/postpro', 'Velocity.xpost')
    romCVXPOST_file = os.path.join(folder, testcase, f'{GreedyFolder}/evaluate', 
                                   f'{romfolder}{numHDM:03d}', f'point{k:03d}/postpro', 'ControlVolume.xpost')
    
    romMachbin_file = os.path.join(folder, testcase, f'{GreedyFolder}/evaluate', 
                                   f'{romfolder}{numHDM:03d}', f'point{k:03d}/results', 'Mach.bin001')
    romDispbin_file = os.path.join(folder, testcase, f'{GreedyFolder}/evaluate', 
                                   f'{romfolder}{numHDM:03d}', f'point{k:03d}/results', 'Displacement.bin001')
    romVelbin_file = os.path.join(folder, testcase, f'{GreedyFolder}/evaluate', 
                                  f'{romfolder}{numHDM:03d}', f'point{k:03d}/results', 'Velocity.bin001')
    
    if (not os.path.exists(romMachXPOST_file) or not os.path.exists(romDispXPOST_file) or 
        not os.path.exists(romVelXPOST_file)) and (os.path.exists(romMachbin_file) and os.path.exists(romDispbin_file) and 
        os.path.exists(romVelbin_file)):
        if hyper: 
            runHROMpostpro(folder, testcase, numHDM, k, ROM_params)
            runsower(folder, testcase, numHDM, k, 'HPROM')
        else: 
            runsower(folder, testcase, numHDM, k, 'PROM')
        print(f'Sowered {romfolder}{numHDM:03d} point{k:03d}')
    
    if os.path.exists(romMachXPOST_file):
        # ROM_CVdata_local[:, 0] = readXPOST(romCVXPOST_file)
        if whichdata == 'Mach':
            ROMdata_local[:, 0] = readXPOST(romMachXPOST_file)
            print(f"Read Mach.xpost for {romfolder}{numHDM:03d}/point{k:03d}.")
        elif whichdata == 'Velocity':
            ROMdata_local[:, :, 0] = readXPOST(romVelXPOST_file)
            print(f"Read Velocity.xpost for {romfolder}{numHDM:03d}/point{k:03d}.")
    else:
        if whichdata == 'Mach':
            ROMdata_local[:, 0] = np.nan
            print(f"Warning: Missing Mach.xpost file for {romfolder}{numHDM:03d}/point{k:03d}. Filled with NaN.")
        elif whichdata == 'Velocity':
            ROMdata_local[:, :, 0] = np.nan
            print(f"Warning: Missing Velocity.xpost file for {romfolder}{numHDM:03d}/point{k:03d}. Filled with NaN.")
    
    return ROMdata_local, ROM_CVdata_local, k  # Return index as well for correct ordering

def read_single_HDM_Data(folder, testcase, numHDM, k, whichdata, numDOF):
    """Read a single HDM run and return the data"""
    
    # Initialize local arrays for this run
    if whichdata == 'Mach':
        HDMdata_local = np.zeros((numDOF, 1))
    elif whichdata == 'Velocity':
        HDMdata_local = np.zeros((numDOF, 3, 1))
    
    HDM_CVdata_local = np.zeros((numDOF, 1))
    
    machxpost_file = os.path.join(folder, testcase, 'InitialHDMruns', f'HDMrun{k:03d}', 'postpro', 'Mach.xpost')
    velxpost_file = os.path.join(folder, testcase, 'InitialHDMruns', f'HDMrun{k:03d}', 'postpro', 'Velocity.xpost')
    dispxpost_file = os.path.join(folder, testcase, 'InitialHDMruns', f'HDMrun{k:03d}', 'postpro', 'Displacement.xpost')
    CVxpost_file = os.path.join(folder, testcase, 'InitialHDMruns', f'HDMrun{k:03d}', 'postpro', 'ControlVolume.xpost')
    
    if (not os.path.exists(machxpost_file) or not os.path.exists(dispxpost_file) or 
        not os.path.exists(velxpost_file)):
        runsower(folder, testcase, numHDM, k, 'HDM')
        print(f'Sowered HDMrun{numHDM:03d} point{k:03d}')

    if os.path.exists(machxpost_file):
        # HDM_CVdata_local[:, 0] = readXPOST(CVxpost_file)
        if whichdata == 'Mach':
            HDMdata_local[:, 0] = readXPOST(machxpost_file)
            print(f"Read Mach.xpost for HDMrun{k:03d}.")
        elif whichdata == 'Velocity':
            HDMdata_local[:, :, 0] = readXPOST(velxpost_file)
            print(f"Read Velocity.xpost for HDMrun{k:03d}.")
    else:
        if whichdata == 'Mach':
            HDMdata_local[:, 0] = np.nan
            print(f"Warning: Missing Mach.xpost file for HDMrun{k:03d}. Filled with NaN.")
        elif whichdata == 'Velocity':
            HDMdata_local[:, :, 0] = np.nan
            print(f"Warning: Missing Velocity.xpost file for HDMrun{k:03d}. Filled with NaN.")

    return HDMdata_local, HDM_CVdata_local, k  # Return index as well for correct ordering

def readData(numHDM, numDOF, folder, testcase, whichdata, hyper, nprocs=1):

    # Set ROM folder name 
    if hyper: romfolder = 'hromruns'
    else: romfolder = 'romruns'

    # Counter number of ROM runs
    numROM = count_folders(os.path.join(folder, testcase, f'{GreedyFolder}/evaluate', f'{romfolder}{numHDM:03d}'))

    # Read log file for parameter information
    if folder == 'combined_Sobol':
        log_file = os.path.join(folder, 'NonDes_Laplace', 'log.out.back')
    else:
        log_file = os.path.join(folder, testcase, 'log.out')
    # HDM_params, ROM_params = readLog(log_file, numHDM, numROM)

    HDM_params, _ = readLog(log_file, numHDM, numROM)
    _, ROM_params = readLog(log_file, numHDM, numROM)

    # Initialize arrays
    if whichdata == 'Mach': 
        HDMdata = np.zeros((numDOF, numHDM))
        PROMdata = np.zeros((numDOF, numROM))
    elif whichdata == 'Velocity': 
        HDMdata = np.zeros((numDOF, 3, numHDM))
        PROMdata = np.zeros((numDOF, 3, numROM))
    
    # Read HDM data in parallel
    n_jobs = min(numHDM, nprocs)
    hdm_results = Parallel(n_jobs=n_jobs)(
        delayed(read_single_HDM_Data)(folder, testcase, numHDM, k, whichdata, numDOF) 
        for k in range(1, numHDM + 1)
    )

    # Combine results
    HDM_CVdata = np.zeros((numDOF, numHDM))
    for HDMdata_local, HDM_CVdata_local, k in hdm_results:
        idx = k - 1
        if whichdata == 'Mach':
            HDMdata[:, idx] = HDMdata_local[:, 0]
        elif whichdata == 'Velocity':
            HDMdata[:, :, idx] = HDMdata_local[:, :, 0]
        HDM_CVdata[:, idx] = HDM_CVdata_local[:, 0]

    # Read PROM data in parallel
    if hyper: n_jobs = nprocs // 8
    else: n_jobs = min(numROM, nprocs)
    rom_results = Parallel(n_jobs=n_jobs)(
        delayed(read_single_PROM_Data)(folder, testcase, numHDM, k, whichdata, numDOF, ROM_params, hyper) 
        for k in range(1, numROM + 1)
    )
    
    # Combine results
    PROM_CVdata = np.zeros((numDOF, numROM))
    for ROMdata_local, ROM_CVdata_local, k in rom_results:
        idx = k - 1
        if whichdata == 'Mach':
            PROMdata[:, idx] = ROMdata_local[:, 0]
        elif whichdata == 'Velocity':
            PROMdata[:, :, idx] = ROMdata_local[:, :, 0]
        PROM_CVdata[:, idx] = ROM_CVdata_local[:, 0]
    
    print("\n")
    
    return HDMdata, PROMdata, HDM_CVdata, PROM_CVdata, HDM_params, ROM_params


if __name__ == "__main__":
    numDOF = 812098
    snapping_tol = 0.01 # 2-norm relative error
    folder = 'combined_Sobol' # 'FarField' or 'WallGeometry'
    testcase = 'Des_NoShift' # 'NonDes_NoShift', 'Des_NoShift', 'NonDes_Laplace', 'Des_Laplace'
    hyper = False
    numHDM = count_folders(os.path.join(folder, testcase, 'InitialHDMruns')) 
    numHDM = 70
    nproc = 4

    if folder == 'combined_Sobol':
        GreedyFolder = 'GreedyRuns_{}'.format(numHDM)
    else:
        GreedyFolder = 'GreedyRuns'

    # ========== Step 1: get snapping results and write to file ==========
    HDMdata, PROMdata, HDM_CVdata, PROM_CVdata, HDM_params, ROM_params = readData(numHDM, numDOF, folder, testcase, whichdata='Velocity', hyper=hyper, nprocs=nproc)
    snapping_output_file = os.path.join(folder, testcase, 'results_{}.snapping'.format(numHDM))
    idx_snapped = checkSnapping(HDMdata, PROMdata, HDM_CVdata, PROM_CVdata, HDM_params, ROM_params, snapping_tol=snapping_tol, output_file=snapping_output_file)
    print(f"Snapped PROM idx: {idx_snapped[:, 0]}")
    print(f"Snapped HDM idx: {idx_snapped[:, 1]}")

    # # ========== Step 2: run xp2exo for snapped runs ==========
    # snap_idx = [1,2] # idx of the PROM runs
    # for k in snap_idx:
    #     run_xp2exo(folder, testcase, numHDM, k)
    #     print(f"Finished xp2exo for romruns{numHDM:03d}/point{k:03d}.")

    # ========== Step 3:Download exo files for paraview (make sure to run this part on local machine) ==========
    # local_dir = f'/home/zyh03/Desktop/Snapping/{folder}/{testcase}/'
    # os.makedirs(local_dir, exist_ok=True)
    # for k in snap_idx:
    #     exo_file = os.path.join(folder, testcase, 'GreedyRuns/evaluate', 'romruns{:03d}'.format(numHDM), 'point{:03d}/postpro'.format(k), 'fluid_solution.exo')
    #     download_exofiles(exo_file, local_dir, k)