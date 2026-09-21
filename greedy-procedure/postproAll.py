import os
import re
import subprocess

def checkSetup(Des, Shift):
    with open("HROMpostpro", "r") as f:
        content = f.read()

    form_match = re.search(r"\bForm\s*=\s*(Descriptor|NonDescriptor)\s*;", content)
    shift_match = re.search(r"\bShiftVectorType\s*=\s*(None|Laplace)\s*;", content)
    laplace_match = re.search(r"\bLaplaceSnapshotData\s*=\s*\"([^\"]*)\"\s*;", content)

    file_des = form_match.group(1)
    file_shift = shift_match.group(1)
    laplace_snapshot_data = laplace_match.group(1).strip()

    if file_des != Des or file_shift != Shift:
        raise ValueError(
            f"Setup mismatch: expected Form={Des}, ShiftVectorType={Shift}, "
            f"but found Form={file_des}, ShiftVectorType={file_shift}"
        )

    if file_shift == "None" and laplace_snapshot_data != "":
        raise ValueError(
            "Setup mismatch: ShiftVectorType=None requires LaplaceSnapshotData=\"\""
        )
    
    if file_shift == "Laplace" and laplace_snapshot_data == "":
        raise ValueError(
            "Setup mismatch: ShiftVectorType=Laplace requires non-empty LaplaceSnapshotData"
        )

    print(
        f"Setup check passed: Form={file_des}, "
        f"ShiftVectorType={file_shift}, LaplaceSnapshotData=\"{laplace_snapshot_data}\""
    )

def run_hrom_points(postpro_points):
    """
    Loop through points 1-3, extract Mach numbers from parameters.txt,
    update HROMpostpro file, and run the simulation.
    """
    
    # Read the original HROMpostpro file
    with open("HROMpostpro", "r") as f:
        original_hrom = f.read()

    # Read the original sower_results file
    with open("sower_results.sh", "r") as f:
        original_sower = f.read()

    # Read the original xp2exo file
    with open("xp2exo.sh", "r") as f:
        original_xp2exo = f.read()
    
    # Loop over points 1-3
    for i in postpro_points:
        point_name = f"point{i:03d}"
        hdm_run_name = f"HDMrun{i:03d}"
        params_file = f"GreedyRuns/evaluate/hromruns002/{point_name}/parameters.txt"
        
        # Read parameters file and extract Mach number from second line
        with open(params_file, "r") as f:
            lines = f.readlines()
        
        mach = lines[1].strip()
        print(f"Processing {point_name} with Mach = {mach}")
        
        # Update the Mach number and point001 in HROMpostpro
        updated_hrom = original_hrom.replace('point001', point_name)
        updated_hrom = re.sub(r'Mach = [0-9.]+;', f'Mach = {mach};', updated_hrom)

        # Write the updated HROMpostpro
        with open("HROMpostpro", "w") as f:
            f.write(updated_hrom)

        updated_sowerresults = original_sower.replace('point001', point_name)
        with open("sower_results.sh", "w") as f:
            f.write(updated_sowerresults)

        updated_xp2exo = original_xp2exo.replace('point001', point_name)
        updated_xp2exo = updated_xp2exo.replace('HDMrun001', hdm_run_name)
        with open("xp2exo.sh", "w") as f:
            f.write(updated_xp2exo)
        
        # Run the command
        subprocess.run("mpirun -n 8 $AEROF HROMpostpro", shell=True)
        subprocess.run("./sower_results.sh", shell=True)
        subprocess.run("./xp2exo.sh", shell=True)
        
        print("Completed processing for", point_name)
    
        # Restore the original files
        with open("HROMpostpro", "w") as f:
            f.write(original_hrom)

        with open("sower_results.sh", "w") as f:
            f.write(original_sower)

        with open("xp2exo.sh", "w") as f:
            f.write(original_xp2exo)
        
        print("All points processed. Original files restored.")

if __name__ == "__main__":
    postpro_points = [1]

    Des = "Descriptor"
    Shift = "Laplace"
    checkSetup(Des, Shift)
    breakpoint()

    run_hrom_points(postpro_points)
