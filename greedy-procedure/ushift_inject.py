import numpy as np
import os

def ushiftXPOST(u_star_xpost, TurbulenceModel, folder_name=None, hrmesh_nodes=None):
    # u_star_xpost is the solution to the BC=[1,0] Laplace problem
    # settings object is the from setup.py

    u_star = np.loadtxt(u_star_xpost, skiprows=3)
    if hrmesh_nodes is not None:
        print("Number of hrmesh_nodes: {}".format(len(hrmesh_nodes)),flush=True)
        u_star = u_star[hrmesh_nodes]

    Ngrid = len(u_star)

    prefix_idx = u_star_xpost.find("-u_star.xpost") # prefix to all xpost
    meshname = u_star_xpost[u_star_xpost.rfind("/")+1 : prefix_idx] # mesh name only
    if folder_name is None:
        u_shift_xpost = "{}-u_shift.xpost".format(u_star_xpost[:prefix_idx])
    else:
        u_shift_xpost = "{}{}-u_shift.xpost".format(folder_name, meshname)

    if TurbulenceModel == "SpalartAllmaras":
        nVar = 6

        vals = np.column_stack(([u_star]*nVar))

        with open(u_shift_xpost, "w") as f:
            f.write("Vector{} {}-u_shift under load for FluidNodes\n".format(nVar, meshname))
            f.write("{}\n".format(Ngrid))
            f.write("1\n")
            np.savetxt(f, vals, comments="")

        print("----u_shift saved as XPOST file: {}----".format(u_shift_xpost), flush=True)

    return "{}".format(u_shift_xpost)

def get_ushift_file(meshpath, folder_name=None, hrmesh_nodes=None, u_star_path=None):
    meshname = os.path.basename(meshpath)
    dotIdx = meshname.rfind(".") # "." right before the mesh extension (ex. .msh)
    if dotIdx == -1:
        u_star = meshname + "-u_star.xpost" # if no extension, just add -u_star.xpost
    else:
        u_star = meshname[:dotIdx] + "-u_star.xpost" # safer way to get u_star.xpost name

    if u_star_path is None:
        u_star_xpost = os.path.join('xdmf_files', u_star) # u_star.xpost was obtained via AirfoilPoisson3D-clean.py
    else:
        u_star_xpost = os.path.join(u_star_path, u_star)

    print("\n#### Generating Laplace XPOST and binaries ####", flush=True)
    u_shift_xpost = ushiftXPOST(u_star_xpost, "SpalartAllmaras", folder_name=folder_name, hrmesh_nodes=hrmesh_nodes)

    return u_shift_xpost

def Laplace_ushiftsnapdata(settings, frg, u_shift_xpost, folder_name=None, hyper=False):
    # Create folder Laplace_bin = 'GreedyRuns/Laplace-bin'
    if folder_name is None:
        Laplace_bin = "{}Laplace-bin".format(settings.MasterDir)
        os.makedirs(Laplace_bin, exist_ok=True)
    else:
        Laplace_bin = "{}".format(folder_name)
        os.makedirs(Laplace_bin, exist_ok=True)

    if hyper is False:
        n_clust = settings.HDMnclust
    else:
        n_clust = settings.HROMnclust

    # Perform sower split, save results into Laplace_bin as ushift.bin
    frg.sower_fluid_split(file2split=u_shift_xpost, out="{}/ushift.bin".format(Laplace_bin), nclust=n_clust, log='/dev/null')
    print("----Laplace solution sowered into binary: {}/ushift.bin----\n".format(Laplace_bin), flush=True)
    