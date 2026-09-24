import argparse

import fenics as fe
# from mpi4py import MPI
import meshio
import numpy as np
import os
# import matplotlib.pyplot as plt
import h5py
import sys

# %%
def parseMeshGeometry(top_file):
    # Input: top file
    
    # Output: nodes, tets, triangles, triangles_names
    # nodes = (N,3) np array
    # tets = (Nt,4) np array
    # triangles = (Nm,3) np array
    # triangles_names = (Nm,) list of strs

    nodes = [] # list[list[float]] of size (nNodes, 3)
    elements = {} # dict[str, list[list[int]]]

    with open(top_file, 'r') as f:
        lines = iter(f.readlines())
        next(lines)  # skip first line: "Nodes FluidNodes"

        # ============= Parse Nodes =============
        print("Parsing Nodes from .top file...", flush=True)
        for line in lines:
            if line.strip().startswith("Elements"):
                section_name = line.strip().split()[1]  # e.g., "Fluid"
                break

            parts = line.strip().split()

            if len(parts) >= 4:
                current_node = [float(x) for x in parts[1:]]  # skip node ID
                nodes.append(current_node)

        nodes = np.array(nodes, dtype=float)
        
        # ============= Parse Elements =============
        print("Parsing Elements from .top file...", flush=True)
        current_section = section_name
        elements[current_section] = []

        for line in lines:
            line = line.strip()

            if line.startswith("Elements"):
                # Start a new elements block
                current_section = line.split()[1]
                elements[current_section] = []

            parts = line.split()
            
            if len(parts) >= 5 and not line.startswith("Elements"): # the headers are only len=4
                current_element = [int(x) for x in parts]  # [element ID, element type, node1, node2, node3, node4]
                elements[current_section].append(current_element)

        # Convert all to numpy arrays
        for key in elements:
            elements[key] = np.array(elements[key], dtype=np.int64)

        # split elements into tets, triangles, and triangles_names
        physical_tags = list(elements.keys())
        tets = elements[physical_tags[0]][:,2:] # array of (Nt,4)

        triangles = np.zeros((1,3)) # initialize with one row of 0s
        triangles_names = []
        for i in range(1, len(physical_tags)):
            current_section_triangles = np.array(elements[physical_tags[i]][:,2:])
            triangles = np.vstack([triangles, current_section_triangles])
            triangles_names.append(np.shape(current_section_triangles)[0]*[physical_tags[i]])

        triangles_names = [tri_names for sublist in triangles_names for tri_names in sublist] # flatten the list
        triangles = triangles[1:, :] # remove the first row of 0s

    # Ensure .top is 1-based indexing
    tets = tets - 1
    triangles = triangles - 1

    return nodes, tets, triangles, triangles_names, physical_tags


def parsePhysicalTags(top_file):
    """Return the element-block names in the order used for FEniCS tags."""
    physical_tags = []
    with open(top_file, 'r') as top:
        for line in top:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == 'Elements':
                physical_tags.append(parts[1])
    return physical_tags


def laplaceBoundaryConditions(physical_tags):
    """Set the CRM Laplace field to one at the fixed boundary and zero at the airfoil."""
    boundary_values = (('InletFixed_2', 1), ('StickMoving_3', 0))
    tags_by_name = {name: index for index, name in enumerate(physical_tags, start=1)}
    missing = [name for name, _ in boundary_values if name not in tags_by_name]
    if missing:
        raise RuntimeError('Missing CRM boundary groups: {}'.format(', '.join(missing)))

    # Symmetry_1 touches every vertex in the one-layer CRM extrusion. Leaving
    # it natural avoids the previous constant u_star field caused by u_star=1 there.
    facet_tags = [tags_by_name[name] for name, _ in boundary_values]
    uBC = [value for _, value in boundary_values]
    return facet_tags, uBC

# %% [markdown]
# # Split into 2 XDMF files, one for tetrahedron, one for triangles

# %%
def generateXDMF(nodes, tets, triangles, triangles_names, physical_tags, top_filename, xdmf_folder, comm=None, rank=None):
    # input: everything from parseMeshGeometry
    # Output: 2 XDMF file for FEniCS to read later

    # map names -> integer tags
    # map surface/volume names to integer tags

    dot_idx = top_filename.find(".top") # find the top file name

    # MPI setup
    if comm is None:
        comm = fe.MPI.comm_world
    if rank is None:
        rank = comm.Get_rank()

    name_to_tag = {name: i for i, name in enumerate(physical_tags, start=1)}
    facet_tags = np.array([name_to_tag[s] for s in triangles_names], dtype=np.int32)

    # volume tag: use ones for a single region
    cell_tags = np.ones(len(tets), dtype=np.int32)

    cell_tags   = np.asarray(cell_tags,  dtype=np.int32)
    facet_tags  = np.asarray(facet_tags, dtype=np.int32)

    # write volume (tetra only)
    os.makedirs(xdmf_folder, exist_ok=True)
    if rank == 0:
        mesh_vol = meshio.Mesh(
            points=nodes.astype(float),
            cells=[("tetra", tets.astype(np.int64))],
            cell_data={"cell_tags": [np.asarray(cell_tags, np.int32)]},
        )
        mesh_xdmf_filename = top_filename[:dot_idx] + "-mesh.xdmf"
        meshio.write(os.path.join(xdmf_folder, mesh_xdmf_filename), mesh_vol)
        print(f"Generated volume mesh XDMF: {xdmf_folder}/{mesh_xdmf_filename}", flush=True)

        # write facets (triangles only) with tags
        mesh_fac = meshio.Mesh(
            points=nodes.astype(float),
            cells=[("triangle", triangles.astype(np.int64))],
            cell_data={"facet_tags": [np.asarray(facet_tags, np.int32)]},
        )
        facets_xdmf_filename = top_filename[:dot_idx] + "-facets.xdmf"
        meshio.write(os.path.join(xdmf_folder, facets_xdmf_filename), mesh_fac)
        print(f"Generated facet mesh XDMF: {xdmf_folder}/{facets_xdmf_filename}", flush=True)

    comm.Barrier()

# %% [markdown]
# # Load in mesh

# %%
def loadMeshFEniCS(top_filename, xdmf_folder="xdmf_files", comm=None, rank=None):
    dot_idx = top_filename.find(".top") # find the top file name

    # MPI setup
    if comm is None:
        comm = fe.MPI.comm_world
    if rank is None:
        rank = comm.Get_rank()

    # Load mesh into FEniCS objects
    mesh = fe.Mesh()
    with fe.XDMFFile(comm, f"{xdmf_folder}/{top_filename[:dot_idx]}-mesh.xdmf") as xdmf:
        xdmf.read(mesh)

    tdim = mesh.topology().dim()

    mvc = fe.MeshValueCollection("size_t", mesh, tdim-1)
    with fe.XDMFFile(comm, f"{xdmf_folder}/{top_filename[:dot_idx]}-facets.xdmf") as xdmf:
        xdmf.read(mvc)

    facets = fe.MeshFunction("size_t", mesh, mvc)

    return mesh, facets

# %% [markdown]
# # Define Boundary Conditions

# %%
def defineBC(mesh, facets, facet_tags, uBC):
    # define Dirichelet boundary conditions
    # Give the same number of facet_tags and uBC, they will be matched one-to-one
    # example: facets_tags = [airfoil_tag, inlet_tag, outlet_tag]
    #                  uBC = [3, 5, 6]
    
    assert len(facet_tags)==len(uBC)

    V = fe.FunctionSpace(mesh, "Lagrange", 1)

    bcs = []
    for i in range(len(facet_tags)):
        bcs.append(fe.DirichletBC(V, fe.Constant(uBC[i]), facets, facet_tags[i]))

    if rank == 0:print(f"Defined {len(bcs)} Dirichlet BCs on facet tags {facet_tags} with values {uBC}", flush=True)
    return V, bcs

# %% [markdown]
# # Solve problem
# $\Delta u = f$ with the boundary conditions defined above

# %%
def solveProblem(mesh, p, V, bcs, f):
    # input: mesh, p, V, bcs, f
    # f is the forcing term, currently only supports constant forcing
    # p is the exponent to cellVolume, which scales the solution u

    # Trial and Test Functions
    u_trial = fe.TrialFunction(V)
    v_test = fe.TestFunction(V)

    A = fe.CellVolume(mesh)**p
    # Weak Form
    forcing = fe.Constant(f)
    weak_form_lhs = fe.inner(A*fe.grad(u_trial), fe.grad(v_test)) * fe.dx
    weak_form_rhs = forcing * v_test * fe.dx

    # # Finite Element Assembly and Linear System solve
    # u_star = fe.Function(V)

    # print("Solving the linear system for u_star...", flush=True)
    # fe.solve(
    #     weak_form_lhs == weak_form_rhs,
    #     u_star,
    #     bcs,
    # )

    u_star = fe.Function(V)

    for bc in bcs:
        bc.apply(u_star.vector())

    A, b = fe.assemble_system(weak_form_lhs, weak_form_rhs, bcs)

    # # Set PETSc options BEFORE creating the solver
    # fe.PETScOptions.set("pc_type", "hypre")
    # fe.PETScOptions.set("pc_hypre_type", "boomeramg")

    # solver = fe.PETScKrylovSolver("cg")
    # solver.set_from_options()

    # solver.set_operator(A)
    # solver.parameters["relative_tolerance"] = 1e-11
    # solver.parameters["absolute_tolerance"] = 1e-12
    # solver.parameters["maximum_iterations"] = 2000
    # fe.PETScOptions.set("ksp_monitor")

    # num_iter = solver.solve(u_star.vector(), b)

    fe.PETScOptions.set("ksp_type", "cg")
    fe.PETScOptions.set("pc_type", "hypre")
    fe.PETScOptions.set("pc_hypre_type", "boomeramg")
    fe.PETScOptions.set("ksp_rtol", 1e-11)
    fe.PETScOptions.set("ksp_atol", 1e-12)
    fe.PETScOptions.set("ksp_max_it", 2000)
    fe.PETScOptions.set("ksp_monitor")        # prints residual every iter — very useful

    solver = fe.PETScKrylovSolver()
    solver.set_from_options()
    solver.set_operator(A)

    num_iter = solver.solve(u_star.vector(), b)

    if rank == 0: print(f"Solved linear system for u_star in {num_iter} iterations", flush=True)

    # residual = np.linalg.norm(A*u_star.vector() - b) # should be close to 0
    # if rank == 0: print(f"Residual of the linear system: {residual:.2e}", flush=True)

    # r = b.copy()
    # A.mult(u_star.vector(), r)   # r = A x
    # r.axpy(-1.0, b)              # r = A x - b

    # r_norm = r.norm("l2")
    # b_norm = b.norm("l2")

    # if rank == 0:
    #     print(f"||r||        = {r_norm:.3e}")
    #     print(f"||b||        = {b_norm:.3e}")
    #     print(f"||r||/||b||  = {r_norm/b_norm:.3e}")   # true relative residual
    #     print(f"max|r|       = {r.max():.3e}")

    return u_star

def exportForParaview(u_star, top_filename, xdmf_folder="xdmf_files"):
    dot_idx = top_filename.find(".top") # find the top file name
    # Export u_star to paraview
    with fe.XDMFFile(comm, f"{xdmf_folder}/{top_filename[:dot_idx]}-u_star.xdmf") as xdmf:
        xdmf.write(u_star)

    if rank == 0: print(f"u_star ready for paraview: {xdmf_folder}/{top_filename[:dot_idx]}-u_star.xdmf", flush=True)

# %%
def saveXPOST(top_filename, xdmf_folder="xdmf_files", comm=None, rank=None):
    dot_idx = top_filename.find(".top") # find the top file name
    
    # MPI setup
    if comm is None:
        comm = fe.MPI.comm_world
    if rank is None:
        rank = comm.Get_rank()
        
    h5 = f"{xdmf_folder}/{top_filename[:dot_idx]}-u_star.h5"
    
    if rank == 0:
        with h5py.File(h5, "r") as f:
            vals = f["/VisualisationVector/0"][:]   # shape (812098, 1)
            vals = np.array(vals.ravel())           # (812098,)
            
        N = np.shape(vals)[0]
        out_path = f"{xdmf_folder}/{top_filename[:dot_idx]}-u_star.xpost"
    
        with open(out_path, "w") as f:
            f.write(f"Scalar {top_filename[:dot_idx]}-u_star under load for FluidNodes\n")
            f.write(f"{N}\n")
            f.write("1\n")
            np.savetxt(f, vals, comments="")
            
        print(f"u_star saved as XPOST file: {xdmf_folder}/{top_filename[:dot_idx]}-u_star.xpost", flush=True)

# %%
if __name__ == "__main__":
    comm  = fe.MPI.comm_world
    rank  = comm.Get_rank()
    size  = comm.Get_size()

    if rank == 0: print(f"Number of MPI processor: {size}", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("top_filename")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    top_filename = args.top_filename
    top_file = os.path.join(top_filename)
    physical_tags = parsePhysicalTags(top_file)
    
    # Extract just the filename (without path) for use in output file naming
    top_filename_only = os.path.basename(top_filename)
    
    if args.output_dir is not None:
        xdmf_folder = args.output_dir
    elif "/" in top_file:
        xdmf_folder = top_file[:top_file.rfind("/")] + "/xdmf_files"
    else:
        xdmf_folder = "./xdmf_files"

    if rank == 0: print(f"Processing mesh file: {top_file}", flush=True)
    
    dot_idx = top_filename_only.find(".top") # find the top file name
    if os.path.exists(f"{xdmf_folder}/{top_filename_only[:dot_idx]}-mesh.xdmf") and os.path.exists(f"{xdmf_folder}/{top_filename_only[:dot_idx]}-facets.xdmf"):
        if rank == 0: print("XDMF files already exist. Skipping mesh generation.", flush=True)
    else:
        nodes, tets, triangles, triangles_names, mesh_tags = parseMeshGeometry(top_file)
        if mesh_tags != physical_tags:
            raise RuntimeError('Inconsistent element-block ordering in {}'.format(top_file))
        generateXDMF(nodes, tets, triangles, triangles_names, mesh_tags, top_filename_only, xdmf_folder = xdmf_folder, comm=comm, rank=rank)
    
    mesh, facets = loadMeshFEniCS(top_filename_only, xdmf_folder=xdmf_folder, comm=comm, rank=rank)
    comm.Barrier()

    facet_tags, uBC = laplaceBoundaryConditions(physical_tags)
    V, bcs = defineBC(mesh, facets, facet_tags, uBC)
    u_star = solveProblem(mesh=mesh, p=1, V=V, bcs=bcs, f=0.0)

    exportForParaview(u_star, top_filename_only, xdmf_folder=xdmf_folder)
    saveXPOST(top_filename_only, xdmf_folder=xdmf_folder, comm=comm, rank=rank)

    if rank == 0: print("All done!", flush=True)
