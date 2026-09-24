#!/usr/bin/env python3
"""Report FEniCS facet tags and constrained degrees of freedom for CRM."""

from pathlib import Path

import fenics as fe
import numpy as np


BASE = Path('GreedyRuns/HDMrun001/xdmf_files/naca0012_Re1p5_deformed')


def main():
    """Load the generated mesh and inspect the three CRM boundary groups."""
    mesh = fe.Mesh()
    comm = mesh.mpi_comm()
    with fe.XDMFFile(comm, '{}-mesh.xdmf'.format(BASE)) as xdmf:
        xdmf.read(mesh)

    facets_data = fe.MeshValueCollection('size_t', mesh, mesh.topology().dim() - 1)
    with fe.XDMFFile(comm, '{}-facets.xdmf'.format(BASE)) as xdmf:
        xdmf.read(facets_data)
    facets = fe.MeshFunction('size_t', mesh, facets_data)

    values, counts = np.unique(facets.array(), return_counts=True)
    print('FEniCS facet tags:', dict(zip(values.tolist(), counts.tolist())))

    space = fe.FunctionSpace(mesh, 'Lagrange', 1)
    for tag in (2, 3, 4):
        boundary = fe.DirichletBC(space, fe.Constant(0.0), facets, tag)
        print('tag {} constrained dofs: {}'.format(tag, len(boundary.get_boundary_values())))


if __name__ == '__main__':
    main()
