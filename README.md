# CRM ROM workbench

Research workspace for the CRM reduced-order-model benchmark. The initial
import preserves the supplied `greedy-procedure` workflow and its local
`pyaeroopt` dependency as source code, without importing the prior Git
history.

## Layout

- `greedy-procedure/`: greedy HDM/ROM/HROM workflow and helper scripts.
- `pyaeroopt/`: local Python interface for FRG tools.
- `mesh/`: required CRM mesh input, deliberately not versioned.

`greedy-procedure/mesh` is a relative link to `../mesh`; therefore the mesh
must be installed at the repository root.

## Mesh data

The supplied mesh is approximately 535 MB, with individual files larger than
GitHub's regular 100 MB limit. It is excluded from Git in this initial public
repository. Copy or link an authorized local mesh directory to `./mesh` before
running the workflow. Do not publish it unless its redistribution terms are
confirmed; Git LFS can be considered later if publication is authorized.

On Sherlock, the existing source mesh can be used as the data source:

```bash
ln -s /scratch/users/sadpr/Code3Aug/CRM_tmp/mesh \
  /scratch/users/sadpr/Code3Aug/crm-rom-workbench/mesh
```

The original `xdmf_files` link pointed to an external Oak directory and is not
included. Install or generate the required XDMF files in
`greedy-procedure/xdmf_files/` for the relevant run configuration.

## Dependencies

`greedy-procedure/README_LaplaceShift.txt` describes the external Laplace
solver dependency. `pyaeroopt/README.md` lists the FRG executables expected in
the environment.
