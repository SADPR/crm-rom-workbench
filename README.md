# NACA 5D ROM workbench

Research workspace for the parametric 2D NACA 4-digit airfoil ROM benchmark built
on AERO-F. The initial import preserves the supplied `greedy-procedure`
workflow and its local `pyaeroopt` dependency as source code, without
importing the prior Git history.

**Naming.** The case is a parametric 2D NACA 4-digit airfoil, not the NASA Common Research Model
(CRM). "crm" survives only where renaming would break things: the repository name
(`crm-rom-workbench`), the Sherlock paths, the local mount (`Sherlock_CRM`), the `CRM_*`
environment variables, the `CRM_Laplace` conda environment, the 2026 manifest's `name` field, and the
file names of this handoff and of `docs/history/`. New names (Slurm jobs, documents, figures,
presentations) use `naca5d`.

Start with `CRM_WORKFLOW_HANDOFF.md`. It records the operating rules, where the
data live, the campaign history, and the current plan.

## Layout

- `greedy-procedure/`: supplied greedy HDM/ROM/HROM workflow and our isolated
  campaign drivers.
- `pyaeroopt/`: local Python interface for FRG tools.
- `docs/history/`: superseded plans and records of earlier campaigns.
- `mesh/`: required NACA mesh input, deliberately not versioned.

`greedy-procedure/mesh` is a relative link to `../mesh`; therefore the mesh
must be installed at the repository root.

## Mesh data

The supplied mesh is approximately 535 MB, with individual files larger than
GitHub's regular 100 MB limit. It is excluded from Git. Each working copy,
including the Sherlock execution clone, uses a private copy in `./mesh`
checked against the delivery by the SHA-256 of `naca0012_Re1p5.top`. Do not
publish it unless its redistribution terms are confirmed.

## Dependencies

`greedy-procedure/README_LaplaceShift.txt` describes the Laplace-shift
environment. `pyaeroopt/README.md` lists the FRG executables expected in the
environment.
