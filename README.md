# NACA 5D ROM workbench

Research workspace for the parametric 2D NACA 4-digit airfoil ROM benchmark built
on AERO-F.

**This public copy holds only my own scripts and notes.** The code of the Farhat Research Group
they build on is not included: the supplied `greedy-procedure` workflow (`runs.py`, `setup.py`,
...), the `pyaeroopt` interface, and my patches to AERO-F. The scripts therefore do not run from
this copy alone.

**Naming.** The case is a parametric 2D NACA 4-digit airfoil, not the NASA Common Research Model
(CRM). "crm" survives only where renaming would break things: the repository name
(`crm-rom-workbench`), the Sherlock paths, the local mount (`Sherlock_CRM`), the `CRM_*`
environment variables, the `CRM_Laplace` conda environment, the 2026 manifest's `name` field, and the
file names of this handoff and of `docs/history/`. New names (Slurm jobs, documents, figures,
presentations) use `naca5d`.

Start with `CRM_WORKFLOW_HANDOFF.md`. It records the operating rules, where the
data live, the campaign history, and the current plan.

## Layout

- `greedy-procedure/`: our campaign drivers (`sobol5d_*.py`, `submit_sobol5d_*`).
- `docs/history/`: superseded plans and records of earlier campaigns.
- `mesh/`: required NACA mesh input, deliberately not versioned.


## Mesh data

The supplied mesh is approximately 535 MB, with individual files larger than
GitHub's regular 100 MB limit. It is excluded from Git. Each working copy,
including the Sherlock execution clone, uses a private copy in `./mesh`
checked against the delivery by the SHA-256 of `naca0012_Re1p5.top`. Do not
publish it unless its redistribution terms are confirmed.

## Dependencies

AERO-F, sower, xp2exo and the FRG `pyaeroopt` interface, with the supplied greedy-procedure
workflow; none of them is part of this copy.
