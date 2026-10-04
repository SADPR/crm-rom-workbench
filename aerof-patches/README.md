# AERO-F patches used by the NACA 5D campaign

The production executable stays untouched. It is
`/scratch/users/sadpr/Code3Aug/aero-f/build-scalapack/bin/aerof.opt`, built from FRG master
`069ef9d8`, and every result before 2026-10-04 used it. A fix is a patch on top of that commit,
applied in its own git worktree, and built in its own directory with the production
configuration (`build_aerof_variant.sbatch`). Only the jobs that need the fix point to it,
through `AEROF_POD` for the clustered POD. The output records the executable and its commit
(`local.json`: `aerof`, `aerof_commit`).

Nothing here is pushed to the FRG Bitbucket. Whether a validated fix goes there, as a branch
or a pull request and never directly to master, is a separate decision.

## 0001: keep merged snapshots in their cluster when adding cluster overlap

- File: `NonlinearRomClustering.C`, `Clustering::addClusterOverlap`.
- Base: `069ef9d8`. The local branch is `fix/cluster-overlap-after-merge`, commit `575b4f54`.
- Bug: with `MinClusterSize`, a merged cluster's center is the weighted mean of the two old
  centers. A snapshot of the merged cluster can then lie closer to another center, and the
  overlap step aborted ("something is wrong with cluster indices in addClusterOverlap()").
  Both 2026-10-04 runs with MinClusterSize 6 and 10 hit it.
- Fix: the snapshot keeps its assigned cluster as home and takes the closer center as its
  neighbor.
- Effect elsewhere: none. Runs that never merge a cluster always had the closest center equal to
  the assigned one, since the old code aborted otherwise, so they follow the same code path.

On Sherlock:

```bash
cd /scratch/users/sadpr/Code3Aug/aero-f
git log --oneline -1          # must be 069ef9d8
git status -sb                # must be clean
git worktree add -b fix/cluster-overlap-after-merge /scratch/users/sadpr/Code3Aug/aerof-overlapfix 069ef9d8
cd /scratch/users/sadpr/Code3Aug/aerof-overlapfix
git am /scratch/users/sadpr/Code3Aug/crm-rom-workbench/aerof-patches/0001-Keep-merged-snapshots-in-their-cluster-when-adding-c.patch
cd /scratch/users/sadpr/Code3Aug/crm-rom-workbench/greedy-procedure
sbatch ../aerof-patches/build_aerof_variant.sbatch /scratch/users/sadpr/Code3Aug/aerof-overlapfix
```

Before using the build, run the regression: the k = 8 POD without MinClusterSize, rebuilt with
the fix in `reductionrun256-c8-regress`, must give the same clusters (`state.index`,
`state.map`) and singular values as `reductionrun256-c8`.
