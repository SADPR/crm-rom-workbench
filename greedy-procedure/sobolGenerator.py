from scipy.stats import qmc
import itertools
import numpy as np
import matplotlib.pyplot as plt

# Axis labels for the 5-D parameter vector, used only by the plot
PARAM_NAMES = ['M', 'Alpha', 'Shape', 'Camber', 'Thickness']


def sobolGenerator(ranges, numHDM, include_corners=True, numToSkip=0, names=None, make_plot=True):
    """
    Generate HDM points using a Sobol quasi-random sequence.

    ranges is one [lower, upper] pair per parameter, i.e.
    list(zip(ParamsLowerBound, ParamsUpperBound)). A parameter whose bounds are
    equal is inactive: it is pinned at that value and takes no dimension in the
    sequence. Collapsing the space from three parameters to two therefore needs
    no change here, only a change to the bounds in setup.py.

    With include_corners, all 2**n_active corners of the active subspace come
    first, followed by enough Sobol points to reach numHDM points in total.

    Returns an (N, ndim) array of full parameter vectors with the first
    numToSkip rows dropped.
    """

    lo = np.array([r[0] for r in ranges], dtype=float)
    hi = np.array([r[-1] for r in ranges], dtype=float)
    ndim = len(lo)
    act = np.nonzero(hi - lo)[0]          # indices of the active parameters
    active = set(act.tolist())

    if names is None:
        names = PARAM_NAMES[:ndim]

    # Corners of the active subspace, inactive parameters held at their value.
    # itertools.product varies the last dimension fastest, which reproduces the
    # corner ordering the three-parameter version wrote out by hand.
    corners = np.array(list(itertools.product(
        *[[lo[j], hi[j]] if j in active else [lo[j]] for j in range(ndim)])))

    if include_corners:
        num_sobol = max(0, numHDM - corners.shape[0])
    else:
        num_sobol = numHDM

    # Generate the Sobol sequence in [0, 1]^n_active and scale it onto the
    # active ranges; inactive components stay pinned at their single value.
    # NOTE: unscrambled Sobol always starts at the origin, which maps to the
    # all-lower-bound corner. Generate one extra point and discard the first to
    # avoid duplicating a corner that was just added by hand.
    sobol_points = np.tile(lo, (num_sobol, 1))
    if num_sobol > 0 and act.size > 0:
        sampler = qmc.Sobol(d=act.size, scramble=False, seed=42)
        u = sampler.random(n=num_sobol + 1)[1:]
        sobol_points[:, act] = lo[act] + u * (hi[act] - lo[act])

    if include_corners:
        HDM_points = np.vstack([corners, sobol_points])
    else:
        HDM_points = sobol_points
        corners = None

    if make_plot:
        plotPoints(HDM_points, corners, lo, hi, act, names, numHDM)

    return HDM_points[numToSkip:]


def plotPoints(points, corners, lo, hi, act, names, numHDM):
    """Scatter the generated points over the active parameter subspace."""

    # Nothing worth drawing for a 0-D or 1-D space, and no honest projection for
    # 4-D or more, so leave those cases unplotted
    if act.size not in (2, 3):
        return

    pad = 0.02 * (hi[act] - lo[act])
    fig = plt.figure(figsize=(10, 8))

    if act.size == 2:
        i, j = act
        ax = fig.add_subplot(111)
        ax.scatter(points[:, i], points[:, j],
                   color='red', s=40, label='HDM Executed Points (Sobol)', alpha=0.8)
        if corners is not None:
            ax.scatter(corners[:, i], corners[:, j],
                       color='blue', marker='s', s=80, label='Corners', alpha=0.9)
        ax.grid(color='gray', linestyle='--', linewidth=0.3, alpha=0.4)
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        ax.set_xlim(lo[i] - pad[0], hi[i] + pad[0])
        ax.set_ylim(lo[j] - pad[1], hi[j] + pad[1])
    else:
        i, j, k = act
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, i], points[:, j], points[:, k],
                   color='red', s=40, label='HDM Executed Points (Sobol)', alpha=0.8)
        if corners is not None:
            ax.scatter(corners[:, i], corners[:, j], corners[:, k],
                       color='blue', marker='s', s=80, label='Corners', alpha=0.9)
        # The 12 edges of the bounding box: span each axis in turn at all four
        # combinations of the bounds of the other two
        for span, a, b in [(i, j, k), (j, i, k), (k, i, j)]:
            for va in (lo[a], hi[a]):
                for vb in (lo[b], hi[b]):
                    edge = {span: [lo[span], hi[span]], a: [va, va], b: [vb, vb]}
                    ax.plot(edge[i], edge[j], edge[k],
                            color='gray', linestyle='--', linewidth=0.3, alpha=0.4)
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        ax.set_zlabel(names[k])
        ax.set_xlim(lo[i] - pad[0], hi[i] + pad[0])
        ax.set_ylim(lo[j] - pad[1], hi[j] + pad[1])
        ax.set_zlim(lo[k], hi[k])

    ax.set_title('HDM Executed Points (Sobol Sequence)')
    ax.legend()
    plt.savefig('HDM_executed_point_Sobol_{}.pdf'.format(numHDM))
    plt.close(fig)
