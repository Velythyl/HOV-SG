"""Upstream's ``pcd_denoise_dbscan`` result without Open3D's all-neighbours DBSCAN.

``hovsg.utils.graph_utils.pcd_denoise_dbscan`` keeps the largest cluster of
``PointCloud.cluster_dbscan(eps, min_points)``. Open3D 0.18 implements that
(``PointCloudCluster.cpp``) by first materialising every point's full radius
neighbourhood, then expanding clusters in index order. On HOV-SG's merged masks
-- concatenations of every frame's observation, never re-downsampled -- the
neighbourhoods hold thousands of points each, and that is where the sequential
merge spends ~95 % of its time (profile in docs/openlex3d-eval-plan.md).

Open3D's labelling is fully determined by three facts, which this module
reproduces without enumerating neighbourhoods:

1. ``j`` is a neighbour of ``i`` iff the nanoflann L2 distance
   ``((dx*dx + dy*dy) + dz*dz) < eps*eps`` (strict, the point itself included).
   A point is *core* iff it has at least ``min_points`` neighbours.
2. Clusters are the connected components of the core-neighbour graph, labelled
   0, 1, ... in order of each component's lowest core index: Open3D's outer loop
   seeds a cluster at the first unlabelled core point, and expansion labels the
   whole component before the loop moves on.
3. A non-core point within ``eps`` of a core point is claimed by the first
   cluster that reaches it, i.e. the lowest label among the clusters of its core
   neighbours; any other point is noise (-1).

The caller then needs only the largest cluster (``Counter.most_common``: ties go
to the label that first occurs in index order) and its size.

Speed comes from a grid of side ``eps / 2``: any two points of one cell are
within ``0.87 eps``, so a cell holding ``min_points`` points makes all of them
core, and all cores of one cell are one component. Only sparse cells need
neighbour counts, and only cell pairs that end up in different components need
an exact closest-pair test. Every decision taken from a k-d tree distance has a
relative margin of 1e-6 around ``eps``; anything inside the margin is
recomputed with the exact predicate (1).
"""

from __future__ import annotations

import itertools
import os

import numpy as np
from scipy.spatial import cKDTree

_MARGIN = 1e-6
#: scipy's ``workers=-1`` means ``os.cpu_count()``: the whole node, not this
#: step's CPUs (48 threads per query on a 4-CPU step). Use the affinity mask,
#: and threads only for queries large enough to pay for them.
_CPUS = max(1, len(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else 1


def workers(n_queries: int) -> int:
    return _CPUS if n_queries >= 20000 else 1


def _d2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """nanoflann's ``L2_Adaptor::evalMetric`` for dim 3: ((dx^2 + dy^2) + dz^2), float64."""
    d = a - b
    return (d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1]) + d[..., 2] * d[..., 2]


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, x: int) -> int:
        parent = self.parent
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return int(root)

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if ra < rb:
            ra, rb = rb, ra
        self.parent[ra] = rb
        return True

    def roots(self) -> np.ndarray:
        parent = self.parent
        while True:
            nxt = parent[parent]
            if np.array_equal(nxt, parent):
                return parent
            parent = nxt
            self.parent = parent


def dbscan_labels(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    """Open3D 0.18 ``cluster_dbscan`` labels for ``points`` (N x 3 float64)."""
    pts = np.ascontiguousarray(points, dtype=np.float64)
    n = pts.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0 or n < min_points:
        return labels
    eps2 = eps * eps
    r_lo, r_hi = eps * (1.0 - _MARGIN), eps * (1.0 + _MARGIN)

    # --- grid --------------------------------------------------------------
    side = eps / 2.0
    cell3 = np.floor(pts / side).astype(np.int64)
    cell3 -= cell3.min(axis=0)
    dims = cell3.max(axis=0) + 5  # room for +-2 offsets without wrap-around
    cell3 += 2
    key = (cell3[:, 0] * dims[1] + cell3[:, 1]) * dims[2] + cell3[:, 2]
    order = np.argsort(key, kind="stable")
    skey = key[order]
    ukeys, starts, counts = np.unique(skey, return_index=True, return_counts=True)
    cell_of = np.empty(n, dtype=np.int64)
    cell_of[order] = np.repeat(np.arange(ukeys.size), counts)

    # --- core points -------------------------------------------------------
    core = counts[cell_of] >= min_points
    tree = None
    sparse = np.flatnonzero(~core)
    if sparse.size:
        tree = cKDTree(pts)
        dist, _ = tree.query(pts[sparse], k=min_points, distance_upper_bound=r_hi, workers=workers(sparse.size))
        kth = dist[:, -1] if dist.ndim == 2 else dist
        core[sparse[kth < r_lo]] = True
        amb = sparse[(kth >= r_lo) & (kth <= r_hi)]
        if amb.size:
            for i, nb in zip(amb.tolist(), tree.query_ball_point(pts[amb], r=r_hi, workers=workers(amb.size))):
                nb = np.asarray(nb, dtype=np.int64)
                if np.count_nonzero(_d2(pts[nb], pts[i]) < eps2) >= min_points:
                    core[i] = True
    core_idx = np.flatnonzero(core)
    if core_idx.size == 0:
        return labels

    # --- components of the core graph, on core cells ----------------------
    core_cells = np.unique(cell_of[core_idx])  # sorted cell ids holding a core
    ncc = core_cells.size
    cc_of_cell = np.full(ukeys.size, -1, dtype=np.int64)
    cc_of_cell[core_cells] = np.arange(ncc)
    # Core points grouped by core cell, for exact closest-pair tests.
    corder = core_idx[np.argsort(cc_of_cell[cell_of[core_idx]], kind="stable")]
    ccnt = np.bincount(cc_of_cell[cell_of[corder]], minlength=ncc)
    cstart = np.concatenate([[0], np.cumsum(ccnt)[:-1]])
    cpts = pts[corder]
    cmin = np.minimum.reduceat(cpts, cstart, axis=0)
    cmax = np.maximum.reduceat(cpts, cstart, axis=0)

    uf = _UnionFind(ncc)
    # Pass 1 (sound, not complete): each core point's nearest cores. A pair
    # under r_lo in two different cells certainly links them.
    ctree = cKDTree(cpts)
    k = min(8, cpts.shape[0])
    if k > 1:
        cd, ci = ctree.query(cpts, k=k, distance_upper_bound=r_hi, workers=workers(cpts.shape[0]))
        own = np.repeat(np.arange(ncc), ccnt)
        ok = (cd < r_lo) & np.isfinite(cd)
        ci = np.where(ok, ci, 0)
        other = own[ci]
        hit = ok & (other != own[:, None])
        a = np.repeat(own, k).reshape(-1, k)[hit]
        b = other[hit]
        if a.size:
            pairs = np.unique(np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1), axis=0)
            for x, y in pairs.tolist():
                uf.union(x, y)
    # Pass 2 (complete): every neighbouring core-cell pair still in different
    # components gets an exact test.
    ckeys = ukeys[core_cells]
    offs = [o for o in itertools.product(range(-2, 3), repeat=3) if o > (0, 0, 0)]
    cand_a, cand_b = [], []
    for dx, dy, dz in offs:
        nkey = ckeys + (dx * dims[1] + dy) * dims[2] + dz
        pos = np.searchsorted(ckeys, nkey)
        pos = np.minimum(pos, ncc - 1)
        found = ckeys[pos] == nkey
        if found.any():
            cand_a.append(np.flatnonzero(found))
            cand_b.append(pos[found])
    if cand_a:
        ca = np.concatenate(cand_a)
        cb = np.concatenate(cand_b)
        # Cheap reject: bounding boxes of the two cells' cores at least eps apart.
        gap = np.maximum(np.maximum(cmin[ca] - cmax[cb], cmin[cb] - cmax[ca]), 0.0)
        near = _d2(gap, np.zeros_like(gap)) < eps2 * (1.0 + 4 * _MARGIN)
        ca, cb = ca[near], cb[near]
        roots = uf.roots()
        diff = roots[ca] != roots[cb]
        for x, y in zip(ca[diff].tolist(), cb[diff].tolist()):
            if uf.find(x) == uf.find(y):
                continue
            px = cpts[cstart[x]:cstart[x] + ccnt[x]]
            py = cpts[cstart[y]:cstart[y] + ccnt[y]]
            if bool(np.any(_d2(px[:, None, :], py[None, :, :]) < eps2)):
                uf.union(x, y)
    comp_cc = uf.roots()

    # --- labels in Open3D's order -----------------------------------------
    comp_core = comp_cc[cc_of_cell[cell_of[core_idx]]]
    uniq, first = np.unique(comp_core, return_index=True)  # core_idx is ascending
    rank = np.empty(uniq.size, dtype=np.int64)
    rank[np.argsort(core_idx[first], kind="stable")] = np.arange(uniq.size)
    labels[core_idx] = rank[np.searchsorted(uniq, comp_core)]

    # --- border points: lowest label among core neighbours -------------------
    noncore = np.flatnonzero(~core)
    if noncore.size:
        for i, nb in zip(noncore.tolist(), ctree.query_ball_point(pts[noncore], r=r_hi, workers=workers(noncore.size))):
            if not nb:
                continue
            nb = np.asarray(nb, dtype=np.int64)
            nb = nb[_d2(cpts[nb], pts[i]) < eps2]
            if nb.size:
                labels[i] = labels[corder[nb]].min()
    return labels


def denoise_dbscan(pcd, eps: float = 0.02, min_points: int = 10):
    """Same result as ``hovsg.utils.graph_utils.pcd_denoise_dbscan(pcd, eps, min_points)``."""
    import open3d as o3d

    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    labels = dbscan_labels(obj_points, eps, min_points)
    valid = labels >= 0
    if not valid.any():
        return pcd
    lab, first, cnt = np.unique(labels[valid], return_index=True, return_counts=True)
    first_pos = np.flatnonzero(valid)[first]
    # Counter.most_common: highest count, ties to the first label in index order.
    best = lab[np.lexsort((first_pos, -cnt))[0]]
    largest_mask = labels == best
    largest_cluster_points = obj_points[largest_mask]
    largest_cluster_colors = obj_colors[largest_mask]
    if len(largest_cluster_points) < 5:
        return pcd
    largest_cluster_pcd = o3d.geometry.PointCloud()
    largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
    largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
    return largest_cluster_pcd
