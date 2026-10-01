"""Exact, faster drop-in for upstream's ``merge_3d_masks`` (sequential merge).

Upstream's sequential merge (``hovsg/utils/graph_utils.py``) is what makes
HOV-SG take ~11 h per Replica scene (OVO-SLAM, Table IV; upstream issue #16).
Merged masks are concatenations that are never re-downsampled, so they grow
with every frame, and every frame upstream

* re-runs ``pcd_denoise_dbscan(eps=0.1, min_points=10)`` -- Open3D's
  all-neighbours DBSCAN -- on *every* mask (92-95 % of the merge on Replica
  office0, stride 10, measured), and
* runs a brute-force faiss ``IndexFlatL2`` search between every pair of masks
  whose boxes overlap (the rest; it grows quadratically).

Both are replaced by exact equivalents; everything else -- the bbox gate, the
overlap matrix, ``connected_components``, the in-place concatenation -- is
upstream's logic, unchanged. The merged masks are byte-identical to upstream's
(points, colours and order); see ``docs`` in the RAGMAP repository for the
evidence.

**Denoise.** ``ragmap_adapter.exact_dbscan`` computes Open3D's labelling from
its definition without enumerating neighbourhoods. And a mask whose contents a
previous denoise left unchanged is not denoised again: the result is a pure
function of the contents, recorded by digest.

**Overlap.** The merge reads from the faiss search only whether each point's
float32 nearest-neighbour distance is ``< radius**2``. An exact float64 k-d tree
on the float32-rounded coordinates decides every point farther than a rigorous
rounding bound from that threshold; the few points inside the band are decided
by faiss itself, over the same database with the same search path (BLAS or
sequential, chosen by the number of queries as upstream's call chooses it).
Trees and pair values are reused while a mask's points are unchanged.

``bbox`` IoUs are computed vectorised with the same float64 operations in the
same order as upstream's ``compute_3d_bbox_iou``.

``HOVSG_FAST_MERGE_VERIFY=1`` also runs upstream's faiss search and DBSCAN on
every call and raises on the first disagreement (slow; for validation only).
"""

from __future__ import annotations

import hashlib
import os
import time

import faiss
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import hovsg.utils.graph_utils as gu
from ragmap_adapter.exact_dbscan import denoise_dbscan, workers

#: faiss's threshold between its sequential and BLAS exhaustive search
#: (``faiss::distance_compute_blas_threshold``), read once.
_BLAS_THRESHOLD = int(getattr(faiss.cvar, "distance_compute_blas_threshold", 20))
_EPS32 = float(np.finfo(np.float32).eps)
#: Largest neighbour distance the per-point rounding bound in `_count_within`
#: allows for (metres; upstream's radius is 1.5 x 0.02 = 0.03).
_REACH_CAP = 0.1

STATS = {"pairs": 0, "points": 0, "band": 0, "verified": 0,
         "dbscan_runs": 0, "dbscan_skipped": 0, "dbscan_pts": 0, "dbscan_skipped_pts": 0}

#: Contents (digest of points and colours) on which upstream's merge denoise
#: is known to be the identity: ``pcd_denoise_dbscan(X)`` returned X itself or
#: a cloud with exactly X's points and colours, in order.
_DENOISE_FIXED: set = set()


def _digest(pcd) -> bytes:
    h = hashlib.blake2b(digest_size=20)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    h.update(np.int64(pts.shape[0]).tobytes())
    h.update(np.ascontiguousarray(pts).tobytes())
    h.update(np.int64(cols.shape[0]).tobytes())
    h.update(np.ascontiguousarray(cols).tobytes())
    return h.digest()


def _points_digest(pcd) -> bytes:
    """Digest of a mask's points only: all that its overlaps depend on."""
    pts = np.ascontiguousarray(np.asarray(pcd.points))
    h = hashlib.blake2b(digest_size=20)
    h.update(np.int64(pts.shape[0]).tobytes())
    h.update(pts.tobytes())
    return h.digest()


def _denoise(pcd):
    """``gu.pcd_denoise_dbscan(pcd, eps=0.1, min_points=10)``, skipped where it is the identity.

    The denoise is a deterministic function of the cloud's contents. Upstream
    reruns it on every mask at every frame, including the masks no frame
    touched, whose input is last frame's output. Once a run returned its input
    unchanged, that content is recorded and the rerun skipped. A mask is
    denoised afresh whenever its contents change, so the result is upstream's.
    """
    n = len(pcd.points)
    key = _digest(pcd)
    if key in _DENOISE_FIXED:
        STATS["dbscan_skipped"] += 1
        STATS["dbscan_skipped_pts"] += n
        return pcd
    STATS["dbscan_runs"] += 1
    STATS["dbscan_pts"] += n
    if os.environ.get("HOVSG_FAST_MERGE_VERIFY") == "1":
        ref = gu.pcd_denoise_dbscan(pcd, eps=0.1, min_points=10)
    t0 = time.perf_counter()
    out = denoise_dbscan(pcd, eps=0.1, min_points=10)
    STATS["dbscan_s"] = STATS.get("dbscan_s", 0.0) + time.perf_counter() - t0
    if os.environ.get("HOVSG_FAST_MERGE_VERIFY") == "1":
        if (ref is pcd) != (out is pcd) or _digest(ref) != _digest(out):
            raise AssertionError(f"exact DBSCAN differs from upstream on a {n}-point mask")
        STATS["dbscan_verified"] = STATS.get("dbscan_verified", 0) + 1
    if out is pcd or _digest(out) == key:
        _DENOISE_FIXED.add(key)
    return out


def merge_point_clouds_list(pcd_list, voxel_size=0.02):
    """Upstream ``merge_point_clouds_list`` (same in-place concatenation) with `_denoise`."""
    merged_pcd = pcd_list[0]
    for pcd in pcd_list[1:]:
        merged_pcd += pcd
    return _denoise(merged_pcd)


def _faiss_min_d2(queries32: np.ndarray, db32: np.ndarray, n_queries_upstream: int) -> np.ndarray:
    """faiss's k=1 L2 distances for ``queries32`` against ``db32``.

    Upstream's call searches ``n_queries_upstream`` queries at once, which picks
    faiss's BLAS path at or above ``_BLAS_THRESHOLD`` queries and the sequential
    path below it. The same path is forced here: below the threshold the band
    is necessarily smaller too; at or above it, the queries are padded by
    repetition to reach it (each query's row is independent of the others).
    """
    n = queries32.shape[0]
    q = queries32
    if n_queries_upstream >= _BLAS_THRESHOLD and n < _BLAS_THRESHOLD:
        q = np.concatenate([queries32] * (-(-_BLAS_THRESHOLD // n)), axis=0)
    index = faiss.IndexFlatL2(3)
    index.add(db32)
    dist, _ = index.search(q, k=1)
    return dist[:n, 0]


def _count_within(a32: np.ndarray, b32: np.ndarray, tree_b: cKDTree, r2_32: np.float32) -> int:
    """``np.sum(D < radius**2)`` of upstream's ``index_b.search(a, k=1)``."""
    a64 = a32.astype(np.float64)
    b64 = tree_b.data
    # Rounding bound on faiss's float32 squared distance (u = 2^-24). Either
    # path stays within about 11 u (|x|^2 + |y|^2) of the exact value of the
    # float32 inputs: the BLAS/fused kernels compute |x|^2 + |y|^2 - 2<x, y>
    # (3u on each norm, 3u (|x|^2 + |y|^2)/2 on the 3-term dot, 2u on the two
    # final operations, FMA or not); the sequential one sums three squared
    # differences (3u d^2). Only database points within `reach` of x can decide
    # x, and those have |y| <= |x| + reach, so the band is taken per query
    # point at 32 u (|x|^2 + (|x| + _REACH_CAP)^2), plus an absolute floor.
    # A point y beyond `reach` has faiss value >= reach^2 - 11 u (|x|^2 +
    # (|x| + d)^2) > r2 for every d >= reach, so it cannot be x's answer.
    u = _EPS32 / 2.0
    norm_a = np.einsum("ij,ij->i", a64, a64)
    margin = 32.0 * u * (norm_a + (np.sqrt(norm_a) + _REACH_CAP) ** 2) + 1e-12
    r2 = float(r2_32)
    lo, hi = r2 - margin, r2 + margin
    reach = float(np.sqrt(np.max(hi))) * (1.0 + 1e-9)
    if reach > _REACH_CAP:
        # Coordinates so large that the band reaches past the cap the bound
        # assumes: fall back to the global one (largest |y|^2 of the database).
        norm_b = float(np.max(np.einsum("ij,ij->i", b64, b64)))
        margin = 32.0 * u * (norm_a + norm_b) + 1e-12
        lo, hi = r2 - margin, r2 + margin
        reach = float(np.sqrt(np.max(hi))) * (1.0 + 1e-9)
    # Only points inside b's bounding box grown by `reach` can be within reach.
    inside = np.all((a64 >= tree_b.mins - reach) & (a64 <= tree_b.maxes + reach), axis=1)
    d2 = np.full(a64.shape[0], np.inf)
    if inside.any():
        d, _ = tree_b.query(a64[inside], k=1, distance_upper_bound=reach, workers=workers(int(inside.sum())))
        d2[inside] = d * d  # inf beyond reach
    sure = int(np.count_nonzero(d2 < lo))
    band = np.flatnonzero((d2 >= lo) & (d2 <= hi))
    STATS["points"] += a32.shape[0]
    if band.size == 0:
        return sure
    STATS["band"] += int(band.size)
    # faiss itself decides the band, against the whole database in upstream's
    # order: its float32 value for a (query, point) pair depends on the shape
    # of the database it is computed in (a 3-point subset gave a different
    # last bit), but not on which other queries share the call.
    dist = _faiss_min_d2(np.ascontiguousarray(a32[band]), b32, a32.shape[0])
    return sure + int(np.count_nonzero(dist < r2_32))


class _Mask:
    """A mask's float32 points and k-d tree, kept while its contents are unchanged."""

    __slots__ = ("a32", "tree")

    def __init__(self, pcd):
        self.a32 = np.asarray(pcd.points).astype(np.float32)
        self.tree = cKDTree(self.a32.astype(np.float64)) if self.a32.shape[0] else None


def _overlap(ma: _Mask, mb: _Mask, radius: float):
    """Same value as ``gu.find_overlapping_ratio_faiss`` on the two masks."""
    if ma.a32.shape[0] == 0 or mb.a32.shape[0] == 0:
        return 0
    # Upstream compares a float32 array with a Python float: NumPy casts the
    # scalar to float32 (value-based casting), so the threshold is float32.
    r2_32 = np.float32(radius**2)
    n1 = np.int64(_count_within(ma.a32, mb.a32, mb.tree, r2_32))
    n2 = np.int64(_count_within(mb.a32, ma.a32, ma.tree, r2_32))
    STATS["pairs"] += 1
    return np.max([n1 / ma.a32.shape[0], n2 / mb.a32.shape[0]])


def overlapping_ratio(pcd1, pcd2, radius=0.02) -> float:
    """Same value as ``gu.find_overlapping_ratio_faiss(pcd1, pcd2, radius)``."""
    return _overlap(_Mask(pcd1), _Mask(pcd2), radius)


#: Per-content caches, pruned to the current masks at every merge. Both values
#: are deterministic functions of the masks' contents (points), so a mask that
#: no frame changed reuses them: its tree, and its overlap with every other
#: unchanged mask.
_MASKS: dict = {}
_PAIRS: dict = {}


def _bbox_iou_matrix(mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """Upstream ``compute_3d_bbox_iou`` (padding 0) for all pairs, same float64 ops."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        omin = np.maximum(mins[:, None, :], mins[None, :, :])
        omax = np.minimum(maxs[:, None, :], maxs[None, :, :])
        osz = np.maximum(omax - omin, 0.0)
        ovol = (osz[..., 0] * osz[..., 1]) * osz[..., 2]
        ext = maxs - mins
        vol = (ext[:, 0] * ext[:, 1]) * ext[:, 2]
        return ovol / ((vol[:, None] + vol[None, :]) - ovol)


def merge_3d_masks(mask_list, overlap_threshold=0.5, radius=0.02, iou_thresh=0.05):
    """Upstream ``merge_3d_masks`` with the faiss search replaced; same output."""
    aa_bb = [pcd.get_axis_aligned_bounding_box() for pcd in mask_list]
    n = len(mask_list)
    overlap_matrix = np.zeros((n, n))
    if n:
        mins = np.array([np.asarray(b.get_min_bound()) for b in aa_bb], dtype=np.float64).reshape(n, 3)
        maxs = np.array([np.asarray(b.get_max_bound()) for b in aa_bb], dtype=np.float64).reshape(n, 3)
        iou = _bbox_iou_matrix(mins, maxs)
        ii, jj = np.nonzero(np.triu(iou > iou_thresh, k=1))
        verify = os.environ.get("HOVSG_FAST_MERGE_VERIFY") == "1"
        t0 = time.perf_counter()
        keys = [_points_digest(pcd) for pcd in mask_list]
        live = set(keys)
        for key in [k for k in _MASKS if k not in live]:
            del _MASKS[key]
        for key in [k for k in _PAIRS if k[0] not in live or k[1] not in live]:
            del _PAIRS[key]
        STATS["digest_s"] = STATS.get("digest_s", 0.0) + time.perf_counter() - t0
        for i, j in zip(ii.tolist(), jj.tolist()):
            t0 = time.perf_counter()
            pair = (keys[i], keys[j], 1.5 * radius)
            if pair in _PAIRS:
                value = _PAIRS[pair]
                STATS["pairs_cached"] = STATS.get("pairs_cached", 0) + 1
            else:
                for idx in (i, j):
                    if keys[idx] not in _MASKS:
                        _MASKS[keys[idx]] = _Mask(mask_list[idx])
                value = _overlap(_MASKS[keys[i]], _MASKS[keys[j]], 1.5 * radius)
                _PAIRS[pair] = value
            STATS["overlap_s"] = STATS.get("overlap_s", 0.0) + time.perf_counter() - t0
            if verify:
                ref = gu.find_overlapping_ratio_faiss(mask_list[i], mask_list[j], radius=1.5 * radius)
                if ref != value:
                    dump = os.environ.get("HOVSG_FAST_MERGE_DUMP")
                    if dump:
                        import pickle
                        with open(dump, "wb") as fh:
                            pickle.dump((np.asarray(mask_list[i].points), np.asarray(mask_list[j].points), 1.5 * radius), fh)
                    raise AssertionError(f"fast overlap {value!r} != upstream {ref!r} for pair ({i}, {j})")
                STATS["verified"] += 1
            overlap_matrix[i, j] = value
    # --- upstream, verbatim but for the memoised denoise ----------------------
    if overlap_matrix.size == 0:
        return mask_list
    graph = overlap_matrix > overlap_threshold
    n_components, component_labels = connected_components(graph)
    component_indices = [np.where(component_labels == i)[0] for i in range(n_components)]
    pcd_list_merged = []
    for indices in component_indices:
        pcd_list_merged.append(merge_point_clouds_list([mask_list[i] for i in indices], voxel_size=0.5 * radius))
    return pcd_list_merged


def install() -> None:
    """Route upstream's ``seq_merge`` / ``hierarchical_merge`` through this merge."""
    gu.merge_3d_masks = merge_3d_masks
