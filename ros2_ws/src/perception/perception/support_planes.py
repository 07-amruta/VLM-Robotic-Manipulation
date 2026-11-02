import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree


def fit_horizontal_plane(points, iters, tol, max_tilt_deg, rng):
    """RANSAC plane with normal within max_tilt_deg of +z. Returns (unit normal, offset d: n.p + d = 0) or None."""
    idx = rng.integers(0, len(points), size=(iters, 3))
    a, b, c = points[idx[:, 0]], points[idx[:, 1]], points[idx[:, 2]]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=1)
    ok = norm > 1e-9
    n[ok] /= norm[ok, None]
    n[n[:, 2] < 0] *= -1
    ok &= n[:, 2] >= np.cos(np.radians(max_tilt_deg))
    if not ok.any():
        return None
    n, a = n[ok], a[ok]
    d = -np.einsum('ij,ij->i', n, a)
    counts = (np.abs(points @ n.T + d) < tol).sum(axis=0)
    best = counts.argmax()
    return n[best], d[best]


def plane_area(xy, rng, sample=5000):
    """Convex-hull area (m^2) of 2D points; 0 if degenerate."""
    if len(xy) < 3:
        return 0.0
    if len(xy) > sample:
        xy = xy[rng.choice(len(xy), sample, replace=False)]
    try:
        return float(ConvexHull(xy).volume)
    except RuntimeError:  # QhullError
        return 0.0


def peel_support_planes(points, tol, margin, iters, min_area, max_planes, max_tilt_deg,
                        fit_sample=20000, seed=None):
    """Iteratively removes large horizontal planes (table, floor, shelf).
    Returns (planes [{normal, offset, z}], remaining: bool mask over `points`).
    Points within tol below to `margin` above each plane are removed with it."""
    rng = np.random.default_rng(seed)
    remaining = np.ones(len(points), dtype=bool)
    planes = []
    for _ in range(max_planes):
        idx = np.flatnonzero(remaining)
        if len(idx) < 3:
            break
        sample = idx if len(idx) <= fit_sample else rng.choice(idx, fit_sample, replace=False)
        fit = fit_horizontal_plane(points[sample], iters, tol, max_tilt_deg, rng)
        if fit is None:
            break
        n, d = fit
        current = points[idx]
        dist = current @ n + d
        inliers = current[np.abs(dist) < tol]
        if plane_area(inliers[:, :2], rng) < min_area:
            break
        planes.append({'normal': n, 'offset': d, 'z': float(inliers[:, 2].mean())})
        remaining[idx[(dist > -tol) & (dist < margin)]] = False
    return planes, remaining


def cluster_labels(points, radius, voxel):
    """Connected-component label per point (points closer than `radius` link). Voxel-downsampled for speed."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    centres = points[first]
    pairs = cKDTree(centres).query_pairs(radius, output_type='ndarray')
    n = len(centres)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels[inverse.reshape(-1)]