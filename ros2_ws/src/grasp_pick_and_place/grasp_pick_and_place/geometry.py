import numpy as np
from scipy.spatial import ConvexHull


def fit_box(points, margin, base=None):
    """Yaw-aligned box around a cloud: minimum-area rectangle in xy, full extent in z.

    Returns (centre, size, yaw, fill). The bottom face sits on `base` (the table top) when it is
    given and lower than the cloud, otherwise on the cloud's lowest point.
    """
    xy = points[:, :2]
    best = None
    for yaw in np.deg2rad(np.arange(0.0, 90.0, 1.0)):
        c, s = np.cos(yaw), np.sin(yaw)
        local = xy @ np.array([[c, -s], [s, c]])
        low, high = local.min(axis=0), local.max(axis=0)
        area = np.prod(high - low)
        if best is None or area < best[0] * 0.98:
            best = (area, yaw, low, high)
    _, yaw, low, high = best
    fill = ConvexHull(xy).volume / np.prod(high - low)
    c, s = np.cos(yaw), np.sin(yaw)
    centre_xy = np.array([[c, -s], [s, c]]) @ ((low + high) / 2.0)
    z_low, z_high = points[:, 2].min(), points[:, 2].max()
    if base is not None:
        z_low = min(z_low, base)
    centre = np.array([centre_xy[0], centre_xy[1], (z_low + z_high + margin) / 2.0])
    size = np.array([*(high - low + 2.0 * margin), z_high - z_low + margin])
    return centre, size, yaw, fill

def is_round(points, yaw):
    """True when the hull follows the ellipse inscribed in the oriented box better than the box outline."""
    xy = points[:, :2]
    c, s = np.cos(yaw), np.sin(yaw)
    local = xy @ np.array([[c, -s], [s, c]])
    hull = local[ConvexHull(xy).vertices]
    t = np.linspace(0.0, 1.0, 20, endpoint=False)[None, :, None]
    edge = (hull[:, None, :] * (1 - t) + np.roll(hull, -1, axis=0)[:, None, :] * t).reshape(-1, 2)
    low, high = local.min(axis=0), local.max(axis=0)
    u, v = (2.0 * (edge - (low + high) / 2.0) / (high - low)).T
    rect = np.abs(np.maximum(abs(u), abs(v)) - 1.0).mean()
    ellipse = np.abs(np.hypot(u, v) - 1.0).mean()
    return ellipse < rect, rect, ellipse

def top_slice(points, fraction):
    """Points in the highest `fraction` of the cloud's height."""
    z = points[:, 2]
    return points[z >= z.max() - fraction * (z.max() - z.min())]


def is_dome(points, fraction, ratio):
    """True when the top slice is much narrower than the whole cloud (a sphere's cap)."""
    widest = np.ptp(points[:, :2], axis=0).max()
    return np.ptp(top_slice(points, fraction)[:, :2], axis=0).max() < ratio * widest


def is_similar(a, b, max_dist=0.01, max_angle=0.1):
    cos = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return np.linalg.norm(a[:3, 3] - b[:3, 3]) < max_dist and np.arccos(cos) < max_angle


def longest_run(flags):
    """(first, last) indices of the longest run of True values, or None."""
    best, start = None, None
    for k, ok in enumerate(list(flags) + [False]):
        if ok and start is None:
            start = k
        elif not ok and start is not None:
            if best is None or k - 1 - start > best[1] - best[0]:
                best = (start, k - 1)
            start = None
    return best