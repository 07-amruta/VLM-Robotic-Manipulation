import numpy as np

MM_TO_M = 0.001  # REP 118: 16UC1 depth is millimetres


def depth_to_metres(data, encoding, height, width):
    """Raw depth image bytes -> float array in metres. Supports 32FC1 (m) and 16UC1 (mm)."""
    if encoding == '32FC1':
        return np.frombuffer(data, dtype=np.float32).reshape(height, width).astype(np.float64)
    if encoding == '16UC1':
        return np.frombuffer(data, dtype=np.uint16).reshape(height, width) * MM_TO_M
    raise ValueError(f'Unsupported depth encoding: {encoding}')


def deproject(depth, k):
    """Depth image (m) and 3x3 intrinsics -> (points Nx3 in the optical frame, pixels Nx2 as u, v).
    Pixels with non-finite or non-positive depth are dropped."""
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    v, u = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 0.0)
    z = depth[valid]
    points = np.stack(((u[valid] - cx) * z / fx, (v[valid] - cy) * z / fy, z), axis=1)
    return points, np.stack((u[valid], v[valid]), axis=1)


def transform_points(points, matrix):
    """Apply a 4x4 homogeneous transform to Nx3 points."""
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def project(points, k):
    """Nx3 points in an optical frame -> (pixels Nx2 as float u, v, in_front mask)."""
    in_front = points[:, 2] > 0.0
    z = np.where(in_front, points[:, 2], 1.0)
    u = k[0, 0] * points[:, 0] / z + k[0, 2]
    v = k[1, 1] * points[:, 1] / z + k[1, 2]
    return np.stack((u, v), axis=1), in_front


def select_in_mask(pixels, mask):
    """Boolean per pixel: True where it lands on a True cell of a 2D mask (nearest pixel)."""
    height, width = mask.shape
    cols = np.round(pixels[:, 0]).astype(int)
    rows = np.round(pixels[:, 1]).astype(int)
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    selected = np.zeros(len(pixels), dtype=bool)
    selected[inside] = mask[rows[inside], cols[inside]]
    return selected