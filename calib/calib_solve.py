import numpy as np
from scipy.spatial.transform import Rotation

CUBE_H = 0.08        # object top face above the box top, m (caliper)
OBJ_R = 0.11         # half the longest footprint dimension plus margin, m
TABLE_Z = 0.07
TOUCH = {'A': (-0.1888, 0.9668), 'B': (0.1704, 0.9509), 'C': (-0.0127, 1.1551)}
THR = 0.005
rng = np.random.default_rng(0)


def fit_plane(p, iters=500):
    s = p[rng.choice(len(p), min(len(p), 60000), replace=False)]
    best = None
    for _ in range(iters):
        a = s[rng.choice(len(s), 3, replace=False)]
        n = np.cross(a[1] - a[0], a[2] - a[0]); k = np.linalg.norm(n)
        if k < 1e-9: continue
        n /= k
        inl = np.abs((s - a[0]) @ n) < THR
        if best is None or inl.sum() > best.sum(): best = inl
    q = s[best]; c = q.mean(0)
    n = np.linalg.svd(q - c)[2][-1]
    return (-n if n @ c > 0 else n), c            # normal faces the camera


def level(n):
    z = np.array([0.0, 0.0, 1.0]); v = np.cross(n, z)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + n @ z)


def top_xy(P, zt, lo, hi):
    P = P[np.all((P[:, :2] >= lo) & (P[:, :2] <= hi), axis=1)]
    h = P[:, 2] - zt
    s = P[(h > CUBE_H - 0.012) & (h < CUBE_H + 0.008)]
    if len(s) < 50:
        raise SystemExit(f'only {len(s)} top-face points: check CUBE_H and the capture')
    q = s[rng.choice(len(s), min(len(s), 3000), replace=False)]
    d = np.linalg.norm(q[:, None, :2] - q[None, :, :2], axis=2)
    centre = q[(d < 0.04).sum(1).argmax(), :2]
    for _ in range(5):          # re-centre until the window covers the whole top face
        top = s[np.linalg.norm(s[:, :2] - centre, axis=1) < OBJ_R]
        centre = top[:, :2].mean(0)
    return centre, len(top)


# Two largest planes: floor and box top. The table is the one nearer the camera.
rest = np.load('/tmp/table.npy')
planes = []
for _ in range(2):
    n, c = fit_plane(rest)
    inl = np.abs((rest - c) @ n) < THR
    planes.append((n, c, rest[inl], -(n @ c)))
    rest = rest[~inl]
(n1, _, p1, d1), (n2, _, p2, d2) = planes
angle = np.degrees(np.arccos(np.clip(n1 @ n2, -1, 1)))
print(f'plane distances {d1:.3f} / {d2:.3f} m, angle {angle:.1f} deg, '
      f'inliers {len(p1)} / {len(p2)}')
if angle > 5:
    raise SystemExit('the two planes are not parallel: check the table capture')
n, c, tab, dist = min(planes, key=lambda pl: pl[3])
floor = max(planes, key=lambda pl: pl[3])
rms = np.std((tab - c) @ n) * 1000
print(f'table: distance {dist:.3f} m, inliers {len(tab)}, RMS {rms:.1f} mm, normal {np.round(n, 3).tolist()}')

R = level(n)
tl = tab @ R.T
zt = tl[:, 2].mean()
lo = np.percentile(tl[:, :2], 1, axis=0) + 0.01     # box-top footprint, inset 1 cm
hi = np.percentile(tl[:, :2], 99, axis=0) - 0.01

L = {}
for k in 'ABC':
    L[k], cnt = top_xy(np.load(f'/tmp/{k}.npy') @ R.T, zt, lo, hi)
    print(k, 'top-face points', cnt)
B = {k: np.array(v) for k, v in TOUCH.items()}
th = np.arctan2(*(B['B'] - B['A'])[::-1]) - np.arctan2(*(L['B'] - L['A'])[::-1])
Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
txy = np.mean([B[k] - Rz @ L[k] for k in 'AB'], axis=0)
for k in 'ABC':
    print(k, 'error mm', round(1000 * float(np.linalg.norm(Rz @ L[k] + txy - B[k])), 1))
print('yaw deg', round(float(np.degrees(th)), 2))

T = np.eye(4)
T[:3, :3] = Rotation.from_euler('z', th).as_matrix() @ R
T[:3, 3] = [txy[0], txy[1], TABLE_Z - zt]
tb = tab @ T[:3, :3].T + T[:3, 3]
print('box top in base_link: x', np.round(np.percentile(tb[:, 0], [1, 99]), 3).tolist(),
      'y', np.round(np.percentile(tb[:, 1], [1, 99]), 3).tolist(), 'z', round(float(tb[:, 2].mean()), 4))

fb = floor[2] @ T[:3, :3].T + T[:3, 3]
print('plane step floor-to-table mm', round((floor[3] - dist) * 1000, 1),
      '| floor z in base_link', round(float(fb[:, 2].mean()), 4), '(expect -0.05)')
Tb = T @ np.linalg.inv(np.load('/tmp/root_T_cloud.npy'))   # base_link -> camera_link
q = Rotation.from_matrix(Tb[:3, :3]).as_quat()
print('static TF base_link -> camera_link (x y z qx qy qz qw):')
print(*np.round(Tb[:3, 3], 5), *np.round(q, 6))