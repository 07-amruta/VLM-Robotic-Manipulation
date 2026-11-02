import struct
import xml.etree.ElementTree as ET

import numpy as np
from ament_index_python.packages import get_package_share_directory


def _rot(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp, cp*sr, cp*cr]])


def _resolve(fn):
    if fn.startswith('file://'):
        return fn[7:]
    if fn.startswith('package://'):
        pkg, rel = fn[10:].split('/', 1)
        return get_package_share_directory(pkg) + '/' + rel
    return fn


def _stl_vertices(path):
    data = open(path, 'rb').read()
    n = struct.unpack('<I', data[80:84])[0]
    if len(data) == 84 + 50 * n:  # binary
        dt = np.dtype([('n', '<3f4'), ('v', '<9f4'), ('a', '<u2')])
        return np.frombuffer(data, dt, count=n, offset=84)['v'].reshape(-1, 3).astype(float)
    t = data.decode(errors='ignore').split()  # ascii
    return np.array([[float(t[i + 1]), float(t[i + 2]), float(t[i + 3])]
                     for i, w in enumerate(t) if w == 'vertex'])


def _vec(s, default):
    return np.array([float(v) for v in (s or default).split()])


def link_boxes(urdf_xml):
    """{link: (lo, hi)} collision AABB in the link frame; mesh and cylinder only."""
    boxes = {}
    for link in ET.fromstring(urdf_xml).iter('link'):
        pts = []
        for col in link.findall('collision'):
            geom, o = col.find('geometry'), col.find('origin')
            xyz = _vec(o.get('xyz') if o is not None else None, '0 0 0')
            rpy = _vec(o.get('rpy') if o is not None else None, '0 0 0')
            mesh, cyl = geom.find('mesh'), geom.find('cylinder')
            if mesh is not None:
                v = _stl_vertices(_resolve(mesh.get('filename'))) * _vec(mesh.get('scale'), '1 1 1')
            elif cyl is not None:
                r, h = float(cyl.get('radius')), float(cyl.get('length'))
                v = np.array([[a * r, b * r, c * h / 2] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
            else:
                continue
            pts.append(v @ _rot(rpy).T + xyz)
        if pts:
            allp = np.vstack(pts)
            boxes[link.get('name')] = (allp.min(0), allp.max(0))
    return boxes


def self_keep_mask(points, boxes, transform_of, margin):
    """True for points outside every link box. transform_of(link) -> 4x4 link->points frame, or None."""
    keep = np.ones(len(points), bool)
    for name, (lo, hi) in boxes.items():
        T = transform_of(name)
        if T is None:
            continue
        q = (points - T[:3, 3]) @ T[:3, :3]  # R^T (p - t) per row
        keep &= ~np.all((q >= lo - margin) & (q <= hi + margin), axis=1)
    return keep