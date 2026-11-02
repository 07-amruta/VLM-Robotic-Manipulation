import numpy as np
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2, PointField

from perception.projection import project, select_in_mask, transform_points


def remove_outliers(points, k, std_ratio):
    """Drops points whose mean k-NN distance exceeds mean + std_ratio * std."""
    if len(points) <= k:
        return points
    dist, _ = cKDTree(points).query(points, k=k + 1)
    mean = dist[:, 1:].mean(axis=1)
    return points[mean <= mean.mean() + std_ratio * mean.std()]


def object_clouds(points, to_mask, to_target, k_mask, instances, outlier_k, outlier_std, min_points):
    """points: Nx3 in the depth optical frame. to_mask / to_target: 4x4 from the depth frame.
    instances: [(track_id, class_name, score, bool mask)]. Returns [(track_id, class_name, score, Mx3 in target)].
    Mask-frame lookup per point, so unaligned depth/colour needs no image warping."""
    pixels, in_front = project(transform_points(points, to_mask), k_mask)
    target = transform_points(points, to_target)
    out = []
    for track_id, name, score, mask in instances:
        cloud = remove_outliers(target[in_front & select_in_mask(pixels, mask)], outlier_k, outlier_std)
        if len(cloud) >= min_points:
            out.append((track_id, name, score, cloud))
    return out


def to_pointcloud2(points, header):
    cloud = PointCloud2()
    cloud.header = header
    cloud.height = 1
    cloud.width = len(points)
    cloud.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                    for i, n in enumerate('xyz')]
    cloud.is_bigendian = False
    cloud.point_step = 12
    cloud.row_step = 12 * cloud.width
    cloud.is_dense = True
    cloud.data = points.astype(np.float32).tobytes()
    return cloud