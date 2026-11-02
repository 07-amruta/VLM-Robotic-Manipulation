import numpy as np

from grasp_pick_and_place.projection import (
    deproject, depth_to_metres, project, select_in_mask, transform_points)

K = np.array([[900.0, 0.0, 32.0], [0.0, 900.0, 24.0], [0.0, 0.0, 1.0]])


def test_round_trip_recovers_pixels():
    depth = np.full((48, 64), 0.5)
    points, pixels = deproject(depth, K)
    projected, in_front = project(points, K)
    assert in_front.all()
    assert np.allclose(projected, pixels)


def test_invalid_depth_dropped():
    depth = np.full((4, 4), 0.5)
    depth[0, 0], depth[1, 1], depth[2, 2] = 0.0, np.nan, np.inf
    points, _ = deproject(depth, K)
    assert len(points) == 13


def test_transform_translation():
    matrix = np.eye(4)
    matrix[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(transform_points(np.zeros((1, 3)), matrix), [[1.0, 2.0, 3.0]])


def test_mask_selection():
    mask = np.zeros((48, 64), dtype=bool)
    mask[10:20, 30:40] = True
    selected = select_in_mask(np.array([[35.0, 15.0], [5.0, 5.0], [-3.0, 15.0]]), mask)
    assert selected.tolist() == [True, False, False]


def test_depth_encodings_agree():
    metres = np.full((2, 2), 0.5, dtype=np.float32)
    millimetres = np.full((2, 2), 500, dtype=np.uint16)
    a = depth_to_metres(metres.tobytes(), '32FC1', 2, 2)
    b = depth_to_metres(millimetres.tobytes(), '16UC1', 2, 2)
    assert np.allclose(a, b)