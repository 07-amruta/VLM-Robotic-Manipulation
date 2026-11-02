import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from perception.config import CAMERA, PERCEPTION


def quaternion_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def lookup_matrix(buffer, target, source):
    """Latest 4x4 matrix mapping points in `source` into `target`."""
    tf = buffer.lookup_transform(target, source, Time())
    t, q = tf.transform.translation, tf.transform.rotation
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_to_matrix(q.x, q.y, q.z, q.w)
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


class ConfiguredNode(Node):
    """Node whose parameters default to the active camera profile and PERCEPTION values."""

    def __init__(self, name):
        super().__init__(name)
        for key, value in {**CAMERA, **PERCEPTION}.items():
            self.declare_parameter(key, value)

    def param(self, name):
        return self.get_parameter(name).value


def run_node(node_class):
    rclpy.init()
    node = node_class()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        if rclpy.ok():
            raise  # a real failure; only swallow errors raised while the context shuts down
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()