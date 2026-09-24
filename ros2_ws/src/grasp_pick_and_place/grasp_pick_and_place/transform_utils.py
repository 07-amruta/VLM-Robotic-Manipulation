import numpy as np
from geometry_msgs.msg import Pose
from rclpy.time import Time
from scipy.spatial.transform import Rotation


def make_matrix(rotation, translation):
    m = np.eye(4)
    m[:3, :3] = rotation
    m[:3, 3] = translation
    return m


def lookup_matrix(buffer, target_frame, source_frame):
    """4x4 pose of source_frame expressed in target_frame, from TF."""
    tf = buffer.lookup_transform(target_frame, source_frame, Time())
    q, t = tf.transform.rotation, tf.transform.translation
    return make_matrix(Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), [t.x, t.y, t.z])


def pose_to_matrix(pose):
    q, p = pose.orientation, pose.position
    return make_matrix(Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), [p.x, p.y, p.z])


def matrix_to_pose(matrix):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(float, matrix[:3, 3])
    (pose.orientation.x, pose.orientation.y,
     pose.orientation.z, pose.orientation.w) = map(
        float, Rotation.from_matrix(matrix[:3, :3]).as_quat())
    return pose