import os
import sys

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PoseStamped
from graspnetAPI import GraspGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener


def load_graspnet(root, checkpoint_path, device):
    # GraspNet-baseline is not pip-installable; its modules import each other by bare name.
    sys.path.insert(0, root)
    for sub in ('models', 'dataset', 'utils'):
        sys.path.append(os.path.join(root, sub))
    from models.graspnet import GraspNet, pred_decode

    net = GraspNet(
        input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
        cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False)
    net.to(device)
    # The official checkpoint is a pickled dict; only load files you trust.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net, pred_decode


def lookup_matrix(buffer, target_frame, source_frame):
    tf = buffer.lookup_transform(target_frame, source_frame, Time())
    q, t = tf.transform.rotation, tf.transform.translation
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    m[:3, 3] = [t.x, t.y, t.z]
    return m


class GraspProposer(Node):
    """Runs GraspNet on the cropped scene cloud and publishes the best grasp in the cloud's frame."""

    def __init__(self):
        super().__init__('grasp_proposer')

        self.declare_parameter('graspnet_root', '/opt/graspnet-baseline')
        self.declare_parameter(
            'checkpoint_path',
            '/workspaces/VLM-Robotic-Manipulation/checkpoints/graspnet/checkpoint-rs.tar')
        self.declare_parameter('optical_frame', 'scene_camera_optical')
        self.declare_parameter('num_point', 20000)
        # Reject grasps whose approach axis points upward, i.e. coming from below the table.
        self.declare_parameter('max_approach_z', 0.0)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net, self.pred_decode = load_graspnet(
            self.get_parameter('graspnet_root').value,
            self.get_parameter('checkpoint_path').value,
            self.device)
        self.get_logger().info(f'GraspNet loaded on {self.device}')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(PointCloud2, 'scene_points', self.on_cloud, latched)
        self.pose_pub = self.create_publisher(PoseStamped, 'grasp_pose', latched)

    def on_cloud(self, msg):
        # Cloud format produced by scene_snapshot: xyz float32
        points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 3)
        if points.shape[0] == 0:
            self.get_logger().warn('Empty cloud, skipping')
            return

        try:
            base_from_optical = lookup_matrix(
                self.tf_buffer, msg.header.frame_id,
                self.get_parameter('optical_frame').value)
        except TransformException as e:
            self.get_logger().error(f'TF lookup failed: {e}')
            return

        # GraspNet expects the cloud in the camera frame it was trained on
        optical_from_base = np.linalg.inv(base_from_optical)
        points_opt = points @ optical_from_base[:3, :3].T + optical_from_base[:3, 3]

        n = self.get_parameter('num_point').value
        count = points_opt.shape[0]
        if count >= n:
            idx = np.random.choice(count, n, replace=False)
        else:
            idx = np.concatenate([np.arange(count), np.random.choice(count, n - count, replace=True)])
        sampled = torch.from_numpy(points_opt[idx][np.newaxis].astype(np.float32)).to(self.device)

        with torch.no_grad():
            end_points = self.net({'point_clouds': sampled})
            grasp_array = self.pred_decode(end_points)[0].cpu().numpy()

        # Grasps back into the cloud's frame
        grasps = GraspGroup(grasp_array)
        rot = base_from_optical[:3, :3]
        rotations = rot @ grasps.rotation_matrices
        positions = grasps.translations @ rot.T + base_from_optical[:3, 3]

        keep = rotations[:, 2, 0] <= self.get_parameter('max_approach_z').value
        if not keep.any():
            self.get_logger().warn('No grasp passed the approach filter')
            return
        best = int(np.argmax(np.where(keep, grasps.scores, -np.inf)))

        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = msg.header.frame_id
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, positions[best])
        (pose.pose.orientation.x, pose.pose.orientation.y,
         pose.pose.orientation.z, pose.pose.orientation.w) = map(
            float, Rotation.from_matrix(rotations[best]).as_quat())
        self.pose_pub.publish(pose)
        self.get_logger().info(
            f'Best grasp: score {grasps.scores[best]:.3f}, '
            f'position {np.round(positions[best], 3).tolist()}')


def main():
    rclpy.init()
    node = GraspProposer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()