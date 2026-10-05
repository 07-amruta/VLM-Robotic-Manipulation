import os
import sys

import numpy as np
import rclpy
import torch
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, PoseStamped
from graspnetAPI import GraspGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformException, TransformListener

from grasp_pick_and_place.transform_utils import lookup_matrix, matrix_to_pose


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


def grasp_pose(position, rotation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix_to_pose(matrix)


def select_candidates(positions, rotations, scores, limit, min_dist, min_angle):
    """Best-first selection that drops grasps close to one already kept."""
    kept = []
    for i in np.argsort(-scores):
        duplicate = False
        for j in kept:
            cos = np.clip((np.trace(rotations[i].T @ rotations[j]) - 1.0) / 2.0, -1.0, 1.0)
            if (np.linalg.norm(positions[i] - positions[j]) < min_dist
                    and np.arccos(cos) < min_angle):
                duplicate = True
                break
        if not duplicate:
            kept.append(int(i))
            if len(kept) == limit:
                break
    return kept


class GraspProposer(Node):
    """Runs GraspNet on the cropped scene cloud and publishes ranked grasps in the cloud's frame.

    GraspNet frame: x = approach, y = jaw closing axis. The executor adapts this to the hand.
    """

    def __init__(self):
        super().__init__('grasp_proposer')

        self.declare_parameter('graspnet_root', '/opt/graspnet-baseline')
        self.declare_parameter(
            'checkpoint_path',
            '/workspaces/VLM-Robotic-Manipulation/checkpoints/graspnet/checkpoint-rs.tar')
        self.declare_parameter('optical_frame', 'scene_camera_optical')
        self.declare_parameter('num_point', 20000)
        # Drop grasps whose approach points upward (z component of the approach above this).
        # 0.0 keeps horizontal and downward approaches; the executor ranks them by reachability
        # and by how close the hand can close to the object's centre.
        self.declare_parameter('max_approach_z', 0.0)
        # Drop grasps centred lower than this above the object's lowest point (the table).
        self.declare_parameter('min_grasp_height', 0.02)
        self.declare_parameter('num_candidates', 30)
        self.declare_parameter('nms_translation', 0.03)
        self.declare_parameter('nms_rotation', 0.5)

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
        self.candidates_pub = self.create_publisher(PoseArray, 'grasp_candidates', latched)

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
            idx = np.concatenate(
                [np.arange(count), np.random.choice(count, n - count, replace=True)])
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
        keep &= positions[:, 2] >= points[:, 2].min() + self.get_parameter('min_grasp_height').value
        if not keep.any():
            self.get_logger().warn('No grasp passed the approach/height filter')
            return
        valid = np.flatnonzero(keep)
        chosen = valid[select_candidates(
            positions[valid], rotations[valid], grasps.scores[valid],
            self.get_parameter('num_candidates').value,
            self.get_parameter('nms_translation').value,
            self.get_parameter('nms_rotation').value)]

        candidates = PoseArray()
        candidates.header.stamp = msg.header.stamp
        candidates.header.frame_id = msg.header.frame_id
        candidates.poses = [grasp_pose(positions[i], rotations[i]) for i in chosen]
        self.candidates_pub.publish(candidates)

        best = int(chosen[0])
        pose = PoseStamped()
        pose.header = candidates.header
        pose.pose = candidates.poses[0]
        self.pose_pub.publish(pose)
        self.get_logger().info(
            f'Best grasp: score {grasps.scores[best]:.3f}, '
            f'position {np.round(positions[best], 3).tolist()}, '
            f'{len(chosen)} candidates')


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