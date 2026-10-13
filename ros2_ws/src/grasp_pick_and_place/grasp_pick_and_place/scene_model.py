import xml.etree.ElementTree as ET

import numpy as np
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    AllowedCollisionEntry, AllowedCollisionMatrix, CollisionObject, PlanningScene,
    PlanningSceneComponents)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rcl_interfaces.srv import GetParameters
from scipy.spatial.transform import Rotation
from shape_msgs.msg import SolidPrimitive

from grasp_pick_and_place.config import OBJECT_ID, SUPPORT_ID
from grasp_pick_and_place.geometry import fit_box, is_dome, is_round, top_slice


class SceneMixin:
    """Object and table models in MoveIt's planning scene, gripper-object collision matrix."""

    def apply_scene(self, scene):
        request = ApplyPlanningScene.Request()
        request.scene = scene
        response = self.call_service(self.apply_scene_client, request)
        if response is None or not response.success:
            raise RuntimeError('Planning scene update was rejected')

    def discover_gripper_links(self):
        override = [n.strip() for n in self.get_parameter('gripper_links').value.split(',')
                    if n.strip()]
        if override:
            return override
        if not self.description_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(
                'Cannot read robot_description from /move_group; set the gripper_links parameter')
        request = GetParameters.Request()
        request.names = ['robot_description']
        response = self.call_service(self.description_client, request)
        urdf = response.values[0].string_value
        prefix = self.get_parameter('gripper_link_prefix').value
        links = [link.get('name') for link in ET.fromstring(urdf).iter('link')]
        found = [name for name in links if name.startswith(prefix)]
        if not found:
            raise RuntimeError(f'No links start with "{prefix}"; set the gripper_links parameter')
        return found

    def remove_object(self):
        """Clears a target_object left over from an earlier run."""
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
        current = self.call_service(self.get_scene_client, request).scene
        if any(o.id == OBJECT_ID for o in current.world.collision_objects):
            remove = CollisionObject()
            remove.id = OBJECT_ID
            remove.operation = CollisionObject.REMOVE
            scene = PlanningScene()
            scene.is_diff = True
            scene.world.collision_objects = [remove]
            self.apply_scene(scene)

    def add_object(self):
        """Fits a box, cylinder or sphere to the scene cloud and adds it to MoveIt's scene."""
        points = self.scene_points.astype(float)
        if points.shape[0] < 20:
            raise RuntimeError('Too few cloud points to model the object')
        margin = self.get_parameter('object_margin').value
        fraction = self.get_parameter('dome_slice').value
        centre, size, yaw, fill = fit_box(
            points, margin, self.get_parameter('support_top').value)
        self.round_object, rect_res, ellipse_res = is_round(points, yaw)
        self.sphere = self.round_object and is_dome(
            points, fraction, self.get_parameter('dome_ratio').value)
        if self.sphere:
            # The underside is hidden from the camera: the pole gives the centre (xy from the top
            # slice, z from the top of the cloud minus the radius) and the widest xy extent the
            # diameter.
            radius = float(size[:2].max() / 2.0)  # includes the margin
            cap = top_slice(points, fraction)[:, :2]
            centre = np.array([*(cap.min(axis=0) + cap.max(axis=0)) / 2.0,
                               points[:, 2].max() - radius + margin])
            size = np.full(3, 2.0 * radius)
        self.object_centre, self.object_size = centre, size
        ex = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        ey = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        self.box_axes = (ex, -ex, ey, -ey)

        obj = CollisionObject()
        obj.header.frame_id = self.scene_frame
        obj.id = OBJECT_ID
        primitive = SolidPrimitive()
        if self.sphere:
            primitive.type = SolidPrimitive.SPHERE
            primitive.dimensions = [float(size[0] / 2.0)]  # radius
        elif self.round_object:
            primitive.type = SolidPrimitive.CYLINDER
            primitive.dimensions = [float(size[2]), float(size[:2].mean() / 2.0)]  # height, radius
        else:
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [float(v) for v in size]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, centre)
        qx, qy, qz, qw = Rotation.from_euler('z', yaw).as_quat()
        pose.orientation.x, pose.orientation.y = float(qx), float(qy)
        pose.orientation.z, pose.orientation.w = float(qz), float(qw)
        obj.primitives = [primitive]
        obj.primitive_poses = [pose]
        obj.operation = CollisionObject.ADD
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [obj]
        self.apply_scene(scene)
        shape = 'sphere' if self.sphere else 'cylinder' if self.round_object else 'box'
        self.get_logger().info(
            f'Object {shape} size {np.round(size, 3).tolist()} '
            f'centre {np.round(centre, 3).tolist()} yaw {np.rad2deg(yaw):.0f} deg, '
            f'footprint fill {fill:.2f} rect {rect_res:.3f} ellipse {ellipse_res:.3f}')

    def add_support(self):
        """Adds the table as a block on the floor up to the support_top height."""
        top = self.get_parameter('support_top').value
        size_x, size_y = self.get_parameter('support_size').value
        centre_x, centre_y = self.get_parameter('support_centre').value
        obj = CollisionObject()
        obj.header.frame_id = self.scene_frame
        obj.id = SUPPORT_ID
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(size_x), float(size_y), float(top)]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (
            float(centre_x), float(centre_y), float(top / 2.0))
        pose.orientation.w = 1.0
        obj.primitives = [primitive]
        obj.primitive_poses = [pose]
        obj.operation = CollisionObject.ADD
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [obj]
        self.apply_scene(scene)

    def set_contact_allowed(self, allowed):
        """Allow or forbid gripper-object contact in MoveIt's collision matrix."""
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        acm = self.call_service(
            self.get_scene_client, request).scene.allowed_collision_matrix

        names = list(acm.entry_names)
        rows = [list(entry.enabled) for entry in acm.entry_values]
        if OBJECT_ID not in names:
            for row in rows:
                row.append(False)
            names.append(OBJECT_ID)
            rows.append([False] * len(names))
        k = names.index(OBJECT_ID)
        for i, name in enumerate(names):
            if name in self.gripper_links:
                rows[k][i] = allowed
                rows[i][k] = allowed

        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = AllowedCollisionMatrix(
            entry_names=names,
            entry_values=[AllowedCollisionEntry(enabled=row) for row in rows],
            default_entry_names=list(acm.default_entry_names),
            default_entry_values=list(acm.default_entry_values))
        self.apply_scene(scene)

    def extent_along(self, direction):
        """Length of the object's box projected on a unit direction."""
        ex, _, ey, _ = self.box_axes
        return float(abs(direction @ ex) * self.object_size[0]
                     + abs(direction @ ey) * self.object_size[1]
                     + abs(direction[2]) * self.object_size[2])