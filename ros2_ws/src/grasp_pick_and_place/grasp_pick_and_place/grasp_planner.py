import numpy as np
from scipy.spatial.transform import Rotation

from grasp_pick_and_place.config import ARM_JOINTS, GRIPPER_JOINTS, OBJECT_ID, SUPPORT_ID
from grasp_pick_and_place.geometry import longest_run


class GraspPlannerMixin:
    """Hand placement around the object and feasibility checks. Nothing here moves the robot."""

    def hand_poses(self, proposal):
        """Turns a proposal into hand poses that envelope the object.

        Only the proposal's approach direction is used. The hand's closing axis is put on the
        proposal's jaw axis, then snapped to the nearest box face normal that is not parallel to
        the approach (or to the level direction for a round object), and the hand's closing centre
        is put on the object's centroid. Both 180-degree wrist variants are returned.
        """
        approach = proposal[:3, 0]
        tilt = -np.arctan2(self.hand_axis[2], self.hand_axis[1])
        rot = proposal[:3, :3] @ Rotation.from_rotvec([tilt, 0.0, 0.0]).as_matrix()
        closing = rot @ self.hand_axis

        turns = []
        for axis in self.box_axes:
            target = axis - approach * (axis @ approach)
            length = np.linalg.norm(target)
            if length < 1e-3:  # face normal parallel to the approach: no closing direction here
                continue
            target /= length
            if self.extent_along(target) > self.hand_gap:  # object too wide to straddle
                continue
            turns.append(np.arctan2(approach @ np.cross(closing, target), closing @ target))

        if not turns:
            return []
        snap = min(turns, key=abs)
        if self.round_object:
            # Every horizontal direction is a "face" of an upright round object: close level
            level = np.cross([0.0, 0.0, 1.0], approach)
            if np.linalg.norm(level) > 1e-6:  # a vertical approach keeps the box-face snap
                level /= np.linalg.norm(level)
                level *= np.sign(closing @ level) or 1.0  # keep the smaller of the two turns
                snap = np.arctan2(approach @ np.cross(closing, level), closing @ level)

        poses = []
        for extra in (0.0, np.pi):
            r = Rotation.from_rotvec(approach * (snap + extra)).as_matrix() @ rot
            pose = np.eye(4)
            pose[:3, :3] = r
            pose[:3, 3] = self.object_centre - r @ self.hand_centre
            poses.append(pose)
        return poses

    def top_down_grasp(self):
        """Fallback proposal: straight down onto the object, closing across its narrower side."""
        ex, _, ey, _ = self.box_axes
        closing = ex if self.object_size[0] <= self.object_size[1] else ey
        approach = np.array([0.0, 0.0, -1.0])
        grasp = np.eye(4)
        grasp[:3, :3] = np.column_stack([approach, closing, np.cross(approach, closing)])
        grasp[:3, 3] = self.object_centre
        return grasp

    def object_contacts(self, joints):
        """Names of the robot bodies touching the object or the table in this arm configuration."""
        response = self.validity_at(joints)
        touched = (OBJECT_ID, SUPPORT_ID)
        def tag(c):
            other = c.contact_body_2 if c.contact_body_1 in touched else c.contact_body_1
            hit = c.contact_body_1 if c.contact_body_1 in touched else c.contact_body_2
            return f'{other.replace("gripper_", "")}>{"T" if hit == SUPPORT_ID else "O"}'
        return sorted({tag(c) for c in response.contacts
                       if c.contact_body_1 in touched or c.contact_body_2 in touched})

    def closed_hand_clear(self, joints):
        """Closes the fingers in MoveIt at the grasp pose `joints`, from open up to the first
        contact with the fitted object plus the squeeze margin, and checks that nothing other
        than the object is touched on the way (table, arm). Returns False if something is.
        If the model never touches the object, there is nothing to check against: returns True."""
        step = self.get_parameter('closure_step').value
        limit = min(self.get_parameter('close_limit').value)
        margin = self.get_parameter(
            'squeeze_margin_sphere' if self.sphere else 'squeeze_margin').value
        opened = self.get_parameter('gripper_open').value[0]
        closures = np.arange(opened, limit + 1e-9, step)

        # Pass 1: first closure at which the fingers reach the object
        self.set_contact_allowed(False)
        reach = None
        for q in closures:
            response = self.validity_at(joints, [q] * len(GRIPPER_JOINTS))
            if any(OBJECT_ID in (c.contact_body_1, c.contact_body_2) for c in response.contacts):
                reach = q
                break
        if reach is None:
            return True

        # Pass 2: with gripper-object contact allowed, every closure up to the grasp must be valid
        self.set_contact_allowed(True)
        last = min(reach + margin, limit)
        for q in [c for c in closures if c <= last] + [last]:
            if not self.validity_at(joints, [q] * len(GRIPPER_JOINTS)).valid:
                return False
        return True

    def find_envelope(self, grasp):
        """Slides the open hand across its closing axis, backing off along the approach (at most
        the object's full extent along it) to the deepest position where it touches nothing.
        Returns (pose, free_width, retreat) or None."""
        step = self.get_parameter('search_step').value
        reach = self.get_parameter('search_reach').value
        offsets = np.arange(-reach, reach + 1e-9, step)
        closing = grasp[:3, :3] @ self.hand_axis
        approach = grasp[:3, 0]
        retreats = np.arange(0.0, self.extent_along(approach) + 1e-9,
                             self.get_parameter('retreat_step').value)
        seed = list(self.joint_positions)
        touched = set()
        for retreat in retreats:
            free, touched = [], set()
            for offset in offsets:
                moved = grasp.copy()
                moved[:3, 3] += closing * offset - approach * retreat
                joints = self.solve_ik(moved, seed, avoid_collisions=False)
                if joints is None:
                    free.append(False)
                    touched.add('ik')
                    continue
                hit = self.object_contacts(joints)
                free.append(not hit)
                touched.update(hit)
            run = longest_run(free)
            if run is not None and (run[1] - run[0] + 1) * step >= self.get_parameter(
                    'min_free_width').value:
                moved = grasp.copy()
                moved[:3, 3] += closing * (offsets[run[0]] + offsets[run[1]]) / 2.0 \
                    - approach * retreat
                return moved, (run[1] - run[0] + 1) * step, retreat
        self.get_logger().info(f'hand search: touched at highest retreat: {sorted(touched)}')
        return None

    def targets(self, grasp):
        # Start outside the object's bounding sphere, whatever its size
        back = 0.5 * np.linalg.norm(self.object_size) + self.get_parameter('approach_margin').value
        pre, lift = grasp.copy(), grasp.copy()
        pre[:3, 3] -= back * grasp[:3, 0]
        lift[:3, 3] += [0.0, 0.0, self.get_parameter('lift_height').value]
        return {'pre': pre, 'grasp': grasp, 'lift': lift}

    def reject(self, reason):
        self.rejections[reason] += 1
        self.get_logger().warn(f'Rejected: {reason}')
        return None

    def pre_solutions(self, target):
        """Distinct collision-free IK solutions for `target`, nearest the current joints first."""
        start = np.array(self.joint_positions)
        rng = np.random.default_rng(0)
        seeds = [start] + [rng.uniform(-np.pi, np.pi, len(ARM_JOINTS))
                           for _ in range(self.get_parameter('ik_seeds').value)]
        found = []
        for seed in seeds:
            joints = self.solve_ik(target, seed)
            if joints is None:
                continue
            joints = np.array(joints)
            if all(np.abs(joints - other).max() > 0.05 for other in found):
                found.append(joints)
        found.sort(key=lambda q: np.abs(q - start)[:5].max())
        return found

    def check_branch(self, joints, goals):
        """Checks the whole pick for one pre-grasp solution. Returns (plan, reason)."""
        start = list(self.joint_positions)
        up_joints = self.get_parameter('up_joints').value

        self.set_contact_allowed(False)
        approach = self.plan_line(joints, goals['grasp'])
        if approach is None:
            return None, 'straight approach is blocked'

        if not self.closed_hand_clear(approach[1]):
            return None, 'closed hand touches something other than the object'

        self.set_contact_allowed(True)  # once closed, the fingers touch the object by design
        lift = self.plan_line(approach[1], goals['lift'])
        if lift is None:
            return None, 'straight lift is blocked'
        if self.plan_joint_path(lift[1], up_joints) is None:
            return None, 'cannot reach the up pose from the lift pose'

        # The planner call is the slow one, so it runs last
        self.set_contact_allowed(False)
        pre_trajectory = self.plan_joint_path(start, joints)
        if pre_trajectory is None:
            return None, 'no collision-free path to the pre-grasp'
        return {'goals': goals, 'pre_trajectory': pre_trajectory}, None

    def place_hand(self, grasp):
        """Finds where the open hand fits around the object for one hand pose, without moving.

        Returns (pose, free_width, retreat, offset) or None. `offset` is the distance of the
        closing centre from the object's centroid as a fraction of the object's half-diagonal
        (0 = on the centroid), used to rank placements.
        """
        self.set_contact_allowed(False)
        found = self.find_envelope(grasp)
        if found is None:
            return self.reject('no contact-free hand position around the object')
        pose, width, retreat = found
        centre = pose[:3, 3] + pose[:3, :3] @ self.hand_centre
        offset = float(np.linalg.norm(centre - self.object_centre)
                       / (0.5 * np.linalg.norm(self.object_size)))
        return pose, width, retreat, offset

    def plan_pick(self, grasp):
        """IK and motion checks for a placed hand pose, without moving. Returns a plan or None."""
        goals = self.targets(grasp)
        solutions = self.pre_solutions(goals['pre'])
        if not solutions:
            return self.reject('no collision-free IK solution for the pre-grasp')
        for joints in solutions:
            plan, reason = self.check_branch(joints, goals)
            if plan is not None:
                return plan
            self.reject(reason)
        return None