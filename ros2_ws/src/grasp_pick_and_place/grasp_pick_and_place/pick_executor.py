from grasp_pick_and_place.config import HARDWARE_MODE
from grasp_pick_and_place.grasp_planner import GraspPlannerMixin
from grasp_pick_and_place.gripper import GripperMixin
from grasp_pick_and_place.motion import MotionMixin
from grasp_pick_and_place.robot_io import RobotIO
from grasp_pick_and_place.scene_model import SceneMixin

if HARDWARE_MODE == 'real':
    from grasp_pick_and_place.gripper_real import RealGripperMixin as Gripper
else:
    Gripper = GripperMixin


class PickExecutor(RobotIO, MotionMixin, Gripper, SceneMixin, GraspPlannerMixin):
    """Fixed machinery for one pick. The sequence lives in run_pick.py."""