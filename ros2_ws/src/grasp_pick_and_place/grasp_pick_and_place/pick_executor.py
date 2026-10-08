from grasp_pick_and_place.gripper import GripperMixin
from grasp_pick_and_place.grasp_planner import GraspPlannerMixin
from grasp_pick_and_place.motion import MotionMixin
from grasp_pick_and_place.robot_io import RobotIO
from grasp_pick_and_place.scene_model import SceneMixin


class PickExecutor(RobotIO, MotionMixin, GripperMixin, SceneMixin, GraspPlannerMixin):
    """Fixed machinery for one pick. The sequence lives in run_pick.py."""