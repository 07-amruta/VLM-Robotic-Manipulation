"""Per-setup values: robot, gripper, scene, shape heuristics, search. Edit only this file."""

ARM_JOINTS = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
GRIPPER_JOINTS = [
    'gripper_finger_1_joint_1', 'gripper_finger_2_joint_1', 'gripper_finger_middle_joint_1',
]
OBJECT_ID = 'target_object'
SUPPORT_ID = 'support_table'
MOVEIT_SUCCESS = 1

ROBOT = {
    'group_name': 'ur_manipulator',
    'ik_link': 'tool0',
    'tcp_frame': 'gripper_tool0',
    'up_joints': [0.0, -1.57, 0.0, -1.57, 0.0, 0.0],
    'velocity_scaling': 0.4,
    'planning_time': 3.0,
    'ik_seeds': 8,
    'ik_timeout': 0.1,
}

MOTION = {
    'approach_margin': 0.05,
    'lift_height': 0.15,
    'min_path_fraction': 0.99,
    'cartesian_step': 0.01,
    'cartesian_joint_speed': 0.3,
    'joint_line_step': 0.05,    # rad, max per checked state
    'joint_line_speed': 1.0,    # rad/s
    'max_joint_step': 0.3,
}

GRIPPER = {
    'gripper_link_prefix': 'gripper_',
    'gripper_links': '',        # comma-separated override
    'gripper_open': [0.05, 0.05, 0.05],
    'gripper_duration': 1.5,
    'close_step': 0.03,
    'close_period': 0.1,
    'close_limit': [1.2, 1.2, 1.2],
    'closure_step': 0.05,       # finger rad per checked closure
    'contact_lag': 0.08,
    'squeeze_margin': 0.01,
    'squeeze_margin_sphere': 0.03,
    'close_timeout': 15.0,
    'motion_eps': 0.01,
    'stall_count': 2,
    'hand_pair_frames': ['gripper_finger_1_link_3', 'gripper_finger_2_link_3'],
    'hand_opposing_frame': 'gripper_finger_middle_link_3',
    'hand_palm_frame': 'gripper_palm',
}

SCENE = {
    'support_centre': [0.0, 0.75],  # xy of the table block, scene frame
    'support_size': [0.8, 0.6],     # xy extent of the table top
    'support_top': 0.30,            # z of the table top, scene frame
    'object_margin': 0.004,
}

SHAPE = {
    'prismatic_fill': 0.9,
    'dome_slice': 0.15,             # top fraction of the cloud's height
    'dome_ratio': 0.8,
}

SEARCH = {
    'search_step': 0.005,
    'search_reach': 0.04,
    'retreat_step': 0.01,
    'min_free_width': 0.005,
}

PARAMS = {**ROBOT, **MOTION, **GRIPPER, **SCENE, **SHAPE, **SEARCH}