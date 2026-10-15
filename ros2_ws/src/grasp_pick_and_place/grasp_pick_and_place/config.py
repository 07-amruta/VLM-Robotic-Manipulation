"""Per-setup values: robot, gripper, scene, shape heuristics, search. Edit only this file."""
import os

HARDWARE_MODE = os.environ.get('HARDWARE_MODE', 'sim')    # 'sim' | 'real'
SCENE_SOURCE = os.environ.get('SCENE_SOURCE', 'camera')   # 'camera' | 'manual' (fixed object, no camera)
if HARDWARE_MODE not in ('sim', 'real'):
    raise ValueError(f"HARDWARE_MODE must be 'sim' or 'real', got {HARDWARE_MODE!r}")
if SCENE_SOURCE not in ('camera', 'manual'):
    raise ValueError(f"SCENE_SOURCE must be 'camera' or 'manual', got {SCENE_SOURCE!r}")

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
    'hand_centre_depth': 0.0 if HARDWARE_MODE == 'sim' else 0.10,  # m, tcp -> pad midpoint along approach; measured on real
}

# Table block per hardware. Real values (base_link frame, metres) must be measured.
SCENES = {
    'sim': {
        'support_centre': [0.0, 0.75],  # xy of the table block
        'support_size': [0.8, 0.6],     # xy extent of the table top
        'support_top': 0.30,            # z of the table top
        'object_margin': 0.004,
    },
    'real': {
        'support_centre': [-0.059, 1.086],
        'support_size': [0.67, 0.46],
        'support_top': 0.07,
        'object_margin': 0.004,
    },
}
SCENE = SCENES[HARDWARE_MODE]

# Fixed object for SCENE_SOURCE='manual' (no camera): box standing on the table, base_link frame.
MANUAL_SCENE = {
    'object_centre_xy': None,
    'object_size': [0.08, 0.08, 0.08],
    'object_yaw': 0.0,
}

_missing = [k for k, v in SCENE.items() if v is None]
if SCENE_SOURCE == 'manual':
    _missing += [k for k, v in MANUAL_SCENE.items() if v is None]
if _missing:
    raise ValueError(f'Set these in config.py before running: {_missing}')

# Workspace crop, derived from SCENE: the table footprint, from just above the table top
# (excludes the table surface) up to the tallest object to keep.
CROP_CLEARANCE = 0.01
CROP_HEIGHT = 0.5

_cx, _cy = SCENE['support_centre']
_sx, _sy = SCENE['support_size']
CROP = {
    'crop_min': [_cx - _sx / 2, _cy - _sy / 2, SCENE['support_top'] + CROP_CLEARANCE],
    'crop_max': [_cx + _sx / 2, _cy + _sy / 2, SCENE['support_top'] + CROP_HEIGHT],
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

GRIPPER_REAL = {
    'gripper_ip': '192.168.1.105',
    'gripper_port': 502,
    'gripper_speed': 50,
    'gripper_force': 6,           # 0-255; set from the force sweep
    'gripper_min_contacts': 3,    # fingers in closing contact that count as held
    'gripper_settle_timeout': 5.0,
    'gripper_open_tolerance': 10,
    'gripper_closed_tolerance':20,
}
PARAMS = {**ROBOT, **MOTION, **GRIPPER, **SCENE, **SHAPE, **SEARCH, **GRIPPER_REAL}

# Camera profiles. Optical frames are declared, never inferred from message headers
# (the sim depth header carries a non-optical frame). The depth range only guards against
# invalid-depth sentinel values; tighten it per camera from its datasheet.
CAMERAS = {
    'sim': {
        'depth_topic': '/scene_camera/depth_image',
        'camera_info_topic': '/scene_camera/camera_info',
        'optical_frame': 'scene_camera_optical',
        'depth_min': 0.0,
        'depth_max': 100.0,
        **CROP,
    },
    'd415': {
        'depth_topic': '/camera/camera/depth/image_rect_raw',
        'camera_info_topic': '/camera/camera/depth/camera_info',
        'optical_frame': 'camera_depth_optical_frame',
        'depth_min': 0.0,
        'depth_max': 2.0,
        'support_z': 0.07,  # measured table top height
        **CROP,
    },
}
ACTIVE_CAMERA = os.environ.get('ACTIVE_CAMERA', 'd415' if HARDWARE_MODE == 'real' else 'sim')
CAMERA = CAMERAS[ACTIVE_CAMERA]

SAFETY = {
    'dry_run': os.environ.get('DRY_RUN', '0') == '1',  # plan only, no arm motion (gripper still opens)
    'confirm_execution': True,                         # operator types y before any arm motion
    'confirm_each_stage': HARDWARE_MODE == 'real',     # operator types y before every stage
}