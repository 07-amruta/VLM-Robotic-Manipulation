"""Camera profiles and perception parameters. No table terms: the scene is bounded by arm reach only."""
import os

_real = os.environ.get('HARDWARE_MODE', 'sim') == 'real'
ACTIVE_CAMERA = os.environ.get('ACTIVE_CAMERA', 'd415' if _real else 'sim')

# Optical frames are declared, never inferred from message headers (sim depth header is non-optical).
# depth_min/max only guard against invalid-depth sentinels.
CAMERAS = {
    'sim': {
        'depth_topic': '/scene_camera/depth_image',
        'camera_info_topic': '/scene_camera/camera_info',
        'optical_frame': 'scene_camera_optical',
        'color_topic': '/scene_camera/image',
        'color_info_topic': '/scene_camera/camera_info',
        'color_frame': 'scene_camera_optical',
        'depth_min': 0.0,
        'depth_max': 100.0,
    },
    'd415': {
        'depth_topic': '/camera/camera/depth/image_rect_raw',
        'camera_info_topic': '/camera/camera/depth/camera_info',
        'optical_frame': 'camera_depth_optical_frame',
        'color_topic': '/camera/camera/color/image_raw',          # verify with camera up
        'color_info_topic': '/camera/camera/color/camera_info',   # verify
        'color_frame': 'camera_color_optical_frame',              # verify
        'depth_min': 0.0,
        'depth_max': 2.0,
    },
}
CAMERA = CAMERAS[ACTIVE_CAMERA]

PERCEPTION = {
    'target_frame': 'base_link',
    'masks_topic': '/instance_masks/geometric',
    # Arm reach box in target_frame (m); floor inside, no table term.
    'reach_min': [-1.3, -1.3, -0.3],
    'reach_max': [1.3, 1.3, 1.2],
    'edge_tol': 0.01,
    'self_filter_margin': 0.02,
    # geometric producer
    'rate_hz': 2.0,
    'plane_tol': 0.008,
    'plane_margin': 0.015,
    'plane_iters': 200,
    'plane_min_area': 0.1,        # m^2 of plane inliers; must exceed any object's top face
    'max_planes': 4,
    'max_tilt_deg': 10.0,
    'cluster_radius': 0.015,
    'cluster_voxel': 0.005,
    'min_cluster_points': 150,
    'max_instances': 20,
    # yolo producer
    'yolo_model': os.environ.get('YOLO_MODEL', '/workspaces/VLM-Robotic-Manipulation/models/yolo11n-seg.pt'),
    'yolo_tracker': 'bytetrack.yaml',
    'yolo_conf': 0.25,
    'yolo_device': 'cuda:0',
    'yolo_quantize': 16,
    'mask_erode_px': 2,
    # consumer
    'objects_rate_hz': 5.0,
    'outlier_k': 10,
    'outlier_std': 2.0,
    'min_object_points': 100,
}