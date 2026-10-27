import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def load_yaml(package, path):
    p = os.path.join(get_package_share_directory(package), path)
    with open(p) as f:
        return yaml.safe_load(f)


def generate_launch_description():
    moveit_config_pkg = "ur10_gripper_moveit_config"
    gripper_pkg = get_package_share_directory("robotiq_3f_gripper_description")

    robot_description = ParameterValue(
        Command([
            FindExecutable(name="xacro"), " ",
            os.path.join(gripper_pkg, "urdf", "ur10_robotiq_3f.urdf.xacro"),
            " ur_type:=ur10 name:=ur10 sim_ignition:=false",
        ]),
        value_type=str,
    )

    with open(os.path.join(get_package_share_directory(moveit_config_pkg), "config", "ur10_gripper.srdf")) as f:
        robot_description_semantic = f.read()

    kinematics = load_yaml(moveit_config_pkg, "config/kinematics.yaml")
    joint_limits = load_yaml(moveit_config_pkg, "config/joint_limits.yaml")
    moveit_controllers = load_yaml(moveit_config_pkg, "config/real/moveit_controllers.yaml")
    ompl_planning = load_yaml(moveit_config_pkg, "config/ompl_planning.yaml")

    # The scaled controller stretches execution when the pendant speed slider is below 100%,
    # so the allowed duration must cover 1 / slider.
    trajectory_execution = {
        "moveit_manage_controllers": True,
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "trajectory_execution.allowed_execution_duration_scaling": 10.0,
        "trajectory_execution.allowed_goal_duration_margin": 5.0,
        "trajectory_execution.allowed_start_tolerance": 0.01,
    }

    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    real_time = {"use_sim_time": False}

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}, real_time],
    )

    # The gripper has no joint-state feedback: publish its fingers at the open pose
    gripper_state_publisher = Node(
        package="grasp_pick_and_place",
        executable="gripper_state_publisher",
        output="screen",
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            {"robot_description": robot_description},
            {"robot_description_semantic": robot_description_semantic},
            {"robot_description_kinematics": kinematics},
            {"robot_description_planning": joint_limits},
            moveit_controllers,
            trajectory_execution,
            planning_scene_monitor_parameters,
            {"move_group": ompl_planning},
            real_time,
            {"publish_robot_description_semantic": True},
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", os.path.join(get_package_share_directory(moveit_config_pkg), "rviz", "moveit_real.rviz")],
        parameters=[
            {"robot_description": robot_description},
            {"robot_description_semantic": robot_description_semantic},
            {"robot_description_kinematics": kinematics},
            planning_scene_monitor_parameters,
            real_time,
        ],
    )

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")),
        launch_arguments={
            "depth_module.depth_profile": "848x480x30",
            "rgb_camera.color_profile": "848x480x30",
        }.items(),
    )

    camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0.56772", "1.43292", "0.94878", "-0.439983", "-0.126562", "0.850905",
                   "-0.257601", "base_link", "camera_link"],
    )

    return LaunchDescription([
        robot_state_publisher, gripper_state_publisher, move_group_node, rviz_node, camera, camera_tf])