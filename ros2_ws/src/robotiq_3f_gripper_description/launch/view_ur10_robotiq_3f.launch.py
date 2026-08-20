#!/usr/bin/env python3
"""
Quick sanity-check launch: loads ur10_robotiq_3f.urdf.xacro, publishes TF via
robot_state_publisher, lets you wiggle every joint (arm + gripper) with
joint_state_publisher_gui, and shows it all in RViz.

This does NOT spawn into Gazebo Ignition -- it's just for verifying the
combined URDF (tool0 attachment, meshes, joint limits) before wiring it into
your existing Gazebo spawn launch file.

Usage:
    ros2 launch robotiq_3f_gripper_description view_ur10_robotiq_3f.launch.py
    ros2 launch robotiq_3f_gripper_description view_ur10_robotiq_3f.launch.py ur_type:=ur10
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    ur_type = LaunchConfiguration("ur_type")
    name = LaunchConfiguration("name")

    declare_ur_type = DeclareLaunchArgument(
        "ur_type", default_value="ur10", description="UR arm type"
    )
    # ur_description's ur_macro.xacro declares "name" via xacro:arg with a
    # default, but UR's own launch files always pass it explicitly on the
    # xacro command line (see ur_robot_driver/launch/ur_rsp.launch.py) --
    # without it xacro raises "Undefined substitution argument name".
    declare_name = DeclareLaunchArgument(
        "name", default_value="ur10", description="robot name / xacro robot tag name"
    )

    xacro_path = PathJoinSubstitution(
        [
            get_package_share_directory("robotiq_3f_gripper_description"),
            "urdf",
            "ur10_robotiq_3f.urdf.xacro",
        ]
    )

    robot_description_content = ParameterValue(
        Command(["xacro ", xacro_path, " ur_type:=", ur_type, " name:=", name]),
        value_type=str,
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description_content}],
    )

    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        output="screen",
    )

    rviz_config = os.path.join(
        get_package_share_directory("robotiq_3f_gripper_description"),
        "rviz",
        "view_ur10_robotiq_3f.rviz",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
    )

    return LaunchDescription(
        [
            declare_ur_type,
            declare_name,
            robot_state_publisher_node,
            joint_state_publisher_gui_node,
            rviz_node,
        ]
    )
