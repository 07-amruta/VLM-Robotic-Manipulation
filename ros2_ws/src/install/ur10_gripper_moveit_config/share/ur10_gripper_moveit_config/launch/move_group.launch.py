import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import yaml


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
            " ur_type:=ur10 name:=ur10 sim_ignition:=true",
        ]),
        value_type=str,
    )

    with open(os.path.join(get_package_share_directory(moveit_config_pkg), "config", "ur10_gripper.srdf")) as f:
        robot_description_semantic = f.read()

    kinematics = load_yaml(moveit_config_pkg, "config/kinematics.yaml")
    joint_limits = load_yaml(moveit_config_pkg, "config/joint_limits.yaml")
    moveit_controllers = load_yaml(moveit_config_pkg, "config/moveit_controllers.yaml")
    ompl_planning = load_yaml(moveit_config_pkg, "config/ompl_planning.yaml")

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
            ompl_planning,
            {"use_sim_time": True},
            {"publish_robot_description_semantic": True},
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", os.path.join(get_package_share_directory(moveit_config_pkg), "launch", "moveit.rviz")],
        parameters=[
            {"robot_description": robot_description},
            {"robot_description_semantic": robot_description_semantic},
            {"robot_description_kinematics": kinematics},
            {"use_sim_time": True},
        ],
    )

    return LaunchDescription([move_group_node, rviz_node])