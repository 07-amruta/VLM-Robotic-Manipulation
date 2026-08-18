import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node


def generate_launch_description():

    # ---------------------------------------------------------
    # Locate required packages
    # ---------------------------------------------------------

    ur_description = get_package_share_directory("ur_description")
    gazebo_ros = get_package_share_directory("gazebo_ros")
    ur10_control = get_package_share_directory("ur10_control")

    # ---------------------------------------------------------
    # Gazebo controller configuration
    # ---------------------------------------------------------

    controllers_file = os.path.join(
        ur10_control,
        "config",
        "gazebo_controllers.yaml",
    )

    # ---------------------------------------------------------
    # Generate UR10 robot description
    # ---------------------------------------------------------

    robot_description = Command([
        FindExecutable(name="xacro"),
        " ",
        os.path.join(
            ur_description,
            "urdf",
            "ur.urdf.xacro",
        ),
        " ",
        "name:=ur10",
        " ",
        "ur_type:=ur10",
        " ",
        "sim_gazebo:=true",
        " ",
        "use_fake_hardware:=false",
        " ",
        "simulation_controllers:=",
        controllers_file,
    ])

    # ---------------------------------------------------------
    # Start Gazebo Classic
    # ---------------------------------------------------------

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                gazebo_ros,
                "launch",
                "gazebo.launch.py",
            )
        )
    )

    # ---------------------------------------------------------
    # Robot State Publisher
    # ---------------------------------------------------------

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": True,
            }
        ],
    )

    # ---------------------------------------------------------
    # Spawn UR10 into Gazebo
    # ---------------------------------------------------------

    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic",
            "robot_description",
            "-entity",
            "ur10",
        ],
        output="screen",
    )

    # ---------------------------------------------------------
    # Joint State Broadcaster
    # ---------------------------------------------------------

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    # ---------------------------------------------------------
    # Joint Trajectory Controller
    # ---------------------------------------------------------

    trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_trajectory_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    # ---------------------------------------------------------
    # Launch everything
    # ---------------------------------------------------------

    return LaunchDescription([
        gazebo,

        robot_state_publisher,

        spawn_robot,

        # Give Gazebo time to load gazebo_ros2_control
        # before starting the controllers.
        TimerAction(
            period=5.0,
            actions=[
                joint_state_broadcaster_spawner,
                trajectory_controller_spawner,
            ],
        ),
    ])