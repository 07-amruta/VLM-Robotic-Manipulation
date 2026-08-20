import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node


def generate_launch_description():

    # ---------------------------------------------------------
    # Locate required packages
    #
    # NOTE: `gz sim --version` failed but `ign gazebo --version`
    # reported "Gazebo Sim, version 6.17.0" -> this is Gazebo
    # FORTRESS, which uses the `ign_` prefixed packages
    # (ros_ign_gazebo / ros_ign_bridge / ign_ros2_control), NOT
    # the newer `gz_` prefixed ones (those start at Garden/Harmonic).
    # ---------------------------------------------------------

    robotiq_gripper_description = get_package_share_directory("robotiq_3f_gripper_description")
    ros_ign_gazebo = get_package_share_directory("ros_ign_gazebo")
    ur10_control = get_package_share_directory("ur10_control")

    # Ignition resolves package:// mesh URIs via GZ_SIM_RESOURCE_PATH /
    # IGN_GAZEBO_RESOURCE_PATH, not the ROS package path -- RViz doesn't
    # need this, but Ignition does, or meshes fail to load (SystemPaths
    # "Unable to find file" errors) even though ros2_control works fine.
    gripper_resource_path = os.path.join(robotiq_gripper_description, "..")

    # ---------------------------------------------------------
    # Gazebo controller configuration (unchanged - controller
    # config is simulator-agnostic, it just describes ros2_control
    # controllers, not which physics backend loads them)
    # ---------------------------------------------------------

    controllers_file = os.path.join(
        ur10_control,
        "config",
        "gazebo_controllers.yaml",
    )

    # ---------------------------------------------------------
    # Generate UR10 robot description
    #
    # KEY FIX: sim_ignition:=true (not sim_gazebo:=true).
    # ur_description's ur.ros2_control.xacro checks sim_ignition
    # specifically to decide whether to load the
    # ign_ros2_control/IgnitionSystem hardware interface plugin.
    # Passing sim_gazebo:=true instead loads the Gazebo CLASSIC
    # plugin (gazebo_ros2_control/GazeboSystem), which won't work
    # against Ignition/Gazebo Sim at all.
    # ---------------------------------------------------------

    robot_description = Command([
        FindExecutable(name="xacro"),
        " ",
        os.path.join(
            robotiq_gripper_description,
            "urdf",
            "ur10_robotiq_3f.urdf.xacro",
        ),
        " ",
        "name:=ur10",
        " ",
        "ur_type:=ur10",
        " ",
        "sim_ignition:=true",
        " ",
        "simulation_controllers:=",
        controllers_file,
    ])

    # ---------------------------------------------------------
    # Start Gazebo Fortress (Ignition Gazebo)
    #
    # Package/launch-file/arg names for Fortress:
    #   ros_ign_gazebo / ign_gazebo.launch.py / ign_args
    # (For Garden/Harmonic this would instead be
    #   ros_gz_sim / gz_sim.launch.py / gz_args)
    # ---------------------------------------------------------

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                ros_ign_gazebo,
                "launch",
                "ign_gazebo.launch.py",
            )
        ),
        launch_arguments={"ign_args": "-r empty.sdf"}.items(),
    )

    # ---------------------------------------------------------
    # Clock bridge
    #
    # Ignition publishes sim time over Ignition Transport, not
    # ROS 2 topics. Without this bridge, use_sim_time:=True nodes
    # (robot_state_publisher, controller_manager, etc.) never
    # receive a /clock and effectively stall.
    # ---------------------------------------------------------

    clock_bridge = Node(
        package="ros_ign_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock"
        ],
        output="screen",
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
    # Spawn UR10 into Gazebo Fortress
    #
    # Fortress uses `ros_ign_gazebo create`, NOT
    # `gazebo_ros spawn_entity.py` (Classic) or
    # `ros_gz_sim create` (Garden/Harmonic).
    # ---------------------------------------------------------

    spawn_robot = Node(
        package="ros_ign_gazebo",
        executable="create",
        arguments=[
            "-topic",
            "robot_description",
            "-name",
            "ur10",
            "-z",
            "0.0",
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

    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "gripper_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    # ---------------------------------------------------------
    # Launch everything
    # ---------------------------------------------------------

    return LaunchDescription([

        SetEnvironmentVariable(
            name="IGN_GAZEBO_RESOURCE_PATH",
            value=os.path.join(robotiq_gripper_description, ".."),
        ),
        SetEnvironmentVariable(
            name="GZ_SIM_RESOURCE_PATH",
            value=os.path.join(robotiq_gripper_description, ".."),
        ),

        gazebo,

        clock_bridge,

        robot_state_publisher,

        spawn_robot,

        # Give Ignition time to load ign_ros2_control and start
        # /controller_manager before spawning controllers into it.
        TimerAction(
            period=5.0,
            actions=[
                joint_state_broadcaster_spawner,
                trajectory_controller_spawner,
                gripper_controller_spawner,
            ],
        ),
    ])