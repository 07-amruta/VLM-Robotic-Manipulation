import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable,
    TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

OBJECT_PREFIX = "test_"


def world_actions(context):
    """Gazebo with only the selected test object, and the scene camera TF read from the world."""
    ros_gz_sim = get_package_share_directory("ros_gz_sim")
    src = os.path.join(get_package_share_directory("robot_bringup"), "worlds", "pick_and_place.sdf")
    name = OBJECT_PREFIX + LaunchConfiguration("object").perform(context)

    tree = ET.parse(src)
    world = tree.getroot().find("world")
    objects = [m for m in world.findall("model") if m.get("name").startswith(OBJECT_PREFIX)]
    if name not in [m.get("name") for m in objects]:
        raise RuntimeError(
            f"Unknown object; available: {[m.get('name')[len(OBJECT_PREFIX):] for m in objects]}")
    for m in objects:
        if m.get("name") != name:
            world.remove(m)
    x, y, z, roll, pitch, yaw = world.find("model[@name='scene_camera']/pose").text.split()

    world_file = f"/tmp/pick_and_place_{name}.sdf"
    tree.write(world_file, encoding="utf-8", xml_declaration=True)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": world_file + " -r"}.items(),
    )
    scene_camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["--x", x, "--y", y, "--z", z, "--roll", roll, "--pitch", pitch, "--yaw", yaw,
                   "--frame-id", "base_link", "--child-frame-id", "scene_camera_link"],
    )
    return [gazebo, scene_camera_tf]


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
    robot_bringup = get_package_share_directory("robot_bringup")

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
        robot_bringup,
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

    robot_description = ParameterValue(
        Command([
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
        ]),
        value_type=str,
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
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock"
        ],
        output="screen",
    )

    camera_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/scene_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image",
            "/scene_camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image",
            "/scene_camera/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked",
            "/scene_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
        ],
        output="screen",
    )

    scene_camera_optical_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "-1.5708", "--pitch", "0", "--yaw", "-1.5708",
            "--frame-id", "scene_camera_link",
            "--child-frame-id", "scene_camera_optical",
        ],
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
        package="ros_gz_sim",
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

        DeclareLaunchArgument(
            "object", default_value="mug",
            description="cylinder|cube|tall_box|small_box|wide_box|sphere|mug"),
        OpaqueFunction(function=world_actions),
        clock_bridge,
        camera_bridge,
        robot_state_publisher,
        spawn_robot,
        scene_camera_optical_tf,

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