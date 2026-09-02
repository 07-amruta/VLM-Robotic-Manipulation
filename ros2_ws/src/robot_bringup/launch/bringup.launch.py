from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('ur10_control'),
            '/launch/gazebo_simulation.launch.py'
        ])
    )

    move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('ur10_gripper_moveit_config'),
            '/launch/move_group.launch.py'
        ])
    )

    return LaunchDescription([
        gazebo_launch,
        move_group_launch,
    ])