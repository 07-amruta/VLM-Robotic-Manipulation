from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PRODUCERS = {'geometric': 'geometric_segmenter', 'yolo': 'instance_segmenter'}


def nodes(context):
    mode = LaunchConfiguration('segmenter').perform(context)
    return [
        Node(package='perception', executable=PRODUCERS[mode], output='screen'),
        Node(package='perception', executable='scene_objects', output='screen',
             parameters=[{'masks_topic': f'/instance_masks/{mode}'}]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('segmenter', default_value='geometric', description='geometric|yolo'),
        OpaqueFunction(function=nodes),
    ])