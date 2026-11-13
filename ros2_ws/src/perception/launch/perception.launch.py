from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PRODUCERS = {'geometric': 'geometric_segmenter', 'yolo': 'instance_segmenter', 'target': 'open_vocab_segmenter'}
THREAD_LIMITS = {v: '1' for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}


def nodes(context):
    mode = LaunchConfiguration('segmenter').perform(context)
    obstacles = LaunchConfiguration('obstacles').perform(context) == 'true'
    producer_env = THREAD_LIMITS if mode == 'geometric' else None
    result = [
        Node(package='perception', executable=PRODUCERS[mode], output='screen',
             additional_env=producer_env),
        Node(package='perception', executable='scene_objects', output='screen',
             parameters=[{'masks_topic': f'/instance_masks/{mode}'}],
             additional_env=THREAD_LIMITS),
    ]
    if obstacles and mode != 'geometric':
        result += [
            Node(package='perception', executable='geometric_segmenter', output='screen',
                 additional_env=THREAD_LIMITS),
            Node(package='perception', executable='scene_objects', name='scene_obstacles',
                 output='screen',
                 parameters=[{'masks_topic': '/instance_masks/geometric'}],
                 remappings=[('scene_objects', 'scene_obstacle_clusters')],
                 additional_env=THREAD_LIMITS),
        ]
    return result


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('segmenter', default_value='geometric',
                              description='geometric|yolo|target'),
        DeclareLaunchArgument('obstacles', default_value='false',
                              description='true|false: also publish geometric clusters on /scene_obstacle_clusters'),
        OpaqueFunction(function=nodes),
    ])