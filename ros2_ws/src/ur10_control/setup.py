from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'ur10_control'

setup(
    name=package_name,
    version='0.1.0',

    packages=find_packages(exclude=['test']),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*')
        ),
    ],

    install_requires=['setuptools'],

    zip_safe=True,

    maintainer='scg',
    maintainer_email='amruta.panda.mit@gmail.com',

    description='ROS 2 package for UR10 control using MoveIt 2.',

    license='Apache-2.0',

    extras_require={
        'test': [
            'pytest',
        ],
    },

    entry_points={
        'console_scripts': [
            'joint_control = ur10_control.joint_control:main',
            'cartesian_control = ur10_control.cartesian_control:main',
            'joint_cartesian_control = ur10_control.joint_cartesian_control:main',
            'cartesian_trajectory_control = ur10_control.cartesian_trajectory_control:main',
            'gripper_test = ur10_control.gripper_test:main',
            'ur_gripper_sequence = ur10_control.ur_gripper_sequence:main',
            'camera_align = ur10_control.camera_align:main',
            'arm_check = ur10_control.arm_check:main',
        ],
    },
)