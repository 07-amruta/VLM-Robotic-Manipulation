from setuptools import find_packages, setup

package_name = 'grasp_pick_and_place'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='amruta.panda.mit@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'scene_snapshot = grasp_pick_and_place.scene_snapshot:main',
            'grasp_proposer = grasp_pick_and_place.grasp_proposer:main',
            'run_pick = grasp_pick_and_place.run_pick:main',
            'manual_scene = grasp_pick_and_place.manual_scene:main',
            'gripper_state_publisher = grasp_pick_and_place.gripper_state_publisher:main',
        ],
    },
)