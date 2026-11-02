import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Amruta',
    maintainer_email='amruta.panda.mit@gmail.com',
    description='Camera-agnostic perception: instance masks to per-object clouds',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'geometric_segmenter = perception.geometric_segmenter:main',
            'instance_segmenter = perception.instance_segmenter:main',
            'scene_objects = perception.scene_objects:main',
        ],
    },
)