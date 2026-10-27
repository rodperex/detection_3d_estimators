import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'detection_3d_estimators'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Rodrigo Pérez-Rodríguez',
    maintainer_email='rodrigo.perez@urjc.es',
    description='2D detections placed in 3D, with the distance of a depth image or estimated from a single image',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection_3d_estimator = detection_3d_estimators.node:main',
        ],
    },
)
