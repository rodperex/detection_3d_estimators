# Copyright 2026 Rodrigo Pérez-Rodríguez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The detection_3d_estimator node with the most common options as arguments; any other parameter
# can be given in a YAML file (params_file). See the README.
#
#   ros2 launch detection_3d_estimators detection_3d_estimator.launch.py estimator:=box_size \
#     camera_info_topic:=/camera/camera_info

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml

ARGS = {
  'estimator': ('ray', 'ray | box_size | depth_anything'),
  'detections_2d_topic': ('/detections_2d', 'The 2D detections (vision_msgs/Detection2DArray)'),
  'detections_3d_topic': ('/detections_3d', 'The 3D detections (vision_msgs/Detection3DArray)'),
  'camera_info_topic': ('/camera_info', 'The calibration of the camera (sensor_msgs/CameraInfo)'),
  'camera_intrinsics': ('', '"[fx, fy, cx, cy]" instead of the CameraInfo (empty: CameraInfo)'),
  'camera_frame_is_optical': ('true', 'The frame of the images is optical (z forward); false: '
                                      'x forward, y left, z up'),
  'camera_frame_id': ('', 'The frame of the detections whose header has none (empty: none)'),
  'depth_image_topic': ('/depth_image', 'depth_image: the depth image, aligned with the color '
                                        'image (16UC1 mm or 32FC1 m)'),
  'image_topic': ('/image_rgb', 'depth_anything: the color image'),
  'unknown_distance_m': ('1.0', 'box_size, depth_anything: the distance of what has no estimate'),
  'params_file': ('', 'A YAML file with more parameters of the node'),
  'use_sim_time': ('false', 'Use the simulation clock'),
}


def launch_setup(context):
  value = {name: LaunchConfiguration(name).perform(context) for name in ARGS}
  parameters = [{
    'estimator': value['estimator'],
    'detections_2d_topic': value['detections_2d_topic'],
    'detections_3d_topic': value['detections_3d_topic'],
    'camera_info_topic': value['camera_info_topic'],
    'camera_frame_is_optical': value['camera_frame_is_optical'].lower() in ('true', '1'),
    'camera_frame_id': value['camera_frame_id'],
    'depth_image_topic': value['depth_image_topic'],
    'image_topic': value['image_topic'],
    'unknown_distance_m': float(value['unknown_distance_m']),
    'use_sim_time': value['use_sim_time'].lower() in ('true', '1'),
  }]
  intrinsics = yaml.safe_load(value['camera_intrinsics'] or '[]')
  if intrinsics:
    parameters[0]['camera_intrinsics'] = [float(v) for v in intrinsics]
  if value['params_file']:
    parameters.append(value['params_file'])
  return [Node(package='detection_3d_estimators', executable='detection_3d_estimator',
               output='screen', parameters=parameters)]


def generate_launch_description():
  return LaunchDescription(
    [DeclareLaunchArgument(name, default_value=default, description=description)
     for name, (default, description) in ARGS.items()] + [OpaqueFunction(function=launch_setup)])
