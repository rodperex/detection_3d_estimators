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

"""detection_3d_estimator: 2D detections placed in 3D.

    /detections_2d (vision_msgs/Detection2DArray)  ->  /detections_3d (vision_msgs/Detection3DArray)

Each detection is placed on the ray of the center of its box, at the distance that the
estimator gives (parameter `estimator`: depth_image with a depth camera; ray, box_size or
depth_anything with a single image; see estimators.py),
in the frame of the image (the header of the detections), with the same id, class and score.
It also publishes a sphere per detection for RViz (detection_centroid_markers).

Any detector that fills a Detection2DArray works (an open-vocabulary detector, a YOLO...): the box
in pixels of the camera image (center and size), the class in results[0].hypothesis.class_id, and
optionally a tracker id in id. A class given as a number (e.g. "0", the index of the detector's
classes) is named with class_names, so that box_size finds its size; detections whose header has
no frame_id are placed in camera_frame_id.

The camera calibration comes from a CameraInfo topic (camera_info_topic), or from the
parameter camera_intrinsics ([fx, fy, cx, cy]) when the CameraInfo does not arrive.
See the README for every parameter.
"""

import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2DArray, Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from detection_3d_estimators.estimators import (BoxSizeEstimator, DepthAnythingEstimator,
                                                DepthImageEstimator, RayEstimator)
from detection_3d_estimators.geometry import Box, Intrinsics, point_at, to_body
from detection_3d_estimators.object_sizes import ObjectSizes

ESTIMATORS = ('depth_image', 'ray', 'box_size', 'depth_anything')


def depth_to_metres(msg):
    """A depth image (16UC1 in millimetres, 32FC1 in metres) as an H x W array of metres (0 or
    not finite: no measure)."""
    if msg.encoding == '16UC1':
        dtype, scale = np.dtype('<u2' if not msg.is_bigendian else '>u2'), 0.001
    elif msg.encoding == '32FC1':
        dtype, scale = np.dtype('<f4' if not msg.is_bigendian else '>f4'), 1.0
    else:
        raise ValueError(f'Unsupported depth encoding {msg.encoding} (16UC1 or 32FC1)')
    data = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    values = data[:, :msg.width * dtype.itemsize].copy().view(dtype).reshape(msg.height, msg.width)
    return values.astype(np.float64) * scale


def image_to_rgb(msg):
    """A sensor_msgs/Image (rgb8, bgr8, rgba8, bgra8) as an H x W x 3 RGB array."""
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(msg.encoding)
    if channels is None:
        raise ValueError(f'Unsupported image encoding {msg.encoding}')
    data = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    image = data[:, :msg.width * channels].reshape(msg.height, msg.width, channels)[:, :, :3]
    return image[:, :, ::-1].copy() if msg.encoding.startswith('bgr') else image.copy()


class Detection3DEstimator(Node):

    def __init__(self):
        super().__init__('detection_3d_estimator')
        p = self.declare_parameters('', [
            ('estimator', 'ray'),
            ('detections_2d_topic', '/detections_2d'),
            ('detections_3d_topic', '/detections_3d'),
            ('camera_info_topic', '/camera_info'),
            ('image_topic', '/image_rgb'),             # depth_anything only
            ('depth_image_topic', '/depth_image'),     # depth_image only
            # depth_image and depth_anything: the part of the box and the percentile of its depths
            ('depth_window_px', 5),
            ('camera_intrinsics', [0.0]),              # [fx, fy, cx, cy]: instead of CameraInfo
            ('camera_frame_is_optical', True),
            # The frame of the detections whose header has none (empty: left empty)
            ('camera_frame_id', ''),
            # The names of the classes of a detector that gives them as numbers (class_id "0" is
            # class_names[0], e.g. the 80 of COCO for a YOLO): the box_size table is by name
            ('class_names', ['']),
            # ray
            ('ray_distance_m', 1.0),
            # box_size
            ('object_sizes_file', ''),
            ('sizes', ['']),                           # 'name:height[:width]', overriding the file
            ('classes', ['']),                         # only these names of the table (empty: all)
            ('cut_margin', 0.025),
            # box_size and depth_anything: the distance of what has no estimate
            ('unknown_distance_m', 1.0),
            # depth_anything
            ('depth_anything_model', 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf'),
            ('device', 'cuda'),
            ('depth_region', 0.6),
            ('depth_percentile', 10.0),
            # RViz
            ('centroid_marker_topic', 'detection_centroid_markers'),
            ('centroid_marker_namespace', 'detection_centroids'),
            ('centroid_marker_scale', 0.05),
            ('centroid_marker_color', [0.0, 1.0, 0.0, 1.0]),
        ])
        self.p = {param.name: param.value for param in p}
        self.estimator = self.make_estimator()
        self.optical = self.p['camera_frame_is_optical']
        self.intrinsics = None
        self.last_image = None
        self.last_depth = None
        self.previous_markers = 0

        self.det3d_pub = self.create_publisher(Detection3DArray, self.p['detections_3d_topic'], 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.p['centroid_marker_topic'], 10)
        if self.estimator.needs_image:
            self.create_subscription(Image, self.p['image_topic'], self.on_image,
                                     qos_profile_sensor_data)
        if self.estimator.needs_depth:
            self.create_subscription(Image, self.p['depth_image_topic'], self.on_depth,
                                     qos_profile_sensor_data)
        intrinsics = [float(v) for v in self.p['camera_intrinsics']]
        if len(intrinsics) == 4 and intrinsics[0] > 0.0:
            self.set_intrinsics(Intrinsics(*intrinsics), 'parameters')
        else:
            self.camera_info_sub = self.create_subscription(
                CameraInfo, self.p['camera_info_topic'], self.on_camera_info,
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.VOLATILE))
            self.get_logger().info(f'Waiting for the camera calibration on {self.p["camera_info_topic"]}')

    def make_estimator(self):
        p = self.p
        name = p['estimator']
        if name == 'depth_image':
            estimator = DepthImageEstimator(p['depth_window_px'], p['depth_region'],
                                            p['depth_percentile'])
            detail = (f'{p["depth_image_topic"]}, percentile {p["depth_percentile"]:g} of the '
                      f'central {p["depth_region"]:g} of each box')
        elif name == 'ray':
            estimator = RayEstimator(p['ray_distance_m'])
            detail = f'every detection at {p["ray_distance_m"]} m'
        elif name == 'box_size':
            path = p['object_sizes_file'] or os.path.join(
                get_package_share_directory('detection_3d_estimators'), 'config', 'object_sizes.yaml')
            sizes = ObjectSizes.from_file(path)
            sizes.override([s for s in p['sizes'] if s])
            sizes.restrict([c for c in p['classes'] if c])
            estimator = BoxSizeEstimator(sizes, p['unknown_distance_m'], p['cut_margin'])
            detail = (f'{len(sizes.sizes)} known sizes ({path}); the rest at '
                      f'{p["unknown_distance_m"]} m')
        elif name == 'depth_anything':
            estimator = DepthAnythingEstimator(p['depth_anything_model'], p['device'],
                                               p['depth_region'], p['depth_percentile'],
                                               p['unknown_distance_m'])
            detail = f'{p["depth_anything_model"]} on {estimator.device}'
        else:
            raise ValueError(f'Unknown estimator "{name}": one of {", ".join(ESTIMATORS)}')
        self.get_logger().info(f'Estimator {name}: {detail}')
        return estimator

    def set_intrinsics(self, intrinsics, source):
        self.intrinsics = intrinsics
        self.get_logger().info(
            f'Camera calibration ({source}): fx={intrinsics.fx:.2f} fy={intrinsics.fy:.2f} '
            f'cx={intrinsics.cx:.2f} cy={intrinsics.cy:.2f}')
        self.create_subscription(Detection2DArray, self.p['detections_2d_topic'],
                                 self.on_detections, 10)

    def on_camera_info(self, msg):
        if self.intrinsics is None:
            self.destroy_subscription(self.camera_info_sub)
            self.set_intrinsics(Intrinsics(msg.k[0], msg.k[4], msg.k[2], msg.k[5],
                                           msg.width, msg.height), 'CameraInfo')

    def on_image(self, msg):
        self.last_image = msg

    def on_depth(self, msg):
        self.last_depth = msg

    def on_detections(self, msg):
        detections = [d for d in msg.detections if d.results]
        boxes = [Box(d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.size_x,
                     d.bbox.size_y) for d in detections]
        labels = [self.class_name(d.results[0].hypothesis.class_id) for d in detections]
        if not msg.header.frame_id and self.p['camera_frame_id']:
            msg.header.frame_id = self.p['camera_frame_id']
        image = None
        if self.estimator.needs_image and detections:
            if self.last_image is None:
                self.get_logger().warn('Detections, but no image yet', throttle_duration_sec=2.0)
                return
            image = image_to_rgb(self.last_image)
        depth = None
        if self.estimator.needs_depth and detections:
            if self.last_depth is None:
                self.get_logger().warn('Detections, but no depth image yet', throttle_duration_sec=2.0)
            else:
                depth = depth_to_metres(self.last_depth)
        estimates = self.estimator.distances(self.intrinsics, boxes, labels, image, depth) \
            if boxes else []

        out = Detection3DArray(header=msg.header)
        markers = MarkerArray()
        for det2d, box, (distance, _) in zip(detections, boxes, estimates):
            if distance is None:   # no estimate (no depth there): left out
                continue
            optical = point_at(self.intrinsics, box.u, box.v, distance)
            x, y, z = optical if self.optical else to_body(optical)
            det3d = Detection3D(header=msg.header, id=det2d.id)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = det2d.results[0].hypothesis.class_id
            hypothesis.hypothesis.score = det2d.results[0].hypothesis.score
            hypothesis.pose.pose.position.x = x
            hypothesis.pose.pose.position.y = y
            hypothesis.pose.pose.position.z = z
            det3d.results.append(hypothesis)
            det3d.bbox.center.position.x = x
            det3d.bbox.center.position.y = y
            det3d.bbox.center.position.z = z
            # The size of the box at the depth of its center (along the optical axis)
            det3d.bbox.size.x = box.width / self.intrinsics.fx * optical[2]
            det3d.bbox.size.y = box.height / self.intrinsics.fy * optical[2]
            det3d.bbox.size.z = 0.01
            out.detections.append(det3d)
            markers.markers.append(self.marker(msg.header, len(out.detections) - 1, x, y, z))
        for old in range(len(out.detections), self.previous_markers):
            markers.markers.append(Marker(header=msg.header, ns=self.p['centroid_marker_namespace'],
                                          id=old, action=Marker.DELETE))
        self.previous_markers = len(out.detections)
        self.det3d_pub.publish(out)
        self.marker_pub.publish(markers)

    def class_name(self, class_id):
        """The name of a class: class_names[i] for a class given as the number i."""
        names = [n for n in self.p['class_names'] if n]
        if names and class_id.strip().isdigit() and int(class_id) < len(names):
            return names[int(class_id)]
        return class_id

    def marker(self, header, marker_id, x, y, z):
        marker = Marker(header=header, ns=self.p['centroid_marker_namespace'], id=marker_id,
                        type=Marker.SPHERE, action=Marker.ADD)
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = x, y, z
        marker.pose.orientation.w = 1.0
        scale = self.p['centroid_marker_scale']
        marker.scale.x = marker.scale.y = marker.scale.z = scale
        r, g, b, a = (list(self.p['centroid_marker_color']) + [0.0, 1.0, 0.0, 1.0])[:4]
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = r, g, b, a
        return marker


def main():
    rclpy.init()
    node = Detection3DEstimator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
