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

"""The node with the detections of a generic detector, as a YOLO publishes them in a
vision_msgs/Detection2DArray: the class by name or by number (COCO index), no tracker id, and
a header with or without frame_id. Needs ROS (rclpy); the node runs in this process."""

import math
import os
import time

import pytest

rclpy = pytest.importorskip('rclpy')
from rclpy.parameter import Parameter  # noqa: E402
from sensor_msgs.msg import CameraInfo  # noqa: E402
from vision_msgs.msg import (Detection2D, Detection2DArray, Detection3DArray,  # noqa: E402
                             ObjectHypothesisWithPose)

from detection_3d_estimators.node import Detection3DEstimator  # noqa: E402

FX = FY = 500.0
CX, CY, W, H = 320.0, 240.0, 640, 480
COCO = os.path.join(os.path.dirname(__file__), '..', 'config', 'class_names_coco.yaml')


def yolo_detection(class_id, u, v, width, height, score=0.9, tracker_id=''):
    d = Detection2D()
    d.id = tracker_id                       # empty: a YOLO without tracker
    d.bbox.center.position.x, d.bbox.center.position.y = u, v
    d.bbox.size_x, d.bbox.size_y = width, height
    h = ObjectHypothesisWithPose()
    h.hypothesis.class_id, h.hypothesis.score = class_id, score
    d.results.append(h)
    return d


def run(parameters, detections, frame_id='camera_color_optical_frame'):
    """The 3D detections that the node publishes for one array of 2D detections."""
    # The parameters of the node, as overrides of the command line
    rclpy.init(args=['--ros-args'] + sum((['-p', f'{k}:={v}'] for k, v in parameters.items()), []))
    try:
        node = Detection3DEstimator()
        probe = rclpy.create_node('probe')
        received = []
        probe.create_subscription(Detection3DArray, '/detections_3d', received.append, 10)
        info_pub = probe.create_publisher(CameraInfo, '/camera_info', 10)
        det_pub = probe.create_publisher(Detection2DArray, '/detections_2d', 10)
        info = CameraInfo(width=W, height=H)
        info.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        executor.add_node(probe)
        end = time.time() + 10.0
        sent = False
        while time.time() < end and not received:
            if node.intrinsics is None:
                info_pub.publish(info)
            elif not sent or int(time.time() * 2) % 2 == 0:
                msg = Detection2DArray(detections=detections)
                msg.header.frame_id = frame_id
                det_pub.publish(msg)
                sent = True
            executor.spin_once(timeout_sec=0.1)
        node.destroy_node()
        probe.destroy_node()
        assert received, 'No 3D detections published'
        return received[0]
    finally:
        rclpy.try_shutdown()


def test_class_by_name_without_tracker_id():
    # A person 340 px tall (1.70 m) at the center: box_size puts it at 500 * 1.70 / 340 = 2.5 m
    out = run({'estimator': 'box_size'}, [yolo_detection('person', CX, CY, 100.0, 340.0)])
    assert len(out.detections) == 1
    d = out.detections[0]
    assert d.id == '' and d.results[0].hypothesis.class_id == 'person'
    assert out.header.frame_id == 'camera_color_optical_frame'
    assert math.isclose(d.results[0].pose.pose.position.z, 2.5, rel_tol=1e-6)


def test_class_by_coco_index_with_class_names():
    # class_id "0" is COCO's person: with class_names the size of a person is used, and the
    # class is published as the detector gave it ("0")
    names = 'person,bicycle,car'
    out = run({'estimator': 'box_size', 'class_names': f'[{names}]'},
              [yolo_detection('0', CX, CY, 100.0, 340.0, tracker_id='7')])
    d = out.detections[0]
    assert d.id == '7' and d.results[0].hypothesis.class_id == '0'
    assert math.isclose(d.results[0].pose.pose.position.z, 2.5, rel_tol=1e-6)


def test_class_by_index_without_class_names_is_unknown():
    out = run({'estimator': 'box_size', 'unknown_distance_m': 3.0},
              [yolo_detection('0', CX, CY, 100.0, 340.0)])
    assert math.isclose(out.detections[0].results[0].pose.pose.position.z, 3.0, rel_tol=1e-6)


def test_header_without_frame_gets_camera_frame_id():
    out = run({'estimator': 'ray', 'camera_frame_id': 'camera_link_optical'},
              [yolo_detection('dog', 420.0, CY, 50.0, 50.0)], frame_id='')
    assert out.header.frame_id == 'camera_link_optical'
    p = out.detections[0].results[0].pose.pose.position
    # On the ray of the box center, at ray_distance_m (1 m) along it
    assert p.x > 0.0 and math.isclose(math.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2), 1.0, rel_tol=1e-6)


def test_coco_file_has_the_80_classes():
    import yaml
    with open(COCO) as f:
        names = yaml.safe_load(f)['detection_3d_estimator']['ros__parameters']['class_names']
    assert len(names) == 80 and names[0] == 'person' and names[79] == 'toothbrush'
