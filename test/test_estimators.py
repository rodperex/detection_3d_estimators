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

import math
import os

import pytest

from detection_3d_estimators.estimators import BoxSizeEstimator, RayEstimator
from detection_3d_estimators.geometry import Box, Intrinsics, point_at, to_body, unit_ray
from detection_3d_estimators.object_sizes import ObjectSizes

SIZES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config',
                     'object_sizes.yaml')
CAMERA = Intrinsics(598.05, 598.05, 319.5, 239.5, 640, 480)   # the Webots NAO top camera


def table():
    return ObjectSizes.from_file(SIZES)


def test_unit_ray_is_unit_and_points_through_the_pixel():
    x, y, z = unit_ray(CAMERA, 500.0, 100.0)
    assert math.isclose(x * x + y * y + z * z, 1.0)
    assert math.isclose(CAMERA.fx * x / z + CAMERA.cx, 500.0)
    assert math.isclose(CAMERA.fy * y / z + CAMERA.cy, 100.0)


def test_ray_at_one_metre_is_the_former_projector():
    # attention_perception's projector without depth: nx = (u - cx) / fx, z = 1 / sqrt(nx^2 +
    # ny^2 + 1), the point (nx z, ny z, z), in the body convention (z, -x, -y)
    u, v = 412.0, 77.0
    nx, ny = (u - CAMERA.cx) / CAMERA.fx, (v - CAMERA.cy) / CAMERA.fy
    z = 1.0 / math.sqrt(nx * nx + ny * ny + 1.0)
    distance, _ = RayEstimator(1.0).distances(CAMERA, [Box(u, v, 10, 10)], ['x'])[0]
    assert to_body(point_at(CAMERA, u, v, distance)) == pytest.approx((z, -nx * z, -ny * z))


@pytest.mark.parametrize('label,name', [
    ('person', 'person'), ('person with blue T-shirt', 'person'),
    ('the red coffee cup on the table', 'cup'), ('dining table', 'dining table'),
    ('a man holding a cup', 'person'), ('cell phone', 'cell phone'), ('two cups', 'cup'),
    ('the woman on the left', 'person'), ('people', 'person'), ('teddy bear', 'teddy bear'),
])
def test_labels_match_their_head_noun(label, name):
    assert table().match(label).name == name


def test_unknown_label():
    assert table().match('unicorn') is None
    estimator = BoxSizeEstimator(table(), unknown_distance=2.0)
    assert estimator.distance(CAMERA, Box(320, 240, 50, 50), 'unicorn') == (2.0, 'unknown')


def test_override_and_restrict():
    sizes = table()
    sizes.override(['person:1.85:0.6', 'robot arm:0.8'])
    assert sizes.match('person').height == 1.85 and sizes.match('person').width == 0.6
    assert sizes.match('the robot arm').height == 0.8
    sizes.restrict(['person'])
    assert sizes.match('cup') is None and sizes.match('woman').name == 'person'
    with pytest.raises(ValueError):
        sizes.restrict(['unicorn'])


def test_box_size_from_the_height_of_a_whole_box():
    sizes = table()
    sizes.override(['person:1.8:0.6'])
    estimator = BoxSizeEstimator(sizes)
    # 1.8 m tall at 3 m: 598.05 * 1.8 / 3 = 358.8 pixels
    distance, how = estimator.distance(CAMERA, Box(320, 240, 120, 358.83), 'person')
    assert distance == pytest.approx(3.0, rel=1e-3) and how.startswith('height')


def test_box_cut_by_the_image_uses_the_closer_of_height_and_width():
    sizes = table()
    sizes.override(['person:1.8:0.6'])
    estimator = BoxSizeEstimator(sizes)
    # At 1.5 m the person (718 pixels) does not fit: the box is cut to the image (480)
    box = Box(320, 240, 598.05 * 0.6 / 1.5, 480)
    distance, how = estimator.distance(CAMERA, box, 'person')
    assert distance == pytest.approx(1.5, rel=1e-3) and how.startswith('width')


def test_depth_in_box_is_a_low_percentile_of_the_central_region():
    import numpy as np
    from detection_3d_estimators.estimators import depth_in_box
    depth = np.full((480, 640), 6.0)          # the wall behind
    depth[100:400, 280:300] = 2.0            # a leg
    depth[100:400, 340:360] = 2.1            # the other leg; the center (320, 240) is the wall
    depth[240, 320] = 0.0                    # a hole
    box = Box(320, 250, 100, 300)
    assert depth[250, 320] == 6.0
    assert depth_in_box(depth, box) == 2.0
    assert depth_in_box(np.zeros((480, 640)), box) is None
    assert depth_in_box(depth, Box(700, 250, 10, 10)) is None


def test_depth_image_scales_the_box_to_the_depth_resolution():
    import numpy as np
    from detection_3d_estimators.estimators import DepthImageEstimator
    color = Intrinsics(1000.0, 1000.0, 640.0, 360.0, 1280, 720)
    depth = np.full((360, 640), 5.0)
    depth[150:210, 290:350] = 3.0            # the object, at the center of the 640x360 depth
    (distance, _), = DepthImageEstimator().distances(color, [Box(640, 360, 100, 100)], ['x'],
                                                     depth=depth)
    assert distance == pytest.approx(3.0)    # straight ahead: the distance is the depth


def test_round_half_away_as_cpp():
    from detection_3d_estimators.estimators import round_half_away
    assert [round_half_away(v) for v in (0.5, 1.5, 2.5, -0.5, -2.5, 2.49)] == [1, 2, 3, -1, -3, 2]
