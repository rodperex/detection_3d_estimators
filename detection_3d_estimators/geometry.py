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

"""The pinhole camera: from a pixel and a distance to a 3D point. No ROS here."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Intrinsics:
    """The camera calibration: focal lengths and principal point (pixels), and the image size
    (0 when unknown: then it is taken as twice the principal point)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 0
    height: int = 0

    @property
    def image_width(self):
        return self.width or 2.0 * self.cx

    @property
    def image_height(self):
        return self.height or 2.0 * self.cy


@dataclass(frozen=True)
class Box:
    """A 2D detection box: its center and size, in pixels."""

    u: float
    v: float
    width: float
    height: float

    @property
    def top(self):
        return self.v - self.height / 2.0

    @property
    def bottom(self):
        return self.v + self.height / 2.0

    @property
    def left(self):
        return self.u - self.width / 2.0

    @property
    def right(self):
        return self.u + self.width / 2.0


def unit_ray(intrinsics, u, v):
    """The direction of the pixel (u, v), as a unit vector in optical coordinates (x right,
    y down, z forward)."""
    nx = (u - intrinsics.cx) / intrinsics.fx
    ny = (v - intrinsics.cy) / intrinsics.fy
    z = 1.0 / math.sqrt(nx * nx + ny * ny + 1.0)
    return nx * z, ny * z, z


def point_at(intrinsics, u, v, distance):
    """The 3D point at this distance (m, from the camera, along the ray) seen at the pixel
    (u, v), in optical coordinates."""
    x, y, z = unit_ray(intrinsics, u, v)
    return x * distance, y * distance, z * distance


def to_body(point):
    """Optical coordinates (x right, y down, z forward) to the body convention of a camera
    frame (x forward, y left, z up), for images stamped with a frame that is not optical."""
    x, y, z = point
    return z, -x, -y
