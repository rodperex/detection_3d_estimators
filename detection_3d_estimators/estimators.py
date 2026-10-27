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

"""The estimators of the distance to a detection. Each gives, for every box, its distance from
the camera along the ray of the box center (m), or None if it has none, and how it was obtained.
No ROS here.

With a depth camera:
    depth_image     the depth image, in the central part of the box (a low percentile)
With a single image (no depth camera):
    ray             a fixed distance: only the direction is known
    box_size        the real size of what was detected (object_sizes.py) over its size in the image
    depth_anything  a monocular depth network (Depth Anything V2, metric)
"""

import math

import numpy as np

from detection_3d_estimators.geometry import Box, unit_ray


def round_half_away(value):
    """Rounding as C++ std::round (halves away from zero), not as Python's round."""
    return int(math.floor(abs(value) + 0.5)) * (1 if value >= 0 else -1)


def depth_in_box(depth, box, window=5, region=0.6, percentile=10.0):
    """The depth of a box in a depth image (H x W, metres along the optical axis; 0 or not
    finite: no measure): a low percentile of the valid depths in its central part (`region` of
    its width and height, at least (2 * window + 1)^2 pixels), so the same part of the target at
    any resolution, and its nearest surface (the center of a person's box falls between the legs,
    where the depth is that of the background). At most about 2500 samples. None if no pixel has
    a depth."""
    rows, cols = depth.shape
    center_u, center_v = round_half_away(box.u), round_half_away(box.v)
    if center_u < 0 or center_v < 0 or center_u >= cols or center_v >= rows:
        return None
    half_u = max(max(0, window), round_half_away(box.width * region / 2.0))
    half_v = max(max(0, window), round_half_away(box.height * region / 2.0))
    step = max(1, int(math.sqrt((2.0 * half_u + 1) * (2.0 * half_v + 1) / 2500.0)))
    patch = depth[max(0, center_v - half_v):min(rows - 1, center_v + half_v) + 1:step,
                  max(0, center_u - half_u):min(cols - 1, center_u + half_u) + 1:step]
    values = patch[np.isfinite(patch) & (patch > 0.0)].astype(np.float64)
    if values.size == 0:
        return None
    index = int(math.floor(min(max(percentile, 0.0), 100.0) / 100.0 * (values.size - 1)))
    return float(np.partition(values, index)[index])


class DepthImageEstimator:
    """A depth camera: the depth of each box in the depth image (depth_in_box), which must be
    aligned with the color image (the same field of view; another resolution is scaled)."""

    needs_image = False
    needs_depth = True

    def __init__(self, window=5, region=0.6, percentile=10.0):
        self.window, self.region, self.percentile = window, region, percentile

    def distances(self, intrinsics, boxes, labels, image=None, depth=None):
        if depth is None:
            return [(None, 'no depth image') for _ in boxes]
        rows, cols = depth.shape
        sx = sy = 1.0   # the pixels of the color image in the depth image (e.g. OAK-D: 1280 -> 640)
        if intrinsics.width > 0 and intrinsics.height > 0 and (cols, rows) != (intrinsics.width,
                                                                                intrinsics.height):
            sx, sy = cols / intrinsics.width, rows / intrinsics.height
        out = []
        for box in boxes:
            scaled = Box(box.u * sx, box.v * sy, box.width * sx, box.height * sy)
            z = depth_in_box(depth, scaled, self.window, self.region, self.percentile)
            # The depth is along the optical axis; the distance, along the ray of the center
            out.append((None, 'no depth') if z is None else
                       (z / unit_ray(intrinsics, box.u, box.v)[2], 'depth image'))
        return out


class RayEstimator:
    """Every detection at the same distance: for a robot that only needs the direction (to turn
    its head or body towards it)."""

    needs_image = False
    needs_depth = False

    def __init__(self, distance=1.0):
        if distance <= 0.0:
            raise ValueError('The distance of the ray must be > 0')
        self.distance = distance

    def distances(self, intrinsics, boxes, labels, image=None, depth=None):
        return [(self.distance, 'ray') for _ in boxes]


class BoxSizeEstimator:
    """The pinhole camera: an object of real height H seen h pixels tall is at fy * H / h.

    The height of the box is used, unless the box is cut by the top or the bottom of the image
    (within `margin` of its height): then the box is shorter than the object, and the height
    would give a distance too far. The width (fx * W / w) is then used too, and the closer of
    both kept: a robot that keeps its distance would rather stop too far than walk into
    someone. What is not in the table of sizes is placed at `unknown_distance` (its direction
    only)."""

    needs_image = False
    needs_depth = False

    def __init__(self, sizes, unknown_distance=1.0, margin=0.025):
        self.sizes = sizes
        self.unknown_distance = unknown_distance
        self.margin = margin

    def distance(self, intrinsics, box, label):
        size = self.sizes.match(label)
        if size is None or (size.height <= 0.0 and size.width <= 0.0) or box.height < 1.0 \
                or box.width < 1.0:
            return self.unknown_distance, 'unknown'
        rows = intrinsics.image_height
        cut_vertically = box.top <= self.margin * rows or box.bottom >= rows - self.margin * rows
        by_height = intrinsics.fy * size.height / box.height if size.height > 0.0 else math.inf
        by_width = intrinsics.fx * size.width / box.width if size.width > 0.0 else math.inf
        if not cut_vertically and size.height > 0.0:
            return by_height, f'height of a {size.name}'
        # Cut by the top or the bottom (or no height known): the closer of both
        if by_width <= by_height:
            return by_width, f'width of a {size.name}'
        return by_height, f'height of a {size.name} (cut by the image)'

    def distances(self, intrinsics, boxes, labels, image=None, depth=None):
        return [self.distance(intrinsics, b, label) for b, label in zip(boxes, labels)]


class DepthAnythingEstimator:
    """Depth Anything V2 (metric), through Hugging Face transformers: a depth for every pixel of
    the image. The depth of a detection is a low percentile of the depths in the central part of
    its box (its nearest surface: the center of a person's box falls between the legs, where the
    background is), turned into the distance along the ray of its center.

    The model is downloaded the first time (Hugging Face cache). Metric models: Metric-Indoor
    (Hypersim, up to 20 m) and Metric-Outdoor (Virtual KITTI, up to 80 m), Small, Base or Large;
    the Small ones are Apache-2.0, the others CC-BY-NC-4.0."""

    needs_image = True
    needs_depth = False

    def __init__(self, model='depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf',
                 device='cuda', region=0.6, percentile=10.0, unknown_distance=1.0):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch = torch
        self.device = device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu'
        self.processor = AutoImageProcessor.from_pretrained(model, use_fast=False)
        self.model = AutoModelForDepthEstimation.from_pretrained(model).to(self.device).eval()
        self.region = region
        self.percentile = percentile
        self.unknown_distance = unknown_distance

    def depth_map(self, rgb):
        """The metric depth (m, along the optical axis) of every pixel of an RGB image (H x W x 3,
        uint8)."""
        inputs = self.processor(images=rgb, return_tensors='pt').to(self.device)
        with self.torch.no_grad():
            predicted = self.model(**inputs).predicted_depth
        depth = self.torch.nn.functional.interpolate(
            predicted.unsqueeze(1), size=rgb.shape[:2], mode='bicubic', align_corners=False)
        return depth.squeeze().cpu().numpy()

    def box_depth(self, depth, box):
        rows, cols = depth.shape
        half_u = max(1, round(box.width * self.region / 2.0))
        half_v = max(1, round(box.height * self.region / 2.0))
        u, v = int(round(box.u)), int(round(box.v))
        patch = depth[max(0, v - half_v):min(rows, v + half_v + 1),
                      max(0, u - half_u):min(cols, u + half_u + 1)]
        values = patch[np.isfinite(patch) & (patch > 0.0)]
        return float(np.percentile(values, self.percentile)) if values.size else None

    def distances(self, intrinsics, boxes, labels, image=None, depth=None):
        if image is None:
            return [(self.unknown_distance, 'no image') for _ in boxes]
        depth = self.depth_map(image)
        out = []
        for box in boxes:
            z = self.box_depth(depth, box)
            if z is None:
                out.append((self.unknown_distance, 'unknown'))
            else:   # the depth is along the optical axis; the distance, along the ray
                out.append((z / unit_ray(intrinsics, box.u, box.v)[2], 'depth_anything'))
        return out
