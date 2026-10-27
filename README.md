# detection_3d_estimators

A ROS 2 node that places 2D detections in 3D: it takes the boxes of any detector
(`vision_msgs/Detection2DArray`) and publishes, for each box, a 3D point in the camera frame
(`vision_msgs/Detection3DArray`), with the same id, class and score.

The direction of each detection comes from the camera calibration: the point lies on the ray
through the center of its box. What differs from one camera to another is **how far along that
ray**, and that is what the *estimator* gives:

| Estimator | Camera | The distance comes from | Typical use |
|---|---|---|---|
| `depth_image` | With depth (RGB-D) | The depth image, in the central part of the box | TurtleBot 4 (OAK-D), Astra, RealSense, any simulated RGB-D camera |
| `ray` | Without depth | Nothing: a fixed distance (1 m) | A robot that only turns towards what it sees (the direction is exact) |
| `box_size` | Without depth | The real size of what was detected over its size in the image | A robot without depth that needs the distance to people or known objects (the NAO keeping its distance to a person) |
| `depth_anything` | Without depth | A monocular depth network (Depth Anything V2, metric) | Distances to anything, at the cost of a GPU network |

It works with closed-set detectors (YOLO, COCO classes: `person`, `cup`) and with
open-vocabulary ones (labels such as `person with a red shirt` or `the coffee cup on the table`).
It runs on ROS 2 Humble and Jazzy (Python, no compilation).

```text
  /detections_2d ──┐
  CameraInfo ──────┤      detection_3d_estimator        ┌──> /detections_3d
  (depth image) ───┼──>   (estimator: depth_image |  ───┤
  (color image) ───┘       ray | box_size |             └──> detection_centroid_markers (RViz)
                           depth_anything)
```

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [What it publishes](#what-it-publishes)
- [Parameters](#parameters)
- [The estimators](#the-estimators)
- [The table of sizes](#the-table-of-sizes)
- [Accuracy](#accuracy)
- [Using it without ROS](#using-it-without-ros)
- [Troubleshooting](#troubleshooting)

## Installation

In the `src` folder of a colcon workspace with ROS 2 Humble or Jazzy:

```sh
git clone https://github.com/rodperex/detection_3d_estimators.git
cd .. && rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install --packages-select detection_3d_estimators
source install/setup.bash
```

It needs `rclpy`, `sensor_msgs`, `vision_msgs`, `visualization_msgs`, NumPy and PyYAML (installed
by rosdep). The `depth_anything` estimator also needs PyTorch and Hugging Face Transformers
(`pip install torch transformers`, or the packages of your environment); the other estimators do
not import them.

## Quick start

Each example assumes a detector publishing `/detections_2d`.

```sh
# A depth camera: the depth image must be aligned with the color image the detector sees
ros2 launch detection_3d_estimators detection_3d_estimator.launch.py estimator:=depth_image \
  camera_info_topic:=/camera/color/camera_info \
  depth_image_topic:=/camera/aligned_depth_to_color/image_raw

# A camera without depth, only the direction (every detection 1 m away)
ros2 launch detection_3d_estimators detection_3d_estimator.launch.py estimator:=ray \
  camera_info_topic:=/camera/camera_info

# Without depth, the distance from the size of what is detected (people only, 1.75 m tall)
ros2 run detection_3d_estimators detection_3d_estimator --ros-args \
  -p estimator:=box_size -p camera_info_topic:=/camera/camera_info \
  -p classes:="['person']" -p sizes:="['person:1.75:0.55']"

# Without depth, a monocular depth network (downloads the model the first time)
ros2 run detection_3d_estimators detection_3d_estimator --ros-args \
  -p estimator:=depth_anything -p camera_info_topic:=/camera/camera_info -p image_topic:=/camera/image
```

The launch file (`launch/detection_3d_estimator.launch.py`) has the most common parameters as
arguments (`estimator`, the topics, `camera_intrinsics`, `camera_frame_is_optical`,
`unknown_distance_m`, `use_sim_time`); any other parameter goes in a YAML file given with
`params_file`:

```yaml
detection_3d_estimator:
  ros__parameters:
    estimator: box_size
    camera_info_topic: /camera/camera_info
    sizes: ["person:1.80:0.55", "walker:0.90:0.60"]
    classes: ["person", "walker"]
    unknown_distance_m: 2.0
```

## With any detector (a YOLO, for example)

The node takes the detections of any detector that publishes a `vision_msgs/Detection2DArray`:

- the box in pixels of the camera image whose calibration it has (`bbox.center.position`,
  `bbox.size_x`, `bbox.size_y`); a detector that runs on a resized image must give the boxes in
  the size of the original image, as most ROS wrappers do;
- the class in `results[0].hypothesis.class_id`, by name (`person`) or by number (`0`); a number
  is named with `class_names` (for a YOLO trained on COCO: `config/class_names_coco.yaml`), which
  only `box_size` needs;
- a tracker id in `id`, if the detector has a tracker (empty is fine: the output keeps it);
- the frame of the camera in `header.frame_id`; if the detector leaves it empty, give it with
  `camera_frame_id`.

For a YOLO that publishes COCO indices on `/yolo/detections`:

```bash
ros2 launch detection_3d_estimators detection_3d_estimator.launch.py estimator:=box_size \
  detections_2d_topic:=/yolo/detections camera_info_topic:=/camera/camera_info \
  params_file:=$(ros2 pkg prefix detection_3d_estimators)/share/detection_3d_estimators/config/class_names_coco.yaml
```

`test/test_generic_detector.py` runs the node with detections of this kind.

## What it publishes

| Topic | Type | Content |
|---|---|---|
| `detections_3d_topic` (`/detections_3d`) | `vision_msgs/Detection3DArray` | One detection per input box that got a distance, in the same order, with the header of the input (the frame and the time of the image). For each: `id` (e.g. the tracker id), `results[0]` with the class, the score and the 3D point (`pose.pose.position`), and `bbox` (center: the same point; size: the box at that depth, its depth 0.01 m) |
| `centroid_marker_topic` (`detection_centroid_markers`) | `visualization_msgs/MarkerArray` | A sphere per detection, for RViz; the spheres of detections that disappear are deleted |

The point is in the **frame of the image** (the `frame_id` of the detections). Two conventions,
chosen with `camera_frame_is_optical`:

- `true` (default): the ROS optical convention, x right, y down, z forward (frames such as
  `camera_color_optical_frame`).
- `false`: the body convention, x forward, y left, z up, for cameras whose images are stamped
  with a frame that is not optical (some simulated cameras; the NAO's `CameraTop_frame`).

To have the points in another frame (`odom`, `base_link`), transform them with tf2 at the time
of the header.

A detection is left out only when its estimator has no distance for it (`depth_image` with no
valid depth in its box). The other estimators always give one: what they cannot estimate goes
to `unknown_distance_m`.

## Parameters

| Parameter | Default | Estimators | Meaning |
|---|---|---|---|
| `estimator` | `ray` | | `depth_image`, `ray`, `box_size` or `depth_anything` |
| `detections_2d_topic` | `/detections_2d` | all | Input boxes (`vision_msgs/Detection2DArray`: `bbox.center`, `bbox.size_x/size_y` in pixels; the class in `results[0].hypothesis.class_id`) |
| `detections_3d_topic` | `/detections_3d` | all | Output |
| `camera_info_topic` | `/camera_info` | all | The calibration of the camera of the detections (the first message is used) |
| `camera_intrinsics` | `[0.0]` (none) | all | `[fx, fy, cx, cy]` instead of the CameraInfo, when it does not arrive. The image size is then taken as twice the principal point |
| `camera_frame_is_optical` | `true` | all | The convention of the output point (above) |
| `camera_frame_id` | (empty) | all | The frame of the output for detections whose header has no `frame_id` |
| `class_names` | (none) | all | The names of the classes of a detector that gives them as numbers: `class_id` `"0"` is the first name. Only to find the size (`box_size`); the output keeps the class as given. `config/class_names_coco.yaml` has the 80 of COCO |
| `depth_image_topic` | `/depth_image` | `depth_image` | The depth image: `16UC1` in millimetres or `32FC1` in metres, aligned with the color image |
| `depth_window_px` | `5` | `depth_image` | The region of the box is at least (2 × this + 1)² pixels |
| `depth_region` | `0.6` | `depth_image`, `depth_anything` | The central part of the box whose depths are used (of its width and height) |
| `depth_percentile` | `10.0` | `depth_image`, `depth_anything` | The percentile of those depths taken: low, the nearest surface |
| `ray_distance_m` | `1.0` | `ray` | The distance of every detection |
| `object_sizes_file` | the package's `config/object_sizes.yaml` | `box_size` | The table of sizes |
| `sizes` | `['']` | `box_size` | Sizes that replace or add to those of the table: `"name:height:width"` (m; width optional) |
| `classes` | `['']` (all) | `box_size` | Only these names of the table; the rest go to `unknown_distance_m` |
| `cut_margin` | `0.025` | `box_size` | A box within this fraction of the image height of its top or bottom is taken as cut by the image |
| `unknown_distance_m` | `1.0` | `box_size`, `depth_anything` | The distance of a detection without an estimate |
| `image_topic` | `/image_rgb` | `depth_anything` | The color image (`rgb8`, `bgr8`, `rgba8`, `bgra8`) |
| `depth_anything_model` | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | `depth_anything` | A Hugging Face model id |
| `device` | `cuda` | `depth_anything` | `cuda` (falls back to `cpu` without a GPU) or `cpu` |
| `centroid_marker_topic`, `centroid_marker_namespace`, `centroid_marker_scale`, `centroid_marker_color` | `detection_centroid_markers`, `detection_centroids`, `0.05`, `[0, 1, 0, 1]` | all | The RViz spheres |

## The estimators

### `depth_image`: a depth camera

The depth of a box is a low percentile (`depth_percentile`, 10) of the valid depths (finite, > 0)
in its central part (`depth_region`, 60 % of its width and height; at least 11 × 11 pixels; at most
about 2500 samples). Not the depth of the center pixel: the center of a person's box often falls
between the legs, where the depth image sees the wall behind; the low percentile is the nearest
surface of the target. A depth image of another resolution than the color image (e.g. 640 × 360
for a 1280 × 720 color image) is scaled. The depth must be **aligned with the color image** (the
same field of view; RealSense `aligned_depth_to_color`, Astra `depth_registration`, OAK-D
`stereo` aligned to the color camera): if it is not, the boxes fall a few pixels off, which the
central region mostly absorbs at a few metres.

### `ray`: only the direction

Every detection at `ray_distance_m` (1 m). The direction is exact; the distance means nothing.
Enough for a robot that turns its head or body towards a detection.

### `box_size`: the size of what was detected

A pinhole camera sees an object of real height *H* at distance *d* as *h* = *f* · *H* / *d*
pixels tall, so *d* = *f* · *H* / *h*, with the focal length *f* of the calibration. Nothing is
calibrated beyond the camera itself: only the real size is assumed, from
[the table of sizes](#the-table-of-sizes), and the error of the distance is the error of that
size (a person of 1.60 m taken as 1.75 m: 9 % too far).

- The **height** of the box is used, unless the box is cut by the top or the bottom of the image
  (the object is close, or the camera points away): a cut box is shorter than the object and
  would give a distance too far. Then the **width** gives a second distance, and the closer of
  both is kept: when unsure, the object is taken as closer (a robot that keeps its distance stops
  too far rather than walking into someone).
- A label is matched to the table by its **head noun** (below), so open-vocabulary labels work:
  `person in a blue T-shirt` is a person, `the red coffee cup on the table` a cup.
- What is not in the table (or not in `classes`) goes to `unknown_distance_m`.

It is wrong when the assumption is: a seated person or a child (seen smaller: too far), an
object partly hidden by another (a table hiding a person's legs: too far), a person seen side-on
whose box is also cut (narrower: too far), or an object whose size varies a lot within its class
(chairs, tables). For people standing it works well (see [Accuracy](#accuracy)).

### `depth_anything`: a monocular depth network

[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), a metric model through
Hugging Face Transformers, gives a depth for every pixel of the color image; the depth of a box
is taken as in `depth_image`. It needs the color image (`image_topic`) and a GPU (about 130 ms per
640 × 480 image on a laptop GPU with the Small model). The model is downloaded the first time.
Models: `depth-anything/Depth-Anything-V2-Metric-Indoor-{Small,Base,Large}-hf` (indoors, up to
20 m) and `...-Metric-Outdoor-...` (up to 80 m); the Small ones are Apache-2.0, the Base and Large
ones CC-BY-NC-4.0 (not for commercial use). It works for any object, but it is less accurate than
`box_size` for people (see [Accuracy](#accuracy)), and synthetic images (simulators) are harder
for it than real ones.

## The table of sizes

`config/object_sizes.yaml` has the 80 classes of COCO (those of YOLO models trained on COCO)
and other things common indoors (door, wheelchair, walker, box...):

```yaml
person: {height: 1.70, width: 0.50, aliases: [man, woman, people, human, child, patient, ...]}
cup: {height: 0.10, width: 0.08, aliases: [mug, glass, coffee cup]}
dining table: {height: 0.75, width: 1.40, aliases: [table, desk]}
```

- `height`: its vertical size as the camera sees it; `width`: its horizontal size (omit it if it
  changes too much with the point of view); `aliases`: other names (singular, lowercase).
- A label is matched by its head noun: the words before the first preposition or relative word
  (`with`, `on`, `in`, `holding`, `who`...) are the noun phrase, and its last word (or words) that
  name something in the table are the head: `red coffee cup` → cup, `person with a cup` → person,
  `dining table` → dining table (the longer name). Plurals are reduced to singular.
- The values are typical adult and household sizes. Adapt them to your setting with `sizes`
  (`"person:1.80:0.55"`), restrict the estimate with `classes`, or give your own file with
  `object_sizes_file`.

## Accuracy

One person standing in front of a NAO in Webots (the world of the attention system), from 1 to
5 m, with the head straight and tilted up (18 images, the boxes of an open-vocabulary detector,
OMDet-Turbo; the person's size measured in the simulator: 1.85 m, 0.6 m):

| Estimator | Mean error | Largest error | Time per image |
|---|---|---|---|
| `box_size` | 2.6 % | 7 % | negligible |
| `depth_anything` (Metric-Indoor-Small) | 16 % | 29 % | ~130 ms (GPU) |
| Depth Anything V2 relative (vits), scaled with the other images | 42 % | 172 % | ~230 ms (GPU) |

`depth_anything` underestimated every distance in these synthetic images; on real images it may
do better. `box_size` on a real robot: the error is that of the assumed size, about ±10 % for
standing adults with the default 1.70 m.

## Using it without ROS

The estimators are plain Python (`detection_3d_estimators.estimators`, `geometry`,
`object_sizes`):

```python
from detection_3d_estimators.estimators import BoxSizeEstimator
from detection_3d_estimators.geometry import Box, Intrinsics, point_at
from detection_3d_estimators.object_sizes import ObjectSizes

camera = Intrinsics(fx=598.05, fy=598.05, cx=319.5, cy=239.5, width=640, height=480)
sizes = ObjectSizes.from_file('config/object_sizes.yaml')
estimator = BoxSizeEstimator(sizes)
distance, how = estimator.distance(camera, Box(u=320, v=240, width=120, height=358), 'person in red')
x, y, z = point_at(camera, 320, 240, distance)   # optical coordinates, metres
```

Tests (no ROS needed): `python3 -m pytest test`.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Nothing on `/detections_3d`, the log says it waits for the calibration | No `CameraInfo` on `camera_info_topic` (check its name and QoS), or it does not cross between ROS distributions: give `camera_intrinsics` |
| Points in the wrong directions (e.g. up instead of forward) | `camera_frame_is_optical` does not match the frame of the images: optical frames (z forward) `true`, body frames (x forward) `false` |
| `depth_image`: detections missing | No valid depth in their boxes (too close or too far for the camera, or a black surface), or the depth image has not arrived (`depth_image_topic`; a warning says so) |
| `depth_image`: distances of the background | The depth is not aligned with the color image |
| `box_size`: everything at `unknown_distance_m` | The labels do not match the table: check the names, or add them with `sizes`; with `classes`, only those names are estimated |
| `depth_anything`: slow | It runs on the CPU: check `device` and that PyTorch sees the GPU |

## License

Apache-2.0. Author: Rodrigo Pérez-Rodríguez (Universidad Rey Juan Carlos).
