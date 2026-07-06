#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Aim-ray math for the in-headset laser dot (pure numpy, no XR imports).

The operator points the controller and the HUD shows the ENDPOINT: where the
OpenXR aim ray (the controller's native pointing frame, -Z forward — the same ray
the Quest home menu uses) intersects the robot's table plane (z = 0 in the robot
base frame, the same convention as the EE-bounds z floor).

Everything here is a pure function so it is unit-testable without a headset:
the teleop loop computes the hit in the ROBOT base frame (where the plane is
defined), then converts it to the ANCHOR (OpenXR world) frame where the HUD's
viz session places quad layers.
"""

from __future__ import annotations

import numpy as np

from lerobot.utils.rotation import Rotation

# Beyond this range the "hit" is a grazing intersection far outside the workspace —
# noise, not aiming. The dot is hidden instead of painted meters away.
MAX_AIM_RANGE_M = 2.5


def aim_ray(aim_pos: np.ndarray, aim_quat_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The controller's pointing ray ``(origin, unit direction)`` from the aim pose.

    OpenXR aim frame convention: -Z is forward (out of the controller's nose).
    """
    origin = np.asarray(aim_pos, dtype=float)
    rot = Rotation.from_quat(np.asarray(aim_quat_xyzw, dtype=float))
    direction = rot.as_matrix() @ np.array([0.0, 0.0, -1.0])
    return origin, direction


def ray_plane_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_z: float = 0.0,
    max_range_m: float = MAX_AIM_RANGE_M,
) -> np.ndarray | None:
    """Intersect a ray with the horizontal plane ``z = plane_z`` (robot base frame).

    Returns the hit point, or None when the ray points away from the plane, is
    (near-)parallel to it, or the hit lies beyond ``max_range_m`` along the ray —
    a grazing-angle intersection that would paint the dot far outside the workspace.
    """
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    dz = float(direction[2])
    if abs(dz) < 1e-6:  # parallel to the plane
        return None
    t = (plane_z - float(origin[2])) / dz
    if t <= 0.0 or t > max_range_m:  # behind the controller, or a grazing far hit
        return None
    return origin + t * direction


def base_point_to_anchor(base_T_anchor: np.ndarray, point_base: np.ndarray) -> np.ndarray:  # noqa: N803
    """Convert a point from the robot BASE frame to the ANCHOR (OpenXR world) frame.

    ``base_T_anchor`` maps anchor -> base (the teleop rebase transform); placing HUD
    geometry needs the inverse direction.
    """
    m = np.asarray(base_T_anchor, dtype=float)
    rot = m[:3, :3]
    # Rigid-transform inverse (R orthonormal): anchor_p = R^T (base_p - t).
    return rot.T @ (np.asarray(point_base, dtype=float) - m[:3, 3])


def plane_up_in_anchor(base_T_anchor: np.ndarray) -> np.ndarray:  # noqa: N803
    """The robot base +Z (table normal), expressed in the anchor frame (unit vector)."""
    rot = np.asarray(base_T_anchor, dtype=float)[:3, :3]
    up = rot.T @ np.array([0.0, 0.0, 1.0])
    return up / np.linalg.norm(up)


def quat_wxyz_z_to(direction: np.ndarray) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) rotating the quad's +Z normal onto ``direction``.

    Used to lay the dot flat on the table: the quad's face normal ends up along the
    table's up vector. Degenerate cases (already aligned / opposite) are handled.
    """
    target = np.asarray(direction, dtype=float)
    target = target / np.linalg.norm(target)
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z, target), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    if dot < -1.0 + 1e-9:
        return (0.0, 1.0, 0.0, 0.0)  # 180 deg about X
    axis = np.cross(z, target)
    axis = axis / np.linalg.norm(axis)
    half = np.arccos(dot) / 2.0
    s = float(np.sin(half))
    return (float(np.cos(half)), float(axis[0] * s), float(axis[1] * s), float(axis[2] * s))
