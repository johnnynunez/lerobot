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

"""Unit tests for the aim-ray laser-dot math (pure numpy, no XR runtime).

Loaded by file path like test_hud.py — the example package is not installed.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

from lerobot.utils.rotation import Rotation

_EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "isaac_teleop_to_so101"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _EXAMPLE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


aim = _load("aim")

_IDENTITY_XYZW = np.array([0.0, 0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------- aim_ray


def test_aim_ray_identity_points_minus_z():
    origin, direction = aim.aim_ray(np.array([0.1, 0.2, 0.3]), _IDENTITY_XYZW)
    assert np.allclose(origin, [0.1, 0.2, 0.3])
    assert np.allclose(direction, [0.0, 0.0, -1.0])


def test_aim_ray_rotated_ninety_about_x_points_plus_y():
    # +90 deg about X maps -Z onto +Y (right-hand rule).
    quat = Rotation.from_rotvec([np.pi / 2, 0.0, 0.0]).as_quat()
    _, direction = aim.aim_ray(np.zeros(3), quat)
    assert np.allclose(direction, [0.0, 1.0, 0.0], atol=1e-9)


# ---------------------------------------------------------------------------- ray_plane_hit


def test_hit_straight_down():
    hit = aim.ray_plane_hit(np.array([0.2, 0.1, 0.5]), np.array([0.0, 0.0, -1.0]))
    assert np.allclose(hit, [0.2, 0.1, 0.0])


def test_hit_diagonal():
    # 45-degree ray from 0.4 m up: lands 0.4 m forward of the origin.
    d = np.array([1.0, 0.0, -1.0]) / np.sqrt(2)
    hit = aim.ray_plane_hit(np.array([0.0, 0.0, 0.4]), d)
    assert np.allclose(hit, [0.4, 0.0, 0.0])


def test_no_hit_pointing_up_or_parallel():
    assert aim.ray_plane_hit(np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, 1.0])) is None
    assert aim.ray_plane_hit(np.array([0.0, 0.0, 0.5]), np.array([1.0, 0.0, 0.0])) is None


def test_no_hit_beyond_max_range():
    # Grazing angle: the geometric hit is ~5.7 m away — outside any workspace.
    d = np.array([1.0, 0.0, -0.07])
    d = d / np.linalg.norm(d)
    assert aim.ray_plane_hit(np.array([0.0, 0.0, 0.4]), d) is None
    # The same geometry passes with an explicitly larger budget.
    assert aim.ray_plane_hit(np.array([0.0, 0.0, 0.4]), d, max_range_m=10.0) is not None


def test_hit_on_custom_plane_height():
    hit = aim.ray_plane_hit(np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, -1.0]), plane_z=0.2)
    assert np.allclose(hit, [0.0, 0.0, 0.2])


# ---------------------------------------------------------------------------- frame conversion


def test_base_point_round_trips_through_anchor():
    # base_T_anchor: anchor rotated +90 deg about Z and offset; converting a base point
    # to anchor and mapping it back through the matrix must return the original point.
    m = np.eye(4)
    m[:3, :3] = Rotation.from_rotvec([0.0, 0.0, np.pi / 2]).as_matrix()
    m[:3, 3] = [0.3, -0.1, 0.25]
    p_base = np.array([0.4, 0.2, 0.0])
    p_anchor = aim.base_point_to_anchor(m, p_base)
    assert np.allclose(m[:3, :3] @ p_anchor + m[:3, 3], p_base)


def test_plane_up_is_unit_and_matches_rotation():
    m = np.eye(4)
    m[:3, :3] = Rotation.from_rotvec([np.pi / 6, 0.0, 0.0]).as_matrix()
    up = aim.plane_up_in_anchor(m)
    assert np.isclose(np.linalg.norm(up), 1.0)
    assert np.allclose(m[:3, :3] @ up, [0.0, 0.0, 1.0], atol=1e-9)


# ---------------------------------------------------------------------------- quat_wxyz_z_to


def test_quat_aligns_z_to_direction():
    for target in ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.577, 0.577, 0.578]):
        target = np.asarray(target) / np.linalg.norm(target)
        w, x, y, z = aim.quat_wxyz_z_to(target)
        rot = Rotation.from_quat(np.array([x, y, z, w]))  # xyzw
        assert np.allclose(rot.as_matrix() @ [0.0, 0.0, 1.0], target, atol=1e-6)


def test_quat_degenerate_cases():
    assert aim.quat_wxyz_z_to(np.array([0.0, 0.0, 1.0])) == (1.0, 0.0, 0.0, 0.0)
    w, x, y, z = aim.quat_wxyz_z_to(np.array([0.0, 0.0, -1.0]))
    rot = Rotation.from_quat(np.array([x, y, z, w]))
    assert np.allclose(rot.as_matrix() @ [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], atol=1e-6)
