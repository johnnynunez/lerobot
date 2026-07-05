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

"""Engage-relative clutch for the XR -> SO-101 teleop loop.

Turns the raw controller grip pose into an absolute base-frame EE target, so the XR
device can stay a thin raw-pose reader. Pure numpy + the local ``Rotation`` helper (no
``isaacteleop``), so it is unit-testable without the XR runtime.
"""

from __future__ import annotations

import numpy as np

from lerobot.utils.rotation import Rotation


class Clutch:
    """Engage-relative clutch for both position AND orientation.

    Latch an origin on engage, then track the base-frame delta from it, applied
    independently to position and orientation. State:

    - ``_last_commanded_pos`` / ``_last_commanded_rot``: last commanded EE pose; held
      while disengaged so the arm freezes where it was left.
    - ``_home_pos`` / ``_home_rot``: latched on engage — the EE pose the delta applies to.
    - ``_origin_pos`` / ``_origin_rot``: latched on engage — the controller pose the delta
      is measured against.

    Each engaged frame :meth:`rebase` returns::

        pos = home_pos + (grip_pos - origin_pos)  # 1:1 controller -> EE translation
        rot = (R_ctrl @ R_origin ^ -1) @ R_home  # base-frame delta, left-composed

    On the engage edge the output is exactly the home pose (no teleport). The orientation
    delta is left-composed (base frame), so hand rotation about base Z maps to EE rotation
    about base Z. A re-clutch latches a fresh home/origin.

    NOTE: ``_home_rot`` is the last *commanded* orientation; the 5-DOF SO-101 cannot fully
    realize it, but the commanded signal stays continuous across a re-clutch (no jump).
    """

    def __init__(self, home_base_T_ee: np.ndarray):  # noqa: N803
        # Seed the held pose from the arm's measured startup EE pose so the first
        # engage latches home there (no jump on the first squeeze).
        home = np.asarray(home_base_T_ee, dtype=float)
        self._last_commanded_pos = home[:3, 3].copy()
        self._last_commanded_rot = Rotation.from_matrix(home[:3, :3])
        self._home_pos = self._last_commanded_pos.copy()
        self._home_rot = self._last_commanded_rot
        self._origin_pos = np.zeros(3, dtype=float)
        self._origin_rot = Rotation.from_quat(np.array([0.0, 0.0, 0.0, 1.0]))

    def engage(self, grip_pos: np.ndarray, grip_quat: np.ndarray) -> None:
        """Latch the engage home (where the arm is now) and controller origin."""
        self._home_pos = self._last_commanded_pos.copy()
        self._home_rot = self._last_commanded_rot
        self._origin_pos = np.asarray(grip_pos, dtype=float).copy()
        self._origin_rot = Rotation.from_quat(np.asarray(grip_quat, dtype=float))

    @property
    def last_commanded_pos(self) -> np.ndarray:
        """The last commanded EE position [m] (held while disengaged) — telemetry hook."""
        return self._last_commanded_pos.copy()

    def rebase(self, grip_pos: np.ndarray, grip_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the absolute base-frame EE target ``(pos [m], quat [xyzw])`` for this frame."""
        pos = self._home_pos + (np.asarray(grip_pos, dtype=float) - self._origin_pos)
        rot_ctrl = Rotation.from_quat(np.asarray(grip_quat, dtype=float))
        rot = (rot_ctrl * self._origin_rot.inv()) * self._home_rot
        self._last_commanded_pos = pos.copy()
        self._last_commanded_rot = rot
        return pos, rot.as_quat()

    def limit_lead(self, measured_pos: np.ndarray, max_lead_m: float) -> tuple[np.ndarray, float]:
        """Anti-windup leash: keep the commanded target within ``max_lead_m`` of the arm.

        The 1:1 clutch integrates controller deltas without knowing the arm's reachable
        workspace: push past it and the commanded target keeps flying while the arm
        saturates, after which the return motion is silently swallowed until the virtual
        target re-enters the envelope — which reads as "the arm ignores me". Clamping the
        lead — and dragging the engage home with it, so future deltas stay continuous —
        makes the arm respond to the FIRST centimeter of the return motion. It also stops
        the windup from surviving a re-clutch (engage latches home on the last COMMANDED
        pose, which without the leash may sit far outside the workspace).

        Call each engaged frame AFTER :meth:`rebase`, with the measured EE position (FK
        of the measured joints, same base frame).

        Returns:
            ``(corrected_pos, excess_m)`` — the possibly pulled-back target and how far
            it was pulled (``0.0`` when already within the leash).
        """
        measured = np.asarray(measured_pos, dtype=float)
        lead = self._last_commanded_pos - measured
        dist = float(np.linalg.norm(lead))
        if dist <= max_lead_m:
            return self._last_commanded_pos.copy(), 0.0
        correction = lead * (1.0 - max_lead_m / dist)
        self._home_pos = self._home_pos - correction
        self._last_commanded_pos = self._last_commanded_pos - correction
        return self._last_commanded_pos.copy(), dist - max_lead_m
