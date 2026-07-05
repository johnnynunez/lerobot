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

"""Per-frame sanity gate for the raw XR controller pose stream.

When the headset's cameras lose sight of a controller (out of the tracking volume,
behind the operator's back, under the table), most runtimes do NOT stop streaming:
they keep extrapolating the pose from the controller's IMU. The result is a "ghost"
pose that drifts away smoothly, then TELEPORTS back the instant the cameras
re-acquire the controller. If the clutch is engaged through that sequence, the
snap-back is integrated as one giant delta and the arm lurches — which reads as
"the robot went crazy after I looked away from the controllers".

:class:`PoseGate` is the pure-numpy defense (no XR imports, unit-testable):

* ``invalid`` — the runtime says the pose is not valid (grip ``is_valid`` flag off)
  or the position is non-finite. The gate resets its history.
* ``glitch``  — the position jumped farther between two consecutive valid frames
  than a human hand can move at the loop rate (occlusion snap-back signature).
* ``settling`` — the pose looks sane again but has not been sane for long enough;
  damps ghost oscillation around re-acquisition.
* ``ok``      — trustworthy frame; the only verdict the clutch may act on.

The owning loop maps any non-``ok`` verdict to "disengage the clutch and hold the
arm"; because engaging latches a fresh controller origin, recovery is jump-free by
construction (the operator just keeps squeezing and the arm resumes from where it is).
"""

from __future__ import annotations

import numpy as np

# Default per-frame position jump treated as a tracking glitch [m]. At 30 FPS,
# 0.25 m/frame is 7.5 m/s — beyond a human hand's teleop motion, but well below a
# typical occlusion snap-back (0.3–1 m in one frame).
DEFAULT_MAX_STEP_M = 0.25

# Default number of consecutive sane frames required after an invalid/glitch spell
# before the pose is trusted again (~0.17 s at 30 FPS).
DEFAULT_SETTLE_FRAMES = 5


class PoseGate:
    """Classify each controller-pose frame as ``ok`` / ``invalid`` / ``glitch`` / ``settling``.

    Call :meth:`check` once per loop frame with the raw grip position and the
    runtime's validity verdict for that frame. The gate is stateful (previous
    position + settle countdown) but pure numpy, so it is unit-testable with
    synthetic streams.
    """

    def __init__(
        self,
        max_step_m: float = DEFAULT_MAX_STEP_M,
        settle_frames: int = DEFAULT_SETTLE_FRAMES,
    ) -> None:
        if max_step_m <= 0.0:
            raise ValueError(f"max_step_m must be > 0, got {max_step_m}")
        if settle_frames < 0:
            raise ValueError(f"settle_frames must be >= 0, got {settle_frames}")
        self._max_step_m = float(max_step_m)
        self._settle_frames = int(settle_frames)
        self._prev: np.ndarray | None = None
        # Start distrustful: the first frames after construction must ride out the
        # settle window too (no history yet to judge a teleport against).
        self._settle = self._settle_frames

    def check(self, pos: np.ndarray, valid: bool) -> str:
        """Return the verdict for this frame: ``ok`` | ``invalid`` | ``glitch`` | ``settling``.

        Args:
            pos: Raw grip position (3,) [m] for this frame.
            valid: The runtime's validity verdict for this frame (tracking live AND
                the pose flagged valid). Pass ``False`` whenever either is off.
        """
        if not valid:
            self._prev = None
            self._settle = self._settle_frames
            return "invalid"
        pos = np.asarray(pos, dtype=float)
        if not np.all(np.isfinite(pos)):
            self._prev = None
            self._settle = self._settle_frames
            return "invalid"
        prev = self._prev
        self._prev = pos.copy()
        if prev is not None and float(np.linalg.norm(pos - prev)) > self._max_step_m:
            # Measure the NEXT frame against this (post-snap) position: a single
            # snap-back costs one glitch frame + the settle window, not a deadlock.
            self._settle = self._settle_frames
            return "glitch"
        if self._settle > 0:
            self._settle -= 1
            return "settling"
        return "ok"
