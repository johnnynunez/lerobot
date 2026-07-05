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

"""Pure-numpy unit tests for the XR controller pose gate.

Imported from the submodule (``...isaac_teleop.pose_gate``) so the tests collect and
run WITHOUT ``isaacteleop`` / the XR runtime installed (same pattern as ``test_clutch``).

The gate guards the clutch against the occlusion ghost/snap-back sequence: the
controller leaves the headset's camera view, the runtime keeps extrapolating a smooth
but wrong pose from the IMU, then the pose TELEPORTS back when the cameras re-acquire
it. The invariants that matter: (1) invalid frames are never ok and wipe the history,
(2) a teleport between two valid frames is flagged even when the runtime says valid,
(3) after any distrust spell the gate demands a settle window before trusting again,
(4) a single snap-back costs one glitch verdict — the gate re-anchors on the post-snap
position instead of deadlocking.
"""

import numpy as np
import pytest

from lerobot.teleoperators.isaac_teleop.pose_gate import PoseGate


def _run(gate: PoseGate, frames):
    """Feed ``[(pos, valid), ...]`` and return the verdict list."""
    return [gate.check(np.asarray(pos, dtype=float), valid) for pos, valid in frames]


# ----------------------------------------------------------------------------
# Steady valid stream
# ----------------------------------------------------------------------------


def test_steady_valid_stream_is_ok():
    gate = PoseGate(settle_frames=0)
    verdicts = _run(gate, [([0.0, 0.0, 1.0 + 0.001 * i], True) for i in range(10)])
    assert verdicts == ["ok"] * 10


def test_settle_window_applies_at_startup():
    # With a settle window, the first frames after construction are "settling":
    # the gate has no history yet, so it withholds trust until the stream proves live.
    gate = PoseGate(settle_frames=3)
    verdicts = _run(gate, [([0.0, 0.0, 1.0], True)] * 5)
    assert verdicts == ["settling", "settling", "settling", "ok", "ok"]


# ----------------------------------------------------------------------------
# Invalid frames (runtime says the pose is a ghost)
# ----------------------------------------------------------------------------


def test_invalid_flag_is_never_ok():
    gate = PoseGate(settle_frames=0)
    assert gate.check(np.zeros(3), False) == "invalid"


def test_non_finite_position_is_invalid_even_when_flagged_valid():
    gate = PoseGate(settle_frames=0)
    assert gate.check(np.array([np.nan, 0.0, 0.0]), True) == "invalid"
    assert gate.check(np.array([np.inf, 0.0, 0.0]), True) == "invalid"


def test_recovery_after_invalid_requires_settle_window():
    # Ghost spell then re-acquisition: the snapped-back pose must ride out the settle
    # window before it is trusted, even though each recovered frame looks sane.
    gate = PoseGate(settle_frames=2)
    _run(gate, [([0.0, 0.0, 1.0], True)] * 3)  # trusted stream
    assert gate.check(np.array([0.3, 0.0, 1.0]), False) == "invalid"  # occlusion
    verdicts = _run(gate, [([0.0, 0.0, 1.0], True)] * 3)
    assert verdicts == ["settling", "settling", "ok"]


def test_invalid_wipes_history_so_snap_back_is_not_a_glitch():
    # The pose moves 1 m while invalid; on recovery the jump must NOT read as a glitch
    # (there is no trusted previous frame to measure against) — settling is the verdict.
    gate = PoseGate(settle_frames=1)
    gate.check(np.array([0.0, 0.0, 1.0]), True)
    gate.check(np.array([0.0, 0.0, 1.0]), False)
    assert gate.check(np.array([1.0, 0.0, 1.0]), True) == "settling"
    assert gate.check(np.array([1.0, 0.0, 1.0]), True) == "ok"


# ----------------------------------------------------------------------------
# Teleport between valid frames (runtime lies: flags the ghost valid)
# ----------------------------------------------------------------------------


def test_teleport_between_valid_frames_is_a_glitch():
    gate = PoseGate(max_step_m=0.25, settle_frames=0)
    gate.check(np.array([0.0, 0.0, 1.0]), True)
    assert gate.check(np.array([0.5, 0.0, 1.0]), True) == "glitch"


def test_glitch_re_anchors_on_post_snap_position():
    # One snap costs one glitch + the settle window, then the stream is trusted again
    # at the NEW position — no deadlock measuring forever against the pre-snap pose.
    gate = PoseGate(max_step_m=0.25, settle_frames=2)
    _run(gate, [([0.0, 0.0, 1.0], True)] * 3)
    assert gate.check(np.array([0.6, 0.0, 1.0]), True) == "glitch"
    verdicts = _run(gate, [([0.6, 0.0, 1.0], True)] * 3)
    assert verdicts == ["settling", "settling", "ok"]


def test_fast_but_human_motion_stays_ok():
    # 0.1 m/frame (3 m/s at 30 FPS) is a vigorous hand motion, not a teleport.
    gate = PoseGate(max_step_m=0.25, settle_frames=0)
    verdicts = _run(gate, [([0.1 * i, 0.0, 1.0], True) for i in range(6)])
    assert verdicts == ["ok"] * 6


# ----------------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------------


def test_rejects_nonpositive_max_step():
    with pytest.raises(ValueError, match="max_step_m"):
        PoseGate(max_step_m=0.0)


def test_rejects_negative_settle_frames():
    with pytest.raises(ValueError, match="settle_frames"):
        PoseGate(settle_frames=-1)
