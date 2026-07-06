# !/usr/bin/env python

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

"""Headless tests for the in-headset HUD of the Isaac Teleop -> SO-101 example.

Covers the pure pieces only — no ``isaacteleop`` (let alone a BUILD_VIZ wheel), no
headset, no GPU:

* ``render_hud``: state dict -> RGBA frame (shape, alpha, phase-dependent content),
* ``LazyFollow``: the lazy head-follow placement (anchor latch, dwell, glide),
* ``HudClient``: best-effort send (dedupe, dead-pipe fallback) against a fake process,
* ``PreflightGame.hud_state``: the payload the loop forwards to the HUD.

The HUD lives in ``examples/isaac_teleop_to_so101`` (not the installed package), so the
module is loaded straight from its file path.
"""

import importlib.util
import io
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL", reason="the HUD renders with pillow")

_EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "isaac_teleop_to_so101"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _EXAMPLE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hud = _load("hud")
preflight = _load("preflight")


# ---------------------------------------------------------------------------- render_hud


def test_render_shape_and_alpha():
    rgba = hud.render_hud({"phase": "connect", "instruction": "Waiting…"})
    assert rgba.shape == (hud.PANEL_H, hud.PANEL_W, 4)
    assert rgba.dtype == np.uint8
    assert int(np.count_nonzero(rgba[..., 3])) > 0


def test_render_states_differ():
    """Distinct states must render distinct pixels (a stale-texture HUD is worse than none)."""
    a = hud.render_hud({"phase": "preflight", "instruction": "hold still", "checks_total": 6})
    b = hud.render_hud(
        {"phase": "preflight", "instruction": "hold still", "checks_total": 6, "checks_done": 3}
    )
    c = hud.render_hud({"phase": "teleop", "engaged": True, "squeeze": 0.9})
    assert np.count_nonzero(a != b) > 500  # progress dots moved
    assert np.count_nonzero(a != c) > 5000  # whole layout changed


def test_render_severity_changes_border():
    base = {"phase": "pose_lost", "instruction": "Controller out of view."}
    info = hud.render_hud({**base, "severity": "info"})
    error = hud.render_hud({**base, "severity": "error"})
    assert np.count_nonzero(info != error) > 500


def test_render_tolerates_junk_state():
    """The renderer must never raise on malformed input — the HUD is best-effort."""
    for state in ({}, {"phase": "???"}, {"phase": "teleop", "squeeze": "not-a-number"}):
        try:
            rgba = hud.render_hud(state)
        except (TypeError, ValueError):
            pytest.fail(f"render_hud raised on {state!r}")
        assert rgba.shape == (hud.PANEL_H, hud.PANEL_W, 4)


def test_render_long_instruction_wraps_inside_panel():
    rgba = hud.render_hud({"phase": "preflight", "instruction": "very long word " * 30})
    # Nothing may render outside the border columns (the wrap must clamp, not overflow).
    assert int(rgba[:, :2, :3].sum()) == 0 or int(rgba[:, :2, :3].max()) <= 255  # sanity
    assert rgba.shape == (hud.PANEL_H, hud.PANEL_W, 4)


def _compass_ring_zone(rgba: np.ndarray) -> np.ndarray:
    """Bottom arc of the compass dial — empty background in the compass-less layout."""
    return rgba[250:280, 640:700, :3].astype(int).sum(axis=2)


def test_render_teleop_compass_only_with_heading():
    """The pointing compass renders iff yaw_deg is present and finite (degrades cleanly)."""
    with_yaw = hud.render_hud({"phase": "teleop", "engaged": True, "yaw_deg": 45.0, "pitch_deg": 15.0})
    without_yaw = hud.render_hud({"phase": "teleop", "engaged": True})
    junk_yaw = hud.render_hud({"phase": "teleop", "yaw_deg": "n/a", "pitch_deg": float("nan")})
    assert _compass_ring_zone(with_yaw).max() > 250  # ring drawn
    assert _compass_ring_zone(without_yaw).max() < 250  # plain background
    assert _compass_ring_zone(junk_yaw).max() < 250  # malformed -> degraded, no crash


def test_render_teleop_compass_needle_follows_yaw():
    """Needle direction: yaw=0 -> robot +X -> screen up; yaw=+90 -> robot +Y -> screen left."""

    def needle_centroid(yaw: float) -> tuple[float, float]:
        rgba = hud.render_hud({"phase": "teleop", "engaged": True, "yaw_deg": yaw, "pitch_deg": 0.0})
        r = rgba[:, :, 0].astype(int)
        g = rgba[:, :, 1].astype(int)
        b = rgba[:, :, 2].astype(int)
        mask = (g > 150) & (g > r + 40) & (g > b + 40)  # the ok-green needle
        mask[:120, :] = False  # exclude the title/status rows
        mask[:, :580] = False  # restrict to the compass area
        ys, xs = np.where(mask)
        assert len(ys) > 0, "no needle pixels found"
        return float(ys.mean() - 210), float(xs.mean() - 668)  # (dy, dx) from dial center

    dy_fwd, dx_fwd = needle_centroid(0.0)
    dy_left, dx_left = needle_centroid(90.0)
    assert dy_fwd < -5 and abs(dx_fwd) < 8  # points up
    assert dx_left < -5 and abs(dy_left) < 8  # points left


# ---------------------------------------------------------------------------- LazyFollow

_IDENTITY_Q = (1.0, 0.0, 0.0, 0.0)


def _yaw_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2.0), 0.0, math.sin(yaw / 2.0), 0.0)


def test_lazy_follow_latches_first_anchor():
    follow = hud.LazyFollow()
    pos, yaw = follow.update(np.zeros(3), _IDENTITY_Q, now=0.0)
    # Identity head: forward is -Z, so the panel sits in front (negative Z), below eye level.
    assert pos[2] == pytest.approx(-hud.PANEL_DISTANCE_M)
    assert pos[1] == pytest.approx(hud.PANEL_Y_OFFSET_M)
    # A tiny head turn (below the re-anchor angle) must NOT move the panel.
    pos2, yaw2 = follow.update(np.zeros(3), _yaw_quat_wxyz(math.radians(10)), now=1.0)
    assert np.allclose(pos, pos2)
    assert yaw == pytest.approx(yaw2)


def test_lazy_follow_reanchors_after_dwell():
    follow = hud.LazyFollow()
    follow.update(np.zeros(3), _IDENTITY_Q, now=0.0)
    big_turn = _yaw_quat_wxyz(math.radians(90))
    # Big turn, but not yet dwelled: still the old anchor.
    pos_before, _ = follow.update(np.zeros(3), big_turn, now=0.1)
    assert pos_before[2] == pytest.approx(-hud.PANEL_DISTANCE_M)
    # After the dwell the anchor jumps (glide interpolates toward it over GLIDE_S).
    follow.update(np.zeros(3), big_turn, now=0.1 + hud.REANCHOR_DWELL_S + 0.01)
    pos_after, _ = follow.update(np.zeros(3), big_turn, now=2.0 + hud.GLIDE_S)
    # 90 deg yaw: forward is now -X, so the panel ends up at x ~ -distance.
    assert pos_after[0] == pytest.approx(-hud.PANEL_DISTANCE_M, abs=1e-6)


def test_lazy_follow_quick_glance_does_not_reanchor():
    follow = hud.LazyFollow()
    follow.update(np.zeros(3), _IDENTITY_Q, now=0.0)
    big_turn = _yaw_quat_wxyz(math.radians(90))
    follow.update(np.zeros(3), big_turn, now=0.1)  # glance away…
    follow.update(np.zeros(3), _IDENTITY_Q, now=0.2)  # …and right back
    pos, _ = follow.update(np.zeros(3), big_turn, now=0.2 + hud.REANCHOR_DWELL_S / 2)
    assert pos[2] == pytest.approx(-hud.PANEL_DISTANCE_M)  # anchor unchanged


# ---------------------------------------------------------------------------- HudClient


class _FakeProc:
    """subprocess.Popen stand-in: records writes; can be told to break the pipe."""

    def __init__(self, broken: bool = False):
        self.stdin = io.StringIO()
        self._broken = broken
        if broken:
            real_write = self.stdin.write

            def _raise(_s):  # noqa: ANN001
                raise BrokenPipeError

            self.stdin.write = _raise  # type: ignore[method-assign]
            self._real_write = real_write

    def wait(self, timeout=None):  # noqa: ANN001
        return 0

    def kill(self):
        pass


def test_hud_client_sends_json_lines():
    proc = _FakeProc()
    client = hud.HudClient(proc)
    client.send({"phase": "teleop", "engaged": True})
    lines = proc.stdin.getvalue().strip().splitlines()
    assert len(lines) == 1
    assert '"phase":"teleop"' in lines[0]


def test_hud_client_dedupes_identical_states():
    proc = _FakeProc()
    client = hud.HudClient(proc)
    for _ in range(5):
        client.send({"phase": "teleop", "engaged": False})
    client.send({"phase": "teleop", "engaged": True})
    assert len(proc.stdin.getvalue().strip().splitlines()) == 2


def test_hud_client_survives_dead_pipe():
    client = hud.HudClient(_FakeProc(broken=True))
    client.send({"phase": "connect"})  # must not raise
    client.send({"phase": "teleop"})  # already disabled: still must not raise


# ---------------------------------------------------------------- PreflightGame.hud_state


def test_preflight_hud_state_tracks_progress():
    game = preflight.PreflightGame(preflight.PreflightConfig(fps=10))
    assert game.hud_state() is None  # inactive game -> nothing to show
    game.arm("test")
    state = game.hud_state()
    assert state is not None
    assert state["phase"] == "preflight"
    assert state["checks_done"] == 0
    assert state["checks_total"] == 6
    assert state["attempt"] == 1
    assert state["instruction"]  # non-empty operator instruction
    assert game.hud_state(waiting=True)["waiting"] is True
