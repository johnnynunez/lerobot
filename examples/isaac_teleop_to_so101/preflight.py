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

"""Interactive pre-flight "calibration game" for the XR -> SO-101 teleop loop.

From inside the headset the operator cannot see the calibration state of the chain: a
stale controller stream, wild tracking jitter, a stuck grip, or a dead follower bus all
look identical to "nothing happens" -- or become the first commanded motion. This module
gates teleop behind a short guided check sequence that proves every link -- tracking
stream, pose stability, motion scale, clutch squeeze, trigger, and the follower's servo
bus -- BEFORE any teleop command reaches the arm.

:class:`PreflightGame` is a pure per-frame state machine over the XR action dict (no XR
or robot imports), unit-testable with synthetic frames, and drivable two ways:

* blocking at startup via :func:`run_blocking` (nothing has commanded the arm yet), and
* one :meth:`PreflightGame.step` per control-loop frame for the mid-session re-arm when
  the operator pauses and resumes the VR client (the "Play" edge) -- the loop holds the
  arm (``compute() -> None``) until the game passes again.

The follower-side check (:func:`check_follower_gripper`) wiggles only the gripper jaw
while holding the measured arm joints: a minimal-inertia proof that commands are
actually executed by the bus/servos, safe in any arm pose.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from dataclasses import dataclass

import numpy as np

from lerobot.utils.robot_utils import precise_sleep

# Reminder cadence while the game waits for a paused/disconnected VR stream [s].
_WAIT_REMINDER_S = 15.0

# One check's verdict: (passed, human-readable detail for the scorecard).
CheckResult = tuple[bool, str]
CheckGen = Generator[None, "Frame", CheckResult]


@dataclass(frozen=True)
class Frame:
    """One tick of XR controller state fed to the game."""

    pos: np.ndarray  # (3,) grip position [m], robot base frame
    quat: np.ndarray  # (4,) grip orientation [qx, qy, qz, qw]
    squeeze: float  # [0, 1]
    trigger: float  # [0, 1]
    tracking: bool
    t: float  # monotonic time [s]


@dataclass
class PreflightConfig:
    """Tunables for the pre-flight checks (defaults sized for SO-101 tabletop teleop)."""

    clutch_threshold: float = 0.5  # squeeze level that engages the clutch (mirror the loop's)
    input_rest: float = 0.1  # squeeze/trigger level considered 'released'
    trigger_pull: float = 0.9  # trigger level considered 'fully pulled'
    stream_s: float = 1.0  # continuous live-stream time required
    still_s: float = 1.5  # hold-still window
    still_tolerance_m: float = 0.03  # max drift inside the hold-still window (jitter gate)
    move_distance_m: float = 0.15  # excursion required by the move-and-return check
    return_tolerance_m: float = 0.05  # how close to the start the move check must return
    max_step_m: float = 0.1  # per-frame teleport bound (mirrors common.MAX_EE_STEP_M)
    check_timeout_s: float = 30.0  # per-check timeout (postponed while not tracking)
    fps: int = 30


class PreflightGame:
    """Sequential guided checks; ALL must pass before teleop is (re-)enabled.

    A failed check restarts the whole sequence (the chain is only trusted as a whole);
    a paused VR stream pauses the per-check timeout instead of failing it. The game is
    user-paced -- it never gives up on its own, so the owning loop keeps holding the arm.
    """

    def __init__(self, config: PreflightConfig | None = None, notify=None) -> None:
        self.cfg = config or PreflightConfig()
        # Optional event callback: called with "check_passed" | "check_failed" | "all_passed".
        # Lets the owner mirror progress to channels readable from inside the headset
        # (haptic pulses, an XR HUD) — the terminal is invisible in VR.
        self._notify = notify or (lambda event: None)
        self._checks: tuple[tuple[str, str], ...] = (
            ("stream", "checking the controller stream (no action needed)…"),
            ("rest", "relax your hand: release the grip and the trigger."),
            ("still", f"hold the controller STILL for {self.cfg.still_s:.1f} s."),
            (
                "move",
                f"push the controller ~{self.cfg.move_distance_m * 100:.0f} cm in the ROBOT's "
                "forward direction (the way its gripper faces), then bring it back.",
            ),
            ("squeeze", "SQUEEZE the grip past the clutch threshold, then release."),
            ("trigger", "pull the TRIGGER fully, then release."),
        )
        self._active = False
        self._attempt = 0
        self._idx = 0
        self._gen: CheckGen | None = None
        self._deadline = 0.0
        self._last_wait_msg = 0.0
        self._results: list[tuple[str, str]] = []
        # Excursion vector of the last PASSED move check (peak_pos - start, current base
        # frame). The operator is told to push toward the ROBOT's front, so its direction
        # is their physical "robot forward" — the owner can yaw-align base_T_anchor with it.
        self.move_excursion: np.ndarray | None = None

    @property
    def active(self) -> bool:
        """True while the game is running and the loop must hold the arm."""
        return self._active

    def arm(self, reason: str) -> None:
        """(Re-)start the game from the first check."""
        self._active = True
        self._attempt = 1
        self._idx = 0
        self._gen = None
        self._results = []
        print("\n" + "=" * 76)
        print(f"PREFLIGHT CHECK — {reason}")
        print("The arm will NOT move until every check passes.")
        print("=" * 76)

    def step(self, action: dict, tracking: bool, t: float | None = None) -> bool:
        """Advance one frame; returns True exactly once, when the final check passes."""
        if not self._active:
            return False
        if t is None:
            t = time.monotonic()
        if self._gen is None:
            self._start_check(t)
        if not tracking:
            # A paused VR client must neither time a check out nor spam failures.
            self._deadline = t + self.cfg.check_timeout_s
            if t - self._last_wait_msg >= _WAIT_REMINDER_S:
                print("[preflight] waiting for the VR stream… (press Play / reconnect the headset)")
                self._last_wait_msg = t
        elif t >= self._deadline:
            self._fail(f"timed out after {self.cfg.check_timeout_s:.0f} s")
            return False
        frame = Frame(
            pos=np.asarray(action["grip_pos"], dtype=float),
            quat=np.asarray(action["grip_quat"], dtype=float),
            squeeze=float(action["squeeze"]),
            trigger=float(action["trigger"]),
            tracking=tracking,
            t=t,
        )
        try:
            assert self._gen is not None  # set by _start_check above
            self._gen.send(frame)
        except StopIteration as stop:
            ok, detail = stop.value
            name = self._checks[self._idx][0]
            if not ok:
                self._fail(detail)
                return False
            print(f"[preflight {self._idx + 1}/{len(self._checks)}] ✓ {name}: {detail}")
            self._results.append((name, detail))
            self._idx += 1
            self._gen = None
            if self._idx == len(self._checks):
                self._finish()
                return True
            self._notify("check_passed")
        return False

    # ------------------------------------------------------------------ internals

    def _start_check(self, t: float) -> None:
        name, instruction = self._checks[self._idx]
        print(f"[preflight {self._idx + 1}/{len(self._checks)}] {instruction}")
        gen: CheckGen = getattr(self, f"_check_{name}")()
        next(gen)  # prime to the first yield
        self._gen = gen
        self._deadline = t + self.cfg.check_timeout_s
        self._last_wait_msg = t

    def _fail(self, detail: str) -> None:
        name = self._checks[self._idx][0]
        self._attempt += 1
        print(f"[preflight] ✗ {name}: {detail}")
        print(f"[preflight] restarting the check sequence (attempt {self._attempt})…")
        self._idx = 0
        self._gen = None
        self._results = []
        self._notify("check_failed")

    def _finish(self) -> None:
        self._active = False
        print("-" * 76)
        for name, detail in self._results:
            print(f"  PASS  {name:<8} {detail}")
        print("PREFLIGHT PASSED — teleop enabled. Squeeze the grip to engage.")
        print("-" * 76)
        self._notify("all_passed")

    # ------------------------------------------------------------------ checks
    # Each check is a generator receiving one Frame per `yield` and returning a
    # CheckResult. Non-tracking frames reset/skip progress but never fail a check
    # (step() postpones the timeout while the stream is down).

    def _check_stream(self) -> CheckGen:
        """Live, finite, non-frozen pose stream for ``stream_s`` of tracked frames."""
        need = int(self.cfg.stream_s * self.cfg.fps)
        ok = frozen = 0
        last: np.ndarray | None = None
        while ok < need:
            f = yield
            if not f.tracking or not (np.all(np.isfinite(f.pos)) and np.all(np.isfinite(f.quat))):
                ok, frozen, last = 0, 0, None
                continue
            if last is not None and np.array_equal(f.pos, last):
                # Real tracking always jitters; a bit-identical pose for a full window
                # is a stale held-last stream, not a steady hand. Frozen frames are not
                # live progress — they must not count toward ``ok``.
                frozen += 1
                if frozen >= need:
                    return False, "pose bit-frozen — stale stream (runtime hung / controller asleep?)"
                continue
            frozen = 0
            last = f.pos
            ok += 1
        return True, f"live for {self.cfg.stream_s:.1f} s"

    def _check_rest(self) -> CheckGen:
        """Grip and trigger both read released — a stuck squeeze would auto-engage the clutch."""
        need = int(0.5 * self.cfg.fps)
        ok = 0
        while ok < need:
            f = yield
            at_rest = f.tracking and f.squeeze < self.cfg.input_rest and f.trigger < self.cfg.input_rest
            ok = ok + 1 if at_rest else 0
        return True, "grip and trigger at rest"

    def _check_still(self) -> CheckGen:
        """Hold-still window bounding tracking jitter/drift (bad calibration shows up here)."""
        need = int(self.cfg.still_s * self.cfg.fps)
        anchor: np.ndarray | None = None
        ok, worst = 0, 0.0
        while ok < need:
            f = yield
            if not f.tracking:
                anchor = None
                continue
            if anchor is None:
                anchor, ok, worst = f.pos, 0, 0.0
                continue
            dev = float(np.linalg.norm(f.pos - anchor))
            if dev > self.cfg.still_tolerance_m:
                anchor, ok, worst = f.pos, 0, 0.0  # moved or jittered: restart the window
                continue
            worst = max(worst, dev)
            ok += 1
        return True, f"held still, jitter {worst * 1000:.1f} mm"

    def _check_move(self) -> CheckGen:
        """Guided excursion-and-return: proves pose scale and catches teleporting tracking.

        Also records the excursion vector (``move_excursion``) — the operator pushes toward
        the robot's front, so this is their physical forward, used for yaw auto-alignment.
        """
        f = yield
        while not f.tracking:
            f = yield
        start, prev = f.pos, f.pos
        peak, peak_pos, reached = 0.0, f.pos, False
        while True:
            f = yield
            if not f.tracking:
                continue
            step = float(np.linalg.norm(f.pos - prev))
            prev = f.pos
            if step > self.cfg.max_step_m:
                return False, f"pose teleported {step * 100:.0f} cm in one frame — tracking glitch"
            dist = float(np.linalg.norm(f.pos - start))
            if dist > peak:
                peak, peak_pos = dist, f.pos
            if not reached and dist >= self.cfg.move_distance_m:
                reached = True
                print("[preflight]   good — now bring the controller back to where it started.")
            if reached and dist <= self.cfg.return_tolerance_m:
                self.move_excursion = np.asarray(peak_pos - start, dtype=float)
                return True, f"moved {peak * 100:.0f} cm out and back"

    def _check_squeeze(self) -> CheckGen:
        """Full squeeze-and-release cycle through the clutch threshold."""
        pulled = False
        while True:
            f = yield
            if not f.tracking:
                continue
            if not pulled and f.squeeze > self.cfg.clutch_threshold:
                pulled = True
                print("[preflight]   squeeze registered — now release the grip.")
            elif pulled and f.squeeze < self.cfg.input_rest:
                return True, "squeeze cycles through the clutch threshold"

    def _check_trigger(self) -> CheckGen:
        """Full trigger pull-and-release cycle (the gripper channel)."""
        pulled = False
        while True:
            f = yield
            if not f.tracking:
                continue
            if not pulled and f.trigger > self.cfg.trigger_pull:
                pulled = True
                print("[preflight]   trigger registered — now release it.")
            elif pulled and f.trigger < self.cfg.input_rest:
                return True, "trigger cycles full range"


def run_blocking(game: PreflightGame, teleop_device, fps: int) -> None:
    """Drive the game from the device's own stream until it passes (startup path).

    Blocking is safe here: nothing has commanded the arm yet and the follower's servos
    hold their last position. User-paced; ``Ctrl-C`` aborts (like the connect wait).

    A tracked-but-invalid pose (controller out of the headset's camera view — the
    runtime streams an IMU-extrapolated ghost) counts as NOT tracking, so ghost frames
    pause the checks instead of feeding them garbage.
    """
    while game.active:
        action = teleop_device.get_action()
        trusted = teleop_device.is_tracking and getattr(teleop_device, "pose_valid", True)
        game.step(action, trusted)
        precise_sleep(1.0 / fps)


def check_follower_gripper(
    robot,
    motor_names: list[str],
    fps: int,
    delta: float = 8.0,
    seconds: float = 0.6,
    return_tolerance: float = 2.5,
) -> CheckResult:
    """Prove the follower executes commands with a minimal-inertia gripper wiggle.

    Holds every arm joint at its measured position and ramps only the gripper jaw
    ``delta`` RANGE_0_100 units away and back, verifying the MEASURED position follows.
    A pass means serial bus, calibration mapping, and servo torque all work; a fail
    means teleop would silently do nothing — or execute wrong.
    """
    obs = robot.get_observation()
    if "gripper.pos" not in obs:
        return True, "skipped (no gripper)"
    hold = {f"{n}.pos": float(obs[f"{n}.pos"]) for n in motor_names}
    start = hold["gripper.pos"]
    target = min(100.0, max(0.0, start + delta if start <= 50.0 else start - delta))
    n_steps = max(1, int(seconds * fps))
    peak = 0.0
    for leg_target in (target, start):
        leg_start = float(robot.get_observation()["gripper.pos"])
        for i in range(1, n_steps + 1):
            v = leg_start + (leg_target - leg_start) * i / n_steps
            robot.send_action({**hold, "gripper.pos": v})
            precise_sleep(1.0 / fps)
            peak = max(peak, abs(float(robot.get_observation()["gripper.pos"]) - start))
    final = float(robot.get_observation()["gripper.pos"])
    span = abs(target - start)
    if peak < 0.5 * span:
        return False, (
            f"gripper commanded {span:.0f} units but measured only {peak:.1f} — "
            "bus/torque/calibration problem"
        )
    if abs(final - start) > return_tolerance:
        return False, f"gripper did not return (off by {abs(final - start):.1f} units) — obstruction?"
    return True, f"gripper wiggled {peak:.1f}/{span:.0f} units and returned"


def describe_calibration(robot) -> str:
    """One-line human-readable follower calibration status for the scorecard."""
    calibrated = getattr(robot, "is_calibrated", None)
    if calibrated is None:
        status = "calibration unknown"
    elif calibrated:
        status = "calibrated"
    else:
        status = "NOT CALIBRATED"
    fpath = getattr(robot, "calibration_fpath", None)
    return f"follower {status}" + (f" ({fpath})" if fpath else "")
