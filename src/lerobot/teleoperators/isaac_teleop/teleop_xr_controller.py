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

"""XR (VR) controller device for NVIDIA Isaac Teleop, exposed to LeRobot.

A deliberately thin reader: exposes the raw controller grip pose off
``ControllersSource`` (statically rebased into the robot base frame by
``ControllerTransform``), plus squeeze and trigger. No retargeters and no clutch —
the clutch rebasing and gripper mapping live downstream in the owning loop, so this
device is stateless across frames.

``isaacteleop`` imports are deferred so this module imports without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.types import RobotAction

from .base import IsaacTeleopTeleoperator
from .config_isaac_teleop import XRControllerConfig

if TYPE_CHECKING:
    from isaacteleop.retargeting_engine.interface import OutputCombiner

# Source-node name for the static base_T_anchor rebase input fed via
# ``TeleopSession.step(external_inputs=...)`` each frame.
_BASE_T_ANCHOR_INPUT = "base_T_anchor"

# Source-node name for the per-frame haptic pulse fed the same way (see send_feedback).
_HAPTIC_INPUT = "haptic_pulse"


class XRController(IsaacTeleopTeleoperator):
    """Raw XR controller grip-pose teleoperator (base-frame), no retargeters.

    Reads the raw grip pose + squeeze + trigger off a ``ControllersSource`` rebased into
    the robot base frame. :meth:`get_action` returns the absolute base-frame grip pose
    untouched; the owning loop owns the clutch and gripper mapping.
    """

    config_class = XRControllerConfig
    name = "isaac_teleop_controller"

    def __init__(self, config: XRControllerConfig):
        super().__init__(config)
        self.config: XRControllerConfig = config

        # Constant base_T_anchor input, built once in connect() (a TensorGroup is heavy and
        # isaacteleop-backed) and reused every step.
        self._external_inputs: dict[str, Any] | None = None
        # Whether the last get_action() read a tracked controller; the owning loop polls this
        # to wait for the operator to connect before driving the arm.
        self._is_tracking = False
        # Whether the runtime flagged the last grip pose as VALID. Tracked but not valid
        # happens when the controller leaves the headset's camera view: the runtime keeps
        # streaming an IMU-extrapolated "ghost" pose that later snaps back. The owning
        # loop must not integrate such frames into the clutch.
        self._pose_valid = False
        # ControllersSource built in _build_pipeline; its tracker is shared with the
        # haptic sink so the session creates a single controller tracker.
        self._controllers = None
        # Pending haptic pulse [amplitude, frequency_hz, duration_s], written by
        # send_feedback() and flushed into the haptic TensorGroup on the next step.
        self._haptic_pulse = np.zeros(3, dtype=np.float32)
        self._haptic_pending = False
        # TensorGroup feeding the haptic sink's ValueInput; built in connect().
        self._haptic_group = None

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> OutputCombiner:
        """Build the raw-grip-pose pipeline: a ``ControllersSource`` rebased into the base
        frame by ``ControllerTransform``, exposed verbatim as ``"controller"``. No retargeters.
        """
        from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
        from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
        from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

        side = self.config.hand_side
        controller_key = f"controller_{side}"

        controllers = ControllersSource(name="controllers")
        self._controllers = controllers  # shared with the haptic sink (single tracker)
        # Static base_T_anchor rebase fed via external_inputs each step.
        xform = ValueInput(_BASE_T_ANCHOR_INPUT, TransformMatrix())
        transformed = controllers.transformed(xform.output("value"))
        ctrl = transformed.output(controller_key)

        return OutputCombiner({"controller": ctrl})

    def _build_sinks(self) -> list:
        """Haptic sink for the active hand, fed per-step via ``external_inputs``.

        Reuses the pipeline's ``ControllersSource`` tracker so the session creates a
        single controller tracker (no OpenXR action-set contention). The pulse value
        comes from an OPTIONAL ``ValueInput`` leaf that :meth:`send_feedback` writes
        into — quiet frames feed an absent group (the sink skips them), so a timed
        pulse from a previous frame keeps ringing instead of being stopped by an
        explicit ``amplitude == 0`` refresh.
        """
        from isaacteleop.haptic_devices.controller import ControllerHapticDevice
        from isaacteleop.retargeting_engine.deviceio_source_nodes import HapticSink
        from isaacteleop.retargeting_engine.interface import OptionalType, ValueInput
        from isaacteleop.retargeting_engine.tensor_types import ControllerHapticPulse

        device = ControllerHapticDevice(self._controllers.get_tracker())  # type: ignore[union-attr]  # set in _build_pipeline
        sink = HapticSink("haptic_sink", device)
        pulse_input = ValueInput(_HAPTIC_INPUT, OptionalType(ControllerHapticPulse()))
        return [sink.connect({self.config.hand_side: pulse_input.output("value")})]

    def _build_external_inputs(self) -> dict[str, Any]:
        """Materialize the constant ``base_T_anchor`` + mutable haptic-pulse inputs (once)."""
        from isaacteleop.retargeting_engine.interface import OptionalTensorGroup, TensorGroup
        from isaacteleop.retargeting_engine.tensor_types import ControllerHapticPulse, TransformMatrix

        tg = TensorGroup(TransformMatrix())
        tg[0] = np.asarray(self.config.base_T_anchor, dtype=np.float32)
        self._haptic_group = OptionalTensorGroup(ControllerHapticPulse())  # starts absent
        return {
            _BASE_T_ANCHOR_INPUT: {"value": tg},
            _HAPTIC_INPUT: {"value": self._haptic_group},
        }

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        # Built after a successful connect so a failed connect leaves no half-state.
        self._external_inputs = self._build_external_inputs()

    def update_base_T_anchor(self, base_T_anchor: np.ndarray) -> None:  # noqa: N802, N803  (frameA_T_frameB convention)
        """Replace the static anchor->base rebase live (e.g. the post-preflight yaw alignment).

        Updates the config (the single source of truth) and, when connected, rebuilds the
        external-inputs TensorGroup so the next :meth:`get_action` step uses the new mapping.
        """
        m = np.asarray(base_T_anchor, dtype=np.float32)
        if m.shape != (4, 4):
            raise ValueError(f"base_T_anchor must be a 4x4 matrix, got shape {m.shape}")
        self.config.base_T_anchor = m.tolist()
        if self._external_inputs is not None:
            self._external_inputs = self._build_external_inputs()

    # ------------------------------------------------------------------
    # Action features
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict:
        return {
            "grip_pos": {
                "dtype": "float32",
                "shape": (3,),
                "names": {"x": 0, "y": 1, "z": 2},
            },
            "grip_quat": {
                "dtype": "float32",
                "shape": (4,),
                "names": {"qx": 0, "qy": 1, "qz": 2, "qw": 3},
            },
            # ``get_action`` returns scalars for these two, so the advertised
            # shape is () (0-d) to stay consistent with the returned values.
            "squeeze": {
                "dtype": "float32",
                "shape": (),
                "names": None,
            },
            "trigger": {
                "dtype": "float32",
                "shape": (),
                "names": None,
            },
        }

    @property
    def feedback_features(self) -> dict:
        return {
            # One frame of controller vibration; refresh every frame to sustain.
            "amplitude": {"dtype": "float32", "shape": (), "names": None},  # [0, 1]
            "frequency_hz": {"dtype": "float32", "shape": (), "names": None},  # 0 = default
            "duration_s": {"dtype": "float32", "shape": (), "names": None},  # 0 = one frame
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Queue a haptic pulse on the active controller, applied on the next step.

        ``{"amplitude": [0..1], "frequency_hz": optional, "duration_s": optional}`` —
        the runtime keeps vibrating for ``duration_s``; steady-state force feedback
        (e.g. gripper effort) should just re-send every frame. ``amplitude == 0``
        explicitly stops an active pulse.
        """
        self._haptic_pulse[0] = np.clip(float(feedback.get("amplitude", 0.0)), 0.0, 1.0)
        self._haptic_pulse[1] = float(feedback.get("frequency_hz", 0.0))
        self._haptic_pulse[2] = float(feedback.get("duration_s", 0.0))
        self._haptic_pending = True

    @property
    def is_tracking(self) -> bool:
        """Whether the last :meth:`get_action` read a tracked controller. ``False`` until the
        headset is connected over CloudXR and its controllers are live; the owning loop polls
        it to wait for the operator before commanding the arm."""
        return self._is_tracking

    @property
    def pose_valid(self) -> bool:
        """Whether the runtime flagged the last grip pose as valid (``XR_SPACE_LOCATION_
        POSITION_VALID``). A tracked-but-invalid pose is an IMU-extrapolated ghost — the
        controller is out of the headset's camera view and its pose will snap back on
        re-acquisition. The owning loop must treat such frames as untrusted (disengage
        the clutch / hold the arm) instead of integrating them."""
        return self._pose_valid

    # ------------------------------------------------------------------
    # Action extraction
    # ------------------------------------------------------------------

    def get_action(self) -> RobotAction:
        """Step the session and return the raw base-frame grip pose.

        Reads the grip pose + squeeze + trigger off the transformed controller stream (with
        the constant ``base_T_anchor`` rebase). When the controller is not tracked, returns
        identity pose and squeeze/trigger = 0.0 so the owning loop freezes the arm.

        Returns:
            ``{"grip_pos": (3,) [m], "grip_quat": (4,) [qx,qy,qz,qw], "squeeze": float,
            "trigger": float, "thumbstick_x": float, "thumbstick_y": float,
            "thumbstick_click": bool, "primary_click": bool}`` — pose in the robot base
            frame; squeeze/trigger in ``[0, 1]``; thumbstick axes in ``[-1, 1]``
            (x right, y up). The extra inputs let the owning loop offer in-headset
            runtime controls (speed scaling, re-align, etc.) without reaching for the
            keyboard the operator cannot see.
        """
        # Flush the queued haptic pulse into the sink's optional input group: present only
        # on frames where send_feedback() queued something (one send = one pulse), absent
        # otherwise so a timed pulse keeps ringing. Copy: TensorGroup stores a reference.
        if self._haptic_group is not None:
            if self._haptic_pending:
                self._haptic_group[0] = self._haptic_pulse.copy()
                self._haptic_pending = False
            else:
                self._haptic_group.set_none()
        result = self._step(execution_events=self._running_events(), external_inputs=self._external_inputs)

        from isaacteleop.retargeting_engine.tensor_types.indices import ControllerInputIndex

        # Optional controller group is None until the headset is connected and its controllers
        # are live; expose that as is_tracking so the loop can wait before driving the arm.
        controller = result["controller"]
        grip_pos = np.zeros(3, dtype=np.float32)
        grip_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        squeeze = 0.0
        trigger = 0.0
        thumb_x = 0.0
        thumb_y = 0.0
        thumb_click = False
        primary_click = False
        self._is_tracking = not getattr(controller, "is_none", False)
        self._pose_valid = False
        if self._is_tracking:
            # A read failure on a partially-populated frame leaves the safe defaults above and
            # reports not-tracked, so the loop freezes the arm rather than trusting a partial frame.
            try:
                grip_pos = np.asarray(controller[ControllerInputIndex.GRIP_POSITION], dtype=np.float32)
                grip_quat = np.asarray(controller[ControllerInputIndex.GRIP_ORIENTATION], dtype=np.float32)
                squeeze = float(controller[ControllerInputIndex.SQUEEZE_VALUE])
                trigger = float(controller[ControllerInputIndex.TRIGGER_VALUE])
                thumb_x = float(controller[ControllerInputIndex.THUMBSTICK_X])
                thumb_y = float(controller[ControllerInputIndex.THUMBSTICK_Y])
                thumb_click = float(controller[ControllerInputIndex.THUMBSTICK_CLICK]) > 0.5
                primary_click = float(controller[ControllerInputIndex.PRIMARY_CLICK]) > 0.5
                # The runtime's own verdict on the grip pose. False = IMU-extrapolated
                # ghost (controller out of camera view); the pose values above are then
                # smooth-looking but untrustworthy and will snap on re-acquisition.
                # float() > 0.5 is robust whether the slot yields a Python bool or a
                # 0-d tensor wrapper (which may not implement __bool__ meaningfully).
                self._pose_valid = float(controller[ControllerInputIndex.GRIP_IS_VALID]) > 0.5
            except (IndexError, KeyError, TypeError, ValueError):
                self._is_tracking = False
                self._pose_valid = False

        return {
            "grip_pos": grip_pos,
            "grip_quat": grip_quat,
            "squeeze": squeeze,
            "trigger": trigger,
            "thumbstick_x": thumb_x,
            "thumbstick_y": thumb_y,
            "thumbstick_click": thumb_click,
            "primary_click": primary_click,
        }
