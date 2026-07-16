"""Servo-style MediaPipe hand retargeting with hand-sign landmarks.

The wrist works like a small joystick:
- hand away from neutral = move gripper;
- hand back to neutral = stop;
- thumb-index pinch = close gripper / grasp sign.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ServoRetargetConfig:
    deadzone_x: float = 0.018
    deadzone_y: float = 0.018
    deadzone_z: float = 0.010

    max_step_x: float = 0.010
    max_step_y: float = 0.014
    max_step_z: float = 0.010

    side_sign: float = -1.0
    depth_sign: float = 1.0
    vertical_sign: float = -1.0

    roll_gain: float = 0.75
    roll_deadzone_rad: float = 0.05
    roll_limit_rad: float = 0.80

    assist_start_horizontal: float = 0.120
    assist_full_horizontal: float = 0.025
    assist_start_gap: float = 0.130
    assist_full_gap: float = 0.025
    assist_gain_x: float = 0.45
    assist_gain_y: float = 0.70
    assist_max_step_x: float = 0.010
    assist_max_step_y: float = 0.016


def _deadzone_ratio(value: float, deadzone: float) -> float:
    if abs(value) <= deadzone:
        return 0.0
    return float(np.clip(np.sign(value) * (abs(value) - deadzone) / 0.12, -1.0, 1.0))


def wrist_to_servo_delta(
    wrist_x: float,
    wrist_y: float,
    wrist_depth: float,
    neutral_x: float,
    neutral_y: float,
    neutral_depth: float,
    cfg: ServoRetargetConfig,
) -> tuple[np.ndarray, dict]:
    dx_img = float(wrist_x - neutral_x)
    dy_img = float(wrist_y - neutral_y)
    dz_img = float(wrist_depth - neutral_depth)

    side = _deadzone_ratio(dx_img, cfg.deadzone_x)
    vertical = _deadzone_ratio(dy_img, cfg.deadzone_y)
    depth = _deadzone_ratio(dz_img, cfg.deadzone_z)

    delta = np.array([
        cfg.depth_sign * depth * cfg.max_step_x,
        cfg.side_sign * side * cfg.max_step_y,
        cfg.vertical_sign * vertical * cfg.max_step_z,
    ], dtype=float)

    return delta, {
        "side": float(side),
        "depth": float(depth),
        "vertical": float(vertical),
        "dx_img": dx_img,
        "dy_img": dy_img,
        "dz_img": dz_img,
    }


def pinch_ratio(thumb_tip, index_tip, open_dist: float = 0.14, close_dist: float = 0.035) -> float:
    dx = float(thumb_tip.x - index_tip.x)
    dy = float(thumb_tip.y - index_tip.y)
    dz = float(thumb_tip.z - index_tip.z)
    dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    ratio = (open_dist - dist) / (open_dist - close_dist)
    return float(np.clip(ratio, 0.0, 1.0))


def finger_extended(landmarks, tip_id: int, pip_id: int) -> bool:
    """Simple image-space finger extension check: tip is above PIP."""
    return float(landmarks[tip_id].y) < float(landmarks[pip_id].y)


def classify_hand_sign(landmarks, pinch: float) -> str:
    """Classify basic signs for debug/control display.

    This is intentionally simple and thesis-safe:
    - PINCH / GRASP_SIGN: thumb-index close;
    - OPEN_HAND: most fingers extended;
    - POINTING: index extended while middle/ring/pinky folded;
    - MOVE_HAND: normal teleop hand.
    """
    if pinch >= 0.78:
        return "PINCH_GRASP_SIGN"

    index = finger_extended(landmarks, 8, 6)
    middle = finger_extended(landmarks, 12, 10)
    ring = finger_extended(landmarks, 16, 14)
    pinky = finger_extended(landmarks, 20, 18)

    extended_count = int(index) + int(middle) + int(ring) + int(pinky)

    if extended_count >= 3:
        return "OPEN_HAND_MOVE"
    if index and not middle and not ring and not pinky:
        return "POINTING_MOVE"
    return "MOVE_HAND"


def hand_roll_angle(landmarks) -> float:
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    return float(np.arctan2(float(pinky_mcp.y - index_mcp.y), float(pinky_mcp.x - index_mcp.x)))


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def wrist_roll_command(current_roll: float, neutral_roll: float, cfg: ServoRetargetConfig) -> float:
    delta = wrap_angle(current_roll - neutral_roll)
    if abs(delta) <= cfg.roll_deadzone_rad:
        delta = 0.0
    else:
        delta = np.sign(delta) * (abs(delta) - cfg.roll_deadzone_rad)
    return float(np.clip(cfg.roll_gain * delta, -cfg.roll_limit_rad, cfg.roll_limit_rad))


def smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def local_assist_alpha(horizontal_gap: float, finger_gap: float, cfg: ServoRetargetConfig) -> float:
    if horizontal_gap >= cfg.assist_start_horizontal:
        return 0.0

    h = 1.0 - (horizontal_gap - cfg.assist_full_horizontal) / (
        cfg.assist_start_horizontal - cfg.assist_full_horizontal
    )
    h = smoothstep01(h)

    if finger_gap >= cfg.assist_start_gap:
        z = 0.0
    else:
        z = 1.0 - (finger_gap - cfg.assist_full_gap) / (
            cfg.assist_start_gap - cfg.assist_full_gap
        )
        z = smoothstep01(z)

    return float(np.clip(max(h, z), 0.0, 1.0))


def apply_box_servo_assist(
    target: np.ndarray,
    x_gap: float,
    y_gap: float,
    horizontal_gap: float,
    finger_gap: float,
    box_center: np.ndarray,
    cfg: ServoRetargetConfig,
) -> tuple[np.ndarray, float]:
    alpha = local_assist_alpha(horizontal_gap, finger_gap, cfg)
    if alpha <= 0.0:
        return target, 0.0

    corrected = target.copy()
    corrected[0] += float(np.clip(alpha * cfg.assist_gain_x * x_gap, -cfg.assist_max_step_x, cfg.assist_max_step_x))
    corrected[1] += float(np.clip(alpha * cfg.assist_gain_y * y_gap, -cfg.assist_max_step_y, cfg.assist_max_step_y))

    # Guide rail near the box centerline.
    corrected[1] = (1.0 - 0.65 * alpha) * corrected[1] + (0.65 * alpha) * float(box_center[1])
    return corrected, alpha