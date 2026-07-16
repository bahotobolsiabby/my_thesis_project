"""Panda controller: safe servo movement + hand-sign landmarks.

This version fixes the box problem from the previous run:
- the gripper is no longer allowed to keep diving far below the box/table;
- when released, the box is locked at or above the table-safe height;
- hand landmarks and hand sign state are drawn on the webcam feed.

Controls:
- move hand away from neutral = servo gripper movement;
- return hand to neutral = stop;
- pinch sign = close gripper / grasp;
- open hand = release/place;
- press r = recalibrate neutral hand;
- press h = return home;
- press q = quit.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp
from controller import Supervisor

from panda_config import WorkspaceConfig, IKConfig, PandaGeometry, JOINT_LIMITS, HOME_POSE
from panda_retargeting import (
    ServoRetargetConfig,
    wrist_to_servo_delta,
    pinch_ratio,
    classify_hand_sign,
    hand_roll_angle,
    wrist_roll_command,
    apply_box_servo_assist,
)
from panda_ik import forward_kinematics, solve_panda_ik, clamp_joint_step, ik_is_stuck, joints_at_limit


JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
SENSOR_NAMES = [f"panda_joint{i}_sensor" for i in range(1, 8)]
FINGER_NAMES = ["panda_finger::left", "panda_finger::right"]

BOX_SIZE_Z = 0.070
TABLE_TOP_Z_WORLD = 0.140
BOX_CENTER_MIN_Z_WORLD = TABLE_TOP_Z_WORLD + BOX_SIZE_Z / 2.0

EE_TO_LOWEST_FINGER = 0.070

GRIPPER_OPEN = 0.040
GRIPPER_CLOSED = 0.000
FINGER_FORCE = 90.0
FINGER_VELOCITY = 0.025

MAX_JOINT_STEP = 0.018
HAND_LOST_HOLD_FRAMES = 45

# Hard wrist lock: instead of only *biasing* joints 5/6/7 toward a straight
# posture through the IK's nullspace term (posture_weight below, which the
# position-solve task can still override), this directly overrides those
# three joints every frame after the IK runs. clamp_joint_step() still rate
# -limits the approach to this pose, so it's a smooth converge-to-straight,
# not an instant snap -- but the end state is always exactly straight,
# matching a simple top-down pick-and-place gripper instead of one that
# tilts with hand roll. Hand-roll wrist control effectively becomes a no-op
# while this is True; set to False to restore free hand-roll control.
LOCK_WRIST_STRAIGHT = False  # use soft wrist straightening instead of destructive hard override
STRAIGHT_WRIST_POSE = HOME_POSE[4:7].copy()  # [panda_joint5, panda_joint6, panda_joint7]

ENABLE_CONTACT_HOLD = True
LOCK_BOX_WHEN_FREE = True  # simulation-first: keep box fixed until grasp; set False later for physics release
PINCH_TO_GRASP = 0.78
PINCH_TO_RELEASE = 0.55  # easier release when hand opens
GRASP_MAX_HORIZONTAL = 0.045
GRASP_MIN_FINGER_GAP = -0.025
GRASP_MAX_FINGER_GAP = 0.050
GRASP_REQUIRE_FRAMES = 3
REGRASP_ARM_PINCH = 0.60          # after release, hand must open below this before another attach
RELEASE_COOLDOWN_FRAMES = 18      # prevents instant re-attach after release

# Safety: prevents the previous issue where finger_gap went -17 cm / -27 cm.
MIN_SAFE_FINGER_GAP = -0.030
MAX_SAFE_DESCENT_WHILE_FREE = 0.006
FREE_GRIPPER_TABLE_CLEARANCE = 0.010

# Easy movement / grip assist:
# When the gripper gets too low or too close to the box/table before grasping,
# this gently pulls the target upward and backward so it can recover instead of
# getting stuck beside/inside the box. This is only active before attachment.
AUTO_PULLBACK_LIFT_ENABLE = True
AUTO_PULLBACK_IF_FINGER_GAP_BELOW = -0.020
AUTO_PULLBACK_IF_HORIZONTAL_ABOVE = 0.045
AUTO_PULLBACK_UP_STEP = 0.014
AUTO_PULLBACK_BACK_STEP = 0.010
AUTO_PULLBACK_BACK_SIGN = -1.0  # robot X negative = back toward Panda base in this world


DEBUG_EVERY = 10
SHOW_WEBCAM = True
DRAW_LANDMARK_IDS = True

# SMOOTH_SIDEWAYS_NOTE: Reduced carry-side boost prevents sudden target_y jumps after grasp.
# SIDEWAYS_PRIORITY_FIX:
# Your logs show side command reaching -1.00 while attached, but the arm/box
# does not visibly translate sideways. Two things can block that:
# 1) hard wrist override after IK can destroy the solved XYZ pose;
# 2) side speed is too small compared with the IK/joint-rate limit.
# This patch disables the hard wrist override and boosts side motion while holding.
SIDEWAYS_PRIORITY_FIX = True
CARRY_SIDE_BOOST = 0
CARRY_SIDE_MAX_EXTRA_STEP = 0
CARRY_SIDE_DEBUG_EVERY_FRAMES = 10


# Dataset recording:
# Saves to:
#   <project_root>/datasets/episodes/episode_YYYYMMDD_HHMMSS/
# where <project_root> is automatically detected from:
#   .../my_thesis_project-main/controllers/panda_test/panda_test.py
DATASET_RECORDING_ENABLE = False
DATASET_SAVE_EVERY_N_FRAMES = 3
DATASET_FLUSH_EVERY_N_STEPS = 20
DATASET_SAVE_WEBCAM_FALLBACK = False
DATASET_IMAGE_FORMAT = "png"
DATASET_PRINT_EVERY_N_STEPS = 20


# Pre-grasp sideways / downward fix:
# The old centering assist was too strong before grasping. When descending near
# the box and the user's hand gives a side command, allow sideways motion instead
# of forcing the gripper back to the box centerline.
PREGRASP_SIDE_OVERRIDE_ENABLE = True
PREGRASP_SIDE_OVERRIDE_THRESHOLD = 0.45
PREGRASP_SIDE_OVERRIDE_MAX_GAP = 0.120
PREGRASP_SIDE_OVERRIDE_X_GAIN = 0.25
PREGRASP_SIDE_OVERRIDE_X_MAX_STEP = 0.004

# Straight-gripper fix:
# Full hard wrist locking can damage the IK solution, but no lock lets the wrist
# tilt and hit the box. This softly blends joints 5/6/7 toward a straight pose,
# especially near the box, without completely destroying the solved XYZ motion.
SOFT_WRIST_STRAIGHTEN_ENABLE = True
SOFT_WRIST_ALPHA_FAR = 0.20
SOFT_WRIST_ALPHA_NEAR = 0.72
SOFT_WRIST_NEAR_FINGER_GAP = 0.120



def read_joint_positions(sensors: list, fallback_q: np.ndarray) -> np.ndarray:
    try:
        return np.array([s.getValue() for s in sensors], dtype=float)
    except Exception:
        return fallback_q.copy()


def get_panda_world(robot: Supervisor) -> np.ndarray:
    node = robot.getFromDef("PANDA")
    if node is None:
        return np.zeros(3, dtype=float)
    return np.array(node.getPosition(), dtype=float)


def node_world(robot: Supervisor, name: str) -> np.ndarray | None:
    node = robot.getFromDef(name)
    if node is None:
        return None
    return np.array(node.getPosition(), dtype=float)


def node_rotation(node) -> list[float] | None:
    try:
        field = node.getField("rotation")
        if field is not None:
            return list(field.getSFRotation())
    except Exception:
        pass
    return None


def set_box_pose(box_node, pos_world: np.ndarray, rot: list[float] | None) -> None:
    safe_pos = pos_world.copy()
    safe_pos[2] = max(float(safe_pos[2]), BOX_CENTER_MIN_Z_WORLD)

    try:
        box_node.getField("translation").setSFVec3f([float(safe_pos[0]), float(safe_pos[1]), float(safe_pos[2])])
    except Exception:
        pass

    if rot is not None:
        try:
            box_node.getField("rotation").setSFRotation(rot)
        except Exception:
            pass

    try:
        box_node.resetPhysics()
    except Exception:
        pass


def robot_frame(robot: Supervisor, world_pos: np.ndarray | None) -> np.ndarray | None:
    if world_pos is None:
        return None
    return world_pos - get_panda_world(robot)


def world_frame(robot: Supervisor, robot_pos: np.ndarray) -> np.ndarray:
    return robot_pos + get_panda_world(robot)


def box_robot(robot: Supervisor) -> np.ndarray | None:
    return robot_frame(robot, node_world(robot, "BOX"))


def gripper_robot(robot: Supervisor) -> np.ndarray | None:
    return robot_frame(robot, node_world(robot, "GRIPPER"))


def box_top_z_robot(robot: Supervisor) -> float | None:
    box = box_robot(robot)
    if box is None:
        return None
    return float(box[2] + BOX_SIZE_Z / 2.0)


def compute_gap(robot: Supervisor, gripper_pos_robot: np.ndarray | None = None) -> dict | None:
    box = box_robot(robot)
    top = box_top_z_robot(robot)
    if box is None or top is None:
        return None

    if gripper_pos_robot is None:
        gripper_pos_robot = gripper_robot(robot)
    if gripper_pos_robot is None:
        return None

    vec = box - gripper_pos_robot
    horizontal = float(np.linalg.norm(vec[:2]))
    finger_gap = float(gripper_pos_robot[2] - EE_TO_LOWEST_FINGER - top)

    return {
        "box": box,
        "top": float(top),
        "gripper": gripper_pos_robot,
        "x_gap": float(vec[0]),
        "y_gap": float(vec[1]),
        "horizontal": horizontal,
        "finger_gap": finger_gap,
    }


def clamp_workspace(target: np.ndarray, ws: WorkspaceConfig) -> np.ndarray:
    out = target.copy()
    out[0] = np.clip(out[0], ws.x_min, ws.x_max)
    out[1] = np.clip(out[1], ws.y_min, ws.y_max)
    out[2] = np.clip(out[2], ws.z_min, ws.z_max)
    return out


def clamp_gripper_safety(robot: Supervisor, target_gripper: np.ndarray, attached: bool) -> np.ndarray:
    """Do not let the target dive far below the box/table.

    The previous log showed finger_gap reaching -17 cm and -27 cm. This clamp
    prevents that while keeping the simple servo feel.
    """
    out = target_gripper.copy()
    top = box_top_z_robot(robot)

    if top is not None:
        min_z_near_box = top + EE_TO_LOWEST_FINGER + MIN_SAFE_FINGER_GAP
        if not attached:
            out[2] = max(out[2], min_z_near_box)
        else:
            # While holding, allow only slightly lower; usually the user lifts.
            out[2] = max(out[2], min_z_near_box - 0.010)

    panda_world_z = get_panda_world(robot)[2]
    world_lowest_finger_z = panda_world_z + out[2] - EE_TO_LOWEST_FINGER
    if world_lowest_finger_z < TABLE_TOP_Z_WORLD + FREE_GRIPPER_TABLE_CLEARANCE:
        out[2] += (TABLE_TOP_Z_WORLD + FREE_GRIPPER_TABLE_CLEARANCE) - world_lowest_finger_z

    return out


def apply_auto_pullback_lift(robot: Supervisor, target_gripper: np.ndarray, attached: bool) -> tuple[np.ndarray, bool]:
    """Gently pull the gripper upward/backward when it is in a bad pre-grasp pose.

    This is a recovery assist, not automatic pick-and-place. It only acts before
    the box is attached and only when the live gap says the gripper is too low
    or too far sideways to grasp cleanly.
    """
    if not AUTO_PULLBACK_LIFT_ENABLE or attached:
        return target_gripper, False

    gap = compute_gap(robot)
    if gap is None:
        return target_gripper, False

    too_low = gap["finger_gap"] < AUTO_PULLBACK_IF_FINGER_GAP_BELOW
    too_sideways = gap["horizontal"] > AUTO_PULLBACK_IF_HORIZONTAL_ABOVE and gap["finger_gap"] < 0.030

    if not (too_low or too_sideways):
        return target_gripper, False

    out = target_gripper.copy()
    out[2] += AUTO_PULLBACK_UP_STEP
    out[0] += AUTO_PULLBACK_BACK_SIGN * AUTO_PULLBACK_BACK_STEP
    return out, True


def should_allow_pregrasp_side_override(attached: bool, gap: dict | None, servo_debug: dict) -> bool:
    """Allow manual sideways correction while descending before gripping."""
    if not PREGRASP_SIDE_OVERRIDE_ENABLE or attached or gap is None:
        return False

    side_cmd = abs(float(servo_debug.get("side", 0.0)))
    finger_gap = float(gap.get("finger_gap", 999.0))

    return (
        side_cmd >= PREGRASP_SIDE_OVERRIDE_THRESHOLD
        and finger_gap <= PREGRASP_SIDE_OVERRIDE_MAX_GAP
    )


def soft_wrist_alpha_from_gap(gap: dict | None) -> float:
    """Use stronger wrist straightening near the object."""
    if not SOFT_WRIST_STRAIGHTEN_ENABLE:
        return 0.0
    if gap is None:
        return SOFT_WRIST_ALPHA_FAR

    finger_gap = float(gap.get("finger_gap", 999.0))
    if finger_gap <= SOFT_WRIST_NEAR_FINGER_GAP:
        return SOFT_WRIST_ALPHA_NEAR
    return SOFT_WRIST_ALPHA_FAR



def measured_gripper_offset(robot: Supervisor, q_actual: np.ndarray, geo: PandaGeometry, prev: np.ndarray) -> np.ndarray:
    live = gripper_robot(robot)
    if live is None:
        return prev.copy()

    fk = forward_kinematics(q_actual, geo)
    measured = live - fk

    if not np.all(np.isfinite(measured)):
        return prev.copy()

    return 0.85 * prev + 0.15 * measured


def update_contact_hold(
    robot: Supervisor,
    attached: bool,
    hold_offset: np.ndarray | None,
    box_lock_world: np.ndarray | None,
    box_lock_rot: list[float] | None,
    pinch: float,
    good_counter: int,
    frame: int,
    regrasp_armed: bool,
    release_cooldown: int,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, list[float] | None, int, bool, int]:
    box_node = robot.getFromDef("BOX")
    if box_node is None:
        return False, None, box_lock_world, box_lock_rot, 0, regrasp_armed, release_cooldown

    box_w = node_world(robot, "BOX")
    grip_w = node_world(robot, "GRIPPER")
    if box_w is None or grip_w is None:
        return False, None, box_lock_world, box_lock_rot, 0, regrasp_armed, release_cooldown

    if attached:
        if pinch <= PINCH_TO_RELEASE:
            box_lock_world = box_w.copy()
            box_lock_world[2] = max(float(box_lock_world[2]), BOX_CENTER_MIN_Z_WORLD)
            box_lock_rot = node_rotation(box_node)

            # Realistic release: do not freeze the box unless LOCK_BOX_WHEN_FREE is enabled.
            # This lets Webots physics handle the released object.
            if LOCK_BOX_WHEN_FREE:
                set_box_pose(box_node, box_lock_world, box_lock_rot)

            print(f"[HOLD] frame={frame} | RELEASED_REALISTIC")
            return False, None, box_lock_world, box_lock_rot, 0, False, RELEASE_COOLDOWN_FRAMES

        if hold_offset is None:
            hold_offset = box_w - grip_w

        held_pos = grip_w + hold_offset
        held_pos[2] = max(float(held_pos[2]), BOX_CENTER_MIN_Z_WORLD)
        set_box_pose(box_node, held_pos, box_lock_rot)

        if frame % DEBUG_EVERY == 0:
            print(f"[HOLD] frame={frame} | HOLDING | pinch={pinch:.2f}")
        return True, hold_offset, box_lock_world, box_lock_rot, good_counter, regrasp_armed, release_cooldown

    if LOCK_BOX_WHEN_FREE and box_lock_world is not None:
        set_box_pose(box_node, box_lock_world, box_lock_rot)

    gap = compute_gap(robot)
    if gap is None:
        return False, None, box_lock_world, box_lock_rot, 0, regrasp_armed, release_cooldown

    if release_cooldown > 0:
        release_cooldown -= 1
    if pinch <= REGRASP_ARM_PINCH and release_cooldown <= 0:
        regrasp_armed = True

    good = (
        regrasp_armed
        and pinch >= PINCH_TO_GRASP
        and gap["horizontal"] <= GRASP_MAX_HORIZONTAL
        and GRASP_MIN_FINGER_GAP <= gap["finger_gap"] <= GRASP_MAX_FINGER_GAP
    )
    good_counter = good_counter + 1 if good else 0

    if frame % DEBUG_EVERY == 0:
        reasons = []
        if not regrasp_armed:
            reasons.append("open_hand_to_rearm")
        if release_cooldown > 0:
            reasons.append(f"cooldown_{release_cooldown}")
        if pinch < PINCH_TO_GRASP:
            reasons.append("pinch")
        if gap["horizontal"] > GRASP_MAX_HORIZONTAL:
            reasons.append("center")
        if gap["finger_gap"] < GRASP_MIN_FINGER_GAP:
            reasons.append("too_low")
        if gap["finger_gap"] > GRASP_MAX_FINGER_GAP:
            reasons.append("too_high")

        print(
            f"[GRASP] frame={frame} | counter={good_counter}/{GRASP_REQUIRE_FRAMES} | "
            f"pinch={pinch:.2f} | horizontal={gap['horizontal']*100:.1f}cm | "
            f"finger_gap={gap['finger_gap']*100:.1f}cm | reasons={','.join(reasons) if reasons else 'ready'}"
        )

    if good_counter >= GRASP_REQUIRE_FRAMES:
        hold_offset = box_w - grip_w
        box_lock_rot = node_rotation(box_node)
        print(f"[HOLD] frame={frame} | ATTACHED")
        return True, hold_offset, box_lock_world, box_lock_rot, good_counter, regrasp_armed, release_cooldown

    return False, None, box_lock_world, box_lock_rot, good_counter, regrasp_armed, release_cooldown


def safe_list(value, length: int | None = None) -> list:
    """Convert numpy/list/tuple values to JSON-safe Python floats."""
    if value is None:
        return []
    try:
        arr = np.array(value, dtype=float).reshape(-1)
        if length is not None:
            arr = arr[:length]
        return [float(x) for x in arr.tolist()]
    except Exception:
        return []


def get_project_root() -> Path:
    """Detect project root from controllers/panda_test/panda_test.py."""
    controller_file = Path(__file__).resolve()
    # .../project/controllers/panda_test/panda_test.py
    # parents[0] = controllers/panda_test
    # parents[1] = controllers
    # parents[2] = project root
    try:
        return controller_file.parents[2]
    except Exception:
        return controller_file.parent


def get_place_zone_robot(robot: Supervisor) -> np.ndarray | None:
    """Return PLACE_ZONE center in Panda robot frame if it exists."""
    place_world = node_world(robot, "PLACE_ZONE")
    if place_world is None:
        return None
    return robot_frame(robot, place_world)


def infer_stage(
    attached: bool,
    pinch: float,
    hand_sign: str,
    gap: dict | None,
    release_cooldown: int,
) -> str:
    """Discrete action label used later by build_iql_transitions.py / IQL.

    The starter IQL files use current_record["stage"] as the action label.
    """
    if release_cooldown > 0:
        return "release"

    if attached:
        if gap is not None and gap.get("finger_gap", 0.0) > 0.015:
            return "lift_carry"
        return "hold_carry"

    if hand_sign == "NO_HAND":
        return "no_hand"

    if gap is None:
        return "teleop"

    horizontal = float(gap["horizontal"])
    finger_gap = float(gap["finger_gap"])

    if pinch >= PINCH_TO_GRASP:
        return "close_gripper"

    if horizontal <= GRASP_MAX_HORIZONTAL and GRASP_MIN_FINGER_GAP <= finger_gap <= GRASP_MAX_FINGER_GAP:
        return "grasp_pose"

    if finger_gap > 0.08:
        return "approach_high"

    if finger_gap > GRASP_MAX_FINGER_GAP:
        return "approach_lower"

    if finger_gap < GRASP_MIN_FINGER_GAP:
        return "recover_up"

    return "approach_align"


def compute_reward(
    robot: Supervisor,
    attached: bool,
    pinch: float,
    gap: dict | None,
) -> float:
    """Small shaped reward for starter offline RL.

    It rewards approaching, alignment, pinch at grasp pose, holding, and moving
    the box toward PLACE_ZONE when that node exists.
    """
    if gap is None:
        return -1.0

    horizontal = float(gap["horizontal"])
    finger_gap = float(gap["finger_gap"])
    distance = float(np.linalg.norm(gap["box"] - gap["gripper"]))

    reward = -distance
    reward -= 0.25 * abs(finger_gap)

    pose_ready = (
        horizontal <= GRASP_MAX_HORIZONTAL
        and GRASP_MIN_FINGER_GAP <= finger_gap <= GRASP_MAX_FINGER_GAP
    )
    if pose_ready:
        reward += 1.0

    if pinch >= PINCH_TO_GRASP and pose_ready:
        reward += 1.0

    if attached:
        reward += 2.0

    place = get_place_zone_robot(robot)
    box = box_robot(robot)
    if attached and place is not None and box is not None:
        place_dist = float(np.linalg.norm(box[:2] - place[:2]))
        reward += max(0.0, 1.0 - place_dist * 5.0)

    return float(reward)


class EpisodeDatasetRecorder:
    """Records Webots teleoperation demonstrations for DINOv2 + IQL.

    Output file:
        datasets/episodes/<episode_name>/episode_data.json

    Compatible with:
        extract_dinov2_features.py
        build_iql_transitions.py
        train_iql_starter.py
        plot_phases_trajectory.py
    """

    def __init__(self, robot: Supervisor, timestep: int, webots_camera=None):
        self.robot = robot
        self.timestep = timestep
        self.webots_camera = webots_camera
        self.project_root = get_project_root()
        self.datasets_dir = self.project_root / "datasets"
        self.episodes_dir = self.datasets_dir / "episodes"
        self.episode_name = "episode_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_dir = self.episodes_dir / self.episode_name
        self.images_dir = self.episode_dir / "images"
        self.episode_json_path = self.episode_dir / "episode_data.json"
        self.summary_json_path = self.episode_dir / "episode_summary.json"

        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.started_wall_time = time.time()
        self.steps = []
        self.step_id = 0
        self.last_flush_step_count = 0

        self.data = {
            "episode_name": self.episode_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "project_root": str(self.project_root),
            "controller_file": str(Path(__file__).resolve()),
            "timestep_ms": int(timestep),
            "save_every_n_frames": int(DATASET_SAVE_EVERY_N_FRAMES),
            "image_source": "webots_camera_if_available_else_webcam",
            "notes": (
                "MediaPipe servo-style teleoperation dataset for Franka Panda. "
                "State includes image_path, joints, gripper, cube, gripper/end-effector pose, "
                "pinch, hand sign, target, action_vector, reward, and done."
            ),
            "steps": self.steps,
        }

        self.flush()
        print(f"[DATASET] recording to: {self.episode_json_path}")

    def _save_webots_camera_image(self, image_path: Path) -> bool:
        if self.webots_camera is None:
            return False

        try:
            raw = self.webots_camera.getImage()
            width = int(self.webots_camera.getWidth())
            height = int(self.webots_camera.getHeight())

            if raw is None or width <= 0 or height <= 0:
                return False

            arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
            bgr = arr[:, :, :3]
            return bool(cv2.imwrite(str(image_path), bgr))
        except Exception as exc:
            print(f"[DATASET-WARN] Webots camera save failed: {exc}")
            return False

    def _save_webcam_image(self, image_path: Path, webcam_image: np.ndarray | None) -> bool:
        if webcam_image is None or not DATASET_SAVE_WEBCAM_FALLBACK:
            return False

        try:
            return bool(cv2.imwrite(str(image_path), webcam_image))
        except Exception as exc:
            print(f"[DATASET-WARN] Webcam fallback save failed: {exc}")
            return False

    def save_image(self, webcam_image: np.ndarray | None) -> str:
        filename = f"step_{self.step_id:04d}.{DATASET_IMAGE_FORMAT}"
        image_path = self.images_dir / filename

        saved = self._save_webots_camera_image(image_path)
        if not saved:
            saved = self._save_webcam_image(image_path, webcam_image)

        if not saved:
            return ""

        return str(image_path)

    def record(
        self,
        frame: int,
        q_actual: np.ndarray,
        q_command: np.ndarray,
        gripper_command: float,
        target_gripper: np.ndarray,
        target_ik: np.ndarray,
        servo_delta: np.ndarray,
        servo_debug: dict,
        pinch: float,
        hand_sign: str,
        attached: bool,
        align: float,
        pullback_active: bool,
        regrasp_armed: bool,
        release_cooldown: int,
        webcam_image: np.ndarray | None,
        done: bool = False,
    ) -> None:
        if not DATASET_RECORDING_ENABLE:
            return

        if not done and frame % DATASET_SAVE_EVERY_N_FRAMES != 0:
            return

        self.step_id += 1

        gap = compute_gap(self.robot)
        box_local = box_robot(self.robot)
        box_world = node_world(self.robot, "BOX")
        gripper_local = gripper_robot(self.robot)
        gripper_world = node_world(self.robot, "GRIPPER")
        place_local = get_place_zone_robot(self.robot)

        box_node = self.robot.getFromDef("BOX")
        cube_rotation = []
        if box_node is not None:
            cube_rotation = node_rotation(box_node) or []

        image_path = self.save_image(webcam_image)

        stage = infer_stage(attached, pinch, hand_sign, gap, release_cooldown)
        reward = compute_reward(self.robot, attached, pinch, gap)

        distance_gripper_cube = None
        if gripper_local is not None and box_local is not None:
            distance_gripper_cube = float(np.linalg.norm(gripper_local - box_local))

        step = {
            "step_id": int(self.step_id),
            "frame": int(frame),
            "sim_time": float(self.robot.getTime()),
            "wall_time_from_start": float(time.time() - self.started_wall_time),

            "image_path": image_path,

            "stage": stage,
            "action_name": stage,
            "action_vector": safe_list(servo_delta, 4),

            "reward": float(reward),
            "done": bool(done),
            "joint_positions": safe_list(q_actual, 7),
            "joint_commands": safe_list(q_command, 7),
            "gripper_positions": {
                "left_finger": float(gripper_command),
                "right_finger": float(gripper_command),
            },
            "cube_translation": safe_list(box_local, 3),

            "cube_world_position": safe_list(box_world, 3),
            "cube_rotation": safe_list(cube_rotation),
            "gripper_position": safe_list(gripper_local, 3),
            "gripper_world_position": safe_list(gripper_world, 3),
            "end_effector_position": safe_list(gripper_local, 3),
            "target_gripper_position": safe_list(target_gripper, 3),
            "target_ik_position": safe_list(target_ik, 3),
            "place_zone_position": safe_list(place_local, 3),

            "pinch": float(pinch),
            "hand_sign": str(hand_sign),
            "attached": bool(attached),
            "align_assist": float(align),
            "pullback_active": bool(pullback_active),
            "regrasp_armed": bool(regrasp_armed),
            "release_cooldown": int(release_cooldown),

            "horizontal_gap": None if gap is None else float(gap["horizontal"]),
            "finger_gap": None if gap is None else float(gap["finger_gap"]),
            "x_gap": None if gap is None else float(gap["x_gap"]),
            "y_gap": None if gap is None else float(gap["y_gap"]),
            "distance_gripper_cube": distance_gripper_cube,

            "servo_debug": {
                "depth": float(servo_debug.get("depth", 0.0)),
                "side": float(servo_debug.get("side", 0.0)),
                "vertical": float(servo_debug.get("vertical", 0.0)),
                "dx_img": float(servo_debug.get("dx_img", 0.0)),
                "dy_img": float(servo_debug.get("dy_img", 0.0)),
                "dz_img": float(servo_debug.get("dz_img", 0.0)),
            },
        }

        self.steps.append(step)

        if len(self.steps) - self.last_flush_step_count >= DATASET_FLUSH_EVERY_N_STEPS or done:
            self.flush()

        if self.step_id % DATASET_PRINT_EVERY_N_STEPS == 0 or done:
            print(
                f"[DATASET] saved_steps={self.step_id} | "
                f"stage={stage} | reward={reward:.3f} | "
                f"json={self.episode_json_path}"
            )

    def flush(self) -> None:
        tmp_path = self.episode_json_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        tmp_path.replace(self.episode_json_path)
        self.last_flush_step_count = len(self.steps)

    def finish(self) -> None:
        if not DATASET_RECORDING_ENABLE:
            return

        if self.steps:
            self.steps[-1]["done"] = True

        self.flush()

        summary = {
            "episode_name": self.episode_name,
            "episode_json": str(self.episode_json_path),
            "num_steps": len(self.steps),
            "images_dir": str(self.images_dir),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "stage_counts": {},
        }

        for step in self.steps:
            stage = step.get("stage", "unknown")
            summary["stage_counts"][stage] = summary["stage_counts"].get(stage, 0) + 1

        with open(self.summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[DATASET] finished episode: {self.episode_json_path}")
        print(f"[DATASET] total saved steps: {len(self.steps)}")



def draw_landmark_ids(image, landmarks) -> None:
    h, w = image.shape[:2]
    for idx, lm in enumerate(landmarks):
        x = int(float(lm.x) * w)
        y = int(float(lm.y) * h)
        cv2.circle(image, (x, y), 2, (255, 255, 255), -1)
        if DRAW_LANDMARK_IDS:
            cv2.putText(image, str(idx), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)


def draw_overlay(frame, lines: list[str]) -> None:
    y = 24
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20


def main() -> None:
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    print("[VERSION] dataset recorder + straight gripper pregrasp-side fix loaded")

    motors = [robot.getDevice(n) for n in JOINT_NAMES]
    sensors = [robot.getDevice(n) for n in SENSOR_NAMES]
    fingers = [robot.getDevice(n) for n in FINGER_NAMES]

    for sensor in sensors:
        try:
            sensor.enable(timestep)
        except Exception:
            pass

    for finger in fingers:
        try:
            finger.setVelocity(FINGER_VELOCITY)
        except Exception:
            pass
        try:
            finger.setAvailableForce(FINGER_FORCE)
        except Exception:
            pass

    webots_camera = None
    try:
        webots_camera = robot.getDevice("camera")
        webots_camera.enable(timestep)
        print("[DATASET] Webots wrist camera enabled for image recording.")
    except Exception:
        print("[DATASET-WARN] Webots camera not available; using webcam fallback for images.")

    recorder = EpisodeDatasetRecorder(robot, timestep, webots_camera)

    ws = WorkspaceConfig()
    ik_cfg = IKConfig()
    geo = PandaGeometry()
    servo_cfg = ServoRetargetConfig()
    # Faster, more visible side movement. This does not auto-pick/place;
    # it only makes the hand's left/right command more responsive.
    servo_cfg.max_step_y = 0.004
    servo_cfg.deadzone_x = 0.055
    servo_cfg.assist_gain_y = 0.30
    servo_cfg.assist_gain_y = min(float(servo_cfg.assist_gain_y), 0.30)

    q = HOME_POSE.copy()
    for motor, qi in zip(motors, q):
        motor.setPosition(float(qi))

    target_gripper = gripper_robot(robot)
    if target_gripper is None:
        target_gripper = forward_kinematics(q, geo).copy()

    fk_to_gripper_offset = np.zeros(3, dtype=float)

    neutral_wrist = None
    neutral_roll = None
    lost_frames = HAND_LOST_HOLD_FRAMES + 1

    attached = False
    hold_offset = None
    regrasp_armed = True
    release_cooldown = 0
    box_lock_world = node_world(robot, "BOX")
    if box_lock_world is not None:
        box_lock_world[2] = max(float(box_lock_world[2]), BOX_CENTER_MIN_Z_WORLD)

    box_node = robot.getFromDef("BOX")
    box_lock_rot = node_rotation(box_node) if box_node is not None else None
    good_counter = 0

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    frame = 0

    try:
        while robot.step(timestep) != -1:
            frame += 1

            q_actual = read_joint_positions(sensors, q)
            fk_to_gripper_offset = measured_gripper_offset(robot, q_actual, geo, fk_to_gripper_offset)

            ok, img = cap.read()
            if not ok:
                img = np.zeros((480, 640, 3), dtype=np.uint8)

            img = cv2.flip(img, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            pinch = 0.0
            align = 0.0
            hand_sign = "NO_HAND"
            servo_debug = {}
            servo_delta_robot = np.zeros(3, dtype=float)
            roll_cmd = 0.0
            pullback_active = False

            if result.multi_hand_landmarks is not None:
                lost_frames = 0
                hand_landmarks = result.multi_hand_landmarks[0]
                landmarks = hand_landmarks.landmark
                wrist = landmarks[0]

                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style(),
                )
                draw_landmark_ids(img, landmarks)

                if neutral_wrist is None:
                    neutral_wrist = np.array([wrist.x, wrist.y, wrist.z], dtype=float)
                    neutral_roll = hand_roll_angle(landmarks)
                    print(f"[CALIBRATE] frame={frame} | neutral hand set")

                pinch = pinch_ratio(landmarks[4], landmarks[8])
                hand_sign = classify_hand_sign(landmarks, pinch)

                delta, servo_debug = wrist_to_servo_delta(
                    wrist.x, wrist.y, wrist.z,
                    neutral_wrist[0], neutral_wrist[1], neutral_wrist[2],
                    servo_cfg,
                )

                # While holding, reduce downward commands. This prevents dragging
                # the box below the table after it was picked.
                if attached and delta[2] < 0.0:
                    delta[2] *= 0.25

                target_gripper = target_gripper + delta

                # Carrying sideways priority:
                # while holding, alignment is already disabled, so the remaining issue
                # is usually that IK/joint motion is too slow to visibly follow the side command.
                # Add a bounded extra Y step only while attached.
                if attached and SIDEWAYS_PRIORITY_FIX:
                    extra_side = float(np.clip(
                        CARRY_SIDE_BOOST * delta[1],
                        -CARRY_SIDE_MAX_EXTRA_STEP,
                        CARRY_SIDE_MAX_EXTRA_STEP,
                    ))
                    target_gripper[1] += extra_side
                    servo_delta_robot = delta.copy()
                    servo_delta_robot[1] += extra_side
                else:
                    servo_delta_robot = delta.copy()

                gap_now = compute_gap(robot)
                box = box_robot(robot)

                # Sideways/downward pre-grasp fix:
                # Before grasping, normal assist centers the gripper on the box.
                # But when the user is descending and intentionally moving sideways,
                # the assist should not cancel that side motion. In that case, keep a
                # small X/back-depth correction only and let Y/sideways follow the hand.
                side_override = should_allow_pregrasp_side_override(attached, gap_now, servo_debug)

                if (not attached) and gap_now is not None and box is not None and side_override:
                    target_gripper[0] += float(np.clip(
                        PREGRASP_SIDE_OVERRIDE_X_GAIN * gap_now["x_gap"],
                        -PREGRASP_SIDE_OVERRIDE_X_MAX_STEP,
                        PREGRASP_SIDE_OVERRIDE_X_MAX_STEP,
                    ))
                    align = 0.25
                elif (not attached) and gap_now is not None and box is not None:
                    target_gripper, align = apply_box_servo_assist(
                        target_gripper,
                        gap_now["x_gap"],
                        gap_now["y_gap"],
                        gap_now["horizontal"],
                        gap_now["finger_gap"],
                        box,
                        servo_cfg,
                    )
                elif attached:
                    align = 0.0

                if neutral_roll is not None:
                    roll_cmd = wrist_roll_command(hand_roll_angle(landmarks), neutral_roll, servo_cfg)
            else:
                lost_frames += 1
                if lost_frames > HAND_LOST_HOLD_FRAMES:
                    neutral_wrist = None
                    neutral_roll = None
                    pinch = 0.0
                    hand_sign = "NO_HAND"

            target_gripper = clamp_workspace(target_gripper, ws)
            target_gripper, pullback_active = apply_auto_pullback_lift(robot, target_gripper, attached)
            target_gripper = clamp_workspace(target_gripper, ws)
            target_gripper = clamp_gripper_safety(robot, target_gripper, attached)

            # Convert desired live GRIPPER point to analytic IK target.
            target_ik = clamp_workspace(target_gripper - fk_to_gripper_offset, ws)

            gripper_cmd = GRIPPER_OPEN - pinch * (GRIPPER_OPEN - GRIPPER_CLOSED)
            for finger in fingers:
                finger.setPosition(float(gripper_cmd))

            gap_for_posture = compute_gap(robot)
            wrist_alpha = soft_wrist_alpha_from_gap(gap_for_posture)

            # Keep the wrist straight near the object. Hand roll is useful for free
            # teleoperation, but near the box it makes the gripper hit the object.
            posture_target = HOME_POSE.copy()
            posture_target[4:7] = STRAIGHT_WRIST_POSE

            if gap_for_posture is not None and gap_for_posture["finger_gap"] > SOFT_WRIST_NEAR_FINGER_GAP:
                posture_target[6] = np.clip(HOME_POSE[6] + roll_cmd, JOINT_LIMITS[6, 0], JOINT_LIMITS[6, 1])

            posture_weight = 0.20
            if gap_for_posture is not None and gap_for_posture["finger_gap"] < SOFT_WRIST_NEAR_FINGER_GAP:
                posture_weight = 0.85

            q_solution = solve_panda_ik(
                target_ik,
                q_actual,
                JOINT_LIMITS,
                HOME_POSE,
                geo,
                ik_cfg,
                posture_target=posture_target,
                posture_weight=posture_weight,
                posture_mask=np.array([0, 0, 0, 0, 1, 1, 1], dtype=float),
            )

            # Soft straight-gripper correction:
            # Unlike the old hard override, this does not completely overwrite the
            # IK result. It blends the wrist toward the straight pose, stronger near
            # the box, so the fingers approach more vertically and are less likely
            # to hit/push the box sideways.
            if SOFT_WRIST_STRAIGHTEN_ENABLE:
                q_solution[4:7] = (
                    (1.0 - wrist_alpha) * q_solution[4:7]
                    + wrist_alpha * STRAIGHT_WRIST_POSE
                )

            if LOCK_WRIST_STRAIGHT:
                q_solution[4:7] = STRAIGHT_WRIST_POSE

            q = clamp_joint_step(q_solution, q, MAX_JOINT_STEP)
            q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

            for motor, qi in zip(motors, q):
                motor.setPosition(float(qi))

            if ENABLE_CONTACT_HOLD:
                attached, hold_offset, box_lock_world, box_lock_rot, good_counter, regrasp_armed, release_cooldown = update_contact_hold(
                    robot,
                    attached,
                    hold_offset,
                    box_lock_world,
                    box_lock_rot,
                    pinch,
                    good_counter,
                    frame,
                    regrasp_armed,
                    release_cooldown,
                )

            recorder.record(
                frame=frame,
                q_actual=q_actual,
                q_command=q,
                gripper_command=gripper_cmd,
                target_gripper=target_gripper,
                target_ik=target_ik,
                servo_delta=servo_delta_robot,
                servo_debug=servo_debug,
                pinch=pinch,
                hand_sign=hand_sign,
                attached=attached,
                align=align,
                pullback_active=pullback_active,
                regrasp_armed=regrasp_armed,
                release_cooldown=release_cooldown,
                webcam_image=img,
                done=False,
            )

            if frame % DEBUG_EVERY == 0:
                gap = compute_gap(robot)
                if gap is not None:
                    print(
                        f"[SERVO] frame={frame} | sign={hand_sign} | "
                        f"pinch={pinch:.2f} | align={align:.2f} | side_override={side_override if result.multi_hand_landmarks is not None else False} | wrist_alpha={wrist_alpha:.2f} | attached={attached} | pullback={pullback_active} | "
                        f"horizontal={gap['horizontal']*100:.1f}cm | "
                        f"finger_gap={gap['finger_gap']*100:.1f}cm | "
                        f"target_y={target_gripper[1]:.3f} | "
                        f"grip_y={gripper_robot(robot)[1] if gripper_robot(robot) is not None else 0.0:.3f} | "
                        f"delta=[{servo_debug.get('depth',0):.2f},{servo_debug.get('side',0):.2f},{servo_debug.get('vertical',0):.2f}]"
                    )

                if ik_is_stuck(target_ik, q_solution, JOINT_LIMITS, geo, ik_cfg):
                    print(f"[IK-STUCK] frame={frame} | joints={joints_at_limit(q_solution, JOINT_LIMITS)}")

            if SHOW_WEBCAM:
                draw_overlay(img, [
                    f"sign={hand_sign}",
                    f"pinch={pinch:.2f} align={align:.2f} attached={attached} armed={regrasp_armed} pullback={pullback_active}",
                    "landmarks shown: 0 wrist, 4 thumb, 8 index, 12 middle, 16 ring, 20 pinky",
                    "straight-grip + side descent + dataset | e=done | q=quit",
                ])
                cv2.imshow("Panda safe servo + hand landmarks", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    recorder.record(
                        frame=frame,
                        q_actual=q_actual,
                        q_command=q,
                        gripper_command=gripper_cmd,
                        target_gripper=target_gripper,
                        target_ik=target_ik,
                        servo_delta=servo_delta_robot,
                        servo_debug=servo_debug,
                        pinch=pinch,
                        hand_sign=hand_sign,
                        attached=attached,
                        align=align,
                        pullback_active=pullback_active,
                        regrasp_armed=regrasp_armed,
                        release_cooldown=release_cooldown,
                        webcam_image=img,
                        done=True,
                    )
                    break
                if key == ord("e"):
                    recorder.record(
                        frame=frame,
                        q_actual=q_actual,
                        q_command=q,
                        gripper_command=gripper_cmd,
                        target_gripper=target_gripper,
                        target_ik=target_ik,
                        servo_delta=servo_delta_robot,
                        servo_debug=servo_debug,
                        pinch=pinch,
                        hand_sign=hand_sign,
                        attached=attached,
                        align=align,
                        pullback_active=pullback_active,
                        regrasp_armed=regrasp_armed,
                        release_cooldown=release_cooldown,
                        webcam_image=img,
                        done=True,
                    )
                    print("[DATASET] episode marked done with key 'e'.")
                if key == ord("r"):
                    neutral_wrist = None
                    neutral_roll = None
                    print("[CALIBRATE] reset requested")
                if key == ord("h"):
                    q = HOME_POSE.copy()
                    for motor, qi in zip(motors, q):
                        motor.setPosition(float(qi))
                    target_gripper = gripper_robot(robot)
                    if target_gripper is None:
                        target_gripper = forward_kinematics(q, geo).copy()
                    print("[HOME] returned")
    finally:
        try:
            recorder.finish()
        except Exception as exc:
            print(f"[DATASET-WARN] recorder finish failed: {exc}")

        cap.release()
        try:
            hands.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()