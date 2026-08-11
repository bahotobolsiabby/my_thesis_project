"""
panda_test_manual_dataset.py

Franka Panda MANUAL MediaPipe hand-control dataset collector.
No inverse kinematics is used.

Control mode:
- MediaPipe writes gesture_command.json.
- Press 1-7 in the MediaPipe window to select a Panda joint.
- Move your hand left/right to move the selected joint.
- Pinch closes the gripper.
- Open fingers opens the gripper.
- H returns the Panda to the home pose.

Dataset behavior:
- One Webots controller run/reset = one episode.
- Wrist-camera images are saved asynchronously.
- Robot state, action, gripper state, cube position, target position,
  MediaPipe command, reward, success, and done are recorded.
- A successful place ends the episode automatically.
"""

from controller import Supervisor
from datetime import datetime

import cv2
import json
import math
import os
import queue
import threading
import time

import numpy as np


# ============================================================
# Webots
# ============================================================
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

print("Panda MANUAL MediaPipe dataset collector started")
print("No inverse kinematics is used.")


# ============================================================
# Paths
# ============================================================
controller_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(
    os.path.join(controller_dir, "..", "..")
)

gesture_file = os.path.join(
    project_dir,
    "gesture_command.json",
)

episodes_root = os.path.join(
    project_dir,
    "datasets",
    "episodes",
)

os.makedirs(episodes_root, exist_ok=True)

episode_name = (
    "episode_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)

episode_dir = os.path.join(
    episodes_root,
    episode_name,
)

episode_images_dir = os.path.join(
    episode_dir,
    "images",
)

os.makedirs(episode_dir, exist_ok=True)
os.makedirs(episode_images_dir, exist_ok=True)

episode_json_path = os.path.join(
    episode_dir,
    "episode_data.json",
)

print(f"Episode directory: {episode_dir}")
print(f"Reading MediaPipe command: {gesture_file}")


# ============================================================
# Control settings
# ============================================================
DEAD_ZONE_X = 0.10
JOINT_SPEED = 0.020
MOTOR_VELOCITY = 2.0

COMMAND_TIMEOUT_SECONDS = 0.35
COMMAND_POLL_SECONDS = 0.005

PRINT_EVERY_STEPS = 50

# Stop a failed/unfinished demonstration after this many simulation seconds.
MAX_EPISODE_SECONDS = 180.0


# ============================================================
# Dataset settings
# ============================================================
SAVE_IMAGES = True

# One dataset transition + one camera image every N Webots simulation steps.
# This keeps image/state/action alignment simple.
RECORD_INTERVAL_STEPS = 5

IMAGE_WRITER_QUEUE_SIZE = 24
PNG_COMPRESSION = 3

# Optional live wrist-camera preview. False reduces CPU/display overhead.
SHOW_WRIST_CAMERA_PREVIEW = False
PREVIEW_INTERVAL_STEPS = 2

# Set these to match how you want the SAVED wrist-camera dataset images.
CAMERA_ROTATE_180 = True
CAMERA_FLIP_HORIZONTAL = True
CAMERA_FLIP_VERTICAL = False


# ============================================================
# Pick-and-place target settings
# ============================================================
BOX_DEF = "BOX"
TARGET_AREA_DEF = "TARGET_AREA"

# Must match the visible target marker in the .wbt world.
TARGET_SIZE_X = 0.18
TARGET_SIZE_Y = 0.18

# Cube must be low enough and the gripper must be open.
TARGET_MAX_HEIGHT_ABOVE_MARKER = 0.10
GRIPPER_RELEASE_THRESHOLD = 0.020

# Prevent a cube that only passes briefly over the marker from counting.
SUCCESS_HOLD_STEPS = 15


# ============================================================
# Panda joints
# ============================================================
joint_names = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

joint_limits = [
    (-2.60, 2.60),   # J1
    (-1.70, 1.70),   # J2
    (-2.60, 2.60),   # J3
    (-3.00, -0.20),  # J4
    (-2.60, 2.60),   # J5
    (-0.10, 3.60),   # J6
    (-2.60, 2.60),   # J7
]

home_pose = [
    0.0,
    -0.4,
    0.0,
    -1.9,
    -1.57,
    1.57,
    0.0,
]


# ============================================================
# Helpers
# ============================================================
def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def default_command():
    return {
        "detected": False,
        "gesture": "none",
        "pinch_distance": None,
        "hand_x": None,
        "hand_y": None,
        "hand_size": None,
        "selected_joint": 1,
        "home_request": False,
        "timestamp": 0.0,
        "frame_id": -1,
        "capture_timestamp": 0.0,
        "pipeline_latency_ms": None,
    }


# ============================================================
# Background MediaPipe JSON reader
# ============================================================
class CommandReaderThread(threading.Thread):
    def __init__(self, command_path):
        super().__init__(
            name="CommandReaderThread",
            daemon=True,
        )
        self.command_path = command_path
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.latest_command = default_command()
        self.last_modified_ns = None

    def run(self):
        while not self.stop_event.is_set():
            try:
                modified_ns = os.stat(
                    self.command_path
                ).st_mtime_ns

                if modified_ns != self.last_modified_ns:
                    with open(
                        self.command_path,
                        "r",
                        encoding="utf-8",
                    ) as file:
                        command = json.load(file)

                    with self.lock:
                        self.latest_command = {
                            **default_command(),
                            **command,
                        }

                    self.last_modified_ns = modified_ns

            except (
                FileNotFoundError,
                PermissionError,
                OSError,
                json.JSONDecodeError,
            ):
                pass

            self.stop_event.wait(
                COMMAND_POLL_SECONDS
            )

    def get_latest(self):
        with self.lock:
            return dict(self.latest_command)

    def stop(self):
        self.stop_event.set()


# ============================================================
# Background image writer
# ============================================================
class AsyncImageWriter:
    def __init__(self):
        self.jobs = queue.Queue(
            maxsize=IMAGE_WRITER_QUEUE_SIZE
        )
        self.dropped_images = 0

        self.thread = threading.Thread(
            target=self._worker,
            name="DatasetImageWriter",
            daemon=True,
        )
        self.thread.start()

    def enqueue(
        self,
        filepath,
        image_bytes,
        width,
        height,
    ):
        try:
            self.jobs.put_nowait(
                (
                    filepath,
                    image_bytes,
                    width,
                    height,
                )
            )
            return True
        except queue.Full:
            self.dropped_images += 1
            return False

    def _worker(self):
        while True:
            item = self.jobs.get()

            if item is None:
                self.jobs.task_done()
                break

            filepath, image_bytes, width, height = item

            try:
                bgra = np.frombuffer(
                    image_bytes,
                    dtype=np.uint8,
                ).reshape((height, width, 4))

                frame = cv2.cvtColor(
                    bgra,
                    cv2.COLOR_BGRA2BGR,
                )

                if CAMERA_ROTATE_180:
                    frame = cv2.rotate(
                        frame,
                        cv2.ROTATE_180,
                    )

                if CAMERA_FLIP_HORIZONTAL:
                    frame = cv2.flip(frame, 1)

                if CAMERA_FLIP_VERTICAL:
                    frame = cv2.flip(frame, 0)

                cv2.imwrite(
                    filepath,
                    frame,
                    [
                        cv2.IMWRITE_PNG_COMPRESSION,
                        PNG_COMPRESSION,
                    ],
                )

            except Exception as error:
                print(
                    "Background image-write error:",
                    error,
                )

            finally:
                self.jobs.task_done()

    def finish(self):
        self.jobs.join()
        self.jobs.put(None)
        self.thread.join(timeout=5.0)


# ============================================================
# Motors and sensors
# ============================================================
motors = []
joint_sensors = []

for name in joint_names:
    motor = robot.getDevice(name)
    motor.setVelocity(MOTOR_VELOCITY)

    sensor = motor.getPositionSensor()
    sensor.enable(timestep)

    motors.append(motor)
    joint_sensors.append(sensor)

    print(f"Found motor/sensor: {name}")

left_finger = robot.getDevice(
    "panda_finger::left"
)
right_finger = robot.getDevice(
    "panda_finger::right"
)

left_finger.setVelocity(0.12)
right_finger.setVelocity(0.12)

left_finger_sensor = left_finger.getPositionSensor()
right_finger_sensor = right_finger.getPositionSensor()

left_finger_sensor.enable(timestep)
right_finger_sensor.enable(timestep)


def set_arm_pose(pose):
    for motor, target in zip(
        motors,
        pose,
    ):
        motor.setPosition(target)


def open_gripper():
    # Avoids the recurring 0.04 > 0.04 warning.
    left_finger.setPosition(0.039)
    right_finger.setPosition(0.039)


def close_gripper():
    left_finger.setPosition(0.001)
    right_finger.setPosition(0.001)


def get_joint_positions():
    return [
        float(sensor.getValue())
        for sensor in joint_sensors
    ]


def get_gripper_positions():
    return {
        "left": float(
            left_finger_sensor.getValue()
        ),
        "right": float(
            right_finger_sensor.getValue()
        ),
    }


# ============================================================
# Wrist camera
# ============================================================
camera = None
camera_available = False

try:
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    camera_available = True

    print(
        "Wrist camera enabled:",
        camera.getWidth(),
        "x",
        camera.getHeight(),
    )

except Exception as error:
    print(
        "WARNING: Wrist camera not available:",
        error,
    )


def corrected_camera_frame_from_bytes(
    image_bytes,
    width,
    height,
):
    bgra = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    ).reshape((height, width, 4))

    frame = cv2.cvtColor(
        bgra,
        cv2.COLOR_BGRA2BGR,
    )

    if CAMERA_ROTATE_180:
        frame = cv2.rotate(
            frame,
            cv2.ROTATE_180,
        )

    if CAMERA_FLIP_HORIZONTAL:
        frame = cv2.flip(frame, 1)

    if CAMERA_FLIP_VERTICAL:
        frame = cv2.flip(frame, 0)

    return frame


def show_camera_preview(step_count):
    if (
        not SHOW_WRIST_CAMERA_PREVIEW
        or not camera_available
        or step_count % PREVIEW_INTERVAL_STEPS != 0
    ):
        return

    image = camera.getImage()

    if image is None:
        return

    frame = corrected_camera_frame_from_bytes(
        image,
        camera.getWidth(),
        camera.getHeight(),
    )

    cv2.imshow(
        "Panda Wrist Camera - Dataset Preview",
        frame,
    )
    cv2.waitKey(1)


# ============================================================
# Cube and target area
# ============================================================
box_node = robot.getFromDef(BOX_DEF)
target_area_node = robot.getFromDef(
    TARGET_AREA_DEF
)

if box_node is None:
    print(
        f"WARNING: Could not find DEF {BOX_DEF}."
    )
else:
    print(f"Found cube DEF {BOX_DEF}")

if target_area_node is None:
    print(
        f"WARNING: Could not find DEF {TARGET_AREA_DEF}. "
        "Dataset will record, but target success cannot be labelled."
    )
else:
    print(
        f"Found target area DEF {TARGET_AREA_DEF}"
    )


def get_world_position(node):
    if node is None:
        return None

    try:
        return [
            float(value)
            for value in node.getPosition()
        ]
    except Exception:
        return None


def target_status():
    cube_position = get_world_position(
        box_node
    )
    target_position = get_world_position(
        target_area_node
    )

    status = {
        "cube_position": cube_position,
        "target_position": target_position,
        "distance_xy": None,
        "inside_target_xy": False,
        "cube_low_enough": False,
        "gripper_released": False,
        "success_now": False,
    }

    if (
        cube_position is None
        or target_position is None
    ):
        return status

    dx = (
        cube_position[0]
        - target_position[0]
    )
    dy = (
        cube_position[1]
        - target_position[1]
    )

    distance_xy = math.sqrt(
        dx * dx + dy * dy
    )

    inside_target_xy = (
        abs(dx) <= TARGET_SIZE_X / 2.0
        and abs(dy) <= TARGET_SIZE_Y / 2.0
    )

    cube_low_enough = (
        cube_position[2]
        <= target_position[2]
        + TARGET_MAX_HEIGHT_ABOVE_MARKER
    )

    gripper = get_gripper_positions()

    gripper_released = (
        gripper["left"]
        >= GRIPPER_RELEASE_THRESHOLD
        and gripper["right"]
        >= GRIPPER_RELEASE_THRESHOLD
    )

    success_now = (
        inside_target_xy
        and cube_low_enough
        and gripper_released
    )

    status.update(
        {
            "distance_xy": float(
                distance_xy
            ),
            "inside_target_xy": bool(
                inside_target_xy
            ),
            "cube_low_enough": bool(
                cube_low_enough
            ),
            "gripper_released": bool(
                gripper_released
            ),
            "success_now": bool(
                success_now
            ),
        }
    )

    return status


# ============================================================
# Manual selected-joint control
# ============================================================
def move_selected_joint(
    current_target_pose,
    selected_joint,
    hand_x,
):
    """
    Returns:
        updated_pose
        joint_action_delta[7]
    """

    updated_pose = current_target_pose.copy()
    action_delta = [0.0] * 7

    if hand_x is None:
        return updated_pose, action_delta

    joint_index = int(selected_joint) - 1

    if (
        joint_index < 0
        or joint_index >= len(updated_pose)
    ):
        return updated_pose, action_delta

    dx = hand_x - 0.5

    if abs(dx) < DEAD_ZONE_X:
        return updated_pose, action_delta

    delta = dx * JOINT_SPEED * 2.0

    minimum, maximum = joint_limits[
        joint_index
    ]

    previous_value = updated_pose[
        joint_index
    ]

    updated_pose[joint_index] = clamp(
        previous_value + delta,
        minimum,
        maximum,
    )

    action_delta[joint_index] = (
        updated_pose[joint_index]
        - previous_value
    )

    return updated_pose, action_delta


# ============================================================
# Dataset helpers
# ============================================================
episode_records = []
record_index = 0

target_pose = home_pose.copy()

# Accumulates all little joint changes between saved image/state frames.
accumulated_joint_action = [0.0] * 7

# 1 = open, -1 = close, 0 = unchanged/unknown.
latest_gripper_action = 1

success_hold_count = 0
episode_success = False

episode_start_sim_time = None


def save_aligned_record(
    step_count,
    latest_command,
    image_writer,
    status,
):
    global record_index
    global accumulated_joint_action

    record_index += 1

    image_relative_path = None

    if camera_available and SAVE_IMAGES:
        image = camera.getImage()

        if image is not None:
            image_filename = (
                f"frame_{record_index:06d}.png"
            )

            image_absolute_path = os.path.join(
                episode_images_dir,
                image_filename,
            )

            accepted = image_writer.enqueue(
                image_absolute_path,
                bytes(image),
                camera.getWidth(),
                camera.getHeight(),
            )

            if accepted:
                image_relative_path = (
                    "images/" + image_filename
                )

    record = {
        "record_index": record_index,
        "webots_step": step_count,
        "simulation_time_s": float(
            robot.getTime()
        ),

        # Visual observation
        "image_path": image_relative_path,

        # Robot state
        "joint_positions": get_joint_positions(),
        "target_pose": [
            float(value)
            for value in target_pose
        ],
        "gripper_positions": (
            get_gripper_positions()
        ),

        # Explicit action between this record and
        # the previous saved record.
        "action": {
            "joint_delta": [
                float(value)
                for value
                in accumulated_joint_action
            ],
            "gripper": int(
                latest_gripper_action
            ),
            "selected_joint": int(
                latest_command.get(
                    "selected_joint",
                    1,
                )
            ),
        },

        # Task state
        "cube_position": status[
            "cube_position"
        ],
        "target_position": status[
            "target_position"
        ],
        "distance_to_target_xy": status[
            "distance_xy"
        ],
        "inside_target_xy": status[
            "inside_target_xy"
        ],
        "cube_low_enough": status[
            "cube_low_enough"
        ],
        "gripper_released": status[
            "gripper_released"
        ],

        # Sparse task reward
        "reward": (
            1.0
            if status["success_now"]
            else 0.0
        ),
        "success": bool(
            status["success_now"]
        ),
        "done": False,

        # Human command metadata
        "mediapipe": {
            "detected": bool(
                latest_command.get(
                    "detected",
                    False,
                )
            ),
            "gesture": latest_command.get(
                "gesture",
                "none",
            ),
            "hand_x": latest_command.get(
                "hand_x"
            ),
            "hand_y": latest_command.get(
                "hand_y"
            ),
            "hand_size": latest_command.get(
                "hand_size"
            ),
            "pinch_distance": (
                latest_command.get(
                    "pinch_distance"
                )
            ),
            "frame_id": latest_command.get(
                "frame_id"
            ),
            "capture_timestamp": (
                latest_command.get(
                    "capture_timestamp"
                )
            ),
            "processed_timestamp": (
                latest_command.get(
                    "timestamp"
                )
            ),
            "pipeline_latency_ms": (
                latest_command.get(
                    "pipeline_latency_ms"
                )
            ),
        },
    }

    episode_records.append(record)

    accumulated_joint_action = [0.0] * 7


def save_episode_json(
    image_writer,
    terminal_reason,
):
    if episode_records:
        episode_records[-1]["done"] = True

    payload = {
        "metadata": {
            "episode_name": episode_name,
            "control_mode": (
                "manual_mediapipe_selected_joint"
            ),
            "inverse_kinematics": False,
            "success": bool(
                episode_success
            ),
            "terminal_reason": terminal_reason,

            "webots_timestep_ms": timestep,

            "joint_names": joint_names,
            "joint_limits": joint_limits,

            "box_def": BOX_DEF,
            "target_area_def": TARGET_AREA_DEF,
            "target_size_xy_m": [
                TARGET_SIZE_X,
                TARGET_SIZE_Y,
            ],

            "record_interval_steps": (
                RECORD_INTERVAL_STEPS
            ),

            "camera_enabled": bool(
                camera_available
            ),
            "camera_resolution": (
                [
                    camera.getWidth(),
                    camera.getHeight(),
                ]
                if camera_available
                else None
            ),
            "camera_rotate_180": (
                CAMERA_ROTATE_180
            ),
            "camera_flip_horizontal": (
                CAMERA_FLIP_HORIZONTAL
            ),
            "camera_flip_vertical": (
                CAMERA_FLIP_VERTICAL
            ),

            "number_of_records": len(
                episode_records
            ),
            "dropped_images": (
                image_writer.dropped_images
            ),
        },

        "steps": episode_records,
    }

    with open(
        episode_json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    print("")
    print("========== EPISODE SAVED ==========")
    print(f"Path: {episode_json_path}")
    print(f"Success: {episode_success}")
    print(
        f"Terminal reason: {terminal_reason}"
    )
    print(
        f"Records: {len(episode_records)}"
    )
    print(
        "Dropped camera images:",
        image_writer.dropped_images,
    )
    print("===================================")


# ============================================================
# Start
# ============================================================
command_reader = CommandReaderThread(
    gesture_file
)
image_writer = AsyncImageWriter()

command_reader.start()

set_arm_pose(target_pose)
open_gripper()

step_count = 0
previous_home_request = False
terminal_reason = "controller_stopped"

episode_start_sim_time = float(
    robot.getTime()
)

print("")
print("Dataset recording is ACTIVE.")
print(
    "One Webots controller run/reset "
    "= one demonstration episode."
)
print(
    f"Saving one image + transition every "
    f"{RECORD_INTERVAL_STEPS} Webots steps."
)
print(
    "Successful placement inside TARGET_AREA "
    "will end the episode automatically."
)
print("")

try:
    while robot.step(timestep) != -1:
        step_count += 1

        show_camera_preview(step_count)

        latest_command = (
            command_reader.get_latest()
        )

        timestamp = latest_command.get(
            "timestamp",
            0.0,
        )

        command_age = (
            time.time() - timestamp
            if isinstance(
                timestamp,
                (int, float),
            )
            else float("inf")
        )

        command_is_fresh = (
            0.0
            <= command_age
            <= COMMAND_TIMEOUT_SECONDS
        )

        detected = (
            bool(
                latest_command.get(
                    "detected",
                    False,
                )
            )
            and command_is_fresh
        )

        gesture = (
            latest_command.get(
                "gesture",
                "none",
            )
            if command_is_fresh
            else "none"
        )

        hand_x = (
            latest_command.get(
                "hand_x"
            )
            if command_is_fresh
            else None
        )

        selected_joint = int(
            latest_command.get(
                "selected_joint",
                1,
            )
        )

        home_request = (
            bool(
                latest_command.get(
                    "home_request",
                    False,
                )
            )
            and command_is_fresh
        )

        # ----------------------------
        # Home request
        # ----------------------------
        if (
            home_request
            and not previous_home_request
        ):
            target_pose = home_pose.copy()
            set_arm_pose(target_pose)

            # Treat the move-to-home command as the
            # difference from the previous target.
            # We accumulate it into the next saved action.
            print("Returned to home pose")

        previous_home_request = (
            home_request
        )

        # ----------------------------
        # Gripper
        # ----------------------------
        if gesture == "close_gripper":
            close_gripper()
            latest_gripper_action = -1

        elif gesture == "open_gripper":
            open_gripper()
            latest_gripper_action = 1

        # ----------------------------
        # Manual selected-joint action
        # ----------------------------
        if detected:
            (
                updated_target_pose,
                action_delta,
            ) = move_selected_joint(
                target_pose,
                selected_joint,
                hand_x,
            )

            target_pose = updated_target_pose
            set_arm_pose(target_pose)

            for index in range(7):
                accumulated_joint_action[
                    index
                ] += action_delta[index]

        # ----------------------------
        # Task success
        # ----------------------------
        status = target_status()

        if status["success_now"]:
            success_hold_count += 1
        else:
            success_hold_count = 0

        if (
            success_hold_count
            >= SUCCESS_HOLD_STEPS
        ):
            episode_success = True

        # ----------------------------
        # Aligned dataset record
        # ----------------------------
        if (
            step_count
            % RECORD_INTERVAL_STEPS
            == 0
        ):
            save_aligned_record(
                step_count,
                latest_command,
                image_writer,
                status,
            )

        # ----------------------------
        # Console
        # ----------------------------
        if (
            step_count
            % PRINT_EVERY_STEPS
            == 0
        ):
            print(
                f"Step {step_count} | "
                f"J{selected_joint} | "
                f"fresh={command_is_fresh} | "
                f"age={command_age * 1000.0:.1f} ms | "
                f"detected={detected} | "
                f"gesture={gesture} | "
                f"distance_to_target_xy="
                f"{status['distance_xy']} | "
                f"success_hold="
                f"{success_hold_count} | "
                f"records={record_index} | "
                f"dropped_images="
                f"{image_writer.dropped_images}"
            )

        # ----------------------------
        # Successful placement ends episode
        # ----------------------------
        if episode_success:
            terminal_reason = (
                "successful_placement"
            )

            print(
                "SUCCESS: Cube released inside "
                "TARGET_AREA."
            )

            # Guarantee a terminal record at the
            # exact final state if current step was
            # not already recorded.
            if (
                step_count
                % RECORD_INTERVAL_STEPS
                != 0
            ):
                save_aligned_record(
                    step_count,
                    latest_command,
                    image_writer,
                    status,
                )

            break

        # ----------------------------
        # Timeout
        # ----------------------------
        elapsed_episode_time = (
            float(robot.getTime())
            - episode_start_sim_time
        )

        if (
            elapsed_episode_time
            >= MAX_EPISODE_SECONDS
        ):
            terminal_reason = "timeout"

            print(
                "Episode timed out before "
                "successful placement."
            )
            break

finally:
    command_reader.stop()
    command_reader.join(timeout=2.0)

    image_writer.finish()

    save_episode_json(
        image_writer,
        terminal_reason,
    )

    cv2.destroyAllWindows()

    print(
        "Dataset controller stopped cleanly."
    )
