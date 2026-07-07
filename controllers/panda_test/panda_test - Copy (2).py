from controller import Supervisor
import os
import json
from datetime import datetime


robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

print("Panda manual MediaPipe controller with depth control started")

# --------------------------------------------------
# Paths
# --------------------------------------------------
controller_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(controller_dir, "..", ".."))

datasets_dir = os.path.join(project_dir, "datasets")
episodes_dir = os.path.join(datasets_dir, "episodes")
images_root_dir = os.path.join(datasets_dir, "images")

gesture_file = os.path.join(project_dir, "gesture_command.json")

os.makedirs(episodes_dir, exist_ok=True)
os.makedirs(images_root_dir, exist_ok=True)

episode_name = "episode_" + datetime.now().strftime("%Y%m%d_%H%M%S")
episode_dir = os.path.join(episodes_dir, episode_name)
episode_images_dir = os.path.join(images_root_dir, episode_name)

os.makedirs(episode_dir, exist_ok=True)
os.makedirs(episode_images_dir, exist_ok=True)

print(f"Saving manual episode to: {episode_dir}")
print(f"Reading MediaPipe gesture file from: {gesture_file}")

# --------------------------------------------------
# Manual control settings
# --------------------------------------------------
MAX_STEPS = 8000

# Bigger dead zone = less shaky, smaller dead zone = more responsive
DEAD_ZONE_XY = 0.045
DEAD_ZONE_DEPTH = 0.02

# Movement speeds
BASE_SPEED = 0.010
HEIGHT_SPEED = 0.006
REACH_SPEED = 0.012

PRINT_EVERY_STEPS = 50

# This is the "neutral" hand size.
# If your hand_size is bigger than this, the robot reaches forward.
# If your hand_size is smaller than this, the robot retracts backward.
# Adjust this based on the hand_size printed in the MediaPipe window.
NEUTRAL_HAND_SIZE = 0.34

# Joint safety limits
JOINT1_MIN = -1.80
JOINT1_MAX = 1.80

JOINT2_MIN = -1.60
JOINT2_MAX = -0.20

JOINT4_MIN = -2.80
JOINT4_MAX = -1.20

# --------------------------------------------------
# Panda joints
# --------------------------------------------------
joint_names = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7"
]

motors = []
joint_sensors = []

for name in joint_names:
    motor = robot.getDevice(name)
    sensor = motor.getPositionSensor()
    sensor.enable(timestep)

    motors.append(motor)
    joint_sensors.append(sensor)

    print(f"Found motor and sensor: {name}")

# --------------------------------------------------
# Gripper
# --------------------------------------------------
left_finger = robot.getDevice("panda_finger::left")
right_finger = robot.getDevice("panda_finger::right")

left_finger_sensor = left_finger.getPositionSensor()
right_finger_sensor = right_finger.getPositionSensor()

left_finger_sensor.enable(timestep)
right_finger_sensor.enable(timestep)

print("Found gripper fingers")

# --------------------------------------------------
# Camera
# --------------------------------------------------
camera = None
camera_available = False

try:
    camera = robot.getDevice("camera")  # change to "wrist_camera" if that is the camera name
    camera.enable(timestep)
    camera_available = True
    print("Camera found and enabled")
    print("Camera width:", camera.getWidth())
    print("Camera height:", camera.getHeight())
except Exception as e:
    print("No camera found, image logging will be skipped")
    print(e)

# --------------------------------------------------
# Cube
# --------------------------------------------------
box_node = robot.getFromDef("BOX")

if box_node is None:
    print("WARNING: Could not find DEF BOX.")
else:
    print("Found cube node with DEF BOX")

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def set_arm_pose(pose):
    for motor, target in zip(motors, pose):
        motor.setPosition(target)


def open_gripper():
    left_finger.setPosition(0.04)
    right_finger.setPosition(0.04)


def close_gripper():
    left_finger.setPosition(0.001)
    right_finger.setPosition(0.001)


def get_joint_positions():
    return [sensor.getValue() for sensor in joint_sensors]


def get_gripper_positions():
    return {
        "left_finger": left_finger_sensor.getValue(),
        "right_finger": right_finger_sensor.getValue()
    }


def get_cube_translation():
    if box_node is None:
        return None

    return list(box_node.getPosition())


SAVE_IMAGES = True
IMAGE_SAVE_INTERVAL = 10

def save_camera_image(step_id):
    if not camera_available:
        return None

    if not SAVE_IMAGES:
        return None

    if step_id % IMAGE_SAVE_INTERVAL != 0:
        return None

    filename = f"step_{step_id:04d}.png"
    filepath = os.path.join(episode_images_dir, filename)

    camera.saveImage(filepath, 100)

    return filepath

def read_mediapipe_command():
    default_data = {
        "detected": False,
        "gesture": "none",
        "pinch_distance": None,
        "hand_x": None,
        "hand_y": None,
        "hand_size": None,
        "timestamp": None
    }

    if not os.path.exists(gesture_file):
        return default_data

    try:
        with open(gesture_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "detected": data.get("detected", False),
            "gesture": data.get("gesture", "none"),
            "pinch_distance": data.get("pinch_distance", None),
            "hand_x": data.get("hand_x", None),
            "hand_y": data.get("hand_y", None),
            "hand_size": data.get("hand_size", None),
            "timestamp": data.get("timestamp", None)
        }

    except Exception:
        # This can happen if MediaPipe writes while Webots reads.
        return default_data


# --------------------------------------------------
# Initial Panda pose
# --------------------------------------------------
home_pose = [
    0.0,    # panda_joint1 - base rotation
    -0.4,   # panda_joint2 - shoulder / height
    0.0,    # panda_joint3
    -1.9,   # panda_joint4 - elbow / reach
    -1.57,  # panda_joint5 - wrist orientation
    1.57,   # panda_joint6 - wrist orientation
    0.0     # panda_joint7
]

target_pose = home_pose.copy()

# --------------------------------------------------
# Episode logging
# --------------------------------------------------
episode_data = {
    "episode_name": episode_name,
    "control_mode": "manual_mediapipe_teleoperation_with_depth",
    "timestep": timestep,
    "joint_names": joint_names,
    "steps": []
}

current_stage = "manual_mediapipe_control"

latest_mediapipe_data = {
    "detected": False,
    "gesture": "none",
    "pinch_distance": None,
    "hand_x": None,
    "hand_y": None,
    "hand_size": None,
    "timestamp": None
}


def log_step(step_id):
    step_record = {
        "step_id": step_id,
        "stage": current_stage,

        "joint_positions": get_joint_positions(),
        "target_pose": target_pose.copy(),
        "gripper_positions": get_gripper_positions(),
        "cube_translation": get_cube_translation(),
        "image_path": save_camera_image(step_id),

        "mediapipe_detected": latest_mediapipe_data.get("detected", False),
        "mediapipe_gesture": latest_mediapipe_data.get("gesture", "none"),
        "pinch_distance": latest_mediapipe_data.get("pinch_distance", None),
        "hand_x": latest_mediapipe_data.get("hand_x", None),
        "hand_y": latest_mediapipe_data.get("hand_y", None),
        "hand_size": latest_mediapipe_data.get("hand_size", None),
        "mediapipe_timestamp": latest_mediapipe_data.get("timestamp", None),

        "reward": 0.0,
        "done": False
    }

    episode_data["steps"].append(step_record)


# --------------------------------------------------
# Start robot
# --------------------------------------------------
set_arm_pose(target_pose)
open_gripper()

print("Moved to home pose")
print("Gripper opened")
print("Manual MediaPipe control with depth is active")
print("Controls:")
print(" - Move hand left/right             = rotate Panda base")
print(" - Move hand up/down                = raise/lower arm")
print(" - Move hand closer/farther camera  = reach forward/backward")
print(" - Pinch fingers                    = close gripper")
print(" - Open fingers                     = open gripper")
print(f"Neutral hand size is currently: {NEUTRAL_HAND_SIZE}")

step_count = 0

# --------------------------------------------------
# Main simulation loop
# --------------------------------------------------
while robot.step(timestep) != -1:
    step_count += 1

    latest_mediapipe_data = read_mediapipe_command()

    detected = latest_mediapipe_data.get("detected", False)
    gesture = latest_mediapipe_data.get("gesture", "none")
    hand_x = latest_mediapipe_data.get("hand_x", None)
    hand_y = latest_mediapipe_data.get("hand_y", None)
    hand_size = latest_mediapipe_data.get("hand_size", None)

    current_stage = "manual_mediapipe_control"

    # --------------------------------------------------
    # Gripper control
    # --------------------------------------------------
    if gesture == "close_gripper":
        close_gripper()

    elif gesture == "open_gripper":
        open_gripper()

    # --------------------------------------------------
    # Arm movement control
    # --------------------------------------------------
    if detected and hand_x is not None and hand_y is not None:
        center_x = 0.5
        center_y = 0.5

        dx = hand_x - center_x
        dy = hand_y - center_y

        # 1. Left/right controls base rotation
        if dx < -DEAD_ZONE_XY:
            target_pose[0] += BASE_SPEED
        elif dx > DEAD_ZONE_XY:
            target_pose[0] -= BASE_SPEED

        # 2. Up/down controls height
        if dy < -DEAD_ZONE_XY:
            # Hand goes up, arm raises
            target_pose[1] += HEIGHT_SPEED
            target_pose[3] += HEIGHT_SPEED

        elif dy > DEAD_ZONE_XY:
            # Hand goes down, arm lowers
            target_pose[1] -= HEIGHT_SPEED
            target_pose[3] -= HEIGHT_SPEED

        # 3. Hand size controls forward/backward reach
        if hand_size is not None:
            depth_error = hand_size - NEUTRAL_HAND_SIZE

            if depth_error > DEAD_ZONE_DEPTH:
                # Hand is closer to webcam, robot reaches forward/down toward cube
                target_pose[4] = -1.57
                target_pose[5] = 1.57
                target_pose[3] -= REACH_SPEED

            elif depth_error < -DEAD_ZONE_DEPTH:
                # Hand is farther from webcam, robot retracts backward/up
                target_pose[3] += REACH_SPEED

        # Apply safety limits
        target_pose[0] = clamp(target_pose[0], JOINT1_MIN, JOINT1_MAX)
        target_pose[1] = clamp(target_pose[1], JOINT2_MIN, JOINT2_MAX)
        target_pose[3] = clamp(target_pose[3], JOINT4_MIN, JOINT4_MAX)

        set_arm_pose(target_pose)

    # --------------------------------------------------
    # Console feedback
    # --------------------------------------------------
    if step_count % PRINT_EVERY_STEPS == 0:
        print(
            f"Step {step_count} | "
            f"Gesture: {gesture} | "
            f"Detected: {detected} | "
            f"hand_x: {hand_x} | "
            f"hand_y: {hand_y} | "
            f"hand_size: {hand_size} | "
            f"target_pose: {[round(x, 3) for x in target_pose]}"
        )

    log_step(step_count)

    # --------------------------------------------------
    # Save episode and stop
    # --------------------------------------------------
    if step_count >= MAX_STEPS:
        if episode_data["steps"]:
            episode_data["steps"][-1]["done"] = True

        output_path = os.path.join(episode_dir, "episode_data.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(episode_data, f, indent=2)

        print(f"Manual episode saved to: {output_path}")
        print("Finished manual MediaPipe episode")
        break
