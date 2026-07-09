from controller import Supervisor
import os
import json
from datetime import datetime
from panda_position_ik import solve_ik_position


robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

print("Panda MediaPipe every-joint controller started")

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

print(f"Saving manual joint-control episode to: {episode_dir}")
print(f"Reading MediaPipe gesture file from: {gesture_file}")

# --------------------------------------------------
# Control settings
# --------------------------------------------------
MAX_STEPS = 10000

# Smaller value = more responsive. Bigger value = less shaky.
DEAD_ZONE_X = 0.1

# How fast each selected joint moves when your hand is left/right of center.
# If movement is too slow, increase this to 0.012 or 0.015.
# If movement is too wild, decrease this to 0.006.
JOINT_SPEED = 0.020

PRINT_EVERY_STEPS = 50

USE_CUBE_IK = True
IK_SOLVE_INTERVAL = 10

ABOVE_CUBE_OFFSET = 0.12
GRASP_CUBE_OFFSET = 0.035
LIFT_CUBE_OFFSET = 0.18

PLACE_POSITION = [0.35, -0.25, 0.25]

# For practice, set SAVE_IMAGES = False to reduce lag.
# For dataset collection, set SAVE_IMAGES = True.
SAVE_IMAGES = True
IMAGE_SAVE_INTERVAL = 5

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

# Approximate safe limits for the Franka Panda joints in Webots.
joint_limits = [
    (-2.60, 2.60),   # J1
    (-1.70, 1.70),   # J2
    (-2.60, 2.60),   # J3
    (-3.00, -0.20),  # J4
    (-2.60, 2.60),   # J5
    (-0.10, 3.60),   # J6
    (-2.60, 2.60)    # J7
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
    camera = robot.getDevice("camera")
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

panda_node = robot.getFromDef("PANDA")

if panda_node is None:
    print("WARNING: Could not find DEF PANDA.")
else:
    print("Found Panda node with DEF PANDA")

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


def get_cube_position_relative_to_panda():
    if box_node is None or panda_node is None:
        return None

    cube_world = list(box_node.getPosition())
    panda_world = list(panda_node.getPosition())

    # Same idea as the IKPy tutorial:
    # x and y are swapped/inverted because the robot base is not aligned
    # with the Webots global axes.
    x = -(cube_world[1] - panda_world[1])
    y = cube_world[0] - panda_world[0]
    z = cube_world[2] - panda_world[2]

    return [x, y, z]

def get_cube_ik_targets():
    cube_pos = get_cube_position_relative_to_panda()

    if cube_pos is None:
        return None, None, None

    cube_x = cube_pos[0]
    cube_y = cube_pos[1]
    cube_z = cube_pos[2]

    above_cube = [
        cube_x,
        cube_y,
        cube_z + ABOVE_CUBE_OFFSET
    ]

    grasp_cube = [
        cube_x,
        cube_y,
        cube_z + GRASP_CUBE_OFFSET
    ]

    lift_cube = [
        cube_x,
        cube_y,
        cube_z + LIFT_CUBE_OFFSET
    ]

    return above_cube, grasp_cube, lift_cube


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
        "selected_joint": 1,
        "home_request": False,
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
            "selected_joint": data.get("selected_joint", 1),
            "home_request": data.get("home_request", False),
            "timestamp": data.get("timestamp", None)
        }

    except Exception:
        return default_data


def move_selected_joint(target_pose, selected_joint, hand_x):
    if hand_x is None:
        return target_pose

    joint_index = int(selected_joint) - 1

    if joint_index < 0 or joint_index >= len(target_pose):
        return target_pose

    center_x = 0.5
    dx = hand_x - center_x

    if abs(dx) < DEAD_ZONE_X:
        return target_pose

    # If hand is right of center, increase joint.
    # If hand is left of center, decrease joint.
    delta = dx * JOINT_SPEED * 2.0

    new_value = target_pose[joint_index] + delta

    min_limit, max_limit = joint_limits[joint_index]
    target_pose[joint_index] = clamp(new_value, min_limit, max_limit)

    return target_pose


# --------------------------------------------------
# Initial Panda pose
# --------------------------------------------------
home_pose = [
    0.0,    # J1 base rotation
    -0.4,   # J2 shoulder
    0.0,    # J3 elbow twist
    -1.9,   # J4 elbow bend
    -1.57,  # J5 wrist
    1.57,   # J6 wrist
    0.0     # J7 wrist rotation
]

target_pose = home_pose.copy()

# --------------------------------------------------
# Episode logging
# --------------------------------------------------
episode_data = {
    "episode_name": episode_name,
    "control_mode": "manual_mediapipe_every_joint_control",
    "timestep": timestep,
    "joint_names": joint_names,
    "joint_limits": joint_limits,
    "steps": []
}

current_stage = "manual_every_joint_control"

latest_mediapipe_data = {
    "detected": False,
    "gesture": "none",
    "pinch_distance": None,
    "hand_x": None,
    "hand_y": None,
    "hand_size": None,
    "selected_joint": 1,
    "home_request": False,
    "timestamp": None
}


def log_step(step_id):
    step_record = {
        "step_id": step_id,
        "stage": current_stage,

        "joint_positions": get_joint_positions(),
        "target_pose": target_pose.copy(),
        "selected_joint": latest_mediapipe_data.get("selected_joint", 1),

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
print("Cube-following IK control is active")
print("Controls:")
print(" - In MediaPipe window, press 1-7 to select Panda joint")
print(" - Move hand left/right to move the selected joint")
print(" - Pinch fingers to close gripper")
print(" - Open fingers to open gripper")
print(" - Press H in MediaPipe window to return to home pose")
print(" - Press Q in MediaPipe window to quit MediaPipe")

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
    selected_joint = latest_mediapipe_data.get("selected_joint", 1)
    home_request = latest_mediapipe_data.get("home_request", False)

    current_stage = "manual_every_joint_control"

    # --------------------------------------------------
    # Home pose request
    # --------------------------------------------------
    if home_request:
        target_pose = home_pose.copy()
        set_arm_pose(target_pose)
        print("Returned to home pose")

    # --------------------------------------------------
    # Gripper control
    # --------------------------------------------------
    if gesture == "close_gripper":
        close_gripper()

    elif gesture == "open_gripper":
        open_gripper()

        # --------------------------------------------------
    # Cube-following IK control OR manual joint control
    # --------------------------------------------------
    if USE_CUBE_IK:
        above_cube, grasp_cube, lift_cube = get_cube_ik_targets()

        ik_target = None

        if above_cube is not None:
            if step_count < 250:
                current_stage = "ik_move_above_cube"
                ik_target = above_cube
                open_gripper()

            elif step_count < 500:
                current_stage = "ik_lower_to_cube"
                ik_target = grasp_cube

            elif step_count < 650:
                current_stage = "close_gripper"
                ik_target = grasp_cube
                close_gripper()

            elif step_count < 900:
                current_stage = "ik_lift_cube"
                ik_target = lift_cube

            elif step_count < 1200:
                current_stage = "ik_move_to_place"
                ik_target = PLACE_POSITION

            elif step_count < 1400:
                current_stage = "open_gripper"
                ik_target = PLACE_POSITION
                open_gripper()

            else:
                current_stage = "finished"
                ik_target = PLACE_POSITION

            if ik_target is not None and step_count % IK_SOLVE_INTERVAL == 0:

                if step_count % 100 == 0:
                    print("Cube world:", get_cube_translation())
                    print("Cube relative IK:", get_cube_position_relative_to_panda())
                    print("IK target:", ik_target)

                target_pose, ik_info = solve_ik_position(
                    target_position=ik_target,
                    initial_q=target_pose,
                    max_iterations=10,
                    tolerance=0.040,
                    damping=0.15,
                    step_scale=0.20,
                    max_joint_step=0.025,
                )

                set_arm_pose(target_pose)

    else:
        if detected:
            target_pose = move_selected_joint(target_pose, selected_joint, hand_x)
            set_arm_pose(target_pose)

    # --------------------------------------------------
    # Console feedback
    # --------------------------------------------------
    if step_count % PRINT_EVERY_STEPS == 0:
        print(
            f"Step {step_count} | "
            f"Selected J{selected_joint} | "
            f"Gesture: {gesture} | "
            f"Detected: {detected} | "
            f"hand_x: {hand_x} | "
            f"hand_y: {hand_y} | "
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

        print(f"Manual joint-control episode saved to: {output_path}")
        print("Finished manual joint-control episode")
        break
