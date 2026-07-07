from controller import Supervisor
import os
import json
from datetime import datetime

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

print("Panda controller started")

# --------------------------------------------------
# Paths
# --------------------------------------------------
controller_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(controller_dir, "..", ".."))
datasets_dir = os.path.join(project_dir, "datasets")
episodes_dir = os.path.join(datasets_dir, "episodes")
images_root_dir = os.path.join(datasets_dir, "images")

os.makedirs(episodes_dir, exist_ok=True)
os.makedirs(images_root_dir, exist_ok=True)

episode_name = "episode_" + datetime.now().strftime("%Y%m%d_%H%M%S")
episode_dir = os.path.join(episodes_dir, episode_name)
episode_images_dir = os.path.join(images_root_dir, episode_name)

os.makedirs(episode_dir, exist_ok=True)
os.makedirs(episode_images_dir, exist_ok=True)

print(f"Saving episode to: {episode_dir}")

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
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    camera_available = True
    print("Camera found and enabled")
except Exception:
    print("No camera named 'camera' found, image logging will be skipped")

# --------------------------------------------------
# Cube
# --------------------------------------------------
box_node = robot.getFromDef("BOX")
if box_node is None:
    print("WARNING: Could not find DEF BOX.")
else:
    print("Found cube node with DEF BOX")

# --------------------------------------------------
# Helpers
# --------------------------------------------------
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

def save_camera_image(step_id):
    if not camera_available:
        return None
    filename = f"step_{step_id:04d}.png"
    filepath = os.path.join(episode_images_dir, filename)
    camera.saveImage(filepath, 100)
    return filepath

# --------------------------------------------------
# Poses
# Wrist orientation changed:
# joint5 = -1.57
# joint6 = 1.57
# joint7 = 0.0
# This usually makes the hand less sideways
# --------------------------------------------------
home_pose = [
    0.0,
    -0.4,
    0.0,
    -1.9,
    -1.57,
    1.57,
    0.0
]

approach_pose = [
    0.10,
    -1.10,
    0.0,
    -2.15,
    -1.57,
    1.57,
    0.0
]

grasp_pose = [
    0.10,
    -1.35,
    0.0,
    -2.45,
    -1.57,
    1.57,
    0.0
]

lift_pose = [
    0.10,
    -0.85,
    0.0,
    -2.00,
    -1.57,
    1.57,
    0.0
]

# --------------------------------------------------
# Logging
# --------------------------------------------------
episode_data = {
    "episode_name": episode_name,
    "timestep": timestep,
    "joint_names": joint_names,
    "steps": []
}

current_stage = "start"

def log_step(step_id):
    step_record = {
        "step_id": step_id,
        "stage": current_stage,
        "joint_positions": get_joint_positions(),
        "gripper_positions": get_gripper_positions(),
        "cube_translation": get_cube_translation(),
        "image_path": save_camera_image(step_id),
        "reward": 0.0,
        "done": False
    }
    episode_data["steps"].append(step_record)

# --------------------------------------------------
# Start
# --------------------------------------------------
set_arm_pose(home_pose)
open_gripper()

print("Moved to home pose")
print("Gripper opened")

step_count = 0
max_steps = 700

while robot.step(timestep) != -1:
    step_count += 1

    if step_count == 70:
        set_arm_pose(approach_pose)
        current_stage = "approach"
        print("Moved above cube")

    elif step_count == 170:
        set_arm_pose(grasp_pose)
        current_stage = "grasp_pose"
        print("Lowered to grasp pose")

    elif step_count == 320:
        close_gripper()
        current_stage = "close_gripper"
        print("Closed gripper")

    elif step_count == 470:
        set_arm_pose(lift_pose)
        current_stage = "lift"
        print("Lifted arm")

    elif step_count == 620:
        set_arm_pose(home_pose)
        current_stage = "return_home"
        print("Returned home")

    log_step(step_count)

    if step_count >= max_steps:
        if episode_data["steps"]:
            episode_data["steps"][-1]["done"] = True

        output_path = os.path.join(episode_dir, "episode_data.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(episode_data, f, indent=2)

        print(f"Episode saved to: {output_path}")
        print("Finished episode")
        break