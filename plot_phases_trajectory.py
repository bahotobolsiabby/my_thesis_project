import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# CHANGE THIS TO YOUR EPISODE FILE
# ============================================================
EPISODE_JSON = r"C:\Users\my_thesis_project\datasets\episodes\episode_20260416_132634\episode_data.json"
OUTPUT_PNG = r"C:\Users\my_thesis_project\trajectory_phases.png"


# ============================================================
# HELPERS
# ============================================================
def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def is_xyz_list(value):
    return (
        isinstance(value, (list, tuple)) and
        len(value) == 3 and
        all(isinstance(x, (int, float)) for x in value)
    )


def get_value_by_possible_keys(d, possible_keys):
    for key in possible_keys:
        if isinstance(d, dict) and key in d:
            return d[key]
    return None


def extract_steps(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["steps", "records", "episode_steps", "data"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    return None


def extract_cube_position(step):
    # direct possible keys
    for key in [
        "cube_translation",
        "cube_position",
        "object_position",
        "box_position",
        "target_object_position"
    ]:
        value = step.get(key)
        if is_xyz_list(value):
            return value

    # nested options
    for parent in ["state", "observation", "robot_state", "environment_state"]:
        if parent in step and isinstance(step[parent], dict):
            nested = step[parent]
            for key in [
                "cube_translation",
                "cube_position",
                "object_position",
                "box_position"
            ]:
                value = nested.get(key)
                if is_xyz_list(value):
                    return value

    return None


def extract_action_name(step):
    for key in ["stage", "action", "action_name", "label", "phase"]:
        if key in step:
            return str(step[key])

    for parent in ["state", "observation", "robot_state"]:
        if parent in step and isinstance(step[parent], dict):
            nested = step[parent]
            for key in ["stage", "action", "action_name", "label", "phase"]:
                if key in nested:
                    return str(nested[key])

    return "unknown"


def extract_step_id(step, fallback_index):
    for key in ["step_id", "step", "timestep", "index"]:
        if key in step:
            return step[key]
    return fallback_index


def extract_gripper_position(step):
    """
    We try to find a 3D gripper/end-effector position.
    If your JSON does not store it, the graph cannot be computed exactly.
    """

    # direct keys
    for key in [
        "gripper_position",
        "end_effector_position",
        "ee_position",
        "tcp_position",
        "wrist_position"
    ]:
        value = step.get(key)
        if is_xyz_list(value):
            return value

    # nested dicts
    for parent in ["state", "observation", "robot_state", "environment_state"]:
        if parent in step and isinstance(step[parent], dict):
            nested = step[parent]
            for key in [
                "gripper_position",
                "end_effector_position",
                "ee_position",
                "tcp_position",
                "wrist_position"
            ]:
                value = nested.get(key)
                if is_xyz_list(value):
                    return value

    return None


def print_structure_preview(data):
    print("\n========== JSON STRUCTURE PREVIEW ==========")
    if isinstance(data, dict):
        print("Top-level keys:", list(data.keys()))
        for key, value in data.items():
            if isinstance(value, list):
                print(f"Key '{key}' is a list with length {len(value)}")
            else:
                print(f"Key '{key}' type: {type(value).__name__}")
    elif isinstance(data, list):
        print(f"Top-level is a list with length {len(data)}")
    else:
        print("Unsupported top-level JSON type:", type(data).__name__)
    print("===========================================\n")


def print_first_step_preview(step):
    print("========== FIRST STEP PREVIEW ==========")
    if isinstance(step, dict):
        for key, value in step.items():
            if isinstance(value, dict):
                print(f"{key}: dict with keys {list(value.keys())}")
            elif isinstance(value, list):
                preview = value[:5] if len(value) > 5 else value
                print(f"{key}: list(len={len(value)}) preview={preview}")
            else:
                print(f"{key}: {value}")
    print("========================================\n")


# ============================================================
# LOAD FILE
# ============================================================
episode_path = Path(EPISODE_JSON)

if not episode_path.exists():
    raise FileNotFoundError(f"Episode file not found: {episode_path}")

with open(episode_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print_structure_preview(data)

steps = extract_steps(data)
if steps is None:
    raise ValueError(
        "Could not find step records.\n"
        "Expected a top-level list or a dict with keys like 'steps' or 'records'."
    )

if len(steps) == 0:
    raise ValueError("No steps found in episode data.")

print_first_step_preview(steps[0])


# ============================================================
# EXTRACT DISTANCES
# ============================================================
step_ids = []
distances = []
actions = []

missing_gripper = 0
missing_cube = 0

for idx, step in enumerate(steps, start=1):
    step_id = extract_step_id(step, idx)
    action = extract_action_name(step)
    gripper_pos = extract_gripper_position(step)
    cube_pos = extract_cube_position(step)

    if gripper_pos is None:
        missing_gripper += 1
        continue

    if cube_pos is None:
        missing_cube += 1
        continue

    dist = euclidean_distance(gripper_pos, cube_pos)

    step_ids.append(step_id)
    distances.append(dist)
    actions.append(action)

print(f"Usable steps: {len(step_ids)}")
print(f"Skipped steps missing gripper position: {missing_gripper}")
print(f"Skipped steps missing cube position: {missing_cube}")

if len(step_ids) == 0:
    raise ValueError(
        "No usable steps found.\n"
        "Your JSON probably does not contain a 3D gripper/end-effector position.\n"
        "Please send me the first 2-3 step records from episode_data.json so I can tailor the script exactly."
    )


# ============================================================
# BUILD PHASE SPANS
# ============================================================
phase_colors = {
    "start": "#f2f2f2",
    "home_open": "#f2f2f2",
    "approach": "#d9edf7",
    "grasp_pose": "#d9edf7",
    "close_gripper": "#fff2cc",
    "lift": "#d5f5e3",
    "return_home": "#d5f5e3",
    "unknown": "#eeeeee"
}

phase_spans = []
current_action = actions[0]
span_start = 0

for i in range(1, len(actions)):
    if actions[i] != current_action:
        phase_spans.append((span_start, i - 1, current_action))
        span_start = i
        current_action = actions[i]
phase_spans.append((span_start, len(actions) - 1, current_action))


# ============================================================
# SPECIAL POINTS
# ============================================================
grasp_index = None
for i, act in enumerate(actions):
    if act == "close_gripper":
        if grasp_index is None or distances[i] < distances[grasp_index]:
            grasp_index = i

if grasp_index is None:
    grasp_index = distances.index(min(distances))

end_index = len(distances) - 1


# ============================================================
# PLOT
# ============================================================
plt.figure(figsize=(10, 5))

for start_i, end_i, phase in phase_spans:
    x0 = step_ids[start_i]
    x1 = step_ids[end_i]
    color = phase_colors.get(phase, "#eeeeee")
    plt.axvspan(x0, x1, color=color, alpha=0.5)

plt.plot(step_ids, distances, color="black", linewidth=1.8, label="Gripper-Object Distance")

plt.scatter(
    step_ids[grasp_index],
    distances[grasp_index],
    color="green",
    s=45,
    zorder=5,
    label="Grasp Point"
)

plt.scatter(
    step_ids[end_index],
    distances[end_index],
    color="red",
    s=45,
    zorder=5,
    label="End Point"
)

plt.xlabel("Step")
plt.ylabel("Distance (m)")
plt.title("Trajectory Phases - Sample Pick-and-Place Episode")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend(loc="best")
plt.tight_layout()

plt.savefig(OUTPUT_PNG, dpi=300)
plt.show()

print(f"Saved graph to: {OUTPUT_PNG}")