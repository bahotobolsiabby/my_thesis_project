import json
from pathlib import Path
import numpy as np

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
FEATURES_ROOT = Path.home() / "dinov2_features_output"
OUTPUT_ROOT = Path.home() / "iql_transitions_output"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print(f"Reading features from: {FEATURES_ROOT}")
print(f"Saving transitions to: {OUTPUT_ROOT}")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def load_feature(feature_path: str):
    feature_path = Path(feature_path)
    arr = np.load(feature_path)

    # flatten just in case
    return arr.reshape(-1).tolist()

def build_state(record: dict):
    image_feature = load_feature(record["feature_path"])

    joint_positions = record.get("joint_positions") or []
    gripper_positions_dict = record.get("gripper_positions") or {}
    cube_translation = record.get("cube_translation") or []

    gripper_positions = [
        gripper_positions_dict.get("left_finger", 0.0),
        gripper_positions_dict.get("right_finger", 0.0),
    ]

    state = {
        "image_feature": image_feature,
        "joint_positions": joint_positions,
        "gripper_positions": gripper_positions,
        "cube_translation": cube_translation,
    }
    return state

def process_episode(episode_dir: Path):
    merged_json_path = episode_dir / "merged_feature_records.json"

    if not merged_json_path.exists():
        print(f"Skipping {episode_dir.name} (no merged_feature_records.json)")
        return None

    with open(merged_json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if len(records) < 2:
        print(f"Skipping {episode_dir.name} (not enough records)")
        return None

    transitions = []

    for i in range(len(records) - 1):
        current_record = records[i]
        next_record = records[i + 1]

        state = build_state(current_record)
        next_state = build_state(next_record)

        transition = {
            "episode_name": episode_dir.name,
            "step_id": current_record.get("step_id"),
            "next_step_id": next_record.get("step_id"),
            "action": current_record.get("stage"),   # simple starter action
            "reward": current_record.get("reward", 0.0),
            "done": current_record.get("done", False),
            "state": state,
            "next_state": next_state,
        }

        transitions.append(transition)

    output_path = OUTPUT_ROOT / f"{episode_dir.name}_transitions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transitions, f, indent=2)

    print(f"Saved {len(transitions)} transitions to: {output_path}")
    return transitions

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    if not FEATURES_ROOT.exists():
        print("Features root does not exist.")
        return

    episode_dirs = sorted([p for p in FEATURES_ROOT.iterdir() if p.is_dir()])

    if not episode_dirs:
        print("No episode folders found.")
        return

    total_transitions = 0

    for episode_dir in episode_dirs:
        transitions = process_episode(episode_dir)
        if transitions is not None:
            total_transitions += len(transitions)

    print(f"\nDone building IQL transitions.")
    print(f"Total transitions saved: {total_transitions}")

if __name__ == "__main__":
    main()