import os
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_DIR / "datasets"
EPISODES_DIR = DATASETS_DIR / "episodes"
FEATURES_DIR = Path.home() / "dinov2_features_output"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "dinov2_vits14"   # smaller starter model

print(f"Using device: {DEVICE}")
print(f"Loading model: {MODEL_NAME}")

# --------------------------------------------------
# LOAD DINOv2 FROM TORCH HUB
# --------------------------------------------------
model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME, force_reload=True)
model = model.to(DEVICE)
model.eval()

# --------------------------------------------------
# IMAGE TRANSFORM
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

# --------------------------------------------------
# FEATURE EXTRACTION FUNCTION
# --------------------------------------------------
@torch.no_grad()
def extract_feature(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    features = model(image_tensor)

    if isinstance(features, torch.Tensor):
        features = features.squeeze(0).cpu().numpy()
    else:
        raise TypeError("Model output is not a tensor as expected.")

    return features

# --------------------------------------------------
# PROCESS ONE EPISODE
# --------------------------------------------------
def process_episode(episode_json_path: Path):
    print(f"\nProcessing: {episode_json_path}")

    with open(episode_json_path, "r", encoding="utf-8") as f:
        episode_data = json.load(f)

    episode_name = episode_data["episode_name"]
    output_dir = FEATURES_DIR / episode_name
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_records = []

    for step in episode_data["steps"]:
        image_path_str = step.get("image_path")

        if not image_path_str:
            print(f"Skipping step {step['step_id']} (no image_path)")
            continue

        image_path = Path(image_path_str)

        if not image_path.exists():
            print(f"Skipping step {step['step_id']} (missing image file: {image_path})")
            continue

        feature_vector = extract_feature(image_path)

        feature_file = output_dir / f"step_{step['step_id']:04d}.npy"
        np.save(feature_file, feature_vector)

        merged_record = {
            "step_id": step["step_id"],
            "stage": step.get("stage"),
            "image_path": str(image_path),
            "feature_path": str(feature_file),
            "feature_dim": int(feature_vector.shape[0]) if len(feature_vector.shape) == 1 else list(feature_vector.shape),
            "joint_positions": step.get("joint_positions"),
            "gripper_positions": step.get("gripper_positions"),
            "cube_translation": step.get("cube_translation"),
            "reward": step.get("reward"),
            "done": step.get("done"),
        }
        merged_records.append(merged_record)

    merged_output_path = output_dir / "merged_feature_records.json"
    with open(merged_output_path, "w", encoding="utf-8") as f:
        json.dump(merged_records, f, indent=2)

    print(f"Saved merged records to: {merged_output_path}")
    print(f"Saved {len(merged_records)} feature files.")

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    episode_json_files = sorted(EPISODES_DIR.glob("*/episode_data.json"))

    if not episode_json_files:
        print("No episode_data.json files found.")
        return

    for episode_json in episode_json_files:
        process_episode(episode_json)

    print("\nDone extracting DINOv2 features.")

if __name__ == "__main__":
    main()