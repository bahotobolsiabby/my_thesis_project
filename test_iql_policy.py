import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
TRANSITIONS_ROOT = Path.home() / "iql_transitions_output"
MODEL_ROOT = Path.home() / "iql_model_output"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print(f"Reading transitions from: {TRANSITIONS_ROOT}")
print(f"Reading model from: {MODEL_ROOT}")

# --------------------------------------------------
# LOAD ACTION MAPPING
# --------------------------------------------------
action_mapping_path = MODEL_ROOT / "action_mapping.json"
with open(action_mapping_path, "r", encoding="utf-8") as f:
    action_to_id = json.load(f)

id_to_action = {v: k for k, v in action_to_id.items()}

# --------------------------------------------------
# LOAD TRAINING INFO
# --------------------------------------------------
training_info_path = MODEL_ROOT / "training_info.json"
with open(training_info_path, "r", encoding="utf-8") as f:
    training_info = json.load(f)

state_dim = training_info["state_dim"]
num_actions = training_info["num_actions"]
hidden_size = 256  # same as training script

print(f"State dim: {state_dim}")
print(f"Num actions: {num_actions}")

# --------------------------------------------------
# SAME NETWORK AS TRAINING
# --------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)

policy_net = MLP(state_dim, num_actions, hidden=hidden_size).to(DEVICE)
policy_net.load_state_dict(torch.load(MODEL_ROOT / "policy_net.pt", map_location=DEVICE))
policy_net.eval()

print("Loaded policy network successfully.")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def flatten_state(state_dict):
    image_feature = state_dict.get("image_feature", [])
    joint_positions = state_dict.get("joint_positions", [])
    gripper_positions = state_dict.get("gripper_positions", [])
    cube_translation = state_dict.get("cube_translation", [])

    vec = np.array(
        list(image_feature) +
        list(joint_positions) +
        list(gripper_positions) +
        list(cube_translation),
        dtype=np.float32
    )
    return vec

# --------------------------------------------------
# PICK ONE TRANSITION FILE
# --------------------------------------------------
transition_files = sorted(TRANSITIONS_ROOT.glob("*_transitions.json"))

if not transition_files:
    raise RuntimeError("No transition files found in iql_transitions_output.")

test_file = transition_files[-1]  # newest by sorted name
print(f"Using transition file: {test_file}")

with open(test_file, "r", encoding="utf-8") as f:
    transitions = json.load(f)

if not transitions:
    raise RuntimeError("Transition file is empty.")

# --------------------------------------------------
# TEST A FEW STATES
# --------------------------------------------------
num_examples = min(5, len(transitions))

for i in range(num_examples):
    item = transitions[i]

    state_vec = flatten_state(item["state"])
    state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = policy_net(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        pred_id = int(torch.argmax(probs, dim=-1).item())
        pred_action = id_to_action[pred_id]

    true_action = item["action"]

    print("\n----------------------------------------")
    print(f"Example {i+1}")
    print(f"Step ID: {item['step_id']}")
    print(f"True action:      {true_action}")
    print(f"Predicted action: {pred_action}")
    print("Action probabilities:")

    probs_np = probs.squeeze(0).cpu().numpy()
    for action_id, prob in enumerate(probs_np):
        action_name = id_to_action[action_id]
        print(f"  {action_name:15s} -> {prob:.4f}")

print("\nDone testing policy predictions.")