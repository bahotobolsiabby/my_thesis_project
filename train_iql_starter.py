import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
TRANSITIONS_ROOT = Path.home() / "iql_transitions_output"
MODEL_OUTPUT_DIR = Path.home() / "iql_model_output"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 10
GAMMA = 0.99
EXPECTILE = 0.7
BETA = 3.0
LR = 1e-3
HIDDEN_SIZE = 256

print(f"Using device: {DEVICE}")
print(f"Reading transitions from: {TRANSITIONS_ROOT}")
print(f"Saving models to: {MODEL_OUTPUT_DIR}")

# --------------------------------------------------
# LOAD ALL TRANSITIONS
# --------------------------------------------------
all_transitions = []

transition_files = sorted(TRANSITIONS_ROOT.glob("*_transitions.json"))

if not transition_files:
    raise RuntimeError("No transition JSON files found in iql_transitions_output.")

for tf in transition_files:
    with open(tf, "r", encoding="utf-8") as f:
        data = json.load(f)
        all_transitions.extend(data)

print(f"Loaded {len(all_transitions)} transitions.")

# --------------------------------------------------
# BUILD ACTION VOCAB
# --------------------------------------------------
action_names = sorted(set(t["action"] for t in all_transitions if t["action"] is not None))
action_to_id = {name: i for i, name in enumerate(action_names)}
id_to_action = {i: name for name, i in action_to_id.items()}

print("Action mapping:")
for k, v in action_to_id.items():
    print(f"  {k} -> {v}")

# Save action mapping
with open(MODEL_OUTPUT_DIR / "action_mapping.json", "w", encoding="utf-8") as f:
    json.dump(action_to_id, f, indent=2)

# --------------------------------------------------
# HELPER: BUILD FLAT STATE VECTOR
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
# PREPARE DATA
# --------------------------------------------------
states = []
actions = []
rewards = []
next_states = []
dones = []

for t in all_transitions:
    s = flatten_state(t["state"])
    ns = flatten_state(t["next_state"])
    a = action_to_id[t["action"]]
    r = float(t["reward"])
    d = float(t["done"])

    states.append(s)
    actions.append(a)
    rewards.append(r)
    next_states.append(ns)
    dones.append(d)

states = np.stack(states)
next_states = np.stack(next_states)
actions = np.array(actions, dtype=np.int64)
rewards = np.array(rewards, dtype=np.float32)
dones = np.array(dones, dtype=np.float32)

state_dim = states.shape[1]
num_actions = len(action_to_id)

print(f"State dim: {state_dim}")
print(f"Num actions: {num_actions}")

# --------------------------------------------------
# DATASET
# --------------------------------------------------
class TransitionDataset(Dataset):
    def __init__(self, states, actions, rewards, next_states, dones):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.rewards = torch.tensor(rewards, dtype=torch.float32)
        self.next_states = torch.tensor(next_states, dtype=torch.float32)
        self.dones = torch.tensor(dones, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

dataset = TransitionDataset(states, actions, rewards, next_states, dones)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --------------------------------------------------
# NETWORKS
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

# Q(s,a) for all actions
q_net = MLP(state_dim, num_actions, hidden=HIDDEN_SIZE).to(DEVICE)

# V(s)
v_net = MLP(state_dim, 1, hidden=HIDDEN_SIZE).to(DEVICE)

# Policy logits pi(a|s)
policy_net = MLP(state_dim, num_actions, hidden=HIDDEN_SIZE).to(DEVICE)

q_optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)
v_optimizer = torch.optim.Adam(v_net.parameters(), lr=LR)
pi_optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)

# --------------------------------------------------
# IQL-STYLE LOSSES
# --------------------------------------------------
def expectile_loss(diff, expectile):
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return weight * (diff ** 2)

# --------------------------------------------------
# TRAIN
# --------------------------------------------------
for epoch in range(EPOCHS):
    q_losses = []
    v_losses = []
    pi_losses = []

    for batch in loader:
        state, action, reward, next_state, done = [x.to(DEVICE) for x in batch]

        # ----------------------------
        # 1) Q update
        # target = r + gamma * (1-d) * V(next_state)
        # ----------------------------
        with torch.no_grad():
            target_v = v_net(next_state).squeeze(-1)
            q_target = reward + GAMMA * (1.0 - done) * target_v

        q_values = q_net(state)
        q_sa = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        q_loss = F.mse_loss(q_sa, q_target)

        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()

        # ----------------------------
        # 2) V update using expectile regression
        # V tries to fit lower/upper expectile of Q(s,a)
        # ----------------------------
        with torch.no_grad():
            q_values_detached = q_net(state)
            q_sa_detached = q_values_detached.gather(1, action.unsqueeze(1)).squeeze(1)

        v = v_net(state).squeeze(-1)
        diff = q_sa_detached - v
        v_loss = expectile_loss(diff, EXPECTILE).mean()

        v_optimizer.zero_grad()
        v_loss.backward()
        v_optimizer.step()

        # ----------------------------
        # 3) Policy update via advantage-weighted behavior cloning
        # ----------------------------
        with torch.no_grad():
            q_values_detached = q_net(state)
            q_sa_detached = q_values_detached.gather(1, action.unsqueeze(1)).squeeze(1)
            v_detached = v_net(state).squeeze(-1)
            adv = q_sa_detached - v_detached
            weights = torch.exp(BETA * adv).clamp(max=100.0)

        logits = policy_net(state)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)

        pi_loss = -(weights * selected_log_probs).mean()

        pi_optimizer.zero_grad()
        pi_loss.backward()
        pi_optimizer.step()

        q_losses.append(q_loss.item())
        v_losses.append(v_loss.item())
        pi_losses.append(pi_loss.item())

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Q loss: {np.mean(q_losses):.6f} | "
        f"V loss: {np.mean(v_losses):.6f} | "
        f"Policy loss: {np.mean(pi_losses):.6f}"
    )

# --------------------------------------------------
# SAVE MODELS
# --------------------------------------------------
torch.save(q_net.state_dict(), MODEL_OUTPUT_DIR / "q_net.pt")
torch.save(v_net.state_dict(), MODEL_OUTPUT_DIR / "v_net.pt")
torch.save(policy_net.state_dict(), MODEL_OUTPUT_DIR / "policy_net.pt")

training_info = {
    "state_dim": int(state_dim),
    "num_actions": int(num_actions),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "gamma": GAMMA,
    "expectile": EXPECTILE,
    "beta": BETA,
    "learning_rate": LR,
}

with open(MODEL_OUTPUT_DIR / "training_info.json", "w", encoding="utf-8") as f:
    json.dump(training_info, f, indent=2)

print("\nTraining finished.")
print(f"Saved model files to: {MODEL_OUTPUT_DIR}")