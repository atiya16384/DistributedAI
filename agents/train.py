import torch
import torch.nn as nn
import torch.optim as optim
from agents.env import PasubioEnv
import csv
import os
import numpy as np
import traci

# --- Hyperparameters ---
EPISODES = 10
STEPS = 1000
LR = 0.001

# --- Environment and Policy Setup ---
env = PasubioEnv(config_file="pasubio/run.sumocfg", max_steps=STEPS, gui=True)
policy = env.policy  # already initialized in env.py
optimizer = optim.Adam(policy.parameters(), lr=LR)
mse_loss = nn.MSELoss()

# --- Create logging dir ---
os.makedirs("outputs", exist_ok=True)
log_path = "outputs/train.csv"

# --- Create CSV header if needed ---
if not os.path.exists(log_path):
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "average_travel_time", "total_reward", "average_speed", "neighbor_speed_sync"])

# --- Training Loop ---
for episode in range(EPISODES):
    obs_dict = env.reset()
    episode_reward = 0

    for t in range(STEPS):
        veh_ids = list(obs_dict.keys())
        obs_tensor = torch.tensor([obs_dict[vid] for vid in veh_ids], dtype=torch.float32)

        # Policy forward pass
        actions, values = policy(obs_tensor)
        scaled_actions = actions.squeeze().detach().numpy() * 20  # scale for SUMO
        act_dict = {vid: float(a) for vid, a in zip(veh_ids, scaled_actions)}

        # Environment step
        next_obs, rew_dict, done_dict, _ = env.step(act_dict)

        # Compute reward + loss
        rewards = torch.tensor([rew_dict[vid] for vid in veh_ids], dtype=torch.float32)
        advantages = rewards - values.squeeze()

        actor_loss = -torch.mean(actions.squeeze() * advantages.detach())
        critic_loss = mse_loss(values.squeeze(), rewards)
        loss = actor_loss + critic_loss

        # Gradient update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        avg_speed = np.mean([traci.vehicle.getSpeed(vid) for vid in veh_ids])
        neighbor_sync = np.mean([
            abs(env._get_vehicle_features(vid)[0] - env._get_vehicle_features(vid)[-1])
            for vid in veh_ids
        ])
        avg_travel_time = np.mean([traci.vehicle.getAccumulatedWaitingTime(vid) for vid in veh_ids])

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                t + (episode * STEPS),
                avg_travel_time,
                episode_reward,
                avg_speed,
                neighbor_sync
            ])

        obs_dict = next_obs
        episode_reward += rewards.sum().item()

        if all(done_dict.values()):
            break

    print(f"Episode {episode + 1} complete | Total Reward: {episode_reward:.2f}")

env.close()
