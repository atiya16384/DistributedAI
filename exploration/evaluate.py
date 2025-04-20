import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("outputs/train.csv")

# Optional smoothing
df['smooth_reward'] = df['total_reward'].rolling(window=5).mean()
df['smooth_travel_time'] = df['average_travel_time'].rolling(window=5).mean()

# Plot 1: Travel time
plt.figure(figsize=(10, 6))
plt.plot(df['step'], df['smooth_travel_time'], label='Smoothed Travel Time', color='blue')
plt.plot(df['step'], df['average_travel_time'], label='Raw Travel Time', alpha=0.3)
plt.title("🛣️ Average Travel Time Over Time")
plt.xlabel("Step")
plt.ylabel("Time (s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: Rewards
plt.figure(figsize=(10, 6))
plt.plot(df['step'], df['smooth_reward'], label='Smoothed Reward', color='green')
plt.plot(df['step'], df['total_reward'], label='Raw Reward', alpha=0.3)
plt.title("📈 Total Reward Over Time")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 3: Speed
plt.figure(figsize=(10, 6))
plt.plot(df['step'], df['average_speed'], label='Avg Speed (km/h)', color='purple')
plt.title("🚗 Average Speed Over Time")
plt.xlabel("Step")
plt.ylabel("Speed")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 4: Neighborhood sync
if 'neighbor_speed_sync' in df.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['neighbor_speed_sync'], label='Neighbor Speed Sync', color='orange')
    plt.title("🤝 Neighbor Synchronization Over Time")
    plt.xlabel("Step")
    plt.ylabel("Speed Difference")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
