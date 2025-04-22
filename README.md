##  Project Components Overview

| Component           | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| **PasubioEnv**    | Custom SUMO-based simulation environment for training vehicle agents       |
| **Agents = Vehicles** | Each vehicle (agent) observes its state, acts, and receives a reward        |
| **Features**         | Vehicles observe speed, position, neighborhood behavior, and topological context |
| **Dynamic Rerouting**| Agents stuck in traffic are rerouted to improve efficiency                |
| **Policy**            | Neural network (actor + critic) decides speed for each vehicle            |
| **Training**          | Based on REINFORCE algorithm with value function updates                  |
| **Evaluation**        | Travel time, speed, and coordination trends are visualized                |

---

### 1. **Goal-Oriented State Representation**
- **What**: Each vehicle stores its final route destination.
- **Why**: Helps agents make decisions that are aware of their goal (e.g., avoid unnecessary reroutes or congestion).
- **How**: 
  - Vehicle destinations are extracted using `traci.vehicle.getRoute(veh_id)`.
  - These are stored and used in rerouting logic if vehicles get stuck.

---

### 2. **Intersection-Specific Representation (ISR)**
- **What**: Vehicles augment their features with topological context like number of lanes and junction complexity.
- **Why**: Allows agents to adapt behavior based on road network structure (e.g., drive cautiously at complex intersections).
- **How**: 
  - Features include:
    - `num_lanes`: via `traci.lane.getNumLanes`
    - `junction_degree`: number of outgoing controlled connections
  - These are normalized and appended to vehicle observations.

---

### 3. **Contrastive Learning for Robustness**
- **What**: A lightweight memory buffer of recent state embeddings is stored per vehicle.
- **Why**: Facilitates representation learning, helps encode behavior across time for generalization.
- **How**: 
  - Embeddings (observations) are stored in a dictionary with rolling window (`embedding_window`).
  - Enables later use with a contrastive loss (optional).

---

### 4. **Collaborative Policy Optimization**
- **What**: Agents observe and optimize based on neighbors' behavior (mean speed, acceleration, distance).
- **Why**: Encourages smooth traffic flow and cooperative behavior across multiple agents.
- **How**: 
  - Neighbor features are computed via `traci.vehicle.getNeighbors`.
  - Rewards include terms like `+ 0.1 * avg_neighbor_speed`.

---

### 5. **Dynamic Rerouting**
- **What**: Vehicles that are stuck beyond a certain threshold are probabilistically rerouted.
- **Why**: Reduces traffic congestion, prevents deadlocks and improves throughput.
- **How**: 
  - Vehicles with high `waitingTime > 20` reroute using `traci.simulation.findRoute(...)`.
  - Reroute is triggered with an 80% chance (`np.random.rand() < 0.8`).

---

### 6. **Neural Policy + Critic**
- **What**: A shared neural network decides agent actions and estimates value functions.
- **Why**: Supports continuous control (speed adjustment) and temporal credit assignment.
- **How**: 
  - Implemented in `policy.py` with:
    - **Actor**: Outputs action (normalized speed)
    - **Critic**: Outputs value for REINFORCE updates
  - Loss = `Actor loss + Critic loss`

---

### 7. **Training & Evaluation**
- **What**: Trains the RL agent and logs data for analysis.
- **How**:
  - Training uses REINFORCE + Value Baseline (`train.py`)
  - Evaluation logs to `outputs/train.csv`git 
  - `evaluate.py` plots:
    - Average Travel Time
    - Total Reward
    - Average Speed
    - Neighbor Synchronization


### Project flow diagram
[ SUMO Network (XML) ]
        ↓
[ PasubioEnv (env.py) ] ← vehicle info → [ SUMO via TraCI ]
        ↓
[ Policy (PyTorch) ]
        ↓
[ train.py ] ← optimization + CSV logging
        ↓
[ outputs/train.csv ] → [ evaluate.py → 📊 graphs ]
