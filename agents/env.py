import os
import traci
import numpy as np
from policy import PolicyNetwork
import torch 


class PasubioEnv:
    def __init__(self, config_file, max_steps=1000, gui=True, control_mode='speed'):
        self.sumo_binary = "sumo-gui" if gui else "sumo"
        self.sumo_config = config_file
        self.max_steps = max_steps
        self.control_mode = control_mode
        self.step_count = 0
        self.vehicles = set()
        self.embedding_memory = {}  # Contrastive representation history
        self.embedding_window = 10  # You can increase/decrease as needed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(input_dim=22).to(self.device) # Adjust to match your obs shape   
        self.destinations = {}  # veh_id -> target edge

    def reset(self):
        if traci.isLoaded():
            traci.close()
        traci.start([self.sumo_binary, "-c", self.sumo_config])
        self.step_count = 0
        traci.simulationStep()
        self.vehicles = set(traci.vehicle.getIDList())
        self.destinations.clear()
        for veh_id in self.vehicles:
            route = traci.vehicle.getRoute(veh_id) 
            if route:
                self.destinations[veh_id] = route[-1]  # store goal edge
        return self._get_observations()

    def _get_vehicle_features(self, veh_id):
        try:
            # Basic motion
            speed = traci.vehicle.getSpeed(veh_id)
            pos = traci.vehicle.getPosition(veh_id)  # x, y
            angle = traci.vehicle.getAngle(veh_id)
            acc = traci.vehicle.getAcceleration(veh_id)

            # Lane and edge info
            lane_id = traci.vehicle.getLaneID(veh_id)
            lane_pos = traci.vehicle.getLanePosition(veh_id)
            lane_index = float(lane_id[-1]) if lane_id[-1].isdigit() else 0.0
            edge_id = traci.vehicle.getRoadID(veh_id)
            is_junction = 1.0 if edge_id.startswith(":") else 0.0
            normalized_lane_id = hash(lane_id) % 1000 / 1000

            # Waiting and physical
            wait_time = traci.vehicle.getWaitingTime(veh_id)
            length = traci.vehicle.getLength(veh_id)

            # Neighborhood (collaborative info)
            neighbors = traci.vehicle.getNeighbors(veh_id, 30)  # smaller range for tight collaboration
            nearby_speeds = []
            nearby_accs = []
            nearby_distances = []

            for n in neighbors:
                n_id = n[0]
                try:
                    nearby_speeds.append(traci.vehicle.getSpeed(n_id))
                    nearby_accs.append(traci.vehicle.getAcceleration(n_id))
                    n_pos = traci.vehicle.getPosition(n_id)
                    distance = np.linalg.norm(np.array(pos) - np.array(n_pos))
                    nearby_distances.append(distance)
                except:
                    continue

            # Aggregate neighbor info
            mean_speed = np.mean(nearby_speeds) if nearby_speeds else 0.0
            mean_acc = np.mean(nearby_accs) if nearby_accs else 0.0
            min_dist = np.min(nearby_distances) if nearby_distances else 50.0  # Default to safe gap

            collab_features = [mean_speed, mean_acc, min_dist]

            # ISR: Intersection and topological context
            try:
                num_lanes = traci.lane.getNumLanes(lane_id)
            except:
                num_lanes = 1

            # Approximate junction complexity (incoming + outgoing edges)
            junction_degree = 0
            try:
                junction_id = traci.lane.getEdgeID(lane_id)
                if junction_id.startswith(":"):
                    connected_lanes = traci.trafficlight.getControlledLanes(junction_id)
                    junction_degree = len(set(connected_lanes))
            except:
                junction_degree = 0

            # ISR features: (these could also be passed through an encoder in future)
            isr_features = [num_lanes, junction_degree]
            
            feature_vector = np.array([
                speed, pos[0], pos[1], angle,
                lane_pos, lane_index,
                acc, wait_time,
                length, len(neighbors),
                is_junction, normalized_lane_id,
                *isr_features, # <- unpack ISR
                *collab_features  # <- unpack collab
            ], dtype=np.float32) 

            self._store_embedding(veh_id, feature_vector)  # <-- log for contrastive learning
            return feature_vector
        except:
            # Use fixed-length zero vector to maintain shape consistency
            return np.zeros(17, dtype=np.float32)  # 12 original + 2 ISR

    def compute_actions(self, observations):
        actions = {}
        for veh_id, obs in observations.items():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, _ = self.policy(obs_tensor)
            actions[veh_id] = float(action.squeeze().cpu().detach().numpy()) * 20  # Scale up to SUMO speed
        return actions

    def _maybe_reroute_vehicle(self, veh_id):
        try:
            wait_time = traci.vehicle.getWaitingTime(veh_id)
            if wait_time > 20 and np.random.rand() < 0.8:  # 80% chance to reroute
                current_edge = traci.vehicle.getRoadID(veh_id)
                dest_edge = self.destinations.get(veh_id, None)
                if dest_edge and current_edge != dest_edge:
                    route = traci.simulation.findRoute(current_edge, dest_edge)
                    if route.edges:
                        traci.vehicle.setRoute(veh_id, route.edges)
                        self.reroute_count = getattr(self, "reroute_count", 0) + 1

        except Exception as e:
            print(f"[REROUTE ERROR] {veh_id}: {e}")


    def _get_observations(self):
        obs = {}
        for veh_id in traci.vehicle.getIDList():
            obs[veh_id] = self._get_vehicle_features(veh_id)
        return obs

    def _get_rewards(self):
        rewards = {}
        for veh_id in self.vehicles:
            if not traci.vehicle.exists(veh_id):
                continue

            speed = traci.vehicle.getSpeed(veh_id)
            wait = traci.vehicle.getWaitingTime(veh_id)
            acc = traci.vehicle.getAcceleration(veh_id)

            # 👥 Collaborative Reward Bonus
            neighbors = traci.vehicle.getNeighbors(veh_id, 30)
            neighbor_speed_sum = 0
            for n in neighbors:
                try:
                    neighbor_speed_sum += traci.vehicle.getSpeed(n[0])
                except:
                    continue
            avg_neighbor_speed = neighbor_speed_sum / len(neighbors) if neighbors else speed

            # Total reward: selfish + collaborative
            reward = (
                speed
                - 0.2 * wait
                - 0.05 * abs(acc)
                + 0.1 * avg_neighbor_speed  # Encourage driving in flow with neighbors
            )
            rewards[veh_id] = reward
        return rewards


    def _get_dones(self):
        dones = {}
        for veh_id in self.vehicles:
            dones[veh_id] = not traci.vehicle.exists(veh_id)
        return dones
    
    def _store_embedding(self, veh_id, embedding):
        #Store recent feature embeddings for contrastive learning.
        if veh_id not in self.embedding_memory:
            self.embedding_memory[veh_id] = []
        self.embedding_memory[veh_id].append(embedding)

        # Keep only recent N
        if len(self.embedding_memory[veh_id]) > self.embedding_window:
            self.embedding_memory[veh_id].pop(0)

    def get_embedding_memory(self):
        #Returns a copy of stored embeddings for all agents.
        return self.embedding_memory


    def _apply_actions(self, actions):
        if actions is None:
            return
        for veh_id, act in actions.items():
            try:
                if self.control_mode == 'speed':
                    traci.vehicle.setSpeed(veh_id, act)
                elif self.control_mode == 'lane':
                    traci.vehicle.changeLane(veh_id, int(act), 50)
                elif self.control_mode == 'hybrid':
                    traci.vehicle.setSpeed(veh_id, act[0])
                    traci.vehicle.changeLane(veh_id, int(act[1]), 50)
            except:
                continue

    def step(self, actions=None):
        self._apply_actions(actions)
        for veh_id in self.vehicles:
            self._maybe_reroute_vehicle(veh_id)
        traci.simulationStep()
        self.step_count += 1

        obs = self._get_observations()
        rewards = self._get_rewards()
        dones = self._get_dones()
        infos = {veh_id: {} for veh_id in self.vehicles}

        if self.step_count >= self.max_steps:
            for veh_id in dones:
                dones[veh_id] = True

        return obs, rewards, dones, infos

    def close(self):
        if traci.isLoaded():
            traci.close()
