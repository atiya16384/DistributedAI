import traci
import subprocess
import time

# Define SUMO command with the correct path
SUMO_CMD = ["sumo-gui", "-c", "simple.sumocfg", "--remote-port", "57205"]

# Start SUMO manually before connecting to TraCI
sumo_process = subprocess.Popen(SUMO_CMD)
time.sleep(2)  # Give SUMO time to start

try:
    print("🚦 Connecting to SUMO...")
    traci.connect(port=57205)  # Correct way to connect TraCI
    print("✅ Connected to SUMO!")

    # Run the simulation for 100 steps
    for step in range(100):
        traci.simulationStep()
        vehicles = traci.vehicle.getIDList()
        print(f"Step {step}: {len(vehicles)} vehicles on the road")

finally:
    traci.close()
    sumo_process.terminate()
    print("✅ SUMO Simulation Finished!")
