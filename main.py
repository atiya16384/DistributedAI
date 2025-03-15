import traci
import subprocess
import time

SUMO_PORT = 57205
SUMO_CMD = ["sumo-gui", "-c", "simple.sumocfg", "--remote-port", str(SUMO_PORT)]

# Start SUMO manually
sumo_process = subprocess.Popen(SUMO_CMD)
while True:
    try:
        traci.connect(port=57205)
        print("✅ Connected to SUMO!")
        break
    except traci.exceptions.FatalTraCIError:
        print("❌ Waiting for SUMO to start...")
        time.sleep(1)


print("🚦 Connecting to SUMO...")

# Retry connecting multiple times
for attempt in range(5):
    try:
        traci.connect(port=SUMO_PORT)
        print("✅ Connected to SUMO!")
        break
    except traci.exceptions.FatalTraCIError:
        print(f"❌ Connection attempt {attempt + 1} failed. Retrying...")
        time.sleep(2)
else:
    raise Exception("❌ Could not connect to SUMO after multiple attempts.")

# Run simulation for 100 steps
for step in range(100):
    traci.simulationStep()
    vehicles = traci.vehicle.getIDList()
    print(f"Step {step}: {len(vehicles)} vehicles on the road")

traci.close()
sumo_process.terminate()
print("✅ SUMO Simulation Finished!")
