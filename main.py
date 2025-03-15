import traci

# Start SUMO GUI
sumo_binary = "sumo-gui"
sumo_config = "3_3_simulation.sumocfg"

traci.start([sumo_binary, "-c", sumo_config])

# Run the simulation for 1000 steps
for step in range(10000):
    traci.simulationStep()

traci.close()
