import traci

# Define SUMO binary (GUI mode)
sumo_binary = "sumo-gui"

# Define the SUMO configuration file
sumo_config = "3_3_simulation.sumocfg"

# Start SUMO
traci.start([sumo_binary, "-c", sumo_config])

# Run simulation for 500 steps
for step in range(500):
    traci.simulationStep()

# Close SUMO
traci.close()
