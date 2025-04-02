import traci

sumo_binary = "sumo-gui"  # or "sumo" for non-GUI mode
sumo_config = "pasubio/run.sumocfg"

traci.start([sumo_binary, "-c", sumo_config])

for step in range(10000):  
    traci.simulationStep()

traci.close()
