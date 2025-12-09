import traci
import sumolib
import sys
import os
import numpy as np
import pandas as pd

# Check SUMO_HOME environment variable
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("[ERROR] Please declare environment variable 'SUMO_HOME'")

class SUMOEnv:
    """
    SUMO Environment adapted for 'BI.net.xml'.
    """

    def __init__(self, net_file, route_file, use_gui=False, log_file=None):
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.log_file = log_file
        
        # --- CONFIG MATCHING YOUR BI.NET.XML ---
        self.tl_id = "TL"         # From BI.net.xml
        self.cycle_time = 90      # Total cycle duration
        self.yellow_time = 4      # Yellow time
        
        # Lane IDs (4 lanes per edge as seen in BI.net.xml)
        # Incoming lanes for Queue detection
        self.lanes_NS = [
            "N2TL_0", "N2TL_1", "N2TL_2", "N2TL_3",
            "S2TL_0", "S2TL_1", "S2TL_2", "S2TL_3"
        ]
        self.lanes_EW = [
            "E2TL_0", "E2TL_1", "E2TL_2", "E2TL_3",
            "W2TL_0", "W2TL_1", "W2TL_2", "W2TL_3"
        ]
        
        self.episode = 0
        self.step_counter = 0
        self.metrics = [] 

    def reset(self):
        try:
            traci.close()
        except:
            pass
            
        sumo_binary = sumolib.checkBinary('sumo-gui') if self.use_gui else sumolib.checkBinary('sumo')
        
        cmd = [
            sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1"
        ]
        
        traci.start(cmd)
        self.step_counter = 0
        self.episode += 1
        
        return self._get_state()

    def step(self, action):
        """
        Control Logic for BI.net.xml
        We use Phase 0 for NS Green and Phase 4 for EW Green.
        """
        # 1. Calculate Phase Durations
        available_green_time = self.cycle_time - (2 * self.yellow_time)
        split_ns = np.clip(action, 0.1, 0.9)
        green_ns = int(available_green_time * split_ns)
        green_ew = available_green_time - green_ns
        
        # 2. Execute Phases (Mapped to BI.net.xml phases)
        
        # Phase 0: NS Green (Defined in your net.xml)
        self._set_phase(0, green_ns)        
        
        # Phase 1: NS Yellow
        self._set_phase(1, self.yellow_time) 
        
        # Phase 4: EW Green (Note: Phase 4 is EW Green in your file)
        self._set_phase(4, green_ew)        
        
        # Phase 5: EW Yellow
        self._set_phase(5, self.yellow_time) 

        # 3. Observe & Reward
        next_state = self._get_state()
        reward = self._compute_reward()
        done = traci.simulation.getMinExpectedNumber() <= 0
        
        self._log_kpi(action, split_ns, reward)

        return next_state, reward, done, {}

    def _set_phase(self, phase_index, duration):
        traci.trafficlight.setPhase(self.tl_id, phase_index)
        traci.trafficlight.setPhaseDuration(self.tl_id, duration)
        for _ in range(int(duration)):
            traci.simulationStep()
            self.step_counter += 1

    def _get_state(self):
        """Returns Queue Length for NS and EW."""
        # Note: getLastStepHaltingNumber counts cars with speed < 0.1 m/s
        q_ns = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes_NS])
        q_ew = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes_EW])
        return np.array([q_ns, q_ew])

    def _compute_reward(self):
        wait_ns = sum([traci.lane.getWaitingTime(lane) for lane in self.lanes_NS])
        wait_ew = sum([traci.lane.getWaitingTime(lane) for lane in self.lanes_EW])
        return -(wait_ns + wait_ew)

    def _log_kpi(self, action, actual_split, reward):
        if self.log_file:
            state = self._get_state()
            wait_ns = sum([traci.lane.getWaitingTime(lane) for lane in self.lanes_NS])
            wait_ew = sum([traci.lane.getWaitingTime(lane) for lane in self.lanes_EW])
            
            self.metrics.append({
                'Episode': self.episode,
                'Step': self.step_counter,
                'Action_Raw': action,
                'Split_NS': actual_split,
                'Queue_NS': state[0],
                'Queue_EW': state[1],
                'Wait_NS': wait_ns,
                'Wait_EW': wait_ew,
                'Reward': reward
            })

    def save_logs(self):
        if self.log_file and self.metrics:
            df = pd.DataFrame(self.metrics)
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            df.to_csv(self.log_file, index=False)
            print(f"[INFO] KPIs saved to {self.log_file}")

    def close(self):
        traci.close()
        print("[INFO] SUMO connection closed.")