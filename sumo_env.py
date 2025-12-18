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
    SUMO Environment for Traffic Signal Control.
    Loads settings dynamically from a configuration dictionary.
    """

    def __init__(self, config, scenario='low'):
        """
        Initializes the environment.
        Args:
            config (dict): The configuration dictionary loaded from yaml.
            scenario (str): 'low' or 'high' traffic demand.
        """
        self.config = config
        
        # 1. Setup Paths
        self.net_file = config['paths']['net_file']
        
        if scenario == 'low':
            self.route_file = config['paths']['route_file_low']
        else:
            self.route_file = config['paths']['route_file_high']
            
        # 2. Setup Simulation Settings
        self.use_gui = config['simulation']['gui']
        self.log_file = config['simulation']['log_file']
        
        # 3. Setup Traffic Light Settings
        tl_conf = config['traffic_light']
        self.tl_id = tl_conf['tl_id']
        self.cycle_time = tl_conf['cycle_time']
        self.yellow_time = tl_conf['yellow_time']
        self.phases = {
            'ns_green': tl_conf['phase_ns_green'],
            'ns_yellow': tl_conf['phase_ns_yellow'],
            'ew_green': tl_conf['phase_ew_green'],
            'ew_yellow': tl_conf['phase_ew_yellow']
        }
        
        # Lane definitions (Specific to BI.net.xml structure)
        self.lanes_NS = ["N2TL_0", "N2TL_1", "N2TL_2", "N2TL_3", "S2TL_0", "S2TL_1", "S2TL_2", "S2TL_3"]
        self.lanes_EW = ["E2TL_0", "E2TL_1", "E2TL_2", "E2TL_3", "W2TL_0", "W2TL_1", "W2TL_2", "W2TL_3"]
        
        self.episode = 0
        self.step_counter = 0
        self.metrics = []

    def reset(self):
        """Restarts the simulation."""
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
        
        print(f"[INFO] Simulation reset. Episode: {self.episode}")
        return self._get_state()

    def step(self, action):
        """
        Executes one control cycle.
        Action: Split ratio for NS Green (0.1 to 0.9).
        """
        # Calculate Phase Durations
        available_green_time = self.cycle_time - (2 * self.yellow_time)
        split_ns = np.clip(action, 0.1, 0.9)
        green_ns = int(available_green_time * split_ns)
        green_ew = available_green_time - green_ns
        
        # Execute Phases using indices from config
        self._set_phase(self.phases['ns_green'], green_ns)        
        self._set_phase(self.phases['ns_yellow'], self.yellow_time) 
        self._set_phase(self.phases['ew_green'], green_ew)        
        self._set_phase(self.phases['ew_yellow'], self.yellow_time) 

        # Observe & Reward
        next_state = self._get_state()
        reward = self._compute_reward()
        
        # Check if simulation should end (no vehicles left)
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
            
            # --- BỔ SUNG: Tính toán thời gian chờ (Waiting Time) ---
            wait_ns = sum([traci.lane.getWaitingTime(lane) for lane in self.lanes_NS])
            wait_ew = sum([traci.lane.getWaitingTime(lane) for lane in self.lanes_EW])
            # -------------------------------------------------------

            self.metrics.append({
                'Episode': self.episode,
                'Step': self.step_counter,
                'Action': action,
                'Queue_NS': state[0],
                'Queue_EW': state[1],
                'Wait_NS': wait_ns,   # <--- Đã thêm lại cột này
                'Wait_EW': wait_ew,   # <--- Đã thêm lại cột này
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