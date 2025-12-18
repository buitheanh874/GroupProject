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
        self.cycle_time = tl_conf.get('cycle_time', 60)
        self.yellow_time = tl_conf.get('yellow_time', 0)
        self.phases = {
            'ns_green': tl_conf['phase_ns_green'],
            'ns_yellow': tl_conf['phase_ns_yellow'],
            'ew_green': tl_conf['phase_ew_green'],
            'ew_yellow': tl_conf['phase_ew_yellow']
        }

        # 4. Lane grouping (controlled vs slip lanes)
        lanes_conf = config.get('lanes', {})
        self.lanes_NS_ctrl = lanes_conf.get('ns_controlled', [])
        self.lanes_EW_ctrl = lanes_conf.get('ew_controlled', [])
        self.lanes_RT_NS = lanes_conf.get('ns_right_turn', [])
        self.lanes_RT_EW = lanes_conf.get('ew_right_turn', [])

        # Fallback to legacy lane definitions if config is missing
        if not self.lanes_NS_ctrl:
            self.lanes_NS_ctrl = [
                "N2TL_0",
                "N2TL_1",
                "N2TL_2",
                "N2TL_3",
                "S2TL_0",
                "S2TL_1",
                "S2TL_2",
                "S2TL_3",
            ]

        if not self.lanes_EW_ctrl:
            self.lanes_EW_ctrl = [
                "E2TL_0",
                "E2TL_1",
                "E2TL_2",
                "E2TL_3",
                "W2TL_0",
                "W2TL_1",
                "W2TL_2",
                "W2TL_3",
            ]

        # 5. Action space (phase split choices)
        self.action_splits = tl_conf.get(
            'action_splits',
            [
                (0.30, 0.70),
                (0.40, 0.60),
                (0.50, 0.50),
                (0.60, 0.40),
                (0.70, 0.30),
            ],
        )

        # 6. State normalization parameters
        norm_conf = config.get('state_normalization', {})
        mu_cfg = norm_conf.get('mu', {})
        sigma_cfg = norm_conf.get('sigma', {})
        self.state_mu = np.array(
            [
                mu_cfg.get('q_NS', 0.0),
                mu_cfg.get('q_EW', 0.0),
                mu_cfg.get('w_NS', 0.0),
                mu_cfg.get('w_EW', 0.0),
            ]
        )
        self.state_sigma = np.array(
            [
                sigma_cfg.get('q_NS', 1.0),
                sigma_cfg.get('q_EW', 1.0),
                sigma_cfg.get('w_NS', 1.0),
                sigma_cfg.get('w_EW', 1.0),
            ]
        )
        self.state_clip = norm_conf.get('clip', 5)
        self.eps = 1e-6
        
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

    def step(self, action_id):
        """
        Executes one control cycle according to the MDP spec.

        Args:
            action_id (int): Index in the discrete action set controlling the green split.
        """
        if action_id < 0 or action_id >= len(self.action_splits):
            raise ValueError(f"Invalid action id {action_id}. Must be in [0, {len(self.action_splits) - 1}].")

        rho_ns, rho_ew = self.action_splits[action_id]
        if not np.isclose(rho_ns + rho_ew, 1.0):
            raise ValueError("Invalid action split configuration: rho_ns + rho_ew must equal 1.0.")

        # Calculate Phase Durations (cycle length includes both yellows)
        available_green_time = max(self.cycle_time - (2 * self.yellow_time), 1)
        green_ns = int(round(available_green_time * rho_ns))
        green_ew = available_green_time - green_ns

        # Cycle-level accumulators
        wait_ns = 0.0
        wait_ew = 0.0
        q_ns_snapshot = 0
        q_ew_snapshot = 0

        # Execute Phases using indices from config
        q_ns_snapshot, q_ew_snapshot, wait_ns, wait_ew = self._run_phase(
            self.phases['ns_green'], green_ns, wait_ns, wait_ew
        )

        if self.yellow_time > 0:
            q_ns_snapshot, q_ew_snapshot, wait_ns, wait_ew = self._run_phase(
                self.phases['ns_yellow'], self.yellow_time, wait_ns, wait_ew
            )

        q_ns_snapshot, q_ew_snapshot, wait_ns, wait_ew = self._run_phase(
            self.phases['ew_green'], green_ew, wait_ns, wait_ew
        )

        if self.yellow_time > 0:
            q_ns_snapshot, q_ew_snapshot, wait_ns, wait_ew = self._run_phase(
                self.phases['ew_yellow'], self.yellow_time, wait_ns, wait_ew
            )

        # Observe & Reward
        state_raw = np.array([q_ns_snapshot, q_ew_snapshot, wait_ns, wait_ew], dtype=np.float32)
        next_state = self._normalize_state(state_raw)
        reward = -(wait_ns + wait_ew)

        # Check if simulation should end (no vehicles left)
        done = traci.simulation.getMinExpectedNumber() <= 0

        self._log_kpi(action_id, rho_ns, state_raw, reward)

        return next_state, reward, done, {"state_raw": state_raw}

    def _run_phase(self, phase_index, duration, wait_ns, wait_ew):
        traci.trafficlight.setPhase(self.tl_id, phase_index)
        traci.trafficlight.setPhaseDuration(self.tl_id, duration)

        q_ns_snapshot = 0
        q_ew_snapshot = 0

        for _ in range(int(duration)):
            traci.simulationStep()
            self.step_counter += 1

            q_ns_snapshot = self._get_queue(self.lanes_NS_ctrl)
            q_ew_snapshot = self._get_queue(self.lanes_EW_ctrl)

            wait_ns += q_ns_snapshot
            wait_ew += q_ew_snapshot

        return q_ns_snapshot, q_ew_snapshot, wait_ns, wait_ew

    def _get_queue(self, lane_list):
        return sum([traci.lane.getLastStepHaltingNumber(lane) for lane in lane_list])

    def _get_state(self):
        q_ns = self._get_queue(self.lanes_NS_ctrl)
        q_ew = self._get_queue(self.lanes_EW_ctrl)
        state_raw = np.array([q_ns, q_ew, 0.0, 0.0], dtype=np.float32)
        return self._normalize_state(state_raw)

    def _normalize_state(self, state_raw):
        state_norm = (state_raw - self.state_mu) / (self.state_sigma + self.eps)
        state_norm = np.clip(state_norm, -self.state_clip, self.state_clip)
        return state_norm

    def _log_kpi(self, action, actual_split, state_raw, reward):
        if self.log_file:
            self.metrics.append({
                'Episode': self.episode,
                'Step': self.step_counter,
                'Action': action,
                'Split_NS': actual_split,
                'Queue_NS': state_raw[0],
                'Queue_EW': state_raw[1],
                'Wait_NS': state_raw[2],
                'Wait_EW': state_raw[3],
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