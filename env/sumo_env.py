from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from env.base_env import BaseEnv
from env.kpi import EpisodeKpiTracker
from env.normalization import StateNormalizer


@dataclass
class SumoLaneGroups:
    lanes_ns_ctrl: List[str]
    lanes_ew_ctrl: List[str]
    lanes_right_turn_slip_ns: List[str] = field(default_factory=list)
    lanes_right_turn_slip_ew: List[str] = field(default_factory=list)
    approach_lanes: Dict[str, List[str]] = field(default_factory=dict)

    def validate(self) -> None:
        if len(self.lanes_ns_ctrl) <= 0:
            raise ValueError("lanes_ns_ctrl must not be empty")
        if len(self.lanes_ew_ctrl) <= 0:
            raise ValueError("lanes_ew_ctrl must not be empty")

    def direction_lanes(self) -> Dict[str, List[str]]:
        if len(self.approach_lanes) > 0:
            result: Dict[str, List[str]] = {}
            for key, lanes in self.approach_lanes.items():
                dir_key = str(key).upper()
                if dir_key in {"N", "E", "S", "W"}:
                    result[dir_key] = [str(x) for x in lanes]
            for key in ["N", "E", "S", "W"]:
                result.setdefault(key, [])
            return result

        result = {"N": [], "E": [], "S": [], "W": []}
        for lane in self.lanes_ns_ctrl + self.lanes_ew_ctrl + self.lanes_right_turn_slip_ns + self.lanes_right_turn_slip_ew:
            direction = self._infer_direction(lane)
            if direction is not None:
                result[direction].append(str(lane))
        for key in ["N", "E", "S", "W"]:
            result.setdefault(key, [])
        return result

    @staticmethod
    def _infer_direction(lane_id: str) -> Optional[str]:
        lane_upper = str(lane_id).upper()
        first = lane_upper[:1]
        if first in {"N", "E", "S", "W"}:
            return first
        if "N" in lane_upper:
            return "N"
        if "E" in lane_upper:
            return "E"
        if "S" in lane_upper:
            return "S"
        if "W" in lane_upper:
            return "W"
        return None


@dataclass
class SumoPhaseProgram:
    ns_green: int
    ew_green: int
    ns_yellow: Optional[int] = None
    ew_yellow: Optional[int] = None
    all_red: Optional[int] = None

    def validate(self, yellow_sec: int, all_red_sec: int) -> None:
        if yellow_sec > 0:
            if self.ns_yellow is None or self.ew_yellow is None:
                raise ValueError("ns_yellow and ew_yellow must be provided when yellow_sec > 0")
        if all_red_sec > 0:
            if self.all_red is None:
                raise ValueError("all_red phase index must be provided when all_red_sec > 0")


@dataclass
class SumoActionDefinition:
    cycle_sec: int
    rho_ns: float
    rho_ew: float


@dataclass
class SumoEnvConfig:
    sumo_binary: str
    net_file: str
    route_file: str
    additional_files: List[str] = field(default_factory=list)
    tls_id: str = "tls0"
    tls_ids: List[str] = field(default_factory=list)
    center_tls_id: Optional[str] = None
    downstream_links: Dict[str, str] = field(default_factory=dict)
    vehicle_weights: Dict[str, float] = field(default_factory=dict)
    step_length_sec: float = 1.0
    green_cycle_sec: int = 60
    yellow_sec: int = 0
    all_red_sec: int = 0
    max_cycles: int = 60
    max_sim_seconds: Optional[int] = None
    seed: int = 0
    rho_min: float = 0.1
    lambda_fairness: float = 0.12
    action_splits: List[Tuple[float, float]] = field(default_factory=list)
    action_table: List[Dict[str, Any]] = field(default_factory=list)
    include_transition_in_waiting: bool = True
    terminate_on_empty: bool = True
    sumo_extra_args: List[str] = field(default_factory=list)
    normalize_state: bool = True
    return_raw_state: bool = False
    enable_kpi_tracker: bool = False
    state_dim: int = 4
    enable_downstream_occupancy: bool = True


class SUMOEnv(BaseEnv):
    def __init__(
        self,
        config: SumoEnvConfig,
        lanes: Union[SumoLaneGroups, Dict[str, SumoLaneGroups]],
        phases: SumoPhaseProgram,
        normalizer: StateNormalizer,
    ):
        self._config = config
        self._phases = phases
        self._normalizer = normalizer

        self._tls_ids = [str(x) for x in config.tls_ids] if len(config.tls_ids) > 0 else [str(config.tls_id)]
        self._center_tls_id = str(config.center_tls_id) if config.center_tls_id is not None else str(self._tls_ids[0])

        self._lanes_by_tls: Dict[str, SumoLaneGroups] = {}
        if isinstance(lanes, dict):
            self._lanes_by_tls = {str(k): v for k, v in lanes.items()}
        else:
            self._lanes_by_tls[str(config.tls_id)] = lanes

        for tls_id in self._tls_ids:
            if str(tls_id) not in self._lanes_by_tls:
                raise ValueError(f"lane_groups must be provided for tls_id: {tls_id}")

        self._lanes_single = self._lanes_by_tls.get(str(config.tls_id))

        self._state_dim = int(config.state_dim)
        if self._state_dim not in (4, 12):
            raise ValueError(f"state_dim must be 4 or 12, got {self._state_dim}")
        if len(self._tls_ids) > 1 and self._state_dim == 4:
            raise ValueError("state_dim must be 12 when using multiple tls_ids")

        self._multi_mode = len(self._tls_ids) > 1 or self._state_dim > 4
        self._legacy_mode = not self._multi_mode

        if self._normalizer.dim != self._state_dim:
            raise ValueError(f"Normalizer dimension {self._normalizer.dim} does not match state_dim {self._state_dim}")

        self._phases.validate(config.yellow_sec, config.all_red_sec)

        self._direction_lanes_by_tls: Dict[str, Dict[str, List[str]]] = {}
        for tls_id, group in self._lanes_by_tls.items():
            group.validate()
            self._direction_lanes_by_tls[tls_id] = group.direction_lanes()

        self._lane_sets_by_tls: Dict[str, List[str]] = {}
        for tls_id, dirs in self._direction_lanes_by_tls.items():
            lanes_all = []
            for lanes_list in dirs.values():
                lanes_all.extend(lanes_list)
            lanes_all.extend(self._lanes_by_tls[tls_id].lanes_right_turn_slip_ns)
            lanes_all.extend(self._lanes_by_tls[tls_id].lanes_right_turn_slip_ew)
            self._lane_sets_by_tls[tls_id] = sorted({str(x) for x in lanes_all})

        self._downstream_links = {str(k).upper(): str(v) for k, v in config.downstream_links.items()}
        for key in self._downstream_links.keys():
            if key not in {"N", "E", "S", "W"}:
                raise ValueError(f"Invalid downstream link key: {key}")

        if float(self._config.step_length_sec) <= 0.0:
            raise ValueError("step_length_sec must be > 0")

        if self._legacy_mode and len(self._config.action_splits) <= 0:
            self._config.action_splits = [
                (0.30, 0.70),
                (0.40, 0.60),
                (0.50, 0.50),
                (0.60, 0.40),
                (0.70, 0.30),
            ]

        self._action_defs = self._build_action_definitions()
        self._validate_action_defs()
        self._validate_config_consistency()

        self._cycle_to_actions: Dict[int, List[int]] = {}
        for idx, action in enumerate(self._action_defs):
            self._cycle_to_actions.setdefault(int(action.cycle_sec), []).append(int(idx))

        self._normalize_state = bool(self._config.normalize_state)
        self._return_raw_state = bool(self._config.return_raw_state)
        self._enable_kpi_tracker = bool(self._config.enable_kpi_tracker)
        self._include_transition_in_waiting = bool(self._config.include_transition_in_waiting)
        self._enable_downstream_occupancy = bool(self._config.enable_downstream_occupancy)

        self._vehicle_weights = {str(k): float(v) for k, v in self._config.vehicle_weights.items()}

        self._traci: Optional[Any] = None
        self._connected = False
        self._cycle_index = 0
        self._episode_seed = int(self._config.seed)
        self._kpi_tracker: Optional[EpisodeKpiTracker] = None
        self._kpi_disabled_warned = False
        self._stepped_seconds = 0.0
        self._last_state_raw: Optional[Any] = None
        self._prev_accum_wait: Dict[str, float] = {}
        self._lane_id_set: set[str] = set()
        self._edge_id_set: set[str] = set()

    @property
    def state_dim(self) -> int:
        return int(self._state_dim)

    @property
    def action_dim(self) -> int:
        return len(self._action_defs)

    @property
    def cycle_to_actions(self) -> Dict[int, List[int]]:
        return {int(k): [int(x) for x in v] for k, v in self._cycle_to_actions.items()}

    @property
    def center_tls_id(self) -> str:
        return str(self._center_tls_id)

    def set_seed(self, seed: int) -> None:
        self._episode_seed = int(seed)

    def set_route_file(self, route_file: str) -> None:
        self._config.route_file = str(route_file)

    def get_last_state_raw(self) -> Optional[Any]:
        return self._last_state_raw

    def reset(self) -> Any:
        self.close()
        self._start_sumo()
        self._validate_lanes()
        self._cycle_index = 0
        self._stepped_seconds = 0.0
        self._kpi_disabled_warned = False
        self._prev_accum_wait = {}
        self._kpi_tracker = self._make_kpi_tracker() if self._enable_kpi_tracker else None

        if self._legacy_mode:
            q_ns = self._read_queue_ns()
            q_ew = self._read_queue_ew()
            w_ns = 0.0
            w_ew = 0.0
            state_raw = np.array([q_ns, q_ew, w_ns, w_ew], dtype=np.float32)
            self._last_state_raw = state_raw.copy()
            state_norm = self._normalizer.normalize(state_raw) if self._normalize_state else state_raw
            return state_raw if self._return_raw_state else state_norm

        state_map: Dict[str, np.ndarray] = {}
        for tls_id in self._tls_ids:
            state_raw = self._build_state_vector(
                tls_id=tls_id,
                last_q_dir=np.zeros(4, dtype=np.float32),
                w_dir=np.zeros(4, dtype=np.float32),
            )
            self._last_state_raw = self._last_state_raw or {}
            if isinstance(self._last_state_raw, dict):
                self._last_state_raw[tls_id] = state_raw.copy()
            state_norm = self._normalizer.normalize(state_raw) if self._normalize_state else state_raw
            state_map[tls_id] = state_raw if self._return_raw_state else state_norm
        return state_map

    def step(self, actions: Any) -> Tuple[Any, Any, bool, Dict[str, Any]]:
        if not self._connected or self._traci is None:
            raise RuntimeError("SUMOEnv is not connected. Call reset() before step().")

        if self._legacy_mode:
            return self._step_legacy(int(actions))
        return self._step_multi(actions)

    def episode_kpi(self) -> Dict[str, Any]:
        if not self._enable_kpi_tracker or self._kpi_tracker is None:
            return {}
        return self._kpi_tracker.summary_dict()

    def close(self) -> None:
        try:
            if self._traci is not None and self._connected:
                self._traci.close(False)
        except Exception:
            pass
        self._traci = None
        self._connected = False
        self._kpi_tracker = None
        self._last_state_raw = None
        self._prev_accum_wait = {}

    def _step_legacy(self, action_id: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if action_id < 0 or action_id >= self.action_dim:
            raise ValueError(f"Invalid action_id: {action_id}")

        action_def = self._action_defs[action_id]
        rho_ns, rho_ew = float(action_def.rho_ns), float(action_def.rho_ew)
        green_cycle_sec = int(self._config.green_cycle_sec)

        min_green_sec = int(round(float(self._config.rho_min) * float(green_cycle_sec)))
        min_green_sec = max(0, min_green_sec)

        g_ns = int(round(float(rho_ns) * float(green_cycle_sec)))
        g_ns = max(min_green_sec, g_ns)
        g_ns = min(g_ns, max(min_green_sec, green_cycle_sec - min_green_sec))
        g_ew = int(green_cycle_sec - g_ns)

        include_transition = bool(self._config.include_transition_in_waiting)

        w_ns = 0.0
        w_ew = 0.0
        decision_steps = 0

        last_q_ns = 0.0
        last_q_ew = 0.0

        intervals = [
            (int(self._phases.ns_green), int(g_ns), True),
        ]

        if self._config.yellow_sec > 0 and self._phases.ns_yellow is not None:
            intervals.append((int(self._phases.ns_yellow), int(self._config.yellow_sec), include_transition))
        if self._config.all_red_sec > 0 and self._phases.all_red is not None:
            intervals.append((int(self._phases.all_red), int(self._config.all_red_sec), include_transition))

        intervals.append((int(self._phases.ew_green), int(g_ew), True))

        if self._config.yellow_sec > 0 and self._phases.ew_yellow is not None:
            intervals.append((int(self._phases.ew_yellow), int(self._config.yellow_sec), include_transition))
        if self._config.all_red_sec > 0 and self._phases.all_red is not None:
            intervals.append((int(self._phases.all_red), int(self._config.all_red_sec), include_transition))

        for phase_index, duration_sec, accumulate_waiting in intervals:
            last_q_ns, last_q_ew, w_ns, w_ew, steps = self._run_interval_single(
                phase_index=phase_index,
                duration_sec=duration_sec,
                w_ns=w_ns,
                w_ew=w_ew,
                accumulate_waiting=accumulate_waiting,
            )
            decision_steps += int(steps)

        decision_cycle_sec = float(decision_steps) * float(self._config.step_length_sec)

        lambda_fairness = float(self._config.lambda_fairness)
        total_wait = float(w_ns + w_ew)

        if lambda_fairness > 0.0:
            max_wait = max(float(w_ns), float(w_ew))
            reward = -(total_wait + lambda_fairness * max_wait) / 3600.0
        else:
            reward = -total_wait / 3600.0

        state_raw = np.array([float(last_q_ns), float(last_q_ew), float(w_ns), float(w_ew)], dtype=np.float32)
        self._last_state_raw = state_raw.copy()

        state_norm = self._normalizer.normalize(state_raw) if self._normalize_state else state_raw
        state = state_raw if self._return_raw_state else state_norm

        self._cycle_index += 1

        done = False

        if self._config.max_sim_seconds is not None and int(self._config.max_sim_seconds) > 0:
            if float(self._stepped_seconds) >= float(self._config.max_sim_seconds):
                done = True

        elif int(self._config.max_cycles) > 0:
            if self._cycle_index >= int(self._config.max_cycles):
                done = True

        if not done and bool(self._config.terminate_on_empty):
            try:
                expected_remaining = int(self._traci.simulation.getMinExpectedNumber())
                if expected_remaining <= 0:
                    done = True
            except Exception:
                pass

        info: Dict[str, Any] = {
            "cycle_index": int(self._cycle_index),
            "action_id": int(action_id),
            "split_rho_ns": float(rho_ns),
            "split_rho_ew": float(rho_ew),
            "g_ns": int(g_ns),
            "g_ew": int(g_ew),
            "green_cycle_sec": int(self._config.green_cycle_sec),
            "decision_cycle_sec": float(decision_cycle_sec),
            "decision_steps": int(decision_steps),
            "step_length_sec": float(self._config.step_length_sec),
            "yellow_sec": int(self._config.yellow_sec),
            "all_red_sec": int(self._config.all_red_sec),
            "wait_ns": float(w_ns),
            "wait_ew": float(w_ew),
            "waiting_total": float(w_ns + w_ew),
            "state_raw": state_raw.tolist(),
            "state_norm": state_norm.tolist(),
            "sim_time": float(self._traci.simulation.getTime()),
            "total_stepped_seconds": float(self._stepped_seconds),
        }

        if done and self._kpi_tracker is not None:
            info["episode_kpi"] = self._kpi_tracker.summary_dict()

        return state, float(reward), bool(done), info

    def _step_multi(self, actions: Any) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict[str, Any]]:
        action_map = self._normalize_actions(actions)
        if len(action_map) != len(self._tls_ids):
            raise ValueError("actions must provide all tls_ids")

        selected_actions: Dict[str, SumoActionDefinition] = {}
        for tls_id, action_id in action_map.items():
            if action_id < 0 or action_id >= self.action_dim:
                raise ValueError(f"Invalid action_id {action_id} for tls {tls_id}")
            selected_actions[tls_id] = self._action_defs[action_id]

        cycle_set = {int(defn.cycle_sec) for defn in selected_actions.values()}
        if len(cycle_set) != 1:
            raise ValueError("All TLS actions must share the same cycle_sec in multi-agent mode")
        cycle_sec = cycle_set.pop()

        intervals_by_tls: Dict[str, List[Tuple[int, int, bool]]] = {}
        g_plan: Dict[str, Tuple[int, int]] = {}
        for tls_id, defn in selected_actions.items():
            g_ns, g_ew = self._compute_green_split(defn)
            intervals = self._build_intervals_for_action(g_ns=g_ns, g_ew=g_ew)
            intervals_by_tls[tls_id] = intervals
            g_plan[tls_id] = (g_ns, g_ew)

        total_steps_per_tls = {tls_id: sum(d for _, d, _ in intervals) for tls_id, intervals in intervals_by_tls.items()}
        step_values = set(total_steps_per_tls.values())
        if len(step_values) != 1:
            raise ValueError("All TLS intervals must produce the same number of steps")
        decision_steps = step_values.pop()
        decision_cycle_sec = float(decision_steps) * float(self._config.step_length_sec)

        for tls_id, intervals in intervals_by_tls.items():
            if len(intervals) == 0:
                continue
            phase_index, duration_steps, _ = intervals[0]
            self._set_phase(tls_id=tls_id, phase_index=phase_index, hold_steps=duration_steps)

        interval_pos: Dict[str, int] = {tls_id: 0 for tls_id in self._tls_ids}
        remaining_steps: Dict[str, int] = {tls_id: intervals_by_tls[tls_id][0][1] if len(intervals_by_tls[tls_id]) > 0 else 0 for tls_id in self._tls_ids}

        last_q_dir: Dict[str, np.ndarray] = {tls_id: np.zeros(4, dtype=np.float32) for tls_id in self._tls_ids}
        w_dir: Dict[str, np.ndarray] = {tls_id: np.zeros(4, dtype=np.float32) for tls_id in self._tls_ids}
        weighted_wait: Dict[str, float] = {tls_id: 0.0 for tls_id in self._tls_ids}

        for _ in range(int(decision_steps)):
            self._traci.simulationStep()
            seen_vehicles: set[str] = set()

            for tls_id in self._tls_ids:
                accumulate_waiting = True
                if len(intervals_by_tls[tls_id]) > 0 and interval_pos[tls_id] < len(intervals_by_tls[tls_id]):
                    accumulate_waiting = bool(intervals_by_tls[tls_id][interval_pos[tls_id]][2])

                dirs = self._direction_lanes_by_tls[tls_id]
                q_values = self._read_queue_directions(dirs)
                last_q_dir[tls_id] = q_values
                if accumulate_waiting:
                    w_dir[tls_id] += q_values

                lane_ids = self._lane_sets_by_tls[tls_id]
                veh_ids = []
                for lane in lane_ids:
                    try:
                        veh_ids.extend(self._traci.lane.getLastStepVehicleIDs(str(lane)))
                    except Exception:
                        continue
                delta_wait = self._accumulate_weighted_wait(veh_ids, seen_vehicles)
                if accumulate_waiting:
                    weighted_wait[tls_id] += delta_wait

            if self._kpi_tracker is not None:
                try:
                    queue_total = 0.0
                    for tls_id in self._tls_ids:
                        queue_total += float(np.sum(last_q_dir[tls_id]))
                    self._kpi_tracker.on_simulation_step(self._traci, queue_length=queue_total)
                except Exception as exc:
                    if not self._kpi_disabled_warned:
                        print(f"[WARN] Disabling KPI tracker after error: {exc}")
                        self._kpi_disabled_warned = True
                    self._kpi_tracker = None

            self._stepped_seconds += float(self._config.step_length_sec)

            for tls_id in self._tls_ids:
                remaining_steps[tls_id] -= 1
                if remaining_steps[tls_id] <= 0:
                    interval_pos[tls_id] += 1
                    if interval_pos[tls_id] < len(intervals_by_tls[tls_id]):
                        phase_index, duration_steps, _ = intervals_by_tls[tls_id][interval_pos[tls_id]]
                        remaining_steps[tls_id] = duration_steps
                        self._set_phase(tls_id=tls_id, phase_index=phase_index, hold_steps=duration_steps)

        rewards: Dict[str, float] = {}
        states: Dict[str, np.ndarray] = {}

        t_step_value = float(cycle_sec + 2 * int(self._config.yellow_sec) + 2 * int(self._config.all_red_sec))
        for tls_id in self._tls_ids:
            rewards[tls_id] = -float(weighted_wait[tls_id]) / float(t_step_value if t_step_value > 0 else decision_cycle_sec)
            state_raw = self._build_state_vector(
                tls_id=tls_id,
                last_q_dir=last_q_dir[tls_id],
                w_dir=w_dir[tls_id],
            )
            state_norm = self._normalizer.normalize(state_raw) if self._normalize_state else state_raw
            state_value = state_raw if self._return_raw_state else state_norm
            states[tls_id] = state_value
            if self._last_state_raw is None or not isinstance(self._last_state_raw, dict):
                self._last_state_raw = {}
            self._last_state_raw[tls_id] = state_raw.copy()

        self._cycle_index += 1

        done = False

        if self._config.max_sim_seconds is not None and int(self._config.max_sim_seconds) > 0:
            if float(self._stepped_seconds) >= float(self._config.max_sim_seconds):
                done = True

        elif int(self._config.max_cycles) > 0:
            if self._cycle_index >= int(self._config.max_cycles):
                done = True

        if not done and bool(self._config.terminate_on_empty):
            try:
                expected_remaining = int(self._traci.simulation.getMinExpectedNumber())
                if expected_remaining <= 0:
                    done = True
            except Exception:
                pass

        info: Dict[str, Any] = {
            "cycle_index": int(self._cycle_index),
            "action_ids": {tls: int(action_map[tls]) for tls in self._tls_ids},
            "g_plan": {tls: {"g_ns": int(g_plan[tls][0]), "g_ew": int(g_plan[tls][1])} for tls in self._tls_ids},
            "cycle_sec": int(cycle_sec),
            "yellow_sec": int(self._config.yellow_sec),
            "decision_cycle_sec": float(decision_cycle_sec),
            "decision_steps": int(decision_steps),
            "step_length_sec": float(self._config.step_length_sec),
            "total_weighted_wait": float(sum(weighted_wait.values())),
            "mean_reward": float(np.mean(list(rewards.values()))),
            "state_raw": {tls: state.tolist() for tls, state in ((k, self._last_state_raw[k]) for k in self._tls_ids)},
            "sim_time": float(self._traci.simulation.getTime()),
            "total_stepped_seconds": float(self._stepped_seconds),
        }

        if done and self._kpi_tracker is not None:
            info["episode_kpi"] = self._kpi_tracker.summary_dict()

        return states, rewards, bool(done), info

    def _build_state_vector(self, tls_id: str, last_q_dir: np.ndarray, w_dir: np.ndarray) -> np.ndarray:
        if tls_id not in self._direction_lanes_by_tls:
            raise ValueError(f"Unknown tls_id: {tls_id}")

        occupancy = np.zeros(4, dtype=np.float32)
        if self._enable_downstream_occupancy and tls_id == self._center_tls_id and len(self._downstream_links) > 0:
            occupancy = self._read_downstream_occupancy()

        state = np.zeros(12, dtype=np.float32)
        state[0:4] = last_q_dir.astype(np.float32)
        state[4:8] = w_dir.astype(np.float32)
        state[8:12] = occupancy
        return state

    def _compute_green_split(self, action_def: SumoActionDefinition) -> Tuple[int, int]:
        cycle = int(action_def.cycle_sec)
        if cycle <= 0:
            raise ValueError("cycle_sec must be > 0")
        min_green_sec = int(round(float(self._config.rho_min) * float(cycle)))
        min_green_sec = max(0, min_green_sec)

        g_ns = int(round(float(action_def.rho_ns) * float(cycle)))
        g_ns = max(min_green_sec, g_ns)
        g_ns = min(g_ns, max(min_green_sec, cycle - min_green_sec))
        g_ew = int(cycle - g_ns)
        return int(g_ns), int(g_ew)

    def _build_intervals_for_action(self, g_ns: int, g_ew: int) -> List[Tuple[int, int, bool]]:
        intervals: List[Tuple[int, int, bool]] = []

        intervals.append((int(self._phases.ns_green), self._sec_to_steps(g_ns), True))
        if self._config.yellow_sec > 0 and self._phases.ns_yellow is not None:
            intervals.append((int(self._phases.ns_yellow), self._sec_to_steps(int(self._config.yellow_sec)), self._include_transition_in_waiting))
        if self._config.all_red_sec > 0 and self._phases.all_red is not None:
            intervals.append((int(self._phases.all_red), self._sec_to_steps(int(self._config.all_red_sec)), self._include_transition_in_waiting))

        intervals.append((int(self._phases.ew_green), self._sec_to_steps(g_ew), True))
        if self._config.yellow_sec > 0 and self._phases.ew_yellow is not None:
            intervals.append((int(self._phases.ew_yellow), self._sec_to_steps(int(self._config.yellow_sec)), self._include_transition_in_waiting))
        if self._config.all_red_sec > 0 and self._phases.all_red is not None:
            intervals.append((int(self._phases.all_red), self._sec_to_steps(int(self._config.all_red_sec)), self._include_transition_in_waiting))

        return intervals

    def _normalize_actions(self, actions: Any) -> Dict[str, int]:
        if isinstance(actions, dict):
            return {str(k): int(v) for k, v in actions.items()}
        if len(self._tls_ids) != 1:
            raise ValueError("Multi-agent mode requires a dict of actions")
        return {self._tls_ids[0]: int(actions)}

    def _accumulate_weighted_wait(self, vehicle_ids: List[str], seen: set[str]) -> float:
        total = 0.0
        for veh_id in vehicle_ids:
            veh = str(veh_id)
            seen.add(veh)
            try:
                accum_wait = float(self._traci.vehicle.getAccumulatedWaitingTime(veh))
            except Exception:
                continue
            prev = self._prev_accum_wait.get(veh, accum_wait)
            delta = max(0.0, float(accum_wait) - float(prev))
            try:
                type_id = str(self._traci.vehicle.getTypeID(veh))
                weight = float(self._vehicle_weights.get(type_id, 1.0))
            except Exception:
                weight = 1.0
            total += delta * weight
            self._prev_accum_wait[veh] = float(accum_wait)

        stale_ids = [vid for vid in self._prev_accum_wait.keys() if vid not in seen]
        for vid in stale_ids:
            self._prev_accum_wait.pop(vid, None)

        return float(total)

    def _run_interval_single(self, phase_index: int, duration_sec: int, w_ns: float, w_ew: float, accumulate_waiting: bool) -> Tuple[float, float, float, float, int]:
        steps = self._sec_to_steps(duration_sec)
        if steps <= 0:
            q_ns = self._read_queue_ns()
            q_ew = self._read_queue_ew()
            return float(q_ns), float(q_ew), float(w_ns), float(w_ew), 0

        self._set_phase(tls_id=str(self._config.tls_id), phase_index=phase_index, hold_steps=int(steps))

        last_q_ns = 0.0
        last_q_ew = 0.0

        for _ in range(int(steps)):
            self._traci.simulationStep()
            q_ns_step = self._read_queue_ns()
            q_ew_step = self._read_queue_ew()
            last_q_ns = float(q_ns_step)
            last_q_ew = float(q_ew_step)

            if accumulate_waiting:
                w_ns += float(q_ns_step)
                w_ew += float(q_ew_step)

            if self._kpi_tracker is not None:
                try:
                    self._kpi_tracker.on_simulation_step(self._traci, queue_length=float(q_ns_step + q_ew_step))
                except Exception as exc:
                    if not self._kpi_disabled_warned:
                        print(f"[WARN] Disabling KPI tracker after error: {exc}")
                        self._kpi_disabled_warned = True
                    self._kpi_tracker = None

            self._stepped_seconds += float(self._config.step_length_sec)

        return float(last_q_ns), float(last_q_ew), float(w_ns), float(w_ew), int(steps)

    def _set_phase(self, tls_id: str, phase_index: int, hold_steps: int) -> None:
        hold_sec = float(int(hold_steps) * float(self._config.step_length_sec))
        self._traci.trafficlight.setPhase(str(tls_id), int(phase_index))
        self._traci.trafficlight.setPhaseDuration(str(tls_id), float(hold_sec))

    def _read_queue_ns(self) -> float:
        total = 0.0
        if self._lanes_single is None:
            return 0.0
        for lane_id in self._lanes_single.lanes_ns_ctrl:
            total += float(self._traci.lane.getLastStepHaltingNumber(str(lane_id)))
        return float(total)

    def _read_queue_ew(self) -> float:
        total = 0.0
        if self._lanes_single is None:
            return 0.0
        for lane_id in self._lanes_single.lanes_ew_ctrl:
            total += float(self._traci.lane.getLastStepHaltingNumber(str(lane_id)))
        return float(total)

    def _read_queue_directions(self, dirs: Dict[str, List[str]]) -> np.ndarray:
        values = []
        for key in ["N", "E", "S", "W"]:
            count = 0.0
            for lane_id in dirs.get(key, []):
                count += float(self._traci.lane.getLastStepHaltingNumber(str(lane_id)))
            values.append(float(count))
        return np.asarray(values, dtype=np.float32)

    def _read_downstream_occupancy(self) -> np.ndarray:
        values = []
        for key in ["N", "E", "S", "W"]:
            link_id = self._downstream_links.get(key)
            if link_id is None:
                values.append(0.0)
                continue
            if link_id in self._lane_id_set:
                try:
                    values.append(float(self._traci.lane.getLastStepOccupancy(str(link_id))))
                    continue
                except Exception:
                    pass
            if link_id in self._edge_id_set:
                try:
                    values.append(float(self._traci.edge.getLastStepOccupancy(str(link_id))))
                    continue
                except Exception:
                    pass
            raise ValueError(f"Downstream link not found in network: {link_id}")
        return np.asarray(values, dtype=np.float32)

    def _sec_to_steps(self, duration_sec: int) -> int:
        dur = float(int(duration_sec))
        step = float(self._config.step_length_sec)
        if dur <= 0.0:
            return 0
        steps = int(np.ceil(dur / step))
        if steps <= 0:
            return 0
        return int(steps)

    def _validate_lanes(self) -> None:
        lane_ids = set([str(x) for x in self._traci.lane.getIDList()]) if self._traci is not None else set()
        edge_ids = set([str(x) for x in self._traci.edge.getIDList()]) if self._traci is not None else set()
        self._lane_id_set = lane_ids
        self._edge_id_set = edge_ids

        for tls_id, group in self._lanes_by_tls.items():
            required = set([str(x) for x in group.lanes_ns_ctrl + group.lanes_ew_ctrl + group.lanes_right_turn_slip_ns + group.lanes_right_turn_slip_ew])
            missing = [lane for lane in required if lane not in lane_ids]
            if len(missing) > 0:
                raise ValueError(f"Missing lanes in SUMO network for tls {tls_id}: {missing}")

    def _make_kpi_tracker(self) -> Optional[EpisodeKpiTracker]:
        try:
            return EpisodeKpiTracker(stop_speed_threshold=0.1)
        except Exception:
            return None

    def _validate_action_defs(self) -> None:
        for index, action in enumerate(self._action_defs):
            if float(action.rho_ns) < 0.0 or float(action.rho_ew) < 0.0:
                raise ValueError(f"Invalid action split at index {index}: rho must be non-negative")
            if abs((float(action.rho_ns) + float(action.rho_ew)) - 1.0) > 1e-6:
                raise ValueError(f"Invalid action split at index {index}: rho_ns + rho_ew must be 1.0")
            if int(action.cycle_sec) <= 0:
                raise ValueError(f"cycle_sec must be > 0 for action index {index}")

    def _build_action_definitions(self) -> List[SumoActionDefinition]:
        if len(self._config.action_table) > 0:
            defs: List[SumoActionDefinition] = []
            for item in self._config.action_table:
                cycle = item.get("cycle_sec")
                rho_ns = item.get("rho_ns", item.get("ns_ratio", None))
                rho_ew = item.get("rho_ew", None)
                if rho_ns is None:
                    raise ValueError("action_table entries must include rho_ns/ns_ratio")
                if rho_ew is None:
                    rho_ew = 1.0 - float(rho_ns)
                defs.append(SumoActionDefinition(cycle_sec=int(cycle), rho_ns=float(rho_ns), rho_ew=float(rho_ew)))
            return defs

        splits = self._config.action_splits if len(self._config.action_splits) > 0 else [
            (0.30, 0.70),
            (0.40, 0.60),
            (0.50, 0.50),
            (0.60, 0.40),
            (0.70, 0.30),
        ]

        if self._multi_mode:
            defs: List[SumoActionDefinition] = []
            for cycle in [30, 60, 90]:
                for rho_ns, rho_ew in splits:
                    defs.append(SumoActionDefinition(cycle_sec=int(cycle), rho_ns=float(rho_ns), rho_ew=float(rho_ew)))
            return defs

        return [
            SumoActionDefinition(cycle_sec=int(self._config.green_cycle_sec), rho_ns=float(rho_ns), rho_ew=float(rho_ew))
            for rho_ns, rho_ew in splits
        ]

    def _start_sumo(self) -> None:
        try:
            import traci
        except Exception as e:
            raise ImportError("TraCI is required to run SUMOEnv. Ensure SUMO is installed and traci is importable.") from e

        self._traci = traci

        command = self._build_sumo_command(seed=self._episode_seed)
        self._traci.start(command)
        self._connected = True
        self._stepped_seconds = 0.0

    def _build_sumo_command(self, seed: int) -> List[str]:
        command: List[str] = [
            str(self._config.sumo_binary),
            "-n",
            str(self._config.net_file),
            "-r",
            str(self._config.route_file),
            "--step-length",
            str(float(self._config.step_length_sec)),
            "--seed",
            str(int(seed)),
            "--no-step-log",
            "true",
            "--time-to-teleport",
            "-1",
        ]

        if len(self._config.additional_files) > 0:
            additional = ",".join([str(x) for x in self._config.additional_files])
            command.extend(["-a", additional])

        if len(self._config.sumo_extra_args) > 0:
            for arg in self._config.sumo_extra_args:
                command.append(str(arg))

        return command

    def _validate_config_consistency(self) -> None:
        has_time_limit = self._config.max_sim_seconds is not None and self._config.max_sim_seconds > 0
        has_cycle_limit = self._config.max_cycles > 0

        if not has_time_limit and not has_cycle_limit and not self._config.terminate_on_empty:
            raise ValueError(
                "At least one termination condition must be set: "
                "max_sim_seconds, max_cycles, or terminate_on_empty"
            )
