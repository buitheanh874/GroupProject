from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from pathlib import Path
import random

import numpy as np

from env.base_env import BaseEnv
from env.kpi import EpisodeKpiTracker
from env.normalization import StateNormalizer
from env.mdp_metrics import CycleMetricsAggregator, compute_normalized_reward

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
    halt_speed_threshold: float = 0.1
    green_cycle_sec: int = 60
    yellow_sec: int = 0
    all_red_sec: int = 0
    max_cycles: int = 60
    max_sim_seconds: Optional[int] = None
    seed: int = 0
    rho_min: float = 0.1
    g_min_sec: int = 5
    lambda_fairness: float = 0.12
    fairness_metric: str = "max"
    action_splits: List[Tuple[float, float]] = field(default_factory=list)
    action_table: List[Dict[str, Any]] = field(default_factory=list)
    include_transition_in_waiting: bool = True
    queue_count_mode: str = "distinct_cycle"
    use_pcu_weighted_wait: Optional[bool] = None
    use_enhanced_reward: bool = False
    reward_exponent: float = 1.0
    enable_anti_flicker: bool = False
    kappa: float = 0.0
    enable_spillback_penalty: bool = False
    beta: float = 0.0
    occ_threshold: float = 0.0
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
            
            ctrl_lanes = set(group.lanes_ns_ctrl + group.lanes_ew_ctrl)
            slip_lanes = set(group.lanes_right_turn_slip_ns + group.lanes_right_turn_slip_ew)
            overlap = ctrl_lanes.intersection(slip_lanes)
            
            if len(overlap) > 0:
                raise ValueError(
                    f"Lane configuration error for TLS '{tls_id}':\n"
                    f"  Controlled lanes and slip lanes must not overlap.\n"
                    f"  Overlapping lanes: {sorted(overlap)}\n"
                    f"  Per MDP spec: Slip lanes (free-flow right-turn) must be excluded from state/reward.\n"
                    f"  Fix: Remove these lanes from either lanes_*_ctrl or lanes_right_turn_slip_*"
                )
            
            if len(slip_lanes) > 0:
                print(
                    f"[INFO] TLS '{tls_id}': {len(slip_lanes)} slip lanes excluded from MDP:\n"
                    f"  {sorted(slip_lanes)}"
                )

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

        self._g_min_sec = int(self._config.g_min_sec)

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
        self._queue_count_mode = str(self._config.queue_count_mode or "distinct_cycle").lower()
        if self._queue_count_mode not in {"distinct_cycle", "snapshot_last_step"}:
            raise ValueError("queue_count_mode must be distinct_cycle or snapshot_last_step")
        self._halt_speed_threshold = float(self._config.halt_speed_threshold)
        self._fairness_metric = str(self._config.fairness_metric or "max").lower()
        if self._fairness_metric not in {"max", "p95"}:
            raise ValueError("fairness_metric must be max or p95")
        self._use_enhanced_reward = bool(self._config.use_enhanced_reward)
        self._reward_exponent = float(self._config.reward_exponent)
        self._enable_anti_flicker = bool(self._config.enable_anti_flicker)
        self._kappa = float(self._config.kappa)
        self._enable_spillback_penalty = bool(self._config.enable_spillback_penalty)
        self._beta = float(self._config.beta)
        self._occ_threshold = float(self._config.occ_threshold)

        self._vehicle_weights = {str(k): float(v) for k, v in self._config.vehicle_weights.items()}
        if self._config.use_pcu_weighted_wait is None:
            self._use_pcu_weighted_wait = len(self._vehicle_weights) > 0
        else:
            self._use_pcu_weighted_wait = bool(self._config.use_pcu_weighted_wait)

        self._traci: Optional[Any] = None
        self._connected = False
        self._cycle_index = 0
        self._episode_seed = int(self._config.seed)
        self._kpi_tracker: Optional[EpisodeKpiTracker] = None
        self._kpi_disabled_warned = False
        self._stepped_seconds = 0.0
        self._last_state_raw: Optional[Any] = None
        self._lane_id_set: set[str] = set()
        self._edge_id_set: set[str] = set()
        self._prev_cycle_sec: Optional[int] = None
        self._route_pool: List[str] = []

    def set_route_file_pool(self, route_files: List[str]) -> None:
        """Set a pool of route files to randomly select from during reset.
        
        This enables demand randomization across training episodes to prevent
        overfitting to a single traffic pattern.
        
        Args:
            route_files: List of route file paths (relative to project root)
            
        Raises:
            ValueError: If route_files is empty or not a list
            FileNotFoundError: If any route file does not exist
        """
        if not isinstance(route_files, list) or len(route_files) == 0:
            raise ValueError("route_files must be a non-empty list")
        
        validated_files = []
        for path_str in route_files:
            route_path = Path(path_str)
            if not route_path.exists():
                raise FileNotFoundError(
                    f"Route file not found: {path_str}\n"
                    f"Current working directory: {Path.cwd()}\n"
                    f"Please check the path or run from project root."
                )
            validated_files.append(str(route_path))
        
        self._route_pool = validated_files
        print(f"[SUMOEnv] Route pool configured with {len(self._route_pool)} files:")
        for idx, route in enumerate(self._route_pool, 1):
            print(f"  {idx}. {Path(route).name}")

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
        """Reset environment for new episode.
        
        If route pool is configured, randomly selects a route file
        before starting SUMO.
        """
        self.close()

        if hasattr(self, "_route_pool") and len(self._route_pool) > 0:
            selected_route = random.choice(self._route_pool)
            self._config.route_file = selected_route
            route_name = Path(selected_route).name
            print(f"[SUMOEnv] Episode {getattr(self, '_episode_count', 0) + 1}: Using route '{route_name}'")
        
        self._start_sumo()
        self._validate_lanes()
        self._cycle_index = 0
        self._stepped_seconds = 0.0
        self._kpi_disabled_warned = False
        self._prev_cycle_sec = None
        self._kpi_tracker = self._make_kpi_tracker() if self._enable_kpi_tracker else None

        if not hasattr(self, '_episode_count'):
            self._episode_count = 0
        self._episode_count += 1

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
        self._prev_cycle_sec = None

    def _step_legacy(self, action_id: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if action_id < 0 or action_id >= self.action_dim:
            raise ValueError(f"Invalid action_id: {action_id}")

        action_def = self._action_defs[action_id]
        rho_ns, rho_ew = float(action_def.rho_ns), float(action_def.rho_ew)
        cycle_sec = int(action_def.cycle_sec)
        min_green_sec = int(round(float(self._config.rho_min) * float(cycle_sec)))
        min_green_sec = max(int(self._g_min_sec), min_green_sec)

        g_ns_raw = int(round(float(rho_ns) * float(cycle_sec)))
        g_ns = max(min_green_sec, min(g_ns_raw, cycle_sec - min_green_sec))
        g_ew = cycle_sec - g_ns

        if g_ns < min_green_sec or g_ew < min_green_sec:
            raise ValueError(f"Action {action_id} violates min green constraint: g_ns={g_ns}, g_ew={g_ew}, min={min_green_sec}")

        include_transition = bool(self._config.include_transition_in_waiting)

        decision_steps = 0

        agg = CycleMetricsAggregator(directions=["NS", "EW"], queue_mode=self._queue_count_mode)

        intervals = self._build_intervals_for_action(g_ns=g_ns, g_ew=g_ew)

        for phase_index, duration_steps, accumulate_waiting in intervals:
            if duration_steps <= 0:
                continue

            self._set_phase(tls_id=str(self._config.tls_id), phase_index=phase_index, hold_steps=int(duration_steps))

            for _ in range(int(duration_steps)):
                self._traci.simulationStep()

                queued_ns = self._queued_for_lanes(self._lanes_single.lanes_ns_ctrl)
                queued_ew = self._queued_for_lanes(self._lanes_single.lanes_ew_ctrl)

                agg.observe(
                    direction="NS",
                    queued_vehicle_ids=queued_ns,
                    step_sec=float(self._config.step_length_sec),
                    accumulate_waiting=bool(accumulate_waiting),
                    weight_lookup=self._vehicle_weight_lookup if self._use_pcu_weighted_wait or self._use_enhanced_reward else None,
                )
                agg.observe(
                    direction="EW",
                    queued_vehicle_ids=queued_ew,
                    step_sec=float(self._config.step_length_sec),
                    accumulate_waiting=bool(accumulate_waiting),
                    weight_lookup=self._vehicle_weight_lookup if self._use_pcu_weighted_wait or self._use_enhanced_reward else None,
                )

                if self._kpi_tracker is not None:
                    try:
                        queue_total = float(len(queued_ns) + len(queued_ew))
                        self._kpi_tracker.on_simulation_step(self._traci, queue_length=queue_total)
                    except Exception as exc:
                        if not self._kpi_disabled_warned:
                            print(f"[WARN] Disabling KPI tracker after error: {exc}")
                            self._kpi_disabled_warned = True
                        self._kpi_tracker = None

                self._stepped_seconds += float(self._config.step_length_sec)
                decision_steps += 1

        queue_counts = agg.queue_counts(order=["NS", "EW"])
        if queue_counts.size >= 2:
            last_q_ns = float(queue_counts[0])
            last_q_ew = float(queue_counts[1])
        waiting_sums = agg.waiting_sums(order=["NS", "EW"])
        w_ns = float(waiting_sums[0]) if waiting_sums.size >= 1 else 0.0
        w_ew = float(waiting_sums[1]) if waiting_sums.size >= 2 else 0.0

        decision_cycle_sec = float(decision_steps) * float(self._config.step_length_sec)

        t_step_value = float(cycle_sec + 2 * float(self._config.yellow_sec) + 2 * float(self._config.all_red_sec))
        wait_exponent = float(self._reward_exponent if self._use_enhanced_reward else 1.0)
        total_wait = agg.waiting_total(exponent=wait_exponent, use_weights=self._use_pcu_weighted_wait)

        lambda_fairness = float(self._config.lambda_fairness)
        fairness_value = 0.0
        fairness_penalty = 0.0
        if lambda_fairness > 1e-9:
            fairness_value = float(agg.fairness_value(metric=self._fairness_metric))
            fairness_penalty = float(lambda_fairness) * float(fairness_value)

        spill_penalty = self._compute_spillback_penalty()
        anti_flicker_penalty = self._compute_anti_flicker_penalty(cycle_sec=cycle_sec)

        reward = compute_normalized_reward(
            wait_total=total_wait,
            t_step=float(t_step_value),
            decision_cycle_sec=float(decision_cycle_sec),
            fairness_penalty=float(fairness_penalty),
            spill_penalty=float(spill_penalty),
            anti_flicker_penalty=float(anti_flicker_penalty),
        )

        state_raw = np.array([float(last_q_ns), float(last_q_ew), float(w_ns), float(w_ew)], dtype=np.float32)
        self._last_state_raw = state_raw.copy()

        state_norm = self._normalizer.normalize(state_raw) if self._normalize_state else state_raw
        state = state_raw if self._return_raw_state else state_norm

        self._cycle_index += 1
        self._prev_cycle_sec = int(cycle_sec)

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
            "green_cycle_sec": int(cycle_sec),
            "decision_cycle_sec": float(decision_cycle_sec),
            "decision_steps": int(decision_steps),
            "step_length_sec": float(self._config.step_length_sec),
            "yellow_sec": int(self._config.yellow_sec),
            "all_red_sec": int(self._config.all_red_sec),
            "t_step": float(t_step_value),
            "fairness_penalty": float(fairness_penalty),
            "fairness_value": float(fairness_value),
            "anti_flicker_penalty": float(anti_flicker_penalty),
            "spill_penalty": float(spill_penalty),
            "total_wait_reward": float(total_wait),
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
            action_id_int = int(action_id)
            if action_id_int < 0 or action_id_int >= self.action_dim:
                raise ValueError(f"Invalid action_id {action_id_int} for tls {tls_id}")
            selected_actions[tls_id] = self._action_defs[action_id_int]

        cycle_set = {int(defn.cycle_sec) for defn in selected_actions.values()}
        if len(cycle_set) != 1:
            raise ValueError(f"All TLS actions must share the same cycle_sec in multi-agent mode. Got: {cycle_set}")
        cycle_sec = cycle_set.pop()

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
        agg_by_tls: Dict[str, CycleMetricsAggregator] = {
            tls_id: CycleMetricsAggregator(directions=["N", "E", "S", "W"], queue_mode=self._queue_count_mode) for tls_id in self._tls_ids
        }

        for _ in range(int(decision_steps)):
            self._traci.simulationStep()

            for tls_id in self._tls_ids:
                accumulate_waiting = True
                if len(intervals_by_tls[tls_id]) > 0 and interval_pos[tls_id] < len(intervals_by_tls[tls_id]):
                    accumulate_waiting = bool(intervals_by_tls[tls_id][interval_pos[tls_id]][2])

                queued_dirs = self._queued_directions_for_tls(tls_id)
                agg = agg_by_tls[tls_id]

                for dir_key in ["N", "E", "S", "W"]:
                    agg.observe(
                        direction=dir_key,
                        queued_vehicle_ids=queued_dirs.get(dir_key, []),
                        step_sec=float(self._config.step_length_sec),
                        accumulate_waiting=bool(accumulate_waiting),
                        weight_lookup=self._vehicle_weight_lookup if self._use_pcu_weighted_wait or self._use_enhanced_reward else None,
                    )

                last_q_dir[tls_id] = agg.queue_counts(order=["N", "E", "S", "W"])

            if self._kpi_tracker is not None:
                try:
                    queue_total = 0.0
                    for tls_id in self._tls_ids:
                        queue_total += float(np.sum(agg_by_tls[tls_id].snapshot_counts(order=["N", "E", "S", "W"])))
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
        wait_exponent = float(self._reward_exponent if self._use_enhanced_reward else 1.0)
        wait_totals: Dict[str, float] = {}
        w_dir: Dict[str, np.ndarray] = {}
        fairness_values: Dict[str, float] = {}
        for tls_id in self._tls_ids:
            agg = agg_by_tls[tls_id]
            last_q_dir[tls_id] = agg.queue_counts(order=["N", "E", "S", "W"])
            wait_totals[tls_id] = agg.waiting_total(exponent=wait_exponent, use_weights=self._use_pcu_weighted_wait)
            fairness_values[tls_id] = agg.fairness_value(metric=self._fairness_metric)
            w_dir[tls_id] = agg.waiting_sums(order=["N", "E", "S", "W"])

        lambda_fairness = float(self._config.lambda_fairness)
        fairness_penalty = 0.0
        if lambda_fairness > 0.0 and len(fairness_values) > 0:
            fairness_penalty = float(lambda_fairness) * float(max(fairness_values.values()))

        spill_penalty = self._compute_spillback_penalty()
        anti_flicker_penalty = self._compute_anti_flicker_penalty(cycle_sec=cycle_sec)

        for tls_id in self._tls_ids:
            rewards[tls_id] = compute_normalized_reward(
                wait_total=float(wait_totals[tls_id]),
                t_step=float(t_step_value),
                decision_cycle_sec=float(decision_cycle_sec),
                fairness_penalty=float(fairness_penalty),
                spill_penalty=float(spill_penalty),
                anti_flicker_penalty=float(anti_flicker_penalty),
            )
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
        self._prev_cycle_sec = int(cycle_sec)

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
            "t_step": float(t_step_value),
            "fairness_penalty": float(fairness_penalty),
            "fairness_value": float(max(fairness_values.values()) if len(fairness_values) > 0 else 0.0),
            "spill_penalty": float(spill_penalty),
            "anti_flicker_penalty": float(anti_flicker_penalty),
            "total_wait_reward": float(sum(wait_totals.values())),
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
        min_green_sec = max(int(self._g_min_sec), min_green_sec)

        g_ns = int(round(float(action_def.rho_ns) * float(cycle)))
        g_ns = max(min_green_sec, g_ns)
        g_ns = min(g_ns, max(min_green_sec, cycle - min_green_sec))
        g_ew = int(cycle - g_ns)
        return int(g_ns), int(g_ew)

    def _build_intervals_for_action(self, g_ns: int, g_ew: int) -> List[Tuple[int, int, bool]]:
        """
        Build TLS phase intervals for one decision cycle.
        
        Returns:
            List of (phase_index, duration_steps, accumulate_waiting) tuples
            
        The third element controls waiting time accumulation:
            - True: Waiting during this phase counts toward reward
            - False: Waiting during this phase is excluded from reward
            
        Rationale for the flag:
            Yellow/all-red times are fixed and uncontrollable by the agent.
            Including them (default) gives realistic delay metrics but penalizes
            agent for mandatory transitions. Excluding them focuses reward on
            controllable green time but may not reflect real-world delay.
            
        Config key: include_transition_in_waiting (default: true)
        
        Important: Queue observation (q_NS, q_EW) is ALWAYS tracked regardless
        of this flag. Only waiting time accumulation (w_NS, w_EW) is affected.
        """
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

    def _vehicle_weight_lookup(self, vehicle_id: str) -> float:
        try:
            type_id = str(self._traci.vehicle.getTypeID(str(vehicle_id)))
            return float(self._vehicle_weights.get(type_id, 1.0))
        except Exception:
            return 1.0

    def _queued_for_lanes(self, lane_ids: Iterable[str]) -> List[str]:
        queued: List[str] = []
        seen: Set[str] = set()
        for lane_id in lane_ids:
            try:
                veh_ids = self._traci.lane.getLastStepVehicleIDs(str(lane_id))
            except Exception:
                continue
            for vid in veh_ids:
                veh = str(vid)
                if veh in seen:
                    continue
                seen.add(veh)
                try:
                    speed = float(self._traci.vehicle.getSpeed(veh))
                except Exception:
                    continue
                if speed < float(self._halt_speed_threshold):
                    queued.append(veh)
        return queued

    def _queued_directions_for_tls(self, tls_id: str) -> Dict[str, List[str]]:
        dirs = self._direction_lanes_by_tls.get(tls_id, {})
        queued: Dict[str, List[str]] = {}
        for key in ["N", "E", "S", "W"]:
            queued[key] = self._queued_for_lanes(dirs.get(key, []))
        return queued

    def _compute_spillback_penalty(self) -> float:
        if not self._enable_spillback_penalty:
            return 0.0
        if not self._enable_downstream_occupancy or len(self._downstream_links) == 0:
            return 0.0
        if self._traci is None:
            return 0.0
    
        occupancy = self._read_downstream_occupancy()
        occ_threshold = float(np.clip(self._occ_threshold, 0.0, 1.0))
        over_thresh = np.maximum(occupancy - occ_threshold, 0.0)
        penalty = float(self._beta) * float(np.sum(over_thresh))
    
        return penalty

    def _compute_anti_flicker_penalty(self, cycle_sec: int) -> float:
        if not self._enable_anti_flicker:
            return 0.0
        if self._prev_cycle_sec is None:
            return 0.0
        return float(self._kappa) if int(cycle_sec) != int(self._prev_cycle_sec) else 0.0

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
                    raw = float(self._traci.lane.getLastStepOccupancy(str(link_id)))
                    occ = float(raw / 100.0) if raw > 1.0 else float(raw)
                    values.append(float(np.clip(occ, 0.0, 1.0)))
                    continue
                except Exception:
                    pass
            if link_id in self._edge_id_set:
                try:
                    raw = float(self._traci.edge.getLastStepOccupancy(str(link_id)))
                    occ = float(raw / 100.0) if raw > 1.0 else float(raw)
                    values.append(float(np.clip(occ, 0.0, 1.0)))
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
            if float(action.rho_ns) < float(self._config.rho_min) or float(action.rho_ew) < float(self._config.rho_min):
                raise ValueError(f"Invalid action split at index {index}: rho must be >= rho_min={self._config.rho_min}")
            min_green_sec = max(int(round(float(self._config.rho_min) * float(action.cycle_sec))), int(self._g_min_sec))
            g_ns = int(round(float(action.rho_ns) * float(action.cycle_sec)))
            g_ew = int(round(float(action.rho_ew) * float(action.cycle_sec)))
            if g_ns < min_green_sec or g_ew < min_green_sec:
                raise ValueError(f"Invalid action split at index {index}: green phases must be >= g_min_sec={self._g_min_sec}")

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
            
            action_table_by_cycle = {
                30: [
                    (0.30, 0.70),
                    (0.50, 0.50),
                    (0.70, 0.30),
                ],
                60: [
                    (0.20, 0.80),
                    (0.30, 0.70),
                    (0.40, 0.60),
                    (0.50, 0.50),
                    (0.60, 0.40),
                    (0.70, 0.30),
                    (0.80, 0.20),
                ],
                90: [
                    (0.30, 0.70),
                    (0.40, 0.60),
                    (0.50, 0.50),
                    (0.60, 0.40),
                    (0.70, 0.30),
                ],
            }
            
            for cycle in [30, 60, 90]:
                for rho_ns, rho_ew in action_table_by_cycle[cycle]:
                    defs.append(SumoActionDefinition(
                        cycle_sec=int(cycle),
                        rho_ns=float(rho_ns),
                        rho_ew=float(rho_ew)
                    ))
            
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