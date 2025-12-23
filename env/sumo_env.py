from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

    def validate(self) -> None:
        if len(self.lanes_ns_ctrl) <= 0:
            raise ValueError("lanes_ns_ctrl must not be empty")
        if len(self.lanes_ew_ctrl) <= 0:
            raise ValueError("lanes_ew_ctrl must not be empty")


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
class SumoEnvConfig:
    sumo_binary: str
    net_file: str
    route_file: str
    additional_files: List[str] = field(default_factory=list)
    tls_id: str = "tls0"
    step_length_sec: float = 1.0
    green_cycle_sec: int = 60
    yellow_sec: int = 0
    all_red_sec: int = 0
    max_cycles: int = 60
    max_sim_seconds: Optional[int] = None
    seed: int = 0
    rho_min: float = 0.1
    action_splits: List[Tuple[float, float]] = field(default_factory=list)
    include_transition_in_waiting: bool = True
    terminate_on_empty: bool = True
    sumo_extra_args: List[str] = field(default_factory=list)
    normalize_state: bool = True
    return_raw_state: bool = False
    enable_kpi_tracker: bool = False


class SUMOEnv(BaseEnv):
    def __init__(self, config: SumoEnvConfig, lanes: SumoLaneGroups, phases: SumoPhaseProgram, normalizer: StateNormalizer):
        self._config = config
        self._lanes = lanes
        self._phases = phases
        self._normalizer = normalizer

        self._lanes.validate()
        self._phases.validate(config.yellow_sec, config.all_red_sec)

        if len(self._config.action_splits) <= 0:
            self._config.action_splits = [
                (0.30, 0.70),
                (0.40, 0.60),
                (0.50, 0.50),
                (0.60, 0.40),
                (0.70, 0.30),
            ]

        self._validate_action_splits()

        self._normalize_state = bool(self._config.normalize_state)
        self._return_raw_state = bool(self._config.return_raw_state)
        self._enable_kpi_tracker = bool(self._config.enable_kpi_tracker)

        if float(self._config.step_length_sec) <= 0.0:
            raise ValueError("step_length_sec must be > 0")
        if int(self._config.green_cycle_sec) <= 0:
            raise ValueError("green_cycle_sec must be > 0")

        self._traci: Optional[Any] = None
        self._connected = False
        self._cycle_index = 0
        self._episode_seed = int(self._config.seed)
        self._kpi_tracker: Optional[EpisodeKpiTracker] = None
        self._kpi_disabled_warned = False
        self._stepped_seconds = 0.0

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def action_dim(self) -> int:
        return len(self._config.action_splits)

    def set_seed(self, seed: int) -> None:
        self._episode_seed = int(seed)

    def set_route_file(self, route_file: str) -> None:
        self._config.route_file = str(route_file)

    def reset(self) -> np.ndarray:
        self.close()
        self._start_sumo()
        self._validate_lanes()
        self._cycle_index = 0
        self._stepped_seconds = 0.0
        self._kpi_disabled_warned = False
        self._kpi_tracker = self._make_kpi_tracker() if self._enable_kpi_tracker else None

        q_ns = self._read_queue_ns()
        q_ew = self._read_queue_ew()
        w_ns = 0.0
        w_ew = 0.0

        state_raw = np.array([q_ns, q_ew, w_ns, w_ew], dtype=np.float32)
        state_norm = self._normalizer.normalize(state_raw) if self._normalize_state else state_raw
        state = state_raw if self._return_raw_state else state_norm
        return state

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if not self._connected or self._traci is None:
            raise RuntimeError("SUMOEnv is not connected. Call reset() before step().")

        if action_id < 0 or action_id >= self.action_dim:
            raise ValueError(f"Invalid action_id: {action_id}")

        rho_ns, rho_ew = self._config.action_splits[action_id]
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
            last_q_ns, last_q_ew, w_ns, w_ew, steps = self._run_interval(
                phase_index=phase_index,
                duration_sec=duration_sec,
                w_ns=w_ns,
                w_ew=w_ew,
                accumulate_waiting=accumulate_waiting,
            )
            decision_steps += int(steps)

        decision_cycle_sec = float(decision_steps) * float(self._config.step_length_sec)
        lambda_fairness = 0.12
        total_wait = float(w_ns + w_ew)
        max_wait = max(float(w_ns), float(w_ew))
        reward = -(total_wait + lambda_fairness * max_wait) / 3600.0

        state_raw = np.array([float(last_q_ns), float(last_q_ew), float(w_ns), float(w_ew)], dtype=np.float32)
        state_norm = self._normalizer.normalize(state_raw) if self._normalize_state else state_raw
        state = state_raw if self._return_raw_state else state_norm

        self._cycle_index += 1

        done = False

        if int(self._config.max_cycles) > 0:
            done_by_cycles = self._cycle_index >= int(self._config.max_cycles)
            if done_by_cycles:
                done = True

        if self._config.max_sim_seconds is not None and int(self._config.max_sim_seconds) > 0:
            done_by_time = float(self._stepped_seconds) >= float(self._config.max_sim_seconds)
            if done_by_time:
                done = True

        if bool(self._config.terminate_on_empty):
            try:
                expected_remaining = int(self._traci.simulation.getMinExpectedNumber())
                done_by_empty = expected_remaining <= 0
                if done_by_empty:
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

    def _run_interval(self, phase_index: int, duration_sec: int, w_ns: float, w_ew: float, accumulate_waiting: bool) -> Tuple[float, float, float, float, int]:
        steps = self._sec_to_steps(duration_sec)
        if steps <= 0:
            q_ns = self._read_queue_ns()
            q_ew = self._read_queue_ew()
            return float(q_ns), float(q_ew), float(w_ns), float(w_ew), 0

        self._set_phase(phase_index=phase_index, hold_steps=int(steps))

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

    def _set_phase(self, phase_index: int, hold_steps: int) -> None:
        hold_sec = float(int(hold_steps) * float(self._config.step_length_sec))
        self._traci.trafficlight.setPhase(str(self._config.tls_id), int(phase_index))
        self._traci.trafficlight.setPhaseDuration(str(self._config.tls_id), float(hold_sec))

    def _read_queue_ns(self) -> float:
        total = 0.0
        for lane_id in self._lanes.lanes_ns_ctrl:
            total += float(self._traci.lane.getLastStepHaltingNumber(str(lane_id)))
        return float(total)

    def _read_queue_ew(self) -> float:
        total = 0.0
        for lane_id in self._lanes.lanes_ew_ctrl:
            total += float(self._traci.lane.getLastStepHaltingNumber(str(lane_id)))
        return float(total)

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
        required = set([str(x) for x in self._lanes.lanes_ns_ctrl + self._lanes.lanes_ew_ctrl + self._lanes.lanes_right_turn_slip_ns + self._lanes.lanes_right_turn_slip_ew])
        missing = [lane for lane in required if lane not in lane_ids]
        if len(missing) > 0:
            raise ValueError(f"Missing lanes in SUMO network: {missing}")

    def _make_kpi_tracker(self) -> Optional[EpisodeKpiTracker]:
        try:
            return EpisodeKpiTracker(stop_speed_threshold=0.1)
        except Exception:
            return None

    def _validate_action_splits(self) -> None:
        for index, split in enumerate(self._config.action_splits):
            if len(split) != 2:
                raise ValueError(f"Invalid action split at index {index}: {split}")

            rho_ns, rho_ew = float(split[0]), float(split[1])

            if abs((rho_ns + rho_ew) - 1.0) > 1e-6:
                raise ValueError(f"Invalid action split at index {index}: rho_ns + rho_ew must be 1.0")

            if rho_ns < float(self._config.rho_min) or rho_ew < float(self._config.rho_min):
                raise ValueError(f"Invalid action split at index {index}: rho below rho_min")
