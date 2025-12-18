from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from env.base_env import BaseEnv
from env.kpi import EpisodeKpiTracker
from env.normalization import StateNormalizer


@dataclass
class SumoLaneGroups:
    lanes_ns_ctrl: List[str]
    lanes_ew_ctrl: List[str]
    lanes_rt_ns: List[str] = field(default_factory=list)
    lanes_rt_ew: List[str] = field(default_factory=list)

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
    cycle_length_sec: int = 60
    yellow_sec: int = 0
    all_red_sec: int = 0
    max_cycles: int = 60
    seed: int = 0
    rho_min: float = 0.1
    action_splits: List[Tuple[float, float]] = field(default_factory=list)
    include_transition_in_waiting: bool = False
    terminate_on_empty: bool = True
    sumo_extra_args: List[str] = field(default_factory=list)


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

        self._traci: Optional[Any] = None
        self._connected = False
        self._cycle_index = 0
        self._episode_seed = int(self._config.seed)
        self._kpi_tracker = EpisodeKpiTracker(stop_speed_threshold=0.1)

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
        self._cycle_index = 0
        self._kpi_tracker = EpisodeKpiTracker(stop_speed_threshold=0.1)

        q_ns = self._read_queue_ns()
        q_ew = self._read_queue_ew()
        w_ns = 0.0
        w_ew = 0.0

        state_raw = np.array([q_ns, q_ew, w_ns, w_ew], dtype=np.float32)
        state_norm = self._normalizer.normalize(state_raw)

        info = {
            "state_raw": state_raw.tolist(),
            "state_norm": state_norm.tolist(),
            "sim_time": float(self._traci.simulation.getTime()) if self._traci is not None else 0.0,
        }

        return state_norm

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if not self._connected or self._traci is None:
            raise RuntimeError("SUMOEnv is not connected. Call reset() before step().")

        if action_id < 0 or action_id >= self.action_dim:
            raise ValueError(f"Invalid action_id: {action_id}")

        rho_ns, rho_ew = self._config.action_splits[action_id]
        cycle_green_sec = int(self._config.cycle_length_sec)

        min_green_sec = int(round(float(self._config.rho_min) * float(cycle_green_sec)))
        min_green_sec = max(0, min_green_sec)

        g_ns = int(round(float(rho_ns) * float(cycle_green_sec)))
        g_ns = max(min_green_sec, g_ns)
        g_ns = min(g_ns, max(min_green_sec, cycle_green_sec - min_green_sec))
        g_ew = int(cycle_green_sec - g_ns)

        include_transition = bool(self._config.include_transition_in_waiting)

        w_ns = 0.0
        w_ew = 0.0

        last_q_ns = 0.0
        last_q_ew = 0.0

        last_q_ns, last_q_ew, w_ns, w_ew = self._run_interval(
            phase_index=int(self._phases.ns_green),
            duration_sec=int(g_ns),
            w_ns=w_ns,
            w_ew=w_ew,
            accumulate_waiting=True,
        )

        if int(self._config.yellow_sec) > 0:
            last_q_ns, last_q_ew, w_ns, w_ew = self._run_interval(
                phase_index=int(self._phases.ns_yellow),
                duration_sec=int(self._config.yellow_sec),
                w_ns=w_ns,
                w_ew=w_ew,
                accumulate_waiting=include_transition,
            )

        if int(self._config.all_red_sec) > 0:
            last_q_ns, last_q_ew, w_ns, w_ew = self._run_interval(
                phase_index=int(self._phases.all_red),
                duration_sec=int(self._config.all_red_sec),
                w_ns=w_ns,
                w_ew=w_ew,
                accumulate_waiting=include_transition,
            )

        last_q_ns, last_q_ew, w_ns, w_ew = self._run_interval(
            phase_index=int(self._phases.ew_green),
            duration_sec=int(g_ew),
            w_ns=w_ns,
            w_ew=w_ew,
            accumulate_waiting=True,
        )

        if int(self._config.yellow_sec) > 0:
            last_q_ns, last_q_ew, w_ns, w_ew = self._run_interval(
                phase_index=int(self._phases.ew_yellow),
                duration_sec=int(self._config.yellow_sec),
                w_ns=w_ns,
                w_ew=w_ew,
                accumulate_waiting=include_transition,
            )

        if int(self._config.all_red_sec) > 0:
            last_q_ns, last_q_ew, w_ns, w_ew = self._run_interval(
                phase_index=int(self._phases.all_red),
                duration_sec=int(self._config.all_red_sec),
                w_ns=w_ns,
                w_ew=w_ew,
                accumulate_waiting=include_transition,
            )

        reward = -float(w_ns + w_ew)

        state_raw = np.array([float(last_q_ns), float(last_q_ew), float(w_ns), float(w_ew)], dtype=np.float32)
        state_norm = self._normalizer.normalize(state_raw)

        self._cycle_index += 1

        done_by_cycles = self._cycle_index >= int(self._config.max_cycles)

        if bool(self._config.terminate_on_empty):
            expected_remaining = int(self._traci.simulation.getMinExpectedNumber())
            done_by_empty = expected_remaining <= 0
        else:
            done_by_empty = False

        done = bool(done_by_cycles or done_by_empty)

        info: Dict[str, Any] = {
            "cycle_index": int(self._cycle_index),
            "action_id": int(action_id),
            "split_rho_ns": float(rho_ns),
            "split_rho_ew": float(rho_ew),
            "g_ns": int(g_ns),
            "g_ew": int(g_ew),
            "w_ns": float(w_ns),
            "w_ew": float(w_ew),
            "state_raw": state_raw.tolist(),
            "state_norm": state_norm.tolist(),
            "sim_time": float(self._traci.simulation.getTime()),
        }

        if done:
            info["episode_kpi"] = self._kpi_tracker.summary_dict()

        return state_norm, reward, done, info

    def episode_kpi(self) -> Dict[str, Any]:
        return self._kpi_tracker.summary_dict()

    def close(self) -> None:
        if self._traci is not None and self._connected:
            self._traci.close()
        self._traci = None
        self._connected = False

    def _start_sumo(self) -> None:
        try:
            import traci
        except Exception as e:
            raise ImportError("TraCI is required to run SUMOEnv. Ensure SUMO is installed and traci is importable.") from e

        self._traci = traci

        command = self._build_sumo_command(seed=self._episode_seed)
        self._traci.start(command)
        self._connected = True

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

    def _run_interval(self, phase_index: int, duration_sec: int, w_ns: float, w_ew: float, accumulate_waiting: bool) -> Tuple[float, float, float, float]:
        if duration_sec <= 0:
            q_ns = self._read_queue_ns()
            q_ew = self._read_queue_ew()
            return float(q_ns), float(q_ew), float(w_ns), float(w_ew)

        self._set_phase(phase_index=phase_index, duration_sec=duration_sec)

        last_q_ns = 0.0
        last_q_ew = 0.0

        for _ in range(int(duration_sec)):
            self._traci.simulationStep()
            q_ns_step = self._read_queue_ns()
            q_ew_step = self._read_queue_ew()
            last_q_ns = float(q_ns_step)
            last_q_ew = float(q_ew_step)

            if accumulate_waiting:
                w_ns += float(q_ns_step)
                w_ew += float(q_ew_step)

            self._kpi_tracker.on_simulation_step(self._traci, queue_length=float(q_ns_step + q_ew_step))

        return float(last_q_ns), float(last_q_ew), float(w_ns), float(w_ew)

    def _set_phase(self, phase_index: int, duration_sec: int) -> None:
        self._traci.trafficlight.setPhase(str(self._config.tls_id), int(phase_index))
        self._traci.trafficlight.setPhaseDuration(str(self._config.tls_id), float(int(duration_sec) + 1))

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

    def _validate_action_splits(self) -> None:
        for index, split in enumerate(self._config.action_splits):
            if len(split) != 2:
                raise ValueError(f"Invalid action split at index {index}: {split}")

            rho_ns, rho_ew = float(split[0]), float(split[1])

            if abs((rho_ns + rho_ew) - 1.0) > 1e-6:
                raise ValueError(f"Invalid action split at index {index}: rho_ns + rho_ew must be 1.0")

            if rho_ns < float(self._config.rho_min) or rho_ew < float(self._config.rho_min):
                raise ValueError(f"Invalid action split at index {index}: rho below rho_min")
