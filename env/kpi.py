from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np


@dataclass
class EpisodeKpi:
    arrived_vehicles: int
    avg_wait_time: float
    avg_travel_time: float
    avg_stops: float
    avg_queue: float
    max_wait_time: float
    p95_wait_time: float
    teleport_started_total: int = 0
    teleport_unique: int = 0
    teleport_rate: float = 0.0
    arrived_corr: int = 0
    teleported_arrived: int = 0
    completion_rate: float = 0.0
    failed_corr: int = 0
    throughput_corr: float = 0.0
    avg_wait_time_corr: float = 0.0
    avg_travel_time_corr: float = 0.0
    p95_wait_time_corr: float = 0.0
    max_wait_time_corr: float = 0.0


class EpisodeKpiTracker:
    """
    Track per-vehicle KPIs across an episode.

    Metrics tracked:
    - avg_wait_time: Average waiting time per arrived vehicle (seconds)
    - avg_travel_time: Average travel time per arrived vehicle (seconds)
    - avg_stops: Average number of stops per arrived vehicle
    - avg_queue: Average queue length across all timesteps
    - max_wait_time: Maximum waiting time of any vehicle (seconds)
    - p95_wait_time: 95th percentile waiting time (seconds)

    Teleport handling (defensive submission):
    - teleport_started_total: Cumulative teleport events
    - teleport_unique: Unique vehicles that teleported at least once
    - teleport_rate: teleport_unique / departed_unique
    - Corrected KPIs (*_corr): Teleported or not-arrived vehicles are capped by teleport_time_cap_sec for averages.

    NOTE: The waiting time here is DIFFERENT from MDP state w_NS/w_EW:
    - KPI tracker: Sum of individual vehicle wait times (vehicle-seconds total, reported as avg per vehicle in seconds)
    - MDP state w_NS/w_EW: Sum of queued vehicles per second within a cycle (vehicle-seconds per cycle)

    Both measure vehicle-seconds but from different aggregation perspectives:
    - KPI: Vehicle-centric aggregation (sum across vehicles, average per vehicle)
    - MDP: Time-centric aggregation (sum across time steps, per cycle)

    Both are valid and useful for different evaluation purposes.
    """
        
    def __init__(
        self,
        stop_speed_threshold: float = 0.1,
        use_subscription: bool = False,
        teleport_time_cap_sec: Optional[float] = None,
    ):
        self._stop_speed_threshold = float(stop_speed_threshold)
        self._teleport_time_cap_sec = teleport_time_cap_sec

        self._vehicle_depart_time: Dict[str, float] = {}
        self._vehicle_stop_count: Dict[str, int] = {}
        self._vehicle_is_stopped: Dict[str, bool] = {}
        self._vehicle_accumulated_wait: Dict[str, float] = {}
        self._active_vehicles: Set[str] = set()
        self._departed_ids: Set[str] = set()
        self._arrived_ids: Set[str] = set()
        self._arrived_order: List[str] = []

        self._total_wait_time = 0.0
        self._total_travel_time = 0.0
        self._total_stop_count = 0
        self._arrived_vehicle_count = 0

        self._queue_sum = 0.0
        self._queue_samples = 0
        
        self._all_wait_times: list[float] = []

        self._teleported_ids: Set[str] = set()
        self._teleport_started_total: int = 0
        self._total_vehicles_seen: int = 0
        self._last_sim_time: float = 0.0
        self._teleport_count_unknown: int = 0

        self._vehicle_teleported: Dict[str, bool] = {}
        self._all_travel_times: list[float] = []
        self._vehicle_is_teleported: Dict[str, bool] = {}

    def on_simulation_step(self, traci_module: Any, queue_length: Optional[float] = None) -> None:
        try:
            current_time = float(traci_module.simulation.getTime())
        except Exception:
            return

        try:
            departed_ids = traci_module.simulation.getDepartedIDList()
            for vehicle_id in departed_ids:
                self._vehicle_depart_time[vehicle_id] = current_time
                self._vehicle_stop_count[vehicle_id] = 0
                self._vehicle_is_stopped[vehicle_id] = False
                self._vehicle_accumulated_wait[vehicle_id] = 0.0
                self._vehicle_is_teleported[vehicle_id] = False
                self._active_vehicles.add(vehicle_id)
                self._total_vehicles_seen += 1
                self._departed_ids.add(vehicle_id)
        except Exception:
            pass

        try:
            teleport_ids = traci_module.simulation.getStartingTeleportIDList()
            for vehicle_id in teleport_ids:
                self._teleported_ids.add(vehicle_id)
                self._teleport_started_total += 1
                self._vehicle_is_teleported[vehicle_id] = True
        except AttributeError:
            try:
                teleport_count = int(traci_module.simulation.getStartingTeleportNumber())
                self._teleport_started_total += teleport_count
                self._teleport_count_unknown += max(0, teleport_count)
            except Exception:
                pass
        except Exception:
            pass
        
        self._last_sim_time = current_time

        try:
            current_vehicle_ids = set(traci_module.vehicle.getIDList())
            vehicles_to_track = self._active_vehicles.intersection(current_vehicle_ids)

            for vehicle_id in vehicles_to_track:
                try:
                    speed = float(traci_module.vehicle.getSpeed(vehicle_id))
                    is_stopped = speed < self._stop_speed_threshold
                    was_stopped = bool(self._vehicle_is_stopped.get(vehicle_id, False))

                    if is_stopped and not was_stopped:
                        self._vehicle_stop_count[vehicle_id] = self._vehicle_stop_count.get(vehicle_id, 0) + 1

                    if is_stopped:
                        wait_increment = 1.0
                        self._vehicle_accumulated_wait[vehicle_id] = (
                            self._vehicle_accumulated_wait.get(vehicle_id, 0.0) + wait_increment
                        )

                    self._vehicle_is_stopped[vehicle_id] = is_stopped

                except Exception:
                    continue
        except Exception:
            pass

        try:
            arrived_ids = traci_module.simulation.getArrivedIDList()
            for vehicle_id in arrived_ids:
                depart_time = self._vehicle_depart_time.get(vehicle_id)
                travel_time = 0.0
                if depart_time is not None:
                    travel_time = max(0.0, current_time - float(depart_time))
                    self._total_travel_time += travel_time

                accumulated_wait = self._vehicle_accumulated_wait.get(vehicle_id, 0.0)
                self._total_wait_time += max(0.0, accumulated_wait)
                
                self._all_wait_times.append(max(0.0, accumulated_wait))
                self._all_travel_times.append(travel_time)

                is_teleported = self._vehicle_is_teleported.get(vehicle_id, False)
                self._vehicle_teleported[vehicle_id] = is_teleported

                stop_count = self._vehicle_stop_count.get(vehicle_id, 0)
                self._total_stop_count += int(stop_count)

                self._arrived_vehicle_count += 1
                self._arrived_ids.add(vehicle_id)
                self._arrived_order.append(vehicle_id)

                self._vehicle_depart_time.pop(vehicle_id, None)
                self._vehicle_stop_count.pop(vehicle_id, None)
                self._vehicle_is_stopped.pop(vehicle_id, None)
                self._vehicle_accumulated_wait.pop(vehicle_id, None)
                self._vehicle_is_teleported.pop(vehicle_id, None)
                self._active_vehicles.discard(vehicle_id)
        except Exception:
            pass

        if queue_length is not None:
            self._queue_sum += float(queue_length)
            self._queue_samples += 1

    def summary(self) -> EpisodeKpi:
        arrived = int(self._arrived_vehicle_count)

        if arrived <= 0:
            avg_wait_time = 0.0
            avg_travel_time = 0.0
            avg_stops = 0.0
            max_wait_time = 0.0
            p95_wait_time = 0.0
        else:
            avg_wait_time = float(self._total_wait_time) / float(arrived)
            avg_travel_time = float(self._total_travel_time) / float(arrived)
            avg_stops = float(self._total_stop_count) / float(arrived)
            
            if len(self._all_wait_times) > 0:
                wait_array = np.array(self._all_wait_times, dtype=np.float32)
                max_wait_time = float(np.max(wait_array))
                p95_wait_time = float(np.percentile(wait_array, 95))
            else:
                max_wait_time = 0.0
                p95_wait_time = 0.0

        if self._queue_samples <= 0:
            avg_queue = 0.0
        else:
            avg_queue = float(self._queue_sum) / float(self._queue_samples)

        teleport_unique = len(self._teleported_ids)
        teleport_started_total = int(self._teleport_started_total)
        total_departed = max(1, len(self._departed_ids))
        teleport_rate = float(teleport_unique) / float(total_departed)

        time_cap = self._teleport_time_cap_sec if self._teleport_time_cap_sec is not None else self._last_sim_time
        time_cap = max(1.0, time_cap)  
        
        arrived_ids = set(self._arrived_ids)
        departed_ids = set(self._departed_ids)
        teleported_ids = set(self._teleported_ids)

        wait_times_corr: List[float] = []
        travel_times_corr: List[float] = []

        for idx, vehicle_id in enumerate(self._arrived_order):
            wait_time = self._all_wait_times[idx] if idx < len(self._all_wait_times) else 0.0
            travel_time = self._all_travel_times[idx] if idx < len(self._all_travel_times) else 0.0
            if vehicle_id in teleported_ids:
                wait_times_corr.append(min(wait_time, time_cap))
                travel_times_corr.append(min(travel_time, time_cap))
            else:
                wait_times_corr.append(wait_time)
                travel_times_corr.append(travel_time)

        missing_ids = departed_ids.difference(arrived_ids)
        failed_ids = set(teleported_ids).union(missing_ids)

        available_extra = max(0, len(departed_ids) - len(failed_ids))
        extra_unknown = min(max(0, self._teleport_count_unknown), available_extra)

        for _ in missing_ids:
            wait_times_corr.append(time_cap)
            travel_times_corr.append(time_cap)

        for _ in range(extra_unknown):
            wait_times_corr.append(time_cap)
            travel_times_corr.append(time_cap)

        if len(wait_times_corr) > 0:
            wait_array_corr = np.array(wait_times_corr, dtype=np.float32)
            travel_array_corr = np.array(travel_times_corr, dtype=np.float32)
            avg_wait_time_corr = float(np.mean(wait_array_corr))
            avg_travel_time_corr = float(np.mean(travel_array_corr))
            p95_wait_time_corr = float(np.percentile(wait_array_corr, 95))
            max_wait_time_corr = float(np.max(wait_array_corr))
        else:
            avg_wait_time_corr = 0.0
            avg_travel_time_corr = 0.0
            p95_wait_time_corr = 0.0
            max_wait_time_corr = 0.0

        arrived_corr_ids = arrived_ids.difference(teleported_ids)
        arrived_corr = len(arrived_corr_ids)
        teleported_arrived = len(arrived_ids.intersection(teleported_ids))
        completion_rate = float(len(arrived_ids)) / float(max(1, len(departed_ids)))
        failed_corr = len(failed_ids) + extra_unknown
        throughput_corr = float(arrived_corr) / float(max(1, self._queue_samples)) if self._queue_samples > 0 else 0.0

        return EpisodeKpi(
            arrived_vehicles=arrived,
            avg_wait_time=avg_wait_time,
            avg_travel_time=avg_travel_time,
            avg_stops=avg_stops,
            avg_queue=avg_queue,
            max_wait_time=max_wait_time,
            p95_wait_time=p95_wait_time,
            teleport_started_total=teleport_started_total,
            teleport_unique=teleport_unique,
            teleport_rate=teleport_rate,
            arrived_corr=arrived_corr,
            teleported_arrived=teleported_arrived,
            completion_rate=completion_rate,
            failed_corr=failed_corr,
            throughput_corr=throughput_corr,
            avg_wait_time_corr=avg_wait_time_corr,
            avg_travel_time_corr=avg_travel_time_corr,
            p95_wait_time_corr=p95_wait_time_corr,
            max_wait_time_corr=max_wait_time_corr,
        )

    def summary_dict(self) -> Dict[str, Any]:
        result = self.summary()
        return {
            "arrived_vehicles": int(result.arrived_vehicles),
            "avg_wait_time": float(result.avg_wait_time),
            "avg_travel_time": float(result.avg_travel_time),
            "avg_stops": float(result.avg_stops),
            "avg_queue": float(result.avg_queue),
            "max_wait_time": float(result.max_wait_time),
            "p95_wait_time": float(result.p95_wait_time),
            "teleport_started_total": int(result.teleport_started_total),
            "teleport_unique": int(result.teleport_unique),
            "teleport_rate": float(result.teleport_rate),
            "arrived_corr": int(result.arrived_corr),
            "teleported_arrived": int(result.teleported_arrived),
            "completion_rate": float(result.completion_rate),
            "failed_corr": int(result.failed_corr),
            "throughput_corr": float(result.throughput_corr),
            "avg_wait_time_corr": float(result.avg_wait_time_corr),
            "avg_travel_time_corr": float(result.avg_travel_time_corr),
            "p95_wait_time_corr": float(result.p95_wait_time_corr),
            "max_wait_time_corr": float(result.max_wait_time_corr),
        }
    
    def record_teleport(self, vehicle_id: str) -> None:
        """Record a teleport event for the given vehicle ID."""
        self._teleported_ids.add(vehicle_id)
        self._teleport_started_total += 1
        self._vehicle_is_teleported[vehicle_id] = True
