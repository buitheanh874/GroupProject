from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set


@dataclass
class EpisodeKpi:
    arrived_vehicles: int
    avg_wait_time: float
    avg_travel_time: float
    avg_stops: float
    avg_queue: float
    max_wait_time: float
    p95_wait_time: float


class EpisodeKpiTracker:
    def __init__(self, stop_speed_threshold: float = 0.1, use_subscription: bool = False):
        self._stop_speed_threshold = float(stop_speed_threshold)

        self._vehicle_depart_time: Dict[str, float] = {}
        self._vehicle_stop_count: Dict[str, int] = {}
        self._vehicle_is_stopped: Dict[str, bool] = {}
        self._vehicle_accumulated_wait: Dict[str, float] = {}
        self._active_vehicles: Set[str] = set()

        self._total_wait_time = 0.0
        self._total_travel_time = 0.0
        self._total_stop_count = 0
        self._arrived_vehicle_count = 0

        self._queue_sum = 0.0
        self._queue_samples = 0
        
        self._all_wait_times: list[float] = []

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
                self._active_vehicles.add(vehicle_id)
        except Exception:
            pass

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
                if depart_time is not None:
                    travel_time = max(0.0, current_time - float(depart_time))
                    self._total_travel_time += travel_time

                accumulated_wait = self._vehicle_accumulated_wait.get(vehicle_id, 0.0)
                self._total_wait_time += max(0.0, accumulated_wait)
                
                self._all_wait_times.append(max(0.0, accumulated_wait))

                stop_count = self._vehicle_stop_count.get(vehicle_id, 0)
                self._total_stop_count += int(stop_count)

                self._arrived_vehicle_count += 1

                self._vehicle_depart_time.pop(vehicle_id, None)
                self._vehicle_stop_count.pop(vehicle_id, None)
                self._vehicle_is_stopped.pop(vehicle_id, None)
                self._vehicle_accumulated_wait.pop(vehicle_id, None)
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
                import numpy as np
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

        return EpisodeKpi(
            arrived_vehicles=arrived,
            avg_wait_time=avg_wait_time,
            avg_travel_time=avg_travel_time,
            avg_stops=avg_stops,
            avg_queue=avg_queue,
            max_wait_time=max_wait_time,
            p95_wait_time=p95_wait_time,
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
        }