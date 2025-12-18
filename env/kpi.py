from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class EpisodeKpi:
    arrived_vehicles: int
    avg_wait_time: float
    avg_travel_time: float
    avg_stops: float
    avg_queue: float


class EpisodeKpiTracker:
    def __init__(self, stop_speed_threshold: float = 0.1):
        self._stop_speed_threshold = float(stop_speed_threshold)
        self._vehicle_depart_time: Dict[str, float] = {}
        self._vehicle_stop_count: Dict[str, int] = {}
        self._vehicle_is_stopped: Dict[str, bool] = {}

        self._total_wait_time = 0.0
        self._total_travel_time = 0.0
        self._total_stop_count = 0
        self._arrived_vehicle_count = 0

        self._queue_sum = 0.0
        self._queue_samples = 0

    def on_simulation_step(self, traci_module: Any, queue_length: Optional[float] = None) -> None:
        current_time = float(traci_module.simulation.getTime())

        departed_ids = traci_module.simulation.getDepartedIDList()
        for vehicle_id in departed_ids:
            self._vehicle_depart_time[vehicle_id] = current_time
            self._vehicle_stop_count[vehicle_id] = 0
            self._vehicle_is_stopped[vehicle_id] = False

        vehicle_ids = traci_module.vehicle.getIDList()
        for vehicle_id in vehicle_ids:
            speed = float(traci_module.vehicle.getSpeed(vehicle_id))
            is_stopped = speed < self._stop_speed_threshold
            was_stopped = bool(self._vehicle_is_stopped.get(vehicle_id, False))

            if is_stopped and not was_stopped:
                current_stop_count = int(self._vehicle_stop_count.get(vehicle_id, 0))
                self._vehicle_stop_count[vehicle_id] = current_stop_count + 1

            self._vehicle_is_stopped[vehicle_id] = is_stopped

        arrived_ids = traci_module.simulation.getArrivedIDList()
        for vehicle_id in arrived_ids:
            depart_time = self._vehicle_depart_time.get(vehicle_id)
            if depart_time is not None:
                self._total_travel_time += max(0.0, current_time - float(depart_time))

            wait_time = self._read_vehicle_wait_time(traci_module, vehicle_id)
            self._total_wait_time += max(0.0, float(wait_time))

            self._total_stop_count += int(self._vehicle_stop_count.get(vehicle_id, 0))
            self._arrived_vehicle_count += 1

            self._vehicle_depart_time.pop(vehicle_id, None)
            self._vehicle_stop_count.pop(vehicle_id, None)
            self._vehicle_is_stopped.pop(vehicle_id, None)

        if queue_length is not None:
            self._queue_sum += float(queue_length)
            self._queue_samples += 1

    def summary(self) -> EpisodeKpi:
        arrived = int(self._arrived_vehicle_count)

        if arrived <= 0:
            avg_wait_time = 0.0
            avg_travel_time = 0.0
            avg_stops = 0.0
        else:
            avg_wait_time = float(self._total_wait_time) / float(arrived)
            avg_travel_time = float(self._total_travel_time) / float(arrived)
            avg_stops = float(self._total_stop_count) / float(arrived)

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
        )

    def summary_dict(self) -> Dict[str, Any]:
        result = self.summary()
        return {
            "arrived_vehicles": int(result.arrived_vehicles),
            "avg_wait_time": float(result.avg_wait_time),
            "avg_travel_time": float(result.avg_travel_time),
            "avg_stops": float(result.avg_stops),
            "avg_queue": float(result.avg_queue),
        }

    def _read_vehicle_wait_time(self, traci_module: Any, vehicle_id: str) -> float:
        vehicle_api = traci_module.vehicle
        if hasattr(vehicle_api, "getAccumulatedWaitingTime"):
            return float(vehicle_api.getAccumulatedWaitingTime(vehicle_id))

        if hasattr(vehicle_api, "getWaitingTime"):
            return float(vehicle_api.getWaitingTime(vehicle_id))

        return 0.0
