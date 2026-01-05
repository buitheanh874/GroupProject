from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class LaneArrivalConfig:
    edge_id: str
    lane_index: int
    base_flow_per_hour: float
    arrival_probability: Optional[float] = None


@dataclass
class IntersectionTurningConfig:
    approach_edge: str
    turning_ratio_left: float
    turning_ratio_straight: float
    turning_ratio_right: float
    
    def validate(self) -> None:
        total = self.turning_ratio_left + self.turning_ratio_straight + self.turning_ratio_right
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Turning ratios must sum to 1.0, got {total}")
        
        for ratio in [self.turning_ratio_left, self.turning_ratio_straight, self.turning_ratio_right]:
            if ratio < 0.0 or ratio > 1.0:
                raise ValueError(f"Turning ratio must be in [0,1], got {ratio}")


class StochasticDemandGenerator:
    def __init__(self, seed: int = 0):
        self.random_state = random.Random(seed)
    
    def randomize_arrival_probability(
        self,
        base_probability: float,
        variation_range: float = 0.15,
    ) -> float:
        if base_probability <= 0.0 or base_probability >= 1.0:
            raise ValueError("base_probability must be in (0,1)")
        
        lower_bound = max(0.01, base_probability - variation_range)
        upper_bound = min(0.99, base_probability + variation_range)
        
        randomized = self.random_state.uniform(lower_bound, upper_bound)
        return float(randomized)
    
    def randomize_turning_ratio(
        self,
        base_left: float,
        base_straight: float,
        base_right: float,
        variation_magnitude: float = 0.08,
    ) -> Tuple[float, float, float]:
        base_config = IntersectionTurningConfig(
            approach_edge="temp",
            turning_ratio_left=base_left,
            turning_ratio_straight=base_straight,
            turning_ratio_right=base_right,
        )
        base_config.validate()
        
        perturbation = [
            self.random_state.uniform(-variation_magnitude, variation_magnitude)
            for _ in range(3)
        ]
        
        perturbed = [
            base_left + perturbation[0],
            base_straight + perturbation[1],
            base_right + perturbation[2],
        ]
        
        min_val = min(perturbed)
        if min_val < 0.0:
            offset = abs(min_val) + 0.01
            perturbed = [p + offset for p in perturbed]
        
        total = sum(perturbed)
        normalized = [p / total for p in perturbed]
        
        return tuple(normalized)
    
    def generate_balanced_turning_ratios(
        self,
        num_directions: int = 4,
        base_left: float = 0.25,
        base_straight: float = 0.50,
        base_right: float = 0.25,
        variation: float = 0.08,
    ) -> List[Tuple[float, float, float]]:
        ratios: List[Tuple[float, float, float]] = []
        
        for _ in range(num_directions):
            ratio_tuple = self.randomize_turning_ratio(
                base_left=base_left,
                base_straight=base_straight,
                base_right=base_right,
                variation_magnitude=variation,
            )
            ratios.append(ratio_tuple)
        
        return ratios
    
    def ensure_flow_conservation(
        self,
        incoming_flow: Dict[str, float],
        turning_ratios: Dict[str, Tuple[float, float, float]],
    ) -> Dict[str, float]:
        outgoing_flow: Dict[str, float] = {}
        
        for approach_edge, flow_rate in incoming_flow.items():
            if approach_edge not in turning_ratios:
                outgoing_flow[f"{approach_edge}_left"] = flow_rate * 0.25
                outgoing_flow[f"{approach_edge}_straight"] = flow_rate * 0.50
                outgoing_flow[f"{approach_edge}_right"] = flow_rate * 0.25
            else:
                left, straight, right = turning_ratios[approach_edge]
                outgoing_flow[f"{approach_edge}_left"] = flow_rate * left
                outgoing_flow[f"{approach_edge}_straight"] = flow_rate * straight
                outgoing_flow[f"{approach_edge}_right"] = flow_rate * right
        
        total_in = sum(incoming_flow.values())
        total_out = sum(outgoing_flow.values())
        
        if total_out > 0:
            correction_factor = total_in / total_out
            outgoing_flow = {k: v * correction_factor for k, v in outgoing_flow.items()}
        
        return outgoing_flow