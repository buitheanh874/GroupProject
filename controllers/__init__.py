"""
Traffic Signal Controllers Package

Provides baseline controllers for comparison with RL:
- FixedTimeController: Fixed timing plan
- MaxPressureSplitController: MaxPressure with discrete splits
- ActuatedController: Gap-out based actuated control
- WebsterController: Webster formula-based timing
"""
from controllers.fixed_time import FixedTimeController, FixedTimeControllerConfig
from controllers.max_pressure import (
    MaxPressureSplitController,
    OriginalMaxPressureController,
    select_action_from_defs,
)
from controllers.actuated import ActuatedController, ActuatedControllerConfig
from controllers.webster import WebsterController, WebsterControllerConfig

__all__ = [
    # Fixed-time
    "FixedTimeController",
    "FixedTimeControllerConfig",
    # MaxPressure
    "MaxPressureSplitController",
    "OriginalMaxPressureController",
    "select_action_from_defs",
    # Actuated
    "ActuatedController",
    "ActuatedControllerConfig",
    # Webster
    "WebsterController",
    "WebsterControllerConfig",
]
