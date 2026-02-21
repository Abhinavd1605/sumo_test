"""
Fixed-Time Baseline Controller for comparison with DNLight.

Implements a simple static signal timing plan that cycles through
green phases with fixed durations. No learning or adaptation.
"""
import numpy as np
from typing import Dict, List, Tuple


class FixedTimeController:
    """
    Static fixed-time signal controller.

    Cycles through green phases with equal duration.
    Each phase gets `green_duration` seconds of green,
    followed by `yellow_duration` seconds of yellow.
    """

    def __init__(self, num_phases: int = 4,
                 green_duration: int = 30,
                 yellow_duration: int = 3):
        """
        Args:
            num_phases: Number of green phases to cycle through.
            green_duration: Seconds of green per phase.
            yellow_duration: Seconds of yellow between phases.
        """
        self.num_phases = num_phases
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.cycle_length = num_phases * (green_duration + yellow_duration)

        self.current_phase = 0
        self.time_in_phase = 0
        self.step_count = 0

    def select_action(self, state=None, training=False) -> int:
        """
        Return the current phase based on fixed timing.

        The controller cycles: phase0 -> phase1 -> phase2 -> phase3 -> phase0...
        Each phase lasts green_duration steps before switching.
        """
        return self.current_phase

    def update(self, step_duration: int = 10):
        """
        Advance the internal timer and switch phases if needed.

        Args:
            step_duration: How many simulation seconds per RL step.
        """
        self.time_in_phase += step_duration
        self.step_count += 1

        # Switch to next phase after green_duration
        if self.time_in_phase >= self.green_duration:
            self.current_phase = (self.current_phase + 1) % self.num_phases
            self.time_in_phase = 0

    def reset(self):
        """Reset the controller for a new episode."""
        self.current_phase = 0
        self.time_in_phase = 0
        self.step_count = 0


class MaxPressureController:
    """
    Max-Pressure adaptive baseline controller.

    Selects the phase with the highest 'pressure' (difference between
    incoming queue and outgoing capacity). A stronger baseline than
    fixed-time but still non-learning.
    """

    def __init__(self, num_phases: int = 4, min_green: int = 10):
        """
        Args:
            num_phases: Number of phases.
            min_green: Minimum green time before switching (seconds).
        """
        self.num_phases = num_phases
        self.min_green = min_green
        self.current_phase = 0
        self.time_in_phase = 0

    def select_action(self, state: np.ndarray = None,
                      training: bool = False) -> int:
        """
        Select phase with maximum pressure from state vector.

        State is assumed to be (num_lanes * 7) with features:
        [phase, queue_length, wait_time, emv, emv_pos, emv_speed, neighbor]
        """
        if state is None:
            return self.current_phase

        # Only switch if minimum green time elapsed
        if self.time_in_phase < self.min_green:
            return self.current_phase

        # Parse queue lengths from state (index 1 of each 7-element group)
        n_features = 7
        n_lanes = len(state) // n_features

        # Compute pressure per phase (sum of queue lengths for lanes in that phase)
        # Simple heuristic: distribute lanes evenly across phases
        lanes_per_phase = max(n_lanes // self.num_phases, 1)

        pressures = []
        for p in range(self.num_phases):
            start_lane = p * lanes_per_phase
            end_lane = min(start_lane + lanes_per_phase, n_lanes)
            pressure = 0.0
            for l in range(start_lane, end_lane):
                idx = l * n_features
                if idx + 1 < len(state):
                    queue = state[idx + 1]  # normalized queue length
                    wait = state[idx + 2]   # normalized wait time
                    pressure += queue + 0.5 * wait
            pressures.append(pressure)

        best_phase = int(np.argmax(pressures))
        return best_phase

    def update(self, step_duration: int = 10):
        """Advance timer."""
        self.time_in_phase += step_duration
        if self.current_phase != self._last_action:
            self.time_in_phase = 0
        self._last_action = self.current_phase

    def reset(self):
        """Reset for new episode."""
        self.current_phase = 0
        self.time_in_phase = 0
        self._last_action = 0
