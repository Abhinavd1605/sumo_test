"""
Compute Carbon Tracker for DNLight.

Tracks the carbon emissions from the computational infrastructure
(GPU inference) using the formula from the paper:

    Gamma_j(t) = CI_j(t) * E_j_fog(t) / 3.6e6

Where:
    CI_j(t)     = carbon intensity of electricity (gCO2/kWh)
    E_j_fog(t)  = energy consumed by computation (Joules)
    3.6e6       = conversion factor (J to kWh)
"""
import time
import subprocess
import numpy as np
from typing import Dict, Optional


# India average grid carbon intensity (gCO2/kWh)
DEFAULT_CARBON_INTENSITY = 708.0

# Conversion: 1 kWh = 3.6e6 Joules
JOULES_PER_KWH = 3.6e6


class CarbonTracker:
    """
    Track computational carbon emissions during training/inference.

    Measures GPU power draw and computes per-step and cumulative
    carbon emissions using the paper's formula.
    """

    def __init__(self, carbon_intensity: float = DEFAULT_CARBON_INTENSITY,
                 default_power_watts: float = 60.0):
        """
        Args:
            carbon_intensity: CI_j(t) in gCO2/kWh.
            default_power_watts: Fallback GPU power if nvidia-smi unavailable.
        """
        self.carbon_intensity = carbon_intensity
        self.default_power_watts = default_power_watts

        # Tracking state
        self.total_energy_joules = 0.0
        self.total_carbon_gco2 = 0.0
        self.step_count = 0
        self.episode_carbon = 0.0

        # Timing
        self._step_start_time = None

        # Cache GPU power reading (refresh every 10 steps)
        self._cached_power = default_power_watts
        self._cache_counter = 0

    def start_step(self):
        """Mark the start of a computation step (inference/training)."""
        self._step_start_time = time.perf_counter()

    def end_step(self) -> Dict[str, float]:
        """
        Mark the end of a computation step and compute emissions.

        Returns:
            Dict with 'energy_joules', 'carbon_gco2', 'power_watts',
                       'duration_seconds'.
        """
        if self._step_start_time is None:
            return {'energy_joules': 0, 'carbon_gco2': 0,
                    'power_watts': 0, 'duration_seconds': 0}

        duration = time.perf_counter() - self._step_start_time
        self._step_start_time = None

        # Get GPU power
        power = self._get_gpu_power()

        # Energy = Power * Time (Joules)
        energy = power * duration

        # Carbon: Gamma = CI * E / 3.6e6 (gCO2)
        carbon = self.carbon_intensity * energy / JOULES_PER_KWH

        # Accumulate
        self.total_energy_joules += energy
        self.total_carbon_gco2 += carbon
        self.episode_carbon += carbon
        self.step_count += 1

        return {
            'energy_joules': energy,
            'carbon_gco2': carbon,
            'power_watts': power,
            'duration_seconds': duration,
        }

    def reset_episode(self):
        """Reset per-episode tracking (keeps cumulative totals)."""
        ep_carbon = self.episode_carbon
        self.episode_carbon = 0.0
        return ep_carbon

    def get_summary(self) -> Dict[str, float]:
        """Get cumulative carbon tracking summary."""
        return {
            'total_energy_joules': self.total_energy_joules,
            'total_energy_kwh': self.total_energy_joules / JOULES_PER_KWH,
            'total_carbon_gco2': self.total_carbon_gco2,
            'total_carbon_kg': self.total_carbon_gco2 / 1000.0,
            'carbon_intensity_gco2_kwh': self.carbon_intensity,
            'total_steps': self.step_count,
            'avg_carbon_per_step_gco2': (
                self.total_carbon_gco2 / max(self.step_count, 1)
            ),
        }

    def _get_gpu_power(self) -> float:
        """
        Query GPU power draw via nvidia-smi.

        Caches the result and refreshes every 10 calls to avoid
        excessive subprocess overhead.
        """
        self._cache_counter += 1
        if self._cache_counter % 10 != 1:
            return self._cached_power

        try:
            result = subprocess.check_output(
                ['nvidia-smi',
                 '--query-gpu=power.draw',
                 '--format=csv,noheader,nounits'],
                timeout=2,
                stderr=subprocess.DEVNULL,
            )
            power = float(result.decode().strip().split('\n')[0])
            self._cached_power = power
            return power
        except Exception:
            return self.default_power_watts
