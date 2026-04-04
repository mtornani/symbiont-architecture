"""
Environment module for the Symbiont Architecture SAM prototype.

Biological analog: The external world that an organism inhabits. Just as a
biological organism receives signals from its environment (temperature, threat
level, food availability), the SAM agent receives a structured world-state
signal D_t that drives its internal homeostatic recalibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class WorldState:
    """A snapshot of the external environment at a single timestep.

    Biological analog: the aggregate sensory input an organism processes —
    threat proximity, resource abundance, social signals — collapsed into
    a structured signal the endocrine system can act on.

    Attributes:
        risk: Environmental threat level in [0, 1]. High risk triggers
              cortisol setpoint elevation (fight-or-flight analog).
        reward: Resource/opportunity signal in [0, 1]. High reward supports
                dopamine setpoint elevation (foraging/exploration analog).
        step: The simulation timestep that produced this state.
    """
    risk: float
    reward: float
    step: int


class Environment:
    """Generates a fluctuating world-state signal D_t over time.

    Biological analog: the ecological niche an organism evolves within.
    The environment produces structured phases — periods of sustained danger
    followed by safe foraging windows — rather than pure noise, because
    biological environments have temporal structure that organisms must
    track to survive.

    The environment uses a regime-switching design:
      - Phase 1 (steps 0–249):    Calm baseline — low risk, moderate reward.
      - Phase 2 (steps 250–549):  Sustained crisis — high risk, low reward.
      - Phase 3 (steps 550–799):  Recovery — declining risk, rising reward.
      - Phase 4 (steps 800–999):  Abundance — low risk, high reward.

    This phased structure is critical for the validation tests: the system
    must demonstrably shift its setpoints across regimes.

    Args:
        n_steps: Total number of simulation steps.
        seed: Random seed for reproducibility.
    """

    def __init__(self, n_steps: int = 1000, seed: int = 42) -> None:
        self.n_steps = n_steps
        self.rng = np.random.default_rng(seed)
        self._states: List[WorldState] = []
        self._generate()

    def _generate(self) -> None:
        """Pre-generate the full trajectory of world states."""
        n = self.n_steps
        risk = np.zeros(n)
        reward = np.zeros(n)

        # Phase 1: Calm baseline
        risk[:250] = 0.15 + 0.05 * self.rng.standard_normal(250)
        reward[:250] = 0.4 + 0.05 * self.rng.standard_normal(250)

        # Phase 2: Sustained crisis
        risk[250:550] = 0.8 + 0.08 * self.rng.standard_normal(300)
        reward[250:550] = 0.15 + 0.05 * self.rng.standard_normal(300)

        # Phase 3: Recovery — smooth transition
        t_recovery = np.linspace(0, 1, 250)
        risk[550:800] = 0.8 * (1 - t_recovery) + 0.1 * t_recovery + 0.05 * self.rng.standard_normal(250)
        reward[550:800] = 0.15 * (1 - t_recovery) + 0.7 * t_recovery + 0.05 * self.rng.standard_normal(250)

        # Phase 4: Abundance
        risk[800:] = 0.1 + 0.04 * self.rng.standard_normal(n - 800)
        reward[800:] = 0.75 + 0.06 * self.rng.standard_normal(n - 800)

        # Clamp to [0, 1]
        risk = np.clip(risk, 0.0, 1.0)
        reward = np.clip(reward, 0.0, 1.0)

        self._states = [
            WorldState(risk=float(risk[i]), reward=float(reward[i]), step=i)
            for i in range(n)
        ]

    def get_state(self, step: int) -> WorldState:
        """Return the world state at a given timestep.

        Args:
            step: Simulation timestep (0-indexed).

        Returns:
            The WorldState for that step.
        """
        return self._states[step]

    @property
    def states(self) -> List[WorldState]:
        """Full trajectory of world states."""
        return list(self._states)
