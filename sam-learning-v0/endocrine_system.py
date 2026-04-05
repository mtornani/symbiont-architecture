"""
Endocrine System module for the Symbiont Learning SAM prototype (Step 3).

Identical to the cluster v0 EndocrineSystem with the corrected oxytocin
setpoint formula. The DES is not the learning target — the neuron weights are.
The DES continues to provide the modulatory context that gates plasticity.

SAFETY: This module does NOT import from or modify sam-neuron-v0 or
sam-cluster-v0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from environment import GlobalWorldState


@dataclass
class EndocrineState:
    """Snapshot of the endocrine system at one timestep.

    Biological analog: blood-panel reading of hormone concentrations and
    their homeostatic targets.
    """
    cortisol: float
    dopamine: float
    oxytocin: float
    cortisol_setpoint: float
    dopamine_setpoint: float
    oxytocin_setpoint: float
    step: int


class EndocrineSystem:
    """Digital Endocrine System with EHD and neuron contribution accumulation.

    Biological analog: HPA axis + mesolimbic pathway + oxytocinergic system,
    with allostatic setpoint recalibration (EHD).

    Identical to sam-cluster-v0 EndocrineSystem (with fixed oxytocin formula).
    """

    def __init__(
        self,
        setpoint_adaptation_rate: float = 0.05,
        hormone_time_constant: float = 0.1,
        inhibition_strength: float = 0.6,
    ) -> None:
        self.alpha = setpoint_adaptation_rate
        self.tau = hormone_time_constant
        self.inhibition = inhibition_strength

        self._cortisol_setpoint: float = 0.3
        self._dopamine_setpoint: float = 0.4
        self._oxytocin_setpoint: float = 0.1

        self._cortisol: float = 0.3
        self._dopamine: float = 0.4
        self._oxytocin: float = 0.1

        self._cortisol_delta: float = 0.0
        self._dopamine_delta: float = 0.0
        self._oxytocin_delta: float = 0.0

        self._history: List[EndocrineState] = []

    def receive_contribution(
        self, cortisol_delta: float, dopamine_delta: float, oxytocin_delta: float
    ) -> None:
        """Accumulate neurochemical contributions from neurons."""
        self._cortisol_delta += cortisol_delta
        self._dopamine_delta += dopamine_delta
        self._oxytocin_delta += oxytocin_delta

    def _recalibrate_setpoints(self, world: GlobalWorldState) -> None:
        """EHD: shift setpoints based on world-state signal."""
        target_cortisol_sp = 0.1 + 0.7 * world.risk
        target_dopamine_sp = 0.1 + 0.7 * world.reward
        safety = 1.0 - world.risk
        target_oxytocin_sp = 0.05 + 0.45 * safety * world.reward

        self._cortisol_setpoint += self.alpha * (target_cortisol_sp - self._cortisol_setpoint)
        self._dopamine_setpoint += self.alpha * (target_dopamine_sp - self._dopamine_setpoint)
        self._oxytocin_setpoint += self.alpha * (target_oxytocin_sp - self._oxytocin_setpoint)

    def _update_hormones(self) -> None:
        """Update hormone levels: apply contributions, then decay toward setpoints."""
        self._cortisol += self._cortisol_delta
        self._dopamine += self._dopamine_delta
        self._oxytocin += self._oxytocin_delta

        self._cortisol_delta = 0.0
        self._dopamine_delta = 0.0
        self._oxytocin_delta = 0.0

        self._cortisol += self.tau * (self._cortisol_setpoint - self._cortisol)

        dopamine_target = self._dopamine_setpoint * (1.0 - self.inhibition * self._cortisol)
        self._dopamine += self.tau * (dopamine_target - self._dopamine)

        self._oxytocin += self.tau * (self._oxytocin_setpoint - self._oxytocin)

        self._cortisol = float(np.clip(self._cortisol, 0.0, 1.0))
        self._dopamine = float(np.clip(self._dopamine, 0.0, 1.0))
        self._oxytocin = float(np.clip(self._oxytocin, 0.0, 1.0))

    def step(self, world: GlobalWorldState) -> EndocrineState:
        """Advance the endocrine system by one timestep."""
        self._recalibrate_setpoints(world)
        self._update_hormones()

        state = EndocrineState(
            cortisol=self._cortisol,
            dopamine=self._dopamine,
            oxytocin=self._oxytocin,
            cortisol_setpoint=self._cortisol_setpoint,
            dopamine_setpoint=self._dopamine_setpoint,
            oxytocin_setpoint=self._oxytocin_setpoint,
            step=world.step,
        )
        self._history.append(state)
        return state

    @property
    def history(self) -> List[EndocrineState]:
        return list(self._history)

    @property
    def cortisol(self) -> float:
        return self._cortisol

    @property
    def dopamine(self) -> float:
        return self._dopamine

    @property
    def oxytocin(self) -> float:
        return self._oxytocin

    @property
    def cortisol_setpoint(self) -> float:
        return self._cortisol_setpoint

    @property
    def dopamine_setpoint(self) -> float:
        return self._dopamine_setpoint

    @property
    def oxytocin_setpoint(self) -> float:
        return self._oxytocin_setpoint
