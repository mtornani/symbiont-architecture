"""
Endocrine System module for the Symbiont Cluster SAM prototype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from environment import GlobalWorldState


@dataclass
class EndocrineState:
    cortisol: float
    dopamine: float
    oxytocin: float
    cortisol_setpoint: float
    dopamine_setpoint: float
    oxytocin_setpoint: float
    step: int


class EndocrineSystem:
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

        # Accumulators for neuron contributions in the current step
        self._cortisol_delta: float = 0.0
        self._dopamine_delta: float = 0.0
        self._oxytocin_delta: float = 0.0

        self._history: List[EndocrineState] = []

    def receive_contribution(self, cortisol_delta: float, dopamine_delta: float, oxytocin_delta: float) -> None:
        self._cortisol_delta += cortisol_delta
        self._dopamine_delta += dopamine_delta
        self._oxytocin_delta += oxytocin_delta

    def _recalibrate_setpoints(self, world: GlobalWorldState) -> None:
        target_cortisol_sp = 0.1 + 0.7 * world.risk
        target_dopamine_sp = 0.1 + 0.7 * world.reward
        # Oxytocin setpoint: driven by safety (inverse risk) and reward.
        # In biological systems, oxytocin rises when threat is low and social
        # interaction is rewarding — it requires both safety AND positive context.
        safety = 1.0 - world.risk
        target_oxytocin_sp = 0.05 + 0.45 * safety * world.reward

        self._cortisol_setpoint += self.alpha * (target_cortisol_sp - self._cortisol_setpoint)
        self._dopamine_setpoint += self.alpha * (target_dopamine_sp - self._dopamine_setpoint)
        self._oxytocin_setpoint += self.alpha * (target_oxytocin_sp - self._oxytocin_setpoint)

    def _update_hormones(self) -> None:
        # 1. Apply immediate neuron contributions (secretions)
        self._cortisol += self._cortisol_delta
        self._dopamine += self._dopamine_delta
        self._oxytocin += self._oxytocin_delta

        # clear accumulators
        self._cortisol_delta = 0.0
        self._dopamine_delta = 0.0
        self._oxytocin_delta = 0.0

        # 2. Decay toward setpoints
        self._cortisol += self.tau * (self._cortisol_setpoint - self._cortisol)

        dopamine_target = self._dopamine_setpoint * (1.0 - self.inhibition * self._cortisol)
        self._dopamine += self.tau * (dopamine_target - self._dopamine)
        
        self._oxytocin += self.tau * (self._oxytocin_setpoint - self._oxytocin)

        self._cortisol = np.clip(self._cortisol, 0.0, 1.0)
        self._dopamine = np.clip(self._dopamine, 0.0, 1.0)
        self._oxytocin = np.clip(self._oxytocin, 0.0, 1.0)

    def step(self, world: GlobalWorldState) -> EndocrineState:
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
