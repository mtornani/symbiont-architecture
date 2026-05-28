"""
Endocrine System for the Symbiont Memory SAM prototype (Step 4).

Extends the DES with melatonin — a fourth hormone that gates memory
consolidation during rest phases.

Biological analog: the pineal gland secretes melatonin during darkness,
which facilitates hippocampal replay and synaptic consolidation. Melatonin
and cortisol are mutually antagonistic — you cannot consolidate memories
effectively while under stress.

SAFETY: Does NOT import from or modify any previous step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from environment import GlobalWorldState


@dataclass
class EndocrineState:
    """Snapshot of the endocrine system including melatonin."""
    cortisol: float
    dopamine: float
    oxytocin: float
    melatonin: float
    cortisol_setpoint: float
    dopamine_setpoint: float
    oxytocin_setpoint: float
    melatonin_setpoint: float
    step: int
    is_rest: bool


class EndocrineSystem:
    """Digital Endocrine System with EHD + melatonin for consolidation gating.

    The melatonin channel:
      - Setpoint rises during rest phases (is_rest=True), drops during wake.
      - Melatonin gates consolidation: consolidation_signal = melatonin * (1 - cortisol).
      - Cortisol suppresses melatonin effectiveness (stress blocks sleep quality).
    """

    def __init__(
        self,
        setpoint_adaptation_rate: float = 0.05,
        hormone_time_constant: float = 0.1,
        inhibition_strength: float = 0.6,
        melatonin_rise_rate: float = 0.15,
    ) -> None:
        self.alpha = setpoint_adaptation_rate
        self.tau = hormone_time_constant
        self.inhibition = inhibition_strength
        self.melatonin_tau = melatonin_rise_rate

        self._cortisol_setpoint: float = 0.3
        self._dopamine_setpoint: float = 0.4
        self._oxytocin_setpoint: float = 0.1
        self._melatonin_setpoint: float = 0.0

        self._cortisol: float = 0.3
        self._dopamine: float = 0.4
        self._oxytocin: float = 0.1
        self._melatonin: float = 0.0

        self._cortisol_delta: float = 0.0
        self._dopamine_delta: float = 0.0
        self._oxytocin_delta: float = 0.0

        self._history: List[EndocrineState] = []

    def receive_contribution(
        self, cortisol_delta: float, dopamine_delta: float, oxytocin_delta: float
    ) -> None:
        self._cortisol_delta += cortisol_delta
        self._dopamine_delta += dopamine_delta
        self._oxytocin_delta += oxytocin_delta

    def _recalibrate_setpoints(self, world: GlobalWorldState) -> None:
        target_cortisol_sp = 0.1 + 0.7 * world.risk
        target_dopamine_sp = 0.1 + 0.7 * world.reward
        safety = 1.0 - world.risk
        target_oxytocin_sp = 0.05 + 0.45 * safety * world.reward

        # Melatonin setpoint: high during rest, zero during wake
        target_melatonin_sp = 0.85 if world.is_rest else 0.0

        self._cortisol_setpoint += self.alpha * (target_cortisol_sp - self._cortisol_setpoint)
        self._dopamine_setpoint += self.alpha * (target_dopamine_sp - self._dopamine_setpoint)
        self._oxytocin_setpoint += self.alpha * (target_oxytocin_sp - self._oxytocin_setpoint)
        # Melatonin adapts faster (biological: rapid onset/offset)
        self._melatonin_setpoint += self.melatonin_tau * (target_melatonin_sp - self._melatonin_setpoint)

    def _update_hormones(self) -> None:
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

        # Melatonin rises faster, suppressed by cortisol
        melatonin_target = self._melatonin_setpoint * (1.0 - 0.5 * self._cortisol)
        self._melatonin += self.melatonin_tau * (melatonin_target - self._melatonin)

        self._cortisol = float(np.clip(self._cortisol, 0.0, 1.0))
        self._dopamine = float(np.clip(self._dopamine, 0.0, 1.0))
        self._oxytocin = float(np.clip(self._oxytocin, 0.0, 1.0))
        self._melatonin = float(np.clip(self._melatonin, 0.0, 1.0))

    def step(self, world: GlobalWorldState) -> EndocrineState:
        self._recalibrate_setpoints(world)
        self._update_hormones()

        state = EndocrineState(
            cortisol=self._cortisol,
            dopamine=self._dopamine,
            oxytocin=self._oxytocin,
            melatonin=self._melatonin,
            cortisol_setpoint=self._cortisol_setpoint,
            dopamine_setpoint=self._dopamine_setpoint,
            oxytocin_setpoint=self._oxytocin_setpoint,
            melatonin_setpoint=self._melatonin_setpoint,
            step=world.step,
            is_rest=world.is_rest,
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
    def melatonin(self) -> float:
        return self._melatonin

    @property
    def cortisol_setpoint(self) -> float:
        return self._cortisol_setpoint

    @property
    def dopamine_setpoint(self) -> float:
        return self._dopamine_setpoint

    @property
    def oxytocin_setpoint(self) -> float:
        return self._oxytocin_setpoint

    @property
    def melatonin_setpoint(self) -> float:
        return self._melatonin_setpoint
