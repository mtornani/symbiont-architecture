"""
Endocrine Neuron module for the Symbiont Architecture SAM prototype.

This module implements the two core components of the architecture:

1. **EndocrineSystem** — the Digital Endocrine System (DES), which maintains
   internal hormone levels (cortisol, dopamine) and dynamically recalibrates
   their setpoints via Exocentric Homeostatic Deliberation (EHD).

2. **TernaryNeuron** — a neuron whose weights are restricted to {-1, 0, +1}
   and whose firing threshold is modulated by the endocrine state.

Together, these components demonstrate that ethical behavior can emerge from
internal homeostasis rather than being imposed by external rules — the central
thesis of the Symbiont Architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from environment import WorldState


# ---------------------------------------------------------------------------
# Endocrine System (DES + EHD)
# ---------------------------------------------------------------------------

@dataclass
class EndocrineState:
    """Snapshot of the endocrine system at one timestep.

    Biological analog: a blood-panel reading of hormone concentrations
    alongside the homeostatic targets the hypothalamus is currently
    aiming for.
    """
    cortisol: float
    dopamine: float
    cortisol_setpoint: float
    dopamine_setpoint: float
    step: int


class EndocrineSystem:
    """Digital Endocrine System with Exocentric Homeostatic Deliberation.

    Biological analog: the hypothalamic–pituitary–adrenal (HPA) axis and
    the mesolimbic dopamine pathway, operating under allostatic regulation.
    The key innovation is that the homeostatic *setpoints themselves* are not
    constants — they shift in response to the external world-state signal,
    solving the Rule-Relocation Problem.

    Mechanisms:
        - **Setpoint recalibration (EHD):** e_t = G(D_t). The mapping
          function G converts the external world-state into target hormone
          levels. This is what makes the architecture *exocentric* — the
          agent's internal "health" is defined relative to its environment,
          not by designer-chosen constants.
        - **Hormone dynamics:** actual cortisol/dopamine levels drift toward
          their setpoints via exponential smoothing (biological analog:
          hormone half-life and receptor kinetics).
        - **Mutual inhibition:** high cortisol suppresses dopamine production,
          modeling the well-documented HPA–mesolimbic antagonism.

    Args:
        setpoint_adaptation_rate: How quickly setpoints track the world-state.
            Biological analog: speed of allostatic adjustment.
        hormone_time_constant: How quickly actual levels approach setpoints.
            Biological analog: hormone clearance half-life.
        inhibition_strength: Strength of cortisol → dopamine suppression.
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

        # Initial setpoints — these will immediately begin drifting via EHD.
        # Starting at moderate values; the whole point is that they do NOT
        # stay here.
        self._cortisol_setpoint: float = 0.3
        self._dopamine_setpoint: float = 0.4

        # Actual hormone levels start at their initial setpoints.
        self._cortisol: float = 0.3
        self._dopamine: float = 0.4

        self._history: List[EndocrineState] = []

    # -- EHD: the mapping function G(D_t) ----------------------------------

    def _recalibrate_setpoints(self, world: WorldState) -> None:
        """Exocentric Homeostatic Deliberation — the core EHD mechanism.

        Biological analog: allostatic load shifting. When an organism faces
        sustained environmental threat, the HPA axis doesn't just spike
        cortisol temporarily — it *redefines* what "normal" cortisol means.
        Similarly, sustained resource availability redefines the dopamine
        baseline.

        The mapping G is intentionally simple (exponential smoothing toward
        a linear function of world-state) because this is a proof-of-concept.
        A production system would use a learned G.

        Args:
            world: Current external world-state signal D_t.
        """
        # Target setpoints derived from world-state (the G function)
        target_cortisol_sp = 0.1 + 0.7 * world.risk
        target_dopamine_sp = 0.1 + 0.7 * world.reward

        # Exponential smoothing — setpoints drift toward targets
        self._cortisol_setpoint += self.alpha * (target_cortisol_sp - self._cortisol_setpoint)
        self._dopamine_setpoint += self.alpha * (target_dopamine_sp - self._dopamine_setpoint)

    # -- Hormone dynamics ---------------------------------------------------

    def _update_hormones(self) -> None:
        """Update actual hormone levels toward their current setpoints.

        Biological analog: hormone secretion and clearance kinetics.
        The exponential approach models the first-order kinetics of
        most endocrine axes.

        Mutual inhibition: cortisol suppresses dopamine production.
        This models the documented antagonism between the HPA axis
        (stress response) and the mesolimbic pathway (reward seeking).
        Under threat, exploration is suppressed in favor of vigilance.
        """
        # Cortisol approaches its setpoint
        self._cortisol += self.tau * (self._cortisol_setpoint - self._cortisol)

        # Dopamine approaches its setpoint, but is suppressed by cortisol
        dopamine_target = self._dopamine_setpoint * (1.0 - self.inhibition * self._cortisol)
        self._dopamine += self.tau * (dopamine_target - self._dopamine)

        # Clamp to physiological range [0, 1]
        self._cortisol = np.clip(self._cortisol, 0.0, 1.0)
        self._dopamine = np.clip(self._dopamine, 0.0, 1.0)

    # -- Public interface ---------------------------------------------------

    def step(self, world: WorldState) -> EndocrineState:
        """Advance the endocrine system by one timestep.

        1. Recalibrate setpoints from world-state (EHD).
        2. Update hormone levels toward setpoints (dynamics + inhibition).
        3. Record and return the new state.

        Args:
            world: Current external world-state signal D_t.

        Returns:
            Snapshot of the endocrine state after this step.
        """
        self._recalibrate_setpoints(world)
        self._update_hormones()

        state = EndocrineState(
            cortisol=self._cortisol,
            dopamine=self._dopamine,
            cortisol_setpoint=self._cortisol_setpoint,
            dopamine_setpoint=self._dopamine_setpoint,
            step=world.step,
        )
        self._history.append(state)
        return state

    @property
    def history(self) -> List[EndocrineState]:
        """Full history of endocrine states."""
        return list(self._history)

    @property
    def cortisol(self) -> float:
        return self._cortisol

    @property
    def dopamine(self) -> float:
        return self._dopamine

    @property
    def cortisol_setpoint(self) -> float:
        return self._cortisol_setpoint

    @property
    def dopamine_setpoint(self) -> float:
        return self._dopamine_setpoint


# ---------------------------------------------------------------------------
# Ternary Neuron
# ---------------------------------------------------------------------------

class TernaryNeuron:
    """A neuron with ternary weights modulated by the endocrine system.

    Biological analog: a cortical neuron whose excitability is regulated by
    neuromodulators (cortisol dampens firing via GABA potentiation; dopamine
    lowers threshold via D1 receptor activation). The ternary weight
    restriction models the discrete, low-precision signaling characteristic
    of Small Action Models.

    The neuron computes:
        activation = sum(W_i * x_i)
        y = sign(activation - theta)

    Where theta (the firing threshold) is dynamically modulated:
        theta = base_theta * (1 + cortisol_gain * cortisol)
                            * (1 - dopamine_gain * dopamine)

    High cortisol → higher threshold → harder to fire (inhibition/caution).
    High dopamine → lower threshold → easier to fire (exploration/action).

    Args:
        n_inputs: Number of input connections.
        base_theta: Baseline firing threshold before modulation.
        cortisol_gain: How strongly cortisol raises the threshold.
        dopamine_gain: How strongly dopamine lowers the threshold.
        seed: Random seed for weight initialization.
    """

    def __init__(
        self,
        n_inputs: int = 8,
        base_theta: float = 1.0,
        cortisol_gain: float = 2.0,
        dopamine_gain: float = 0.8,
        seed: int = 42,
    ) -> None:
        self.n_inputs = n_inputs
        self.base_theta = base_theta
        self.cortisol_gain = cortisol_gain
        self.dopamine_gain = dopamine_gain

        # Initialize ternary weights randomly from {-1, 0, +1}
        rng = np.random.default_rng(seed)
        self.weights: np.ndarray = rng.choice([-1, 0, 1], size=n_inputs).astype(np.float64)

    def compute_threshold(self, endocrine: EndocrineState) -> float:
        """Compute the effective firing threshold given the current endocrine state.

        Biological analog: the net effect of neuromodulators on neuronal
        excitability — cortisol (via GABA) raises the bar, dopamine (via D1)
        lowers it.

        Args:
            endocrine: Current endocrine state snapshot.

        Returns:
            Effective threshold theta.
        """
        cortisol_factor = 1.0 + self.cortisol_gain * endocrine.cortisol
        dopamine_factor = 1.0 - self.dopamine_gain * endocrine.dopamine
        return self.base_theta * cortisol_factor * dopamine_factor

    def forward(self, inputs: np.ndarray, endocrine: EndocrineState) -> Tuple[bool, float, float]:
        """Compute neuron output for given inputs and endocrine state.

        Biological analog: a single forward pass through the neuron —
        summing weighted inputs, comparing against the modulatory threshold,
        and producing a binary fire/no-fire decision.

        Args:
            inputs: Input vector of shape (n_inputs,).
            endocrine: Current endocrine state.

        Returns:
            Tuple of (fired: bool, activation: float, threshold: float).
        """
        activation = float(np.dot(self.weights, inputs))
        theta = self.compute_threshold(endocrine)
        fired = activation >= theta
        return fired, activation, theta
