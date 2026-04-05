"""
Endocrine Neuron with Hebbian Plasticity — Symbiont Learning SAM (Step 3).

This module extends the TernaryNeuron from Step 2 with a Hebbian learning rule
that is modulated by the endocrine system. The key innovation: weights are no
longer static — they change based on experience, but the *rate* of learning is
gated by hormones.

Biological analog: synaptic plasticity modulated by neuromodulators.
  - Dopamine gates Long-Term Potentiation (LTP): high dopamine → learn faster.
  - Cortisol gates Long-Term Depression (LTD) / consolidation: high cortisol →
    freeze weights (protective consolidation under stress).
  - Oxytocin modulates social learning bias: high oxytocin → weight changes
    favor coordination-compatible patterns.

The learning rule: Ternary Hebbian Update
  For each synapse i, after the neuron fires:
    delta_i = sign(input_i)   if input_i contributed to firing
    delta_i = 0               otherwise

  The delta is applied probabilistically, gated by the plasticity signal:
    plasticity = dopamine * (1 - cortisol)
    P(update) = plasticity * base_learning_rate

  Weights remain ternary: after update, clip to {-1, 0, +1}.

SAFETY: This module does NOT import from or modify sam-neuron-v0 or
sam-cluster-v0. The core forward() logic is identical to v0.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from endocrine_system import EndocrineState


class TernaryNeuron:
    """A ternary neuron with endocrine-gated Hebbian plasticity.

    The forward() method is identical to v0/v1. The new learn() method
    implements the plasticity rule.

    Biological analog: a cortical pyramidal neuron with:
      - Ternary synaptic efficacies (excitatory / silent / inhibitory)
      - Neuromodulator-gated synaptic plasticity
      - Hebbian correlation-based learning
    """

    def __init__(
        self,
        n_inputs: int = 8,
        base_theta: float = 1.0,
        cortisol_gain: float = 2.0,
        dopamine_gain: float = 0.8,
        base_learning_rate: float = 0.3,
        seed: int = 42,
    ) -> None:
        self.n_inputs = n_inputs
        self.base_theta = base_theta
        self.cortisol_gain = cortisol_gain
        self.dopamine_gain = dopamine_gain
        self.base_lr = base_learning_rate

        rng = np.random.default_rng(seed)
        self.weights: np.ndarray = rng.choice([-1, 0, 1], size=n_inputs).astype(np.float64)
        self._initial_weights: np.ndarray = self.weights.copy()

        self._rng = np.random.default_rng(seed + 100)

        # Weight history for visualization
        self._weight_history: List[np.ndarray] = [self.weights.copy()]
        self._plasticity_history: List[float] = []
        self._update_count: int = 0

    def compute_threshold(self, endocrine: EndocrineState) -> float:
        """Compute effective firing threshold. Identical to v0."""
        cortisol_factor = 1.0 + self.cortisol_gain * endocrine.cortisol
        dopamine_factor = 1.0 - self.dopamine_gain * endocrine.dopamine
        return self.base_theta * cortisol_factor * dopamine_factor

    def forward(
        self, inputs: np.ndarray, endocrine: EndocrineState
    ) -> Tuple[bool, float, float]:
        """Compute neuron output. Identical to v0."""
        activation = float(np.dot(self.weights, inputs))
        theta = self.compute_threshold(endocrine)
        fired = activation >= theta
        return fired, activation, theta

    def learn(
        self, inputs: np.ndarray, fired: bool, endocrine: EndocrineState
    ) -> float:
        """Apply endocrine-gated Hebbian learning rule.

        Biological analog: after a postsynaptic spike, evaluate each synapse:
          - If the presynaptic input was active (nonzero) and the neuron fired,
            strengthen the synapse in the direction of the input (Hebbian LTP).
          - If the neuron did NOT fire, no synaptic update occurs (no spike,
            no plasticity signal).

        The plasticity rate is modulated by endocrine state:
          plasticity = dopamine * (1 - cortisol)
        This captures the biological fact that:
          - Dopamine is required for LTP (reward-driven learning)
          - Cortisol blocks plasticity (stress-induced consolidation)

        Args:
            inputs: The input vector that was presented.
            fired: Whether the neuron fired on this input.
            endocrine: Current endocrine state.

        Returns:
            The effective plasticity signal (for logging).
        """
        plasticity = endocrine.dopamine * (1.0 - endocrine.cortisol)
        self._plasticity_history.append(plasticity)

        if not fired:
            self._weight_history.append(self.weights.copy())
            return plasticity

        effective_lr = self.base_lr * plasticity

        for i in range(self.n_inputs):
            if inputs[i] == 0.0:
                continue
            if self._rng.random() < effective_lr:
                # Hebbian: strengthen in direction of input
                new_w = self.weights[i] + np.sign(inputs[i])
                self.weights[i] = float(np.clip(new_w, -1.0, 1.0))
                self._update_count += 1

        self._weight_history.append(self.weights.copy())
        return plasticity

    def contribute(
        self, fired: bool, input_risk: float, input_reward: float
    ) -> Tuple[float, float, float]:
        """Compute neurochemical deltas for the shared EndocrineSystem.

        Identical to cluster v0.
        """
        if not fired:
            return 0.0, 0.0, 0.0

        cort_delta = 0.05 * input_risk
        dopa_delta = 0.05 * input_reward
        oxy_delta = 0.01

        return cort_delta, dopa_delta, oxy_delta

    @property
    def weight_history(self) -> List[np.ndarray]:
        return self._weight_history

    @property
    def plasticity_history(self) -> List[float]:
        return self._plasticity_history

    @property
    def total_updates(self) -> int:
        return self._update_count

    @property
    def weight_drift(self) -> float:
        """L1 distance from initial weights — measures total learning."""
        return float(np.sum(np.abs(self.weights - self._initial_weights)))
