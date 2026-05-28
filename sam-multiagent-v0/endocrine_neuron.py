"""
Endocrine Neuron with STM/LTM Memory — Symbiont Memory SAM (Step 4).

Extends the Hebbian neuron from Step 3 with a synaptic tagging model that
distinguishes short-term memory (volatile weight changes) from long-term
memory (consolidated weights).

Biological analog: synaptic tagging and capture (STC). A Hebbian event
creates a "tag" at the synapse (STM). The tag decays within hours unless
it is "captured" by plasticity-related proteins synthesized during sleep.
Consolidation requires both: (1) a sufficiently strong tag (repeated
reinforcement), and (2) permissive neuromodulatory conditions (low cortisol,
high melatonin — i.e., restful sleep).

Key data structures:
  weights[i]:       current ternary weight {-1, 0, +1}
  ltm_baseline[i]:  last consolidated ternary weight (what the weight reverts
                     to if STM decays)
  stability[i]:     integer counter 0..max_stability tracking how reinforced
                     the current weight change is

SAFETY: Does NOT import from or modify any previous step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from endocrine_system import EndocrineState


@dataclass
class ConsolidationResult:
    """Statistics from a single consolidation call."""
    submitted: int      # weights eligible for consolidation
    consolidated: int   # promoted to LTM
    pruned: int         # reverted to LTM baseline
    skipped: int        # already stable, no action needed


class TernaryNeuron:
    """Ternary neuron with endocrine-gated Hebbian plasticity and STM/LTM.

    forward() is identical to v0. learn() adds stability tracking.
    consolidate() is new: called during rest phases to promote or prune.
    """

    def __init__(
        self,
        n_inputs: int = 8,
        base_theta: float = 1.0,
        cortisol_gain: float = 2.0,
        dopamine_gain: float = 0.8,
        base_learning_rate: float = 0.3,
        max_stability: int = 10,
        consolidation_threshold: int = 3,
        decay_probability: float = 0.02,
        base_consolidation_rate: float = 0.7,
        seed: int = 42,
    ) -> None:
        self.n_inputs = n_inputs
        self.base_theta = base_theta
        self.cortisol_gain = cortisol_gain
        self.dopamine_gain = dopamine_gain
        self.base_lr = base_learning_rate
        self.max_stability = max_stability
        self.consolidation_threshold = consolidation_threshold
        self.decay_prob = decay_probability
        self.base_consolidation_rate = base_consolidation_rate

        rng = np.random.default_rng(seed)
        self.weights: np.ndarray = rng.choice([-1, 0, 1], size=n_inputs).astype(np.float64)
        self._initial_weights: np.ndarray = self.weights.copy()

        # LTM baseline — the "safe" state weights revert to if STM decays
        self._ltm_baseline: np.ndarray = self.weights.copy()

        # Per-weight stability counter: 0 = at LTM baseline, >0 = volatile STM
        self._stability: np.ndarray = np.zeros(n_inputs, dtype=np.int32)

        self._rng = np.random.default_rng(seed + 100)

        # History tracking
        self._weight_history: List[np.ndarray] = [self.weights.copy()]
        self._stability_history: List[np.ndarray] = [self._stability.copy()]
        self._ltm_history: List[np.ndarray] = [self._ltm_baseline.copy()]
        self._plasticity_history: List[float] = []
        self._consolidation_results: List[ConsolidationResult] = []
        self._update_count: int = 0

    def compute_threshold(self, endocrine: EndocrineState) -> float:
        """Identical to v0."""
        cortisol_factor = 1.0 + self.cortisol_gain * endocrine.cortisol
        dopamine_factor = 1.0 - self.dopamine_gain * endocrine.dopamine
        return self.base_theta * cortisol_factor * dopamine_factor

    def forward(
        self, inputs: np.ndarray, endocrine: EndocrineState
    ) -> Tuple[bool, float, float]:
        """Identical to v0."""
        activation = float(np.dot(self.weights, inputs))
        theta = self.compute_threshold(endocrine)
        fired = activation >= theta
        return fired, activation, theta

    def _decay_stm(self) -> None:
        """Probabilistic decay of short-term memory during wake.

        Biological analog: synaptic tags degrade over time without
        protein synthesis. Unreinforced weight changes fade.
        """
        for i in range(self.n_inputs):
            if 0 < self._stability[i] < self.consolidation_threshold:
                if self._rng.random() < self.decay_prob:
                    self._stability[i] -= 1
                    if self._stability[i] <= 0:
                        # Revert to LTM baseline
                        self.weights[i] = self._ltm_baseline[i]
                        self._stability[i] = 0

    def learn(
        self, inputs: np.ndarray, fired: bool, endocrine: EndocrineState
    ) -> float:
        """Endocrine-gated Hebbian learning with stability tracking.

        If a weight update matches the current STM direction (same weight
        value as current), the stability counter increments (reinforcement)
        instead of changing the weight.
        """
        plasticity = endocrine.dopamine * (1.0 - endocrine.cortisol)
        self._plasticity_history.append(plasticity)

        # STM decay happens every wake step
        self._decay_stm()

        if not fired:
            self._weight_history.append(self.weights.copy())
            self._stability_history.append(self._stability.copy())
            self._ltm_history.append(self._ltm_baseline.copy())
            return plasticity

        effective_lr = self.base_lr * plasticity

        for i in range(self.n_inputs):
            if inputs[i] == 0.0:
                continue
            # LTM-consolidated weights resist casual overwriting.
            # Biological analog: long-term memories are structurally
            # stabilized and require significant contradictory evidence
            # to reconsolidate.
            if self._stability[i] >= self.max_stability:
                continue
            if self._rng.random() < effective_lr:
                desired_w = float(np.clip(
                    self.weights[i] + np.sign(inputs[i]), -1.0, 1.0
                ))
                if desired_w == self.weights[i]:
                    # Already at desired value — reinforce stability tag.
                    # Cap at consolidation_threshold: only sleep can promote
                    # past this point to LTM.
                    if self._stability[i] < self.consolidation_threshold:
                        self._stability[i] += 1
                else:
                    # New weight change — reset stability counter
                    self.weights[i] = desired_w
                    self._stability[i] = 1
                self._update_count += 1

        self._weight_history.append(self.weights.copy())
        self._stability_history.append(self._stability.copy())
        self._ltm_history.append(self._ltm_baseline.copy())
        return plasticity

    def consolidate(self, endocrine: EndocrineState) -> ConsolidationResult:
        """Memory consolidation during rest phase.

        For each weight with stability > 0:
          - If stability >= threshold: high chance of consolidation
          - If stability < threshold: evaluated probabilistically
          - Consolidation probability: (stability/threshold) * melatonin * (1-cortisol)

        Consolidated weights: current value becomes new LTM baseline.
        Pruned weights: revert to previous LTM baseline.
        """
        sleep_quality = endocrine.melatonin * (1.0 - endocrine.cortisol)
        submitted = 0
        consolidated = 0
        pruned = 0
        skipped = 0

        for i in range(self.n_inputs):
            if self._stability[i] == 0:
                skipped += 1
                continue

            # Already fully consolidated
            if self._stability[i] >= self.max_stability:
                skipped += 1
                continue

            submitted += 1
            score = self._stability[i] / self.consolidation_threshold
            p_consolidate = min(score * sleep_quality * self.base_consolidation_rate, 0.95)

            if self._rng.random() < p_consolidate:
                # Promote to LTM
                self._ltm_baseline[i] = self.weights[i]
                self._stability[i] = self.max_stability
                consolidated += 1
            else:
                # Gradual pruning: decrement stability each failed attempt.
                # Biological analog: incomplete replay weakens the tag over
                # time rather than erasing it instantly.
                self._stability[i] -= 1
                if self._stability[i] <= 0:
                    # Fully decayed — revert to LTM baseline
                    self.weights[i] = self._ltm_baseline[i]
                    self._stability[i] = 0
                    pruned += 1

        result = ConsolidationResult(submitted, consolidated, pruned, skipped)
        self._consolidation_results.append(result)

        self._weight_history.append(self.weights.copy())
        self._stability_history.append(self._stability.copy())
        self._ltm_history.append(self._ltm_baseline.copy())
        self._plasticity_history.append(0.0)  # no plasticity during rest

        return result

    def contribute(
        self, fired: bool, input_risk: float, input_reward: float
    ) -> Tuple[float, float, float]:
        """Identical to cluster v0."""
        if not fired:
            return 0.0, 0.0, 0.0
        return 0.05 * input_risk, 0.05 * input_reward, 0.01

    @property
    def weight_history(self) -> List[np.ndarray]:
        return self._weight_history

    @property
    def stability_history(self) -> List[np.ndarray]:
        return self._stability_history

    @property
    def ltm_history(self) -> List[np.ndarray]:
        return self._ltm_history

    @property
    def plasticity_history(self) -> List[float]:
        return self._plasticity_history

    @property
    def consolidation_results(self) -> List[ConsolidationResult]:
        return self._consolidation_results

    @property
    def total_updates(self) -> int:
        return self._update_count

    @property
    def weight_drift(self) -> float:
        return float(np.sum(np.abs(self.weights - self._initial_weights)))

    @property
    def stm_count(self) -> int:
        """Number of weights currently in STM state."""
        return int(np.sum(
            (self._stability > 0) & (self._stability < self.max_stability)
        ))

    @property
    def ltm_count(self) -> int:
        """Number of weights that have been consolidated."""
        return int(np.sum(self._stability >= self.max_stability))
