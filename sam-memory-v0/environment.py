"""
Environment module for the Symbiont Memory SAM prototype (Step 4).

Extends the learning environment with explicit wake/sleep cycles. Rest phases
provide the conditions for memory consolidation: low risk, low reward, near-zero
sensory input.

Biological analog: the day/night ecological cycle. During the day, the organism
forages, encounters threats, and forms short-term memories. During sleep, the
hippocampus replays recent experiences and selectively consolidates useful
associations into long-term storage.

SAFETY: This module does NOT import from or modify any previous step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class GlobalWorldState:
    """Macro environment D_t with sleep/wake phase flag."""
    risk: float
    reward: float
    step: int
    is_rest: bool


@dataclass
class NeuronContext:
    """Localized environment for a specific neuron at a given step."""
    inputs: np.ndarray
    local_risk: float
    local_reward: float


@dataclass
class PatternLabel:
    """Metadata about the stimulus (network never sees this)."""
    pattern_type: str   # "threat", "reward", "neutral", "rest"
    pattern_id: int


class Environment:
    """Generates world states with wake/sleep cycles and learnable patterns.

    8 phases across 1200 steps:
      0-199:    Calm Wake — neutral/reward patterns, baseline learning.
      200-249:  Rest 1 — first consolidation window (low cortisol expected).
      250-499:  Crisis Wake — threat patterns dominate, learning freezes.
      500-549:  Rest 2 — post-crisis consolidation (cortisol may be elevated).
      550-749:  Recovery Wake — mixed patterns, learning resumes.
      750-799:  Rest 3 — recovery consolidation.
      800-1099: Abundance Wake — reward patterns dominate, rapid learning.
      1100-1199: Rest 4 (Deep Sleep) — final consolidation.

    Distributed Inhibition test: Steps 100-150, localized threat to N0.
    """

    def __init__(
        self,
        n_steps: int = 1200,
        n_neurons: int = 4,
        n_inputs: int = 8,
        seed: int = 42,
    ) -> None:
        self.n_steps = n_steps
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.rng = np.random.default_rng(seed)

        self.global_states: List[GlobalWorldState] = []
        self.neuron_contexts: List[List[NeuronContext]] = []
        self.pattern_labels: List[PatternLabel] = []
        self._generate()

    # -- Pattern generators -------------------------------------------------

    def _make_threat_pattern_a(self) -> np.ndarray:
        inp = self.rng.choice([-1.0, 0.0, 1.0], size=self.n_inputs)
        inp[0] = 1.0; inp[1] = 1.0; inp[4] = -1.0
        return inp

    def _make_threat_pattern_b(self) -> np.ndarray:
        inp = self.rng.choice([-1.0, 0.0, 1.0], size=self.n_inputs)
        inp[0] = 1.0; inp[1] = 0.0; inp[5] = -1.0
        return inp

    def _make_reward_pattern_a(self) -> np.ndarray:
        inp = self.rng.choice([-1.0, 0.0, 1.0], size=self.n_inputs)
        inp[2] = 1.0; inp[3] = 1.0; inp[6] = 1.0
        return inp

    def _make_reward_pattern_b(self) -> np.ndarray:
        inp = self.rng.choice([-1.0, 0.0, 1.0], size=self.n_inputs)
        inp[2] = 1.0; inp[3] = 0.0; inp[7] = 1.0
        return inp

    def _make_neutral(self) -> np.ndarray:
        return self.rng.choice([-1.0, 0.0, 1.0], size=self.n_inputs)

    def _make_rest_input(self) -> np.ndarray:
        """Near-zero sensory input during sleep."""
        return np.zeros(self.n_inputs)

    def _select_pattern(
        self, risk: float, reward: float, is_rest: bool
    ) -> Tuple[np.ndarray, PatternLabel]:
        if is_rest:
            return self._make_rest_input(), PatternLabel("rest", -1)

        r = self.rng.random()
        if risk > 0.6:
            if r < 0.4:
                return self._make_threat_pattern_a(), PatternLabel("threat", 0)
            elif r < 0.8:
                return self._make_threat_pattern_b(), PatternLabel("threat", 1)
            else:
                return self._make_neutral(), PatternLabel("neutral", -1)
        elif reward > 0.6:
            if r < 0.4:
                return self._make_reward_pattern_a(), PatternLabel("reward", 0)
            elif r < 0.8:
                return self._make_reward_pattern_b(), PatternLabel("reward", 1)
            else:
                return self._make_neutral(), PatternLabel("neutral", -1)
        elif risk > 0.3 and reward > 0.3:
            if r < 0.3:
                return self._make_threat_pattern_a(), PatternLabel("threat", 0)
            elif r < 0.6:
                return self._make_reward_pattern_a(), PatternLabel("reward", 0)
            else:
                return self._make_neutral(), PatternLabel("neutral", -1)
        else:
            return self._make_neutral(), PatternLabel("neutral", -1)

    def _generate(self) -> None:
        n = self.n_steps
        global_risk = np.zeros(n)
        global_reward = np.zeros(n)
        is_rest = np.zeros(n, dtype=bool)

        # Phase definitions
        # Calm Wake: 0-199
        global_risk[:200] = 0.15 + 0.05 * self.rng.standard_normal(200)
        global_reward[:200] = 0.4 + 0.05 * self.rng.standard_normal(200)

        # Rest 1: 200-249
        global_risk[200:250] = 0.05 + 0.02 * self.rng.standard_normal(50)
        global_reward[200:250] = 0.05 + 0.02 * self.rng.standard_normal(50)
        is_rest[200:250] = True

        # Crisis Wake: 250-499
        global_risk[250:500] = 0.8 + 0.08 * self.rng.standard_normal(250)
        global_reward[250:500] = 0.15 + 0.05 * self.rng.standard_normal(250)

        # Rest 2: 500-549
        global_risk[500:550] = 0.05 + 0.02 * self.rng.standard_normal(50)
        global_reward[500:550] = 0.05 + 0.02 * self.rng.standard_normal(50)
        is_rest[500:550] = True

        # Recovery Wake: 550-749
        t_rec = np.linspace(0, 1, 200)
        global_risk[550:750] = (
            0.7 * (1 - t_rec) + 0.1 * t_rec
            + 0.05 * self.rng.standard_normal(200)
        )
        global_reward[550:750] = (
            0.2 * (1 - t_rec) + 0.65 * t_rec
            + 0.05 * self.rng.standard_normal(200)
        )

        # Rest 3: 750-799
        global_risk[750:800] = 0.05 + 0.02 * self.rng.standard_normal(50)
        global_reward[750:800] = 0.05 + 0.02 * self.rng.standard_normal(50)
        is_rest[750:800] = True

        # Abundance Wake: 800-1099
        global_risk[800:1100] = 0.1 + 0.04 * self.rng.standard_normal(300)
        global_reward[800:1100] = 0.75 + 0.06 * self.rng.standard_normal(300)

        # Rest 4 (Deep Sleep): 1100-1199
        global_risk[1100:] = 0.03 + 0.01 * self.rng.standard_normal(n - 1100)
        global_reward[1100:] = 0.03 + 0.01 * self.rng.standard_normal(n - 1100)
        is_rest[1100:] = True

        global_risk = np.clip(global_risk, 0.0, 1.0)
        global_reward = np.clip(global_reward, 0.0, 1.0)

        for i in range(n):
            gr = float(global_risk[i])
            gw = float(global_reward[i])
            rest = bool(is_rest[i])
            self.global_states.append(
                GlobalWorldState(risk=gr, reward=gw, step=i, is_rest=rest)
            )

            base_pattern, label = self._select_pattern(gr, gw, rest)
            self.pattern_labels.append(label)

            contexts = []
            for j in range(self.n_neurons):
                if rest:
                    contexts.append(NeuronContext(
                        self._make_rest_input(), 0.05, 0.05
                    ))
                    continue

                l_risk = float(np.clip(gr + 0.05 * self.rng.standard_normal(), 0, 1))
                l_reward = float(np.clip(gw + 0.05 * self.rng.standard_normal(), 0, 1))

                # Distributed Inhibition injection
                if 100 <= i < 150 and j == 0:
                    l_risk = 0.95; l_reward = 0.05
                elif 100 <= i < 150 and j != 0:
                    l_risk = 0.1; l_reward = 0.4

                inputs = base_pattern.copy()
                noise_mask = self.rng.random(self.n_inputs) < 0.15
                noise_vals = self.rng.choice([-1.0, 0.0, 1.0], size=self.n_inputs)
                inputs = np.where(noise_mask, noise_vals, inputs)

                inputs[0] = 1.0 if l_risk > 0.6 else (-1.0 if l_risk < 0.3 else 0.0)
                inputs[1] = 1.0 if l_risk > 0.8 else 0.0
                inputs[2] = 1.0 if l_reward > 0.6 else (-1.0 if l_reward < 0.3 else 0.0)
                inputs[3] = 1.0 if l_reward > 0.8 else 0.0

                contexts.append(NeuronContext(inputs, l_risk, l_reward))
            self.neuron_contexts.append(contexts)

    def get_state(self, step: int) -> Tuple[GlobalWorldState, List[NeuronContext]]:
        return self.global_states[step], self.neuron_contexts[step]
