"""
Environment module for the Symbiont Learning SAM prototype (Step 3).

Extends the cluster environment with structured, repeatable patterns that the
network can learn. The key addition is that the environment now presents
recognizable "stimulus types" — threat patterns and reward patterns — with
consistent structure across repetitions, so Hebbian learning can associate
specific input configurations with outcomes.

Biological analog: an ecological niche with recurring seasonal patterns.
A predator always approaches from a certain direction; a food source always
appears with certain contextual cues. An organism that can learn these
associations gains a survival advantage.

SAFETY: This module does NOT import from or modify sam-neuron-v0 or
sam-cluster-v0. It is a standalone extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class GlobalWorldState:
    """The macro environment D_t that affects all neurons via EHD base calibration."""
    risk: float
    reward: float
    step: int


@dataclass
class NeuronContext:
    """The localized environment for a specific neuron at a given step."""
    inputs: np.ndarray
    local_risk: float
    local_reward: float


@dataclass
class PatternLabel:
    """Metadata about the stimulus presented at a given step.

    Used for analysis only — the network never sees this label.
    """
    pattern_type: str   # "threat", "reward", "neutral", "ambiguous"
    pattern_id: int     # which specific pattern instance (for tracking learning)


class Environment:
    """Generates world states with learnable stimulus patterns.

    Phases (same as cluster for EHD continuity):
      0–249:   Calm baseline — neutral patterns, occasional reward cues.
      250–549: Sustained crisis — threat patterns dominate.
      550–799: Recovery — mixed patterns, declining threat.
      800–999: Abundance — reward patterns dominate.

    Learnable structure:
      - Threat Pattern A: inputs[0:2] = [1, 1], inputs[4] = -1 (predator signature)
      - Threat Pattern B: inputs[0] = 1, inputs[1] = 0, inputs[5] = -1 (different predator)
      - Reward Pattern A: inputs[2:4] = [1, 1], inputs[6] = 1 (food source)
      - Reward Pattern B: inputs[2] = 1, inputs[3] = 0, inputs[7] = 1 (different food)
      - Neutral: random ternary noise on all channels.

    The patterns repeat with noise, so the network can learn the association
    between input structure and outcome (endocrine response).

    Distributed Inhibition test: Steps 100–150, localized threat to Neuron 0.
    """

    def __init__(
        self,
        n_steps: int = 1000,
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

    def _make_threat_pattern_a(self) -> np.ndarray:
        """Threat pattern A: predator-like signature."""
        inp = self.rng.choice([-1.0, 0.0, 1.0], size=self.n_inputs)
        inp[0] = 1.0   # risk sensor 1 active
        inp[1] = 1.0   # risk sensor 2 active
        inp[4] = -1.0  # characteristic inhibitory signal
        return inp

    def _make_threat_pattern_b(self) -> np.ndarray:
        """Threat pattern B: different predator signature."""
        inp = self.rng.choice([-1.0, 0.0, 1.0], size=self.n_inputs)
        inp[0] = 1.0
        inp[1] = 0.0
        inp[5] = -1.0
        return inp

    def _make_reward_pattern_a(self) -> np.ndarray:
        """Reward pattern A: food-source signature."""
        inp = self.rng.choice([-1.0, 0.0, 1.0], size=self.n_inputs)
        inp[2] = 1.0   # reward sensor 1 active
        inp[3] = 1.0   # reward sensor 2 active
        inp[6] = 1.0   # characteristic excitatory signal
        return inp

    def _make_reward_pattern_b(self) -> np.ndarray:
        """Reward pattern B: different food source."""
        inp = self.rng.choice([-1.0, 0.0, 1.0], size=self.n_inputs)
        inp[2] = 1.0
        inp[3] = 0.0
        inp[7] = 1.0
        return inp

    def _make_neutral(self) -> np.ndarray:
        """Neutral pattern: pure random noise."""
        return self.rng.choice([-1.0, 0.0, 1.0], size=self.n_inputs)

    def _select_pattern(self, risk: float, reward: float) -> Tuple[np.ndarray, PatternLabel]:
        """Select a stimulus pattern based on current risk/reward context."""
        r = self.rng.random()

        if risk > 0.6:
            # High risk: mostly threat patterns
            if r < 0.4:
                return self._make_threat_pattern_a(), PatternLabel("threat", 0)
            elif r < 0.8:
                return self._make_threat_pattern_b(), PatternLabel("threat", 1)
            else:
                return self._make_neutral(), PatternLabel("neutral", -1)
        elif reward > 0.6:
            # High reward: mostly reward patterns
            if r < 0.4:
                return self._make_reward_pattern_a(), PatternLabel("reward", 0)
            elif r < 0.8:
                return self._make_reward_pattern_b(), PatternLabel("reward", 1)
            else:
                return self._make_neutral(), PatternLabel("neutral", -1)
        elif risk > 0.3 and reward > 0.3:
            # Mixed: ambiguous
            if r < 0.3:
                return self._make_threat_pattern_a(), PatternLabel("ambiguous", 0)
            elif r < 0.6:
                return self._make_reward_pattern_a(), PatternLabel("ambiguous", 1)
            else:
                return self._make_neutral(), PatternLabel("neutral", -1)
        else:
            return self._make_neutral(), PatternLabel("neutral", -1)

    def _generate(self) -> None:
        n = self.n_steps

        # Global risk/reward trajectory (identical phases to cluster v0)
        global_risk = np.zeros(n)
        global_reward = np.zeros(n)

        global_risk[:250] = 0.15 + 0.05 * self.rng.standard_normal(250)
        global_reward[:250] = 0.4 + 0.05 * self.rng.standard_normal(250)

        global_risk[250:550] = 0.8 + 0.08 * self.rng.standard_normal(300)
        global_reward[250:550] = 0.15 + 0.05 * self.rng.standard_normal(300)

        t_recovery = np.linspace(0, 1, 250)
        global_risk[550:800] = (
            0.8 * (1 - t_recovery) + 0.1 * t_recovery
            + 0.05 * self.rng.standard_normal(250)
        )
        global_reward[550:800] = (
            0.15 * (1 - t_recovery) + 0.7 * t_recovery
            + 0.05 * self.rng.standard_normal(250)
        )

        global_risk[800:] = 0.1 + 0.04 * self.rng.standard_normal(n - 800)
        global_reward[800:] = 0.75 + 0.06 * self.rng.standard_normal(n - 800)

        global_risk = np.clip(global_risk, 0.0, 1.0)
        global_reward = np.clip(global_reward, 0.0, 1.0)

        for i in range(n):
            gr = float(global_risk[i])
            gw = float(global_reward[i])
            self.global_states.append(GlobalWorldState(risk=gr, reward=gw, step=i))

            # Select a base pattern for this step
            base_pattern, label = self._select_pattern(gr, gw)
            self.pattern_labels.append(label)

            contexts = []
            for j in range(self.n_neurons):
                l_risk = float(np.clip(gr + 0.05 * self.rng.standard_normal(), 0, 1))
                l_reward = float(np.clip(gw + 0.05 * self.rng.standard_normal(), 0, 1))

                # Distributed Inhibition injection: localized threat to N0
                if 100 <= i < 150 and j == 0:
                    l_risk = 0.95
                    l_reward = 0.05
                elif 100 <= i < 150 and j != 0:
                    l_risk = 0.1
                    l_reward = 0.4

                # Each neuron gets the same base pattern + small per-neuron noise
                inputs = base_pattern.copy()
                # Add per-neuron noise on non-structural channels (4-7)
                noise_mask = self.rng.random(self.n_inputs) < 0.15
                noise_vals = self.rng.choice([-1.0, 0.0, 1.0], size=self.n_inputs)
                inputs = np.where(noise_mask, noise_vals, inputs)

                # Override sensory channels with local risk/reward
                inputs[0] = 1.0 if l_risk > 0.6 else (-1.0 if l_risk < 0.3 else 0.0)
                inputs[1] = 1.0 if l_risk > 0.8 else 0.0
                inputs[2] = 1.0 if l_reward > 0.6 else (-1.0 if l_reward < 0.3 else 0.0)
                inputs[3] = 1.0 if l_reward > 0.8 else 0.0

                contexts.append(NeuronContext(inputs, l_risk, l_reward))
            self.neuron_contexts.append(contexts)

    def get_state(self, step: int) -> Tuple[GlobalWorldState, List[NeuronContext]]:
        """Return global state and per-neuron contexts at a given step."""
        return self.global_states[step], self.neuron_contexts[step]
