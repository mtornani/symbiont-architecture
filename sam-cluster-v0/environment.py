"""
Environment module for the Symbiont Cluster SAM prototype.

Extends the v0 environment to support multi-agent (cluster) dynamics.
It provides a global world-state (D_t) for EHD scaling, AND per-neuron
local contexts to test Distributed Inhibition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class GlobalWorldState:
    """The macro environment D_t that affects all neurons via EHD base calibration."""
    risk: float
    reward: float
    step: int


@dataclass
class NeuronContext:
    """The localized environment for a specific neuron at a given step.
    
    Contains the specific input vector it perceives, plus the underlying local 
    risk/reward embedded in that input, which it uses to compute endocrine deposits.
    """
    inputs: np.ndarray
    local_risk: float
    local_reward: float


class Environment:
    """Generates the world states and per-neuron contexts.

    Phases:
      0–249:   Calm baseline
      250–549: Sustained crisis (global)
      550–799: Recovery
      800–999: Abundance
    
      In addition, to test distributed inhibition:
      Steps 100-150: Localized threat for Neuron 0 ONLY.
    """

    def __init__(self, n_steps: int = 1000, n_neurons: int = 4, n_inputs: int = 8, seed: int = 42) -> None:
        self.n_steps = n_steps
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.rng = np.random.default_rng(seed)
        
        self.global_states: List[GlobalWorldState] = []
        self.neuron_contexts: List[List[NeuronContext]] = []
        self._generate()

    def _generate(self) -> None:
        n = self.n_steps
        global_risk = np.zeros(n)
        global_reward = np.zeros(n)

        # Baseline logic is same as v0
        global_risk[:250] = 0.15 + 0.05 * self.rng.standard_normal(250)
        global_reward[:250] = 0.4 + 0.05 * self.rng.standard_normal(250)

        global_risk[250:550] = 0.8 + 0.08 * self.rng.standard_normal(300)
        global_reward[250:550] = 0.15 + 0.05 * self.rng.standard_normal(300)

        t_recovery = np.linspace(0, 1, 250)
        global_risk[550:800] = 0.8 * (1 - t_recovery) + 0.1 * t_recovery + 0.05 * self.rng.standard_normal(250)
        global_reward[550:800] = 0.15 * (1 - t_recovery) + 0.7 * t_recovery + 0.05 * self.rng.standard_normal(250)

        global_risk[800:] = 0.1 + 0.04 * self.rng.standard_normal(n - 800)
        global_reward[800:] = 0.75 + 0.06 * self.rng.standard_normal(n - 800)

        global_risk = np.clip(global_risk, 0.0, 1.0)
        global_reward = np.clip(global_reward, 0.0, 1.0)

        for i in range(n):
            self.global_states.append(
                GlobalWorldState(risk=float(global_risk[i]), reward=float(global_reward[i]), step=i)
            )

            # Structural inputs: give meaning to the input dimensions.
            base_inputs = self.rng.choice([-1.0, 0.0, 1.0], size=self.n_inputs)
            
            contexts = []
            for j in range(self.n_neurons):
                # Normally, local context matches global context roughly
                l_risk = float(np.clip(global_risk[i] + 0.05 * self.rng.standard_normal(), 0, 1))
                l_reward = float(np.clip(global_reward[i] + 0.05 * self.rng.standard_normal(), 0, 1))
                
                # Distributed Inhibition Test Injection: 
                # Localized threat to Neuron 0 only between steps 100-150
                if 100 <= i < 150 and j == 0:
                    l_risk = 0.95
                    l_reward = 0.05
                elif 100 <= i < 150 and j != 0:
                    l_risk = 0.1  # others are safe
                    l_reward = 0.4

                # Ensure base inputs are shared so weight differentiation dominates
                inputs = base_inputs.copy()
                
                # Inputs 0 and 1 are sensory neurons for Risk
                inputs[0] = 1.0 if l_risk > 0.6 else (-1.0 if l_risk < 0.3 else 0.0)
                inputs[1] = 1.0 if l_risk > 0.8 else 0.0
                
                # Inputs 2 and 3 are sensory neurons for Reward
                inputs[2] = 1.0 if l_reward > 0.6 else (-1.0 if l_reward < 0.3 else 0.0)
                inputs[3] = 1.0 if l_reward > 0.8 else 0.0

                contexts.append(NeuronContext(inputs, l_risk, l_reward))
            self.neuron_contexts.append(contexts)

    def get_state(self, step: int) -> tuple[GlobalWorldState, List[NeuronContext]]:
        return self.global_states[step], self.neuron_contexts[step]
