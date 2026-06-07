"""
Environment module for Symbiont Multi-Agent SAM (Step 5).

Generates per-cluster GlobalWorldState and NeuronContext across 6 phases.
Cluster-0 is perturbed in phase 1 while others remain calm, testing whether
inter-cluster oxytocin assists recovery without spreading cortisol.

SAFETY: Does NOT import from or modify any previous step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class GlobalWorldState:
    """Macro environment snapshot — identical signature to sam-memory-v0."""
    risk: float
    reward: float
    step: int
    is_rest: bool


@dataclass
class NeuronContext:
    """Per-neuron localized environment — identical signature to sam-memory-v0."""
    inputs: np.ndarray
    local_risk: float
    local_reward: float


class MultiAgentEnvironment:
    """
    6-phase, N-cluster environment (800 steps total).

    Phase schedule:
      0  (  0-199): Calm       — all clusters stable (low risk, moderate reward)
      1  (200-299): Perturb    — cluster-0 high risk; others unchanged
      2  (300-349): Rest 1     — all clusters sleep (is_rest=True)
      3  (350-549): Recovery   — cluster-0 risk decays linearly back to baseline
      4  (550-749): Abundance  — all clusters high reward, low risk
      5  (750-799): Rest 2     — deep sleep
    """

    N_STEPS: int = 800

    PHASE_BOUNDS: List[Tuple[int, int, str]] = [
        (0,   200, "calm"),
        (200, 300, "perturb"),
        (300, 350, "rest1"),
        (350, 550, "recovery"),
        (550, 750, "abundance"),
        (750, 800, "rest2"),
    ]

    def __init__(
        self,
        n_clusters: int = 3,
        n_neurons: int = 4,
        n_inputs: int = 8,
        seed: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.n_neurons  = n_neurons
        self.n_inputs   = n_inputs
        self.rng        = np.random.default_rng(seed)

        self._worlds:   List[List[GlobalWorldState]]    = []
        self._contexts: List[List[List[NeuronContext]]] = []
        self._generate()

    def _encode_inputs(
        self, base: np.ndarray, local_risk: float, local_reward: float
    ) -> np.ndarray:
        inp = base.copy()
        noise_mask = self.rng.random(self.n_inputs) < 0.15
        noise_vals = self.rng.choice([-1.0, 0.0, 1.0], size=self.n_inputs)
        inp = np.where(noise_mask, noise_vals, inp)
        inp[0] = 1.0 if local_risk   > 0.6 else (-1.0 if local_risk   < 0.3 else 0.0)
        inp[1] = 1.0 if local_risk   > 0.8 else 0.0
        inp[2] = 1.0 if local_reward > 0.6 else (-1.0 if local_reward < 0.3 else 0.0)
        inp[3] = 1.0 if local_reward > 0.8 else 0.0
        return inp

    def _rest_ctx(self) -> NeuronContext:
        return NeuronContext(
            inputs=np.zeros(self.n_inputs),
            local_risk=0.05,
            local_reward=0.05,
        )

    def _generate(self) -> None:
        n  = self.N_STEPS
        nc = self.n_clusters

        risk    = np.zeros((nc, n))
        reward  = np.zeros((nc, n))
        is_rest = np.zeros(n, dtype=bool)

        # Phase 0: Calm (0-199)
        for c in range(nc):
            risk[c,   :200] = np.clip(0.15 + 0.04 * self.rng.standard_normal(200), 0, 1)
            reward[c, :200] = np.clip(0.40 + 0.05 * self.rng.standard_normal(200), 0, 1)

        # Phase 1: Perturbation (200-299) — cluster-0 high stress
        risk[0,   200:300] = np.clip(0.85 + 0.05 * self.rng.standard_normal(100), 0, 1)
        reward[0, 200:300] = np.clip(0.10 + 0.04 * self.rng.standard_normal(100), 0, 1)
        for c in range(1, nc):
            risk[c,   200:300] = np.clip(0.15 + 0.04 * self.rng.standard_normal(100), 0, 1)
            reward[c, 200:300] = np.clip(0.40 + 0.05 * self.rng.standard_normal(100), 0, 1)

        # Phase 2: Rest 1 (300-349)
        is_rest[300:350] = True
        for c in range(nc):
            risk[c,   300:350] = np.clip(0.05 + 0.02 * self.rng.standard_normal(50), 0, 1)
            reward[c, 300:350] = np.clip(0.05 + 0.02 * self.rng.standard_normal(50), 0, 1)

        # Phase 3: Recovery (350-549) — cluster-0 decays back to baseline
        t_rec = np.linspace(0, 1, 200)
        risk[0,   350:550] = np.clip(
            0.80 * (1 - t_rec) + 0.15 * t_rec + 0.05 * self.rng.standard_normal(200), 0, 1
        )
        reward[0, 350:550] = np.clip(
            0.10 * (1 - t_rec) + 0.40 * t_rec + 0.05 * self.rng.standard_normal(200), 0, 1
        )
        for c in range(1, nc):
            risk[c,   350:550] = np.clip(0.15 + 0.04 * self.rng.standard_normal(200), 0, 1)
            reward[c, 350:550] = np.clip(0.40 + 0.05 * self.rng.standard_normal(200), 0, 1)

        # Phase 4: Abundance (550-749)
        for c in range(nc):
            risk[c,   550:750] = np.clip(0.10 + 0.04 * self.rng.standard_normal(200), 0, 1)
            reward[c, 550:750] = np.clip(0.75 + 0.06 * self.rng.standard_normal(200), 0, 1)

        # Phase 5: Rest 2 / Deep Sleep (750-799)
        is_rest[750:] = True
        for c in range(nc):
            risk[c,   750:] = np.clip(0.03 + 0.01 * self.rng.standard_normal(50), 0, 1)
            reward[c, 750:] = np.clip(0.03 + 0.01 * self.rng.standard_normal(50), 0, 1)

        for i in range(n):
            rest = bool(is_rest[i])
            base = self.rng.choice([-1.0, 0.0, 1.0], size=self.n_inputs)

            self._worlds.append([
                GlobalWorldState(
                    risk=float(risk[c, i]),
                    reward=float(reward[c, i]),
                    step=i,
                    is_rest=rest,
                )
                for c in range(nc)
            ])

            ctxs_step: List[List[NeuronContext]] = []
            for c in range(nc):
                if rest:
                    ctxs_step.append([self._rest_ctx() for _ in range(self.n_neurons)])
                    continue
                gr = float(risk[c, i])
                gw = float(reward[c, i])
                neuron_ctxs = []
                for _ in range(self.n_neurons):
                    l_risk   = float(np.clip(gr + 0.05 * self.rng.standard_normal(), 0, 1))
                    l_reward = float(np.clip(gw + 0.05 * self.rng.standard_normal(), 0, 1))
                    inputs   = self._encode_inputs(base, l_risk, l_reward)
                    neuron_ctxs.append(NeuronContext(inputs=inputs, local_risk=l_risk, local_reward=l_reward))
                ctxs_step.append(neuron_ctxs)
            self._contexts.append(ctxs_step)

    def get_state(
        self, step: int
    ) -> Tuple[List[GlobalWorldState], List[List[NeuronContext]]]:
        return self._worlds[step], self._contexts[step]

    def phase_label(self, step: int) -> str:
        for start, end, label in self.PHASE_BOUNDS:
            if start <= step < end:
                return label
        return "unknown"

    @property
    def worlds(self) -> List[List[GlobalWorldState]]:
        return self._worlds
