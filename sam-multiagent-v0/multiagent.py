"""
Multi-Agent Coordination module for the Symbiont Architecture (Step 5).

Introduces SymbiontColony: N MemoryClusters with shared oxytocin signaling.
Each cluster retains its own DES and EHD. Coordination is purely hormonal —
oxytocin from thriving clusters reaches stressed neighbors, facilitating
recovery without spreading cortisol.

Biological analog: the HPA axis operates per-individual, but oxytocinergic
signaling between individuals (touch, vocal cues) can down-regulate cortisol
responses in the receiver without causing cortisol elevation in the sender.

Timing: inter-cluster contributions are deposited into DES accumulators after
each cluster's step() call and applied on the next cycle — a natural one-step
chemical latency.

SAFETY: Does NOT import from or modify any previous step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from cluster import MemoryCluster
from endocrine_neuron import ConsolidationResult
from endocrine_system import EndocrineState
from environment import GlobalWorldState, NeuronContext


@dataclass
class ColonyStepResult:
    """Full output of one SymbiontColony step."""
    endocrine_states:   List[EndocrineState]
    fired_lists:        List[List[bool]]
    broadcasts:         List[float]          # oxytocin * (1 - cortisol) per cluster
    incoming:           List[float]          # inter-cluster delta received per cluster
    plasticity_lists:   List[List[float]]
    consolidation_lists: List[List[ConsolidationResult]]
    step: int


class SymbiontColony:
    """
    N MemoryClusters with fully-connected oxytocin coordination.

    Each step:
      1. Every cluster runs its forward+learn (or consolidate) pass.
      2. Each cluster computes broadcast_i = oxytocin_i * (1 - cortisol_i).
      3. Each cluster receives mean(broadcast_j, j≠i) * coupling_strength
         as an oxytocin delta, applied at the next step.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_neurons_per_cluster: int = 4,
        n_inputs: int = 8,
        coupling_strength: float = 0.1,
        base_seed: int = 42,
    ) -> None:
        self.n_clusters       = n_clusters
        self.n_neurons        = n_neurons_per_cluster
        self.n_inputs         = n_inputs
        self.coupling_strength = coupling_strength
        self.base_seed        = base_seed

        self.clusters: List[MemoryCluster] = [
            MemoryCluster(
                n_neurons=n_neurons_per_cluster,
                n_inputs=n_inputs,
                base_seed=base_seed + i * 100,
            )
            for i in range(n_clusters)
        ]

        self._last_broadcasts: List[float] = [0.0] * n_clusters
        self._history: List[ColonyStepResult] = []

    def step(
        self,
        worlds: List[GlobalWorldState],
        contexts_per_cluster: List[List[NeuronContext]],
    ) -> ColonyStepResult:
        assert len(worlds) == self.n_clusters
        assert len(contexts_per_cluster) == self.n_clusters

        endocrine_states: List[EndocrineState]              = []
        fired_lists:      List[List[bool]]                  = []
        plasticity_lists: List[List[float]]                 = []
        consol_lists:     List[List[ConsolidationResult]]   = []

        # 1. Each cluster runs independently
        for cluster, world, ctxs in zip(self.clusters, worlds, contexts_per_cluster):
            state, fired, plasticity, consolidations = cluster.step(world, ctxs)
            endocrine_states.append(state)
            fired_lists.append(fired)
            plasticity_lists.append(plasticity)
            consol_lists.append(consolidations)

        # 2. Compute broadcast signals (stress gates social output)
        broadcasts: List[float] = [
            float(c.des.oxytocin * (1.0 - c.des.cortisol))
            for c in self.clusters
        ]
        self._last_broadcasts = broadcasts

        # 3. Deposit incoming signal into each cluster's DES accumulator
        #    Applied on the NEXT cluster.step() call — one-step latency
        incoming: List[float] = []
        n = self.n_clusters
        for i, cluster in enumerate(self.clusters):
            others    = [b for j, b in enumerate(broadcasts) if j != i]
            mean_recv = sum(others) / len(others) if others else 0.0
            delta     = mean_recv * self.coupling_strength
            cluster.des.receive_contribution(0.0, 0.0, delta)
            incoming.append(delta)

        result = ColonyStepResult(
            endocrine_states   = endocrine_states,
            fired_lists        = fired_lists,
            broadcasts         = broadcasts,
            incoming           = incoming,
            plasticity_lists   = plasticity_lists,
            consolidation_lists = consol_lists,
            step               = worlds[0].step,
        )
        self._history.append(result)
        return result

    def get_aggregate_endocrine(self) -> Dict[str, float]:
        """Colony-wide endocrine summary for the most recent step."""
        if not self._history:
            return {}
        last      = self._history[-1]
        cortiols  = [s.cortisol  for s in last.endocrine_states]
        dopamines = [s.dopamine  for s in last.endocrine_states]
        oxytocins = [s.oxytocin  for s in last.endocrine_states]
        melatonins = [s.melatonin for s in last.endocrine_states]
        return {
            "mean_cortisol":  float(np.mean(cortiols)),
            "mean_dopamine":  float(np.mean(dopamines)),
            "mean_oxytocin":  float(np.mean(oxytocins)),
            "mean_melatonin": float(np.mean(melatonins)),
            "std_oxytocin":   float(np.std(oxytocins)),
            "broadcast_mean": float(np.mean(self._last_broadcasts)),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_clusters":       self.n_clusters,
            "n_neurons":        self.n_neurons,
            "n_inputs":         self.n_inputs,
            "coupling_strength": self.coupling_strength,
            "base_seed":        self.base_seed,
            "clusters": [
                {
                    "neurons": [
                        {
                            "weights":      neuron.weights.tolist(),
                            "ltm_baseline": neuron._ltm_baseline.tolist(),
                            "stability":    neuron._stability.tolist(),
                        }
                        for neuron in cluster.neurons
                    ]
                }
                for cluster in self.clusters
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbiontColony":
        colony = cls(
            n_clusters           = data["n_clusters"],
            n_neurons_per_cluster = data["n_neurons"],
            n_inputs             = data["n_inputs"],
            coupling_strength    = data["coupling_strength"],
            base_seed            = data["base_seed"],
        )
        for c_idx, c_data in enumerate(data["clusters"]):
            cluster = colony.clusters[c_idx]
            for n_idx, n_data in enumerate(c_data["neurons"]):
                neuron              = cluster.neurons[n_idx]
                neuron.weights      = np.array(n_data["weights"],      dtype=np.float64)
                neuron._ltm_baseline = np.array(n_data["ltm_baseline"], dtype=np.float64)
                neuron._stability   = np.array(n_data["stability"],    dtype=np.int32)
        return colony

    @property
    def history(self) -> List[ColonyStepResult]:
        return list(self._history)
