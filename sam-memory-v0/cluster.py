"""
Memory Cluster for the Symbiont Architecture (Step 4).

Extends the LearningCluster with wake/sleep mode switching. During wake,
the cluster operates as in Step 3 (forward + contribute + learn). During
rest, it calls consolidate() on each neuron instead of learn().

SAFETY: Does NOT import from or modify any previous step.
"""

from __future__ import annotations

from typing import List, Tuple

from endocrine_neuron import TernaryNeuron, ConsolidationResult
from endocrine_system import EndocrineSystem, EndocrineState
from environment import GlobalWorldState, NeuronContext


class MemoryCluster:
    """Cluster of TernaryNeurons with shared DES, plasticity, and consolidation."""

    def __init__(
        self,
        n_neurons: int = 4,
        n_inputs: int = 8,
        base_seed: int = 42,
        base_learning_rate: float = 0.3,
    ) -> None:
        self.n_neurons = n_neurons

        self.neurons: List[TernaryNeuron] = []
        for i in range(n_neurons):
            self.neurons.append(
                TernaryNeuron(
                    n_inputs=n_inputs,
                    seed=base_seed + i * 10,
                    base_learning_rate=base_learning_rate,
                )
            )

        self.des = EndocrineSystem()

        self.current_state = EndocrineState(
            cortisol=self.des.cortisol,
            dopamine=self.des.dopamine,
            oxytocin=self.des.oxytocin,
            melatonin=self.des.melatonin,
            cortisol_setpoint=self.des.cortisol_setpoint,
            dopamine_setpoint=self.des.dopamine_setpoint,
            oxytocin_setpoint=self.des.oxytocin_setpoint,
            melatonin_setpoint=self.des.melatonin_setpoint,
            step=-1,
            is_rest=False,
        )

    def step(
        self, world: GlobalWorldState, contexts: List[NeuronContext]
    ) -> Tuple[EndocrineState, List[bool], List[float], List[ConsolidationResult]]:
        """Run one step: wake (forward+learn) or rest (consolidate).

        Returns:
            (endocrine_state, fired_list, plasticity_list, consolidation_list)
            consolidation_list is empty during wake steps.
        """
        assert len(contexts) == self.n_neurons

        fired_list: List[bool] = []
        plasticity_list: List[float] = []
        consolidation_list: List[ConsolidationResult] = []

        # 1. Forward pass and contributions
        for i, neuron in enumerate(self.neurons):
            fired, _, _ = neuron.forward(contexts[i].inputs, self.current_state)
            fired_list.append(fired)

            c_delta, d_delta, o_delta = neuron.contribute(
                fired, contexts[i].local_risk, contexts[i].local_reward
            )
            self.des.receive_contribution(c_delta, d_delta, o_delta)

        # 2. Coordination oxytocin
        n_fired = sum(fired_list)
        if n_fired > 1:
            self.des.receive_contribution(0.0, 0.0, 0.05 * n_fired)
        elif n_fired == 1:
            self.des.receive_contribution(0.0, 0.0, -0.02)

        # 3. Update DES
        self.current_state = self.des.step(world)

        # 4. Learning or consolidation
        if world.is_rest:
            for neuron in self.neurons:
                result = neuron.consolidate(self.current_state)
                consolidation_list.append(result)
                plasticity_list.append(0.0)
        else:
            for i, neuron in enumerate(self.neurons):
                p = neuron.learn(contexts[i].inputs, fired_list[i], self.current_state)
                plasticity_list.append(p)

        return self.current_state, fired_list, plasticity_list, consolidation_list
