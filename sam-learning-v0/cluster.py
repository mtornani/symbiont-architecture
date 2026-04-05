"""
Learning Cluster module for the Symbiont Architecture (Step 3).

Extends the SymbiontCluster from Step 2 with Hebbian plasticity. The cluster
now learns: after each step, neurons update their ternary weights based on
what they perceived and whether they fired, gated by the endocrine state.

The learning dynamics create a feedback loop:
  1. Environment presents patterns → neurons fire (or don't)
  2. Firing deposits neurochemicals → endocrine state shifts
  3. Endocrine state gates plasticity → weights update (or freeze)
  4. Updated weights change future firing patterns → cycle repeats

This loop means the cluster *adapts its behavior* based on experience, and
the *rate of adaptation* is controlled by the DES. Under threat (high cortisol),
learning freezes — the organism consolidates. Under safety with reward (high
dopamine), learning accelerates — the organism explores.

SAFETY: This module does NOT import from or modify sam-neuron-v0 or
sam-cluster-v0.
"""

from __future__ import annotations

from typing import List, Tuple

from endocrine_neuron import TernaryNeuron
from endocrine_system import EndocrineSystem, EndocrineState
from environment import GlobalWorldState, NeuronContext


class LearningCluster:
    """A cluster of TernaryNeurons with shared DES and Hebbian plasticity.

    Biological analog: a cortical microcolumn where neurons share
    neuromodulatory context (via volume transmission of dopamine, cortisol,
    oxytocin) and each neuron independently adjusts its synaptic weights
    based on local Hebbian signals gated by the shared modulatory state.
    """

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

        # Initialize current state without phantom history entry
        self.current_state = EndocrineState(
            cortisol=self.des.cortisol,
            dopamine=self.des.dopamine,
            oxytocin=self.des.oxytocin,
            cortisol_setpoint=self.des.cortisol_setpoint,
            dopamine_setpoint=self.des.dopamine_setpoint,
            oxytocin_setpoint=self.des.oxytocin_setpoint,
            step=-1,
        )

    def step(
        self, world: GlobalWorldState, contexts: List[NeuronContext]
    ) -> Tuple[EndocrineState, List[bool], List[float]]:
        """Run one step: forward pass → contributions → learning → DES update.

        Returns:
            Tuple of (endocrine_state, fired_list, plasticity_list).
        """
        assert len(contexts) == self.n_neurons

        fired_list: List[bool] = []
        plasticity_list: List[float] = []

        # 1. Forward pass and contributions
        for i, neuron in enumerate(self.neurons):
            fired, _, _ = neuron.forward(contexts[i].inputs, self.current_state)
            fired_list.append(fired)

            c_delta, d_delta, o_delta = neuron.contribute(
                fired, contexts[i].local_risk, contexts[i].local_reward
            )
            self.des.receive_contribution(c_delta, d_delta, o_delta)

        # 2. Coordination bonus/penalty for oxytocin
        n_fired = sum(fired_list)
        if n_fired > 1:
            coordination_bonus = 0.05 * n_fired
            self.des.receive_contribution(0.0, 0.0, coordination_bonus)
        elif n_fired == 1:
            self.des.receive_contribution(0.0, 0.0, -0.02)

        # 3. Update DES
        self.current_state = self.des.step(world)

        # 4. Hebbian learning (AFTER DES update, so learning uses new state)
        for i, neuron in enumerate(self.neurons):
            p = neuron.learn(contexts[i].inputs, fired_list[i], self.current_state)
            plasticity_list.append(p)

        return self.current_state, fired_list, plasticity_list
