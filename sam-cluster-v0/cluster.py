"""
Symbiont Cluster module for the SAM prototype.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

from endocrine_neuron import TernaryNeuron
from endocrine_system import EndocrineSystem, EndocrineState
from environment import GlobalWorldState, NeuronContext


class SymbiontCluster:
    """Orchestrates a micro-network of TernaryNeurons sharing one EndocrineSystem."""

    def __init__(self, n_neurons: int = 4, n_inputs: int = 8, base_seed: int = 42):
        self.n_neurons = n_neurons
        
        # Instantiate diverse neurons based on different seeds
        # Some might be 'Explorers', some 'Guardians' depending on the generated weights
        self.neurons: List[TernaryNeuron] = []
        for i in range(n_neurons):
            self.neurons.append(TernaryNeuron(n_inputs=n_inputs, seed=base_seed + i * 10))
            
        self.des = EndocrineSystem()

        # Step variables
        self.current_state: EndocrineState = self.des.step(GlobalWorldState(0.0, 0.0, 0))

    def step(self, world: GlobalWorldState, contexts: List[NeuronContext]) -> Tuple[EndocrineState, List[bool]]:
        """Run one step of the cluster.
        
        1. Neurons process inputs and potentially fire based on current endocrine state.
        2. Firing neurons deposit neurochemicals into the EndocrineSystem.
        3. EndocrineSystem updates state for next step.
        """
        assert len(contexts) == self.n_neurons
        
        fired_list = []
        
        # 1 & 2: Forward pass and contributions
        for i, neuron in enumerate(self.neurons):
            fired, _, _ = neuron.forward(contexts[i].inputs, self.current_state)
            fired_list.append(fired)
            
            # Compute contribution
            c_delta, d_delta, o_delta = neuron.contribute(
                fired, contexts[i].local_risk, contexts[i].local_reward
            )
            self.des.receive_contribution(c_delta, d_delta, o_delta)
            
        # Coordinated firing boosts oxytocin
        # If more than 1 neuron fires together, we consider it coordination
        n_fired = sum(fired_list)
        if n_fired > 1:
            coordination_bonus = 0.05 * n_fired
            self.des.receive_contribution(0.0, 0.0, coordination_bonus)
        elif n_fired == 1:
            # Isolated firing reduces oxytocin (or rather, no bonus and natural decay)
            # We can actively decrease it for isolation as per prompt
            self.des.receive_contribution(0.0, 0.0, -0.02)
            
        # 3. Step the endocrine system with the global world state
        self.current_state = self.des.step(world)
        
        return self.current_state, fired_list
