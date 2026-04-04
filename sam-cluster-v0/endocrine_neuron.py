"""
Endocrine Neuron module for the Symbiont Cluster SAM prototype.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

from endocrine_system import EndocrineState


class TernaryNeuron:
    """A neuron with ternary weights modulated by the endocrine system.
    
    Copied EXACTLY from v0 for backward compatibility, then extended with 
    the `contribute` method to support cluster dynamics (Neurochemical deposits).
    """

    def __init__(
        self,
        n_inputs: int = 8,
        base_theta: float = 1.0,
        cortisol_gain: float = 2.0,
        dopamine_gain: float = 0.8,
        seed: int = 42,
    ) -> None:
        self.n_inputs = n_inputs
        self.base_theta = base_theta
        self.cortisol_gain = cortisol_gain
        self.dopamine_gain = dopamine_gain

        # Initialize ternary weights randomly from {-1, 0, +1}
        rng = np.random.default_rng(seed)
        self.weights: np.ndarray = rng.choice([-1, 0, 1], size=n_inputs).astype(np.float64)

    def compute_threshold(self, endocrine: EndocrineState) -> float:
        """Compute the effective firing threshold given the current endocrine state."""
        # Using the v0 logic exactly
        cortisol_factor = 1.0 + self.cortisol_gain * endocrine.cortisol
        dopamine_factor = 1.0 - self.dopamine_gain * endocrine.dopamine
        return self.base_theta * cortisol_factor * dopamine_factor

    def forward(self, inputs: np.ndarray, endocrine: EndocrineState) -> Tuple[bool, float, float]:
        """Compute neuron output for given inputs and endocrine state."""
        # Using the v0 logic exactly
        activation = float(np.dot(self.weights, inputs))
        theta = self.compute_threshold(endocrine)
        fired = activation >= theta
        return fired, activation, theta

    def contribute(self, fired: bool, input_risk: float, input_reward: float) -> Tuple[float, float, float]:
        """
        [NEW IN STEP 2]
        Compute the neurochemical deltas to deposit into the shared EndocrineSystem.
        
        If the neuron fires while perceiving high local risk, it secretes cortisol.
        If it fires perceiving high local reward, it secretes dopamine.
        """
        cort_delta = 0.0
        dopa_delta = 0.0
        oxy_delta = 0.0
        
        if fired:
            # The more risk it perceived when firing, the more cortisol it dumps.
            # Scaling factors tuned to allow observable spikes.
            cort_delta = 0.05 * input_risk
            
            # The more reward it perceived when firing, the more dopamine it dumps.
            dopa_delta = 0.05 * input_reward
            
            # Base oxytocin for just firing. Coordination boosts this later in the cluster.
            oxy_delta = 0.01

        return cort_delta, dopa_delta, oxy_delta
