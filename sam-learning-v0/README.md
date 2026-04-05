# Step 3: Hebbian Plasticity — Learning Cluster

This directory implements **Step 3** of the Symbiont Architecture prototype. Building upon the cluster from [sam-cluster-v0](../sam-cluster-v0/), we add **endocrine-gated Hebbian learning**: ternary weights are no longer static — they adapt based on experience, with the learning rate controlled by the Digital Endocrine System.

## What's New

- **Hebbian Learning Rule:** After each firing event, synaptic weights update toward the input pattern (neurons that fire together wire together).
- **Endocrine Gating:** The plasticity signal `p = dopamine * (1 - cortisol)` controls the learning rate. High stress freezes learning (consolidation). High reward with low stress accelerates learning (exploration).
- **Ternary Constraint Preserved:** Weights always remain in {-1, 0, +1} — learning flips weights between ternary states, never producing continuous values.
- **Learnable Patterns:** The environment now presents structured, repeatable stimuli (threat signatures, reward signatures) that the network can associate with outcomes.

## Components

| File | Description | Biological Analog |
|------|-------------|-------------------|
| `endocrine_neuron.py` | TernaryNeuron with `learn()` method | Pyramidal neuron with synaptic plasticity |
| `endocrine_system.py` | Shared DES with EHD (from Step 2) | HPA axis + mesolimbic pathway |
| `cluster.py` | LearningCluster orchestrator | Cortical microcolumn |
| `environment.py` | Structured patterns + phases | Ecological niche with recurring patterns |
| `simulation.py` | Runner + 6-panel plot + 7 tests | Experimental protocol |

## Running the Simulation

```bash
pip install numpy matplotlib
python simulation.py
```

Generates `learning_simulation_output.png` and `test_results.json`.

## Validation Tests

| # | Test | What It Proves |
|---|------|----------------|
| 1 | Backward Compatibility | `forward()` matches v0 formula exactly |
| 2 | Learning Occurred | Weights drifted from initial values |
| 3 | Ternary Constraint | All weights remain in {-1, 0, +1} throughout |
| 4 | Endocrine Gating | Plasticity is higher in abundance than crisis |
| 5 | Learning Phase Separation | More weight changes during high-dopamine phases |
| 6 | EHD at Scale | Cortisol setpoint rises during sustained crisis |
| 7 | Distributed Inhibition | Cluster-wide firing drop during local threat |

## Key Insight

Learning rate is not a hyperparameter — it is an emergent property of the agent's internal state. A stressed agent does not learn; a safe, rewarded agent learns quickly. This parallels biological findings that cortisol impairs LTP while dopamine facilitates it.
