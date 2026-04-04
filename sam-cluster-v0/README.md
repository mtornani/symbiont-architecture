# Phase 1: Micro-Network (Symbiont Cluster)

This directory implements **Step 2** of the Symbiont Architecture prototype. Building upon the single endocrine neuron from [sam-neuron-v0](../sam-neuron-v0/), we scale to a micro-network called the **Symbiont Cluster**.

## Goal
Demonstrate *Distributed Inhibition* and emergent multi-agent coordination.

## Components
- **SymbiontCluster**: Manages an array of varied `TernaryNeuron`s sharing a single global `EndocrineSystem`.
- **Endocrine Contribution**: Neurons actively deposit neurochemicals (cortisol, dopamine, oxytocin) into the shared system when they fire under specific local contexts (high risk, high reward).
- **Distributed Inhibition**: We demonstrate that when a single neuron (N0) encounters a localized threat, its cortisol secretion spikes the shared endocrine state, safely inhibiting the rest of the cluster from firing inappropriately.

## Running the Simulation
```bash
python simulation.py
```
This generates `cluster_simulation_output.png` and runs 5 strict validation tests (backward compatibility, distributed inhibition, EHD at scale, neuron differentiation, and oxytocin tracking).
