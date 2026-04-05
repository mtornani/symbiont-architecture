# Step 4: Memory Consolidation — Sleep/Wake Cycles

This directory implements **Step 4** of the Symbiont Architecture prototype. Building upon the Hebbian plasticity from [sam-learning-v0](../sam-learning-v0/), we add **memory consolidation via sleep/wake cycles**: weight changes are initially volatile (short-term memory) and must be consolidated during rest phases to become permanent (long-term memory).

## What's New

- **Short-Term Memory (STM):** Hebbian weight changes are tagged with a stability counter. Without reinforcement or consolidation, they decay back to the LTM baseline.
- **Long-Term Memory (LTM):** During rest phases, weights with sufficient stability are promoted to permanent storage. Their current value becomes the new baseline.
- **Sleep/Wake Cycles:** The environment includes 4 rest phases interspersed with wake phases. During rest, the cluster consolidates instead of learning.
- **Melatonin:** A fourth hormone that rises during rest and gates consolidation. Consolidation probability: `(stability/threshold) * melatonin * (1 - cortisol)`.
- **Cortisol blocks consolidation:** Post-crisis rest phases (high residual cortisol) consolidate fewer memories than calm rest phases.

## Components

| File | Description | Biological Analog |
|------|-------------|-------------------|
| `endocrine_neuron.py` | TernaryNeuron with STM/LTM tracking + `consolidate()` | Neuron with synaptic tagging and capture |
| `endocrine_system.py` | DES with melatonin channel | HPA axis + pineal gland |
| `cluster.py` | MemoryCluster with wake/sleep mode switching | Cortical microcolumn with sleep replay |
| `environment.py` | 8-phase wake/sleep schedule (1200 steps) | Day/night ecological cycle |
| `simulation.py` | Runner + 7-panel plot + 8 tests | Experimental protocol |

## Running the Simulation

```bash
pip install numpy matplotlib
python simulation.py
```

Generates `memory_simulation_output.png` and `test_results.json`.

## Validation Tests

| # | Test | What It Proves |
|---|------|----------------|
| 1 | Backward Compatibility | `forward()` matches v0 formula exactly |
| 2 | Ternary Constraint | All weights remain in {-1, 0, +1} throughout |
| 3 | STM Decay/Consolidation | Volatile weights don't persist indefinitely |
| 4 | Consolidation Occurred | Some weights reached LTM status |
| 5 | Cortisol Blocks Consolidation | Post-calm rest consolidates more than post-crisis rest |
| 6 | Endocrine Gating | More plasticity in abundance than crisis |
| 7 | LTM Persistence | Consolidated weights survive through subsequent wake phases |
| 8 | Sleep Architecture | Melatonin elevated during rest phases |

## Key Insight

Memory is not just learning — it is *selective stabilization*. The agent doesn't remember everything it learned. It remembers what was reinforced during safe conditions and forgets noise. Stress disrupts both learning (cortisol blocks plasticity) and consolidation (cortisol blocks sleep quality). This dual protection means a traumatized agent needs recovery before it can form new stable memories — exactly as in biological systems.
