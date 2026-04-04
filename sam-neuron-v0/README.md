# Endocrine Neuron — Symbiont Architecture Proof-of-Concept

## What is the Symbiont Architecture?

The Symbiont Architecture is a theoretical framework proposing a third paradigm for AI safety and ethics. Rather than imposing ethical constraints externally (as in Constitutional AI or RLHF) or hardcoding internal rules, it models ethics as an *emergent property* of internal homeostasis — analogous to how a biologically healthy organism naturally exhibits adaptive, non-destructive behavior because its endocrine system is functioning correctly. A "healthy" AI agent behaves ethically not because it is told to, but because its internal state naturally converges on prosocial action.

## The Rule-Relocation Problem and EHD

If an AI system's internal parameters (e.g., thresholds, biases, reward weights) are fixed by the designer, then ethics hasn't truly *emerged* — the designer's rules have merely been *relocated* from an external constraint layer into the agent's internals. This is the **Rule-Relocation Problem**.

**Exocentric Homeostatic Deliberation (EHD)** solves this by making internal setpoints *dynamic*: they are continuously recalibrated based on signals from the external environment. The agent's definition of "healthy" shifts with context, just as a biological organism's allostatic baseline adapts to sustained environmental change. The setpoints are neither static constants nor free parameters — they are *functions of the world-state*.

## How to Run

**Requirements:** Python 3.9+, NumPy, Matplotlib.

```bash
pip install numpy matplotlib
python simulation.py
```

The simulation runs 1000 timesteps across four environmental phases, generates `simulation_output.png`, and prints validation test results.

## How to Read the Output

The generated plot (`simulation_output.png`) has four subplots:

1. **External World-State D_t** — The environmental signals (risk and reward) that the agent perceives. Four phases are visible: calm baseline, sustained crisis, recovery, and abundance.

2. **★ EHD: Dynamic Setpoint Recalibration** — **This is the key plot.** It shows that the homeostatic setpoints (the targets the endocrine system aims for) are *not static lines* — they shift in response to the environment. During crisis, the cortisol setpoint rises; during abundance, the dopamine setpoint rises. The dashed lines show the initial values for contrast.

3. **Cortisol & Dopamine vs Dynamic Setpoints** — Actual hormone levels (solid) tracking their dynamic setpoints (dashed). Note the mutual inhibition: during high cortisol, dopamine is suppressed below its setpoint.

4. **Neuron Activation Events** — Binary firing events and smoothed firing rate. The neuron fires less during crisis (high cortisol raises threshold) and more during abundance (dopamine lowers threshold). This demonstrates that endocrine modulation directly governs behavior.

## Biology-to-Code Mapping

| Biological Concept | Code Class/Method | Description |
|---|---|---|
| Hypothalamic–pituitary–adrenal (HPA) axis | `EndocrineSystem` | Maintains cortisol and dopamine levels with dynamic setpoints |
| Allostatic setpoint adjustment | `EndocrineSystem._recalibrate_setpoints()` | EHD mechanism: setpoints shift based on world-state signal |
| Hormone secretion/clearance kinetics | `EndocrineSystem._update_hormones()` | Exponential smoothing toward current setpoints |
| HPA–mesolimbic antagonism | `inhibition_strength` parameter | Cortisol suppresses dopamine production |
| Cortical neuron with neuromodulation | `TernaryNeuron` | Ternary-weight neuron whose threshold is modulated by hormones |
| Neuromodulatory excitability control | `TernaryNeuron.compute_threshold()` | Cortisol raises threshold; dopamine lowers it |
| Ecological niche / environment | `Environment` | Generates structured world-state signal D_t |
| Sensory snapshot | `WorldState` | Risk and reward signals at one timestep |
| Blood-panel hormone reading | `EndocrineState` | Snapshot of hormone levels and setpoints |

## License

Research artifact — part of the [Symbiont Architecture](https://mtornani.github.io/symbiont-architecture/) project.
