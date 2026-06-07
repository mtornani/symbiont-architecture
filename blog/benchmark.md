---
layout: default
title: "When Homeostasis Fails: A Pre-Registered Benchmark on Out-of-Distribution Recovery"
description: "We pre-registered a hypothesis, built a severe test, ran 30 independent trials. The hypothesis was falsified. Here is what we learned and what comes next."
author: Mirko Tornani
date: 2026-06-07
keywords: "EHD benchmark, homeostatic control, Q-learning, out-of-distribution recovery, falsification, cortisol-damping, adaptive homeostasis, active inference, free energy principle"
---

<meta name="description" content="Pre-registered benchmark: EHD homeostatic regulation vs Q-learning on a gain-shift task. Hypothesis falsified. Three separate outcomes: precision, reliability, speed. Honest negative result.">
<meta property="og:title" content="When Homeostasis Fails: A Pre-Registered Benchmark on Out-of-Distribution Recovery">
<meta property="og:description" content="The hypothesis: homeostatic regulation recovers faster than reward learning after a distribution shift. Falsified on reliability and speed. Precise but fragile out-of-distribution.">
<meta property="og:type" content="article">
<link rel="canonical" href="https://mtornani.github.io/symbiont-architecture/blog/benchmark">

# When Homeostasis Fails: A Pre-Registered Benchmark on Out-of-Distribution Recovery

**Mirko Tornani** | June 2026

---

The hypothesis was falsified. Let me say that up front, before the setup, before the numbers.

We pre-registered a claim: the EHD homeostatic agent (cortisol-damping mechanism, `action = -tanh(error × (1 − 0.5 × cortisol))`) recovers faster than a Q-learning baseline after an out-of-distribution perturbation. We built a test designed to be severe enough to show a difference, ran 30 independent trials, and separated the result into three distinct outcomes. Two of the three outcomes went to Q-learning. One went to EHD.

Publishing this matters more than any CONFIRMED verdict would.

---

## The Hypothesis

The EHD cortisol-damping mechanism is designed to respond automatically to overshoot. When the environment changes and the agent's calibrated actions produce too-large effects, the resulting error drives cortisol up, which reduces the effective gain of the control law. The agent doesn't need to learn a new policy — its physiology adjusts the policy in real time.

The explicit pre-registered claim: **the homeostatic agent recovers equilibrium faster than a Q-learning agent after a distribution shift it has never seen**, because cortisol-damping provides zero-latency gain correction while Q-learning requires exploration and table updates.

---

## The Test

**Environment.** A 1D regulation task: `x_{t+1} = x_t + gain_t × action + noise`. Setpoint = 0. Pre-shift gain = 1.0. At step 1000, gain silently shifts to 4.0 — four times the value both agents trained on. Neither agent is told about the shift.

**Pre-shift parity verification.** A plateau gate (ratio of error in last 200 pre-shift steps vs the 200 before that) confirmed both agents had reached steady state before any comparison was made.

**Two agents.** The homeostatic agent (EHD/MemoryCluster, `sam-multiagent-v0`) uses cortisol to modulate action gain in real time. The Q-learning baseline uses tabular Q-learning with 30 state bins, 7 discrete actions in [-1, 1], epsilon annealing from 0.20 to 0.05 over the pre-shift phase, learning rate 0.30. The Q-agent's epsilon schedule was declared in the pre-registration.

**Recovery metric.** Each agent's recovery time is measured against its *own* pre-shift baseline: the first post-shift step where `|error| < pre_baseline_i + 0.1`. This prevents the EHD's structural precision advantage from contaminating the recovery comparison.

**30 independent seeds.** Same noise trajectory per seed pair. Welch's one-tailed t-test for the precision and speed comparisons.

---

## Three Outcomes

### 1. Precision at Regime

**EHD: 0.04 ± 0.004. Q-learning: 0.71 ± 0.63. EHD wins (p < 0.0001).**

At steady state, the EHD agent is an order of magnitude more precise. This was expected: a continuous proportional control law with endocrine gain modulation outperforms a discrete Q-table with 5% residual exploration noise. The EHD holds x near-exactly at the setpoint. Q-learning oscillates within a range determined by its action resolution and epsilon floor.

This outcome is genuine. Precision is where homeostatic regulation has a real and demonstrable advantage.

### 2. Reliability of Recovery

**EHD: 10/30 seeds never recover. Q-learning: 0/30.**

When gain quadruples, the EHD agent's response is bimodal. In 20 of 30 trials, initial shock is small (mean 0.10), the agent recovers within its 0.14 threshold in the first step, and the episode ends with error near zero. In 10 of 30 trials, the agent enters sustained oscillation and never returns within its own threshold over 200 post-shift steps.

Q-learning always recovers. Median recovery: 2 steps. Maximum: 32 steps across all 30 seeds.

**Q-learning wins on reliability.**

### 3. Speed of Recovery

**Not cleanly determinable.** Initial shocks diverge by 14×: EHD mean shock = 0.10 (the cortisol law produces near-zero corrective actions when x ≈ 0, which gain=4 amplifies only slightly), Q-learning mean shock = 1.47 (Q-table actions optimized for gain=1.0 cause large overshoots when gain=4.0 is applied).

Because the EHD agent's speed mean is contaminated by 10 never-recovered trials (these pull the mean to 67.3), comparing means is not informative. Among the 20/30 seeds that do recover, EHD median = 1 step vs Q-learning median = 2 steps — but those are the easy cases where EHD barely moved from its setpoint. The comparison cannot be isolated from the different shock amplitudes. **No winner declared on speed.**

---

## Why It Failed: The Mechanism

The -tanh control law has a fixed sign and a fixed shape. Cortisol modulates its *amplitude* through the term `(1 − 0.5 × cortisol)`, but it cannot change the *curvature* of the response to match a new gain regime. When the world's gain quadruples, the action that was calibrated to move x by 0.1 now moves it by 0.4. The cortisol-damping response reduces the action magnitude — but not enough, and not fast enough, to prevent the oscillations that gain=4 induces.

The deeper issue is structural: **the EHD agent regulates but does not learn**. Its Hebbian plasticity is blocked during stress — high cortisol suppresses the plasticity signal `p = dopamine × (1 − cortisol)` to near zero. The very mechanism that protects against stress-induced bad learning also prevents adaptation when adaptation is what the situation requires.

Q-learning does learn. Its Q-table updates on every step, including post-shift. The agent discovers within a few trials that the actions calibrated for gain=1.0 now overshoot, and adjusts. It doesn't regulate — it adapts.

Precision requires calibration. Robustness requires adaptation. This architecture has the first without the second.

---

## What This Points To

The next hypothesis is **adaptive homeostasis**: an agent that updates its control law, not just its endocrine setpoints. The setpoints in EHD already shift dynamically (`e_t = G(D_t)` — the world-state reshapes what "healthy" means). What does not shift is the *structure* of the response: the -tanh, the gain modulation formula, the fixed mapping from hormonal state to action.

A genuinely adaptive homeostatic agent would need to update this mapping itself — to learn not just what equilibrium to target, but how to move toward it in a world whose physics have changed.

**Active inference** (Friston, 2010) provides a formal candidate framework. In the free energy formulation, the agent maintains a generative model of the world and acts to minimize prediction error *and* to update the model when the environment is non-stationary. The endocrine system could be reinterpreted as a prior over expected world states — one that updates based on accumulated prediction error, not just current error. When gain quadruples and the prior becomes miscalibrated, the agent revises the prior rather than oscillating against it.

This is an open direction and the next hypothesis to formalize and test — not a solution, not a promised architecture. The benchmark above is the first evidence that the fixed-law formulation is insufficient for severe out-of-distribution shifts. That is the precise gap that adaptive homeostasis would need to close.

---

## What an Honest Negative Gives You

A pre-registered hypothesis that falls in a structured way is more informative than a confirmed hypothesis with ambiguous controls. We now know:

- EHD precision advantage is real and large (0.04 vs 0.71)
- EHD reliability under gain×4 shift is not acceptable (10/30 failure rate)
- The failure mode is oscillation, not divergence — the system stays bounded but fails to converge
- The failure is architectural: fixed control law + stress-blocked plasticity = no adaptation path
- Q-learning's advantage is not speed per se but *guaranteedconvergence* through online learning

None of this was obvious before running the test. The benchmark earned its negative result.

---

## Code and Links

All code is open source, reproducible, and runs with Python + NumPy + Matplotlib.

- **Benchmark script**: [`benchmark_homeostasis_vs_reward.py`](https://github.com/mtornani/symbiont-architecture/blob/main/benchmark_homeostasis_vs_reward.py) — v4, 30 seeds, three-outcome report
- **EHD prototypes**: Steps 1-5 in [`sam-neuron-v0/`](https://github.com/mtornani/symbiont-architecture/tree/main/sam-neuron-v0) through [`sam-multiagent-v0/`](https://github.com/mtornani/symbiont-architecture/tree/main/sam-multiagent-v0)
- **Previous post**: [Exocentric Homeostatic Deliberation — five-step proof of concept](ehd)
- **Repository**: [github.com/mtornani/symbiont-architecture](https://github.com/mtornani/symbiont-architecture)

## About the Author

**Mirko Tornani** — Sports Science (University of Bologna), UEFA B License. Independent researcher, Republic of San Marino.

- [GitHub](https://github.com/mtornani)
- [LinkedIn](https://www.linkedin.com/in/mirkotornani/)

## Citation

```
Tornani, M. (2026). When Homeostasis Fails: A Pre-Registered Benchmark on
Out-of-Distribution Recovery. Symbiont Architecture project, June 2026.
https://github.com/mtornani/symbiont-architecture
```
