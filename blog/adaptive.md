---
layout: default
title: "The Repair: One Parameter Closes the Reliability Gap Without Losing Precision"
description: "The previous benchmark falsified fixed-law homeostasis on reliability. This one tests the minimal repair: a single online gain estimate updated from prediction error. Two pre-registered falsification conditions. Neither was met."
author: Mirko Tornani
date: 2026-06-08
keywords: "adaptive homeostasis, online gain estimation, prediction error, active inference, EHD benchmark, out-of-distribution recovery, homeostatic control, gain estimation, small action model"
---

<meta name="description" content="Minimal repair to EHD fixed-law fragility: add one parameter (ĝ, online gain estimate from prediction error). Two PREREG conditions tested. Neither falsified. Precision preserved, reliability restored, speed cost disclosed.">
<meta property="og:title" content="The Repair: One Parameter Closes the Reliability Gap Without Losing Precision">
<meta property="og:description" content="EHD fixed-law: 10/30 seeds never recover after gain×4. Add online gain estimate from prediction error: 0/30. Precision: 0.04 vs Q's 0.71. Speed cost: adaptive takes longer. Honest three-block result.">
<meta property="og:type" content="article">
<link rel="canonical" href="https://mtornani.github.io/symbiont-architecture/blog/adaptive">

# The Repair: One Parameter Closes the Reliability Gap Without Losing Precision

**Mirko Tornani** | June 2026

---

The previous post ended with a failure and a question.

The failure: a homeostatic agent with a fixed control law (`action = -tanh(error × (1 − 0.5 × cortisol))`) recovers precisely in 20 of 30 seeds after a gain×4 shift, then enters sustained oscillation in 10 of 30. It never recovers in those 10. [Q-learning, by contrast, recovers in all 30.](benchmark) The diagnosis was architectural: fixed sign and shape, plus stress-blocked Hebbian plasticity, leaves the agent with no adaptation path when the world's physics change.

The question: can that fragility be repaired without dismantling the precision advantage? Adding robustness is cheap if you are willing to lose what made the agent interesting in the first place. The question is whether you can keep both.

This post is the answer. A single parameter added to the existing agent closes the reliability gap completely. Both pre-registered falsification conditions were tested. Neither was met.

---

## The Pre-Registration

Two falsification conditions, stated before running:

**Condition A — Reliability.** H1A: `mai_recuperato_adaptive < mai_recuperato_fixed` (10/30, from the closed benchmark). Falsified if the adaptive agent fails as often or more often than the fixed agent.

**Condition B — Precision preserved.** H1B: `pre_baseline_adaptive < pre_baseline_q` (Welch one-tailed, p < 0.05). Falsified if the adaptive agent's pre-shift error is not significantly better than Q-learning's. The rationale: if robustness comes at the cost of the precision that was EHD's only demonstrated advantage, the modification is not a repair — it is a lateral move.

Both conditions had to survive for the hypothesis to remain unrefuted. One failure is enough to falsify it.

Same environment as before: 1D regulation task, gain shifts silently from 1.0 to 4.0 at step 1,000. Same 30 seeds. Same noise trajectories. Same three-block metric structure. Same plateau gate.

---

## The Modification

The adaptive agent is the fixed EHD agent with one additional variable: `ĝ`, an online estimate of the world's gain.

At each step:

```
x_predicted  = x_prev + ĝ × action_prev
pred_err     = x_current − x_predicted
ĝ           += LR_GAIN × pred_err × action_prev     # gradient descent on MSE
ĝ            = clip(ĝ, 0.5, 10.0)
action       = clip(−tanh(error × k) / max(ĝ, 0.5), −1, 1)
               where k = 1 − 0.5 × cortisol         # EHD unchanged
```

`LR_GAIN = 0.05`. `ĝ` initializes at 1.0 (the correct pre-shift prior). The cortisol gate on Hebbian plasticity is unchanged — synaptic weight learning remains blocked under stress. Only the gain estimate updates, because gain estimation is model inference, not synaptic learning: the agent should update its world model even when stressed.

**The equity constraint.** The agent is not told about the shift and does not receive the new gain value. It discovers the change from prediction error — the same signal Q-learning uses (reward = −|error|) to update its Q-table. Providing the true gain would invalidate the comparison.

This is the minimum kernel of active inference: prediction error rewrites the generative model. No full Friston framework, no variational inference, no free energy minimization. One update rule, one scalar.

---

## Three Outcomes

### 1. Reliability

**Adaptive: 0/30. Fixed: 10/30. Q: 0/30.**

The fragility is gone. In all 30 seeds, the adaptive agent returns within its own pre-shift baseline plus 0.1 within the 200-step post-shift window. The bimodal distribution that characterized the fixed agent — either near-instant recovery (small initial error, gain barely matters) or permanent oscillation (large perturbation, gain×4 causes runaway) — does not appear.

The mechanism: when gain quadruples and the agent's uncorrected actions would cause oscillation, the prediction error (`x_actual − x_prev − ĝ × action_prev`) is large and structured. This drives `ĝ` upward, which scales down future actions, which dampens the oscillation before it becomes sustained. The gain estimate converges from 1.0 toward 4.0 within roughly 10–30 steps of the shift, depending on the seed.

### 2. Precision

**Adaptive: 0.040 ± 0.004. Fixed: 0.040 ± 0.004. Q: 0.706 ± 0.626. Welch p < 0.0001.**

Pre-shift, `ĝ ≈ 1.0` (the update rule keeps it near the true gain in the stationary regime), so the adaptive agent's action is identical to the fixed agent's. Both achieve the same steady-state precision — an order of magnitude better than Q-learning's structural floor of 0.71, driven by tabular discretization and residual epsilon exploration.

Condition B is not falsified. The adaptive modification adds no pre-shift cost.

### 3. Speed

**Q: mean 4.0 step, median 2. Adaptive: mean 15.7 step, median 1. Q wins (p_inverse = 0.004).**

The adaptive agent is slower than Q on mean recovery time, and this is expected. Most seeds (the ones that would have recovered in 1 step under the fixed law) still recover in 1 step under the adaptive law — the initial perturbation is small enough that gain×4 barely matters. The remaining seeds require `ĝ` to converge before the corrected actions can stabilize the trajectory, which takes 10–70 steps. The maximum is 70 steps; no seed reaches the 200-step cap. The distribution is skewed right, not bimodal.

Q-learning's advantage here is structural: it learns *actions* directly from error, updating its Q-table on every step including post-shift. The adaptive agent has to first estimate how the world has changed, then act on that estimate. Two-step inference is slower than one-step policy update.

The shock divergence from the closed benchmark persists: adaptive mean initial shock = 0.10, Q = 1.47. Both EHD agents start near their setpoint, so the gain shift barely moves them on the first step. Q-learning's calibrated actions for gain=1.0 overshoot immediately under gain=4.0. This means the speed comparison remains asymmetric — the adaptive agent's slower convergence happens from a much smaller starting error. Read the speed numbers in that context.

---

## What the Verdict Means

Both pre-registered falsification conditions survived. The hypothesis is not falsified: adding online gain estimation from prediction error repairs the 10/30 fragility while preserving the 18× precision advantage over Q-learning.

The cost is speed on a subset of seeds (those where gain correction actually matters). This is honest and expected. The agent is doing more work — it is estimating the world's physics from scratch on every perturbation, not receiving them.

Nothing in this result implies that `ĝ` generalizes to other perturbation types. This is one 1D task with one shock (multiplicative gain shift, ratio 4×). A sign flip in gain, an additive offset to the setpoint, a nonlinear coupling term, or a slower drift would each pose a different estimation problem. Whether the same one-parameter update rule handles those cases is the next experimental question, not an assumed answer.

The sequence matters: the first post published a falsification because it happened. This post publishes a conditional non-falsification for the same reason. Neither result was guaranteed in advance.

---

## Code and Links

All code is open source and runs with Python + NumPy + Matplotlib.

- **Adaptive benchmark**: [`benchmark_adaptive_homeostasis.py`](https://github.com/mtornani/symbiont-architecture/blob/main/benchmark_adaptive_homeostasis.py)
- **Previous post (premise)**: [When Homeostasis Fails — the falsified benchmark](benchmark)
- **EHD five-step proof of concept**: [Exocentric Homeostatic Deliberation](ehd)
- **Repository**: [github.com/mtornani/symbiont-architecture](https://github.com/mtornani/symbiont-architecture)

## About the Author

**Mirko Tornani** — Sports Science (University of Bologna), UEFA B License. Independent researcher, Republic of San Marino.

- [GitHub](https://github.com/mtornani)
- [LinkedIn](https://www.linkedin.com/in/mirkotornani/)

## Citation

```
Tornani, M. (2026). The Repair: One Parameter Closes the Reliability Gap
Without Losing Precision. Symbiont Architecture project, June 2026.
https://github.com/mtornani/symbiont-architecture
```
