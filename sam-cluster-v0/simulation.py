"""
Simulation runner for the Symbiont Cluster SAM prototype.
"""

from __future__ import annotations

import sys
import json
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cluster import SymbiontCluster
from environment import Environment


def run_simulation(n_steps: int = 1000, n_neurons: int = 4, seed: int = 42) -> dict:
    env = Environment(n_steps=n_steps, n_neurons=n_neurons, seed=seed)
    cluster = SymbiontCluster(n_neurons=n_neurons, base_seed=seed)

    # Recording arrays
    global_risks = []
    global_rewards = []
    
    cortisol_levels = []
    dopamine_levels = []
    oxytocin_levels = []
    
    cortisol_setpoints = []
    dopamine_setpoints = []
    oxytocin_setpoints = []
    
    fired_events_matrix = [] # list of lists, shape (n_steps, n_neurons)
    
    # Track coordination for test 5
    coordination_metrics = []

    for t in range(n_steps):
        world, contexts = env.get_state(t)
        global_risks.append(world.risk)
        global_rewards.append(world.reward)

        endo_state, fired_list = cluster.step(world, contexts)
        
        cortisol_levels.append(endo_state.cortisol)
        dopamine_levels.append(endo_state.dopamine)
        oxytocin_levels.append(endo_state.oxytocin)
        
        cortisol_setpoints.append(endo_state.cortisol_setpoint)
        dopamine_setpoints.append(endo_state.dopamine_setpoint)
        oxytocin_setpoints.append(endo_state.oxytocin_setpoint)
        
        fired_events_matrix.append(fired_list)
        
        n_fired = sum(fired_list)
        coordination_metrics.append(n_fired)

    return {
        "n_steps": n_steps,
        "n_neurons": n_neurons,
        "global_risks": np.array(global_risks),
        "global_rewards": np.array(global_rewards),
        "cortisol": np.array(cortisol_levels),
        "dopamine": np.array(dopamine_levels),
        "oxytocin": np.array(oxytocin_levels),
        "cortisol_sp": np.array(cortisol_setpoints),
        "dopamine_sp": np.array(dopamine_setpoints),
        "oxytocin_sp": np.array(oxytocin_setpoints),
        "fired_matrix": np.array(fired_events_matrix),
        "coordination": np.array(coordination_metrics),
        "env": env
    }


def generate_plot(data: dict, output_path: str = "cluster_simulation_output.png") -> None:
    steps = np.arange(data["n_steps"])
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    fig.suptitle("Symbiont Cluster — Micro-Network Distributed Inhibition", fontsize=16, fontweight="bold")

    c_risk, c_reward = "#d62728", "#2ca02c"
    c_cortisol, c_dopamine, c_oxytocin = "#e377c2", "#17becf", "#9467bd"

    phase_bounds = [(0, 250, "Calm"), (250, 550, "Global Crisis"),
                    (550, 800, "Recovery"), (800, 1000, "Abundance")]

    for ax in axes:
        for start, end, label in phase_bounds:
            ax.axvspan(start, end, alpha=0.06, color="gray")
        ax.axvspan(100, 150, alpha=0.15, color="red", label="Localized Threat (N0)" if ax == axes[0] else "")

    # Subplot 1: External world state
    ax1 = axes[0]
    ax1.plot(steps, data["global_risks"], color=c_risk, alpha=0.7, label="Global Risk (D_t)")
    ax1.plot(steps, data["global_rewards"], color=c_reward, alpha=0.7, label="Global Reward (D_t)")
    ax1.set_title("1. External World State (Global + Local threats)", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_ylim(-0.05, 1.1)

    # Subplot 2a: Dynamic setpoints (EHD proof)
    ax2a = axes[1]
    ax2a.plot(steps, data["cortisol_sp"], color=c_cortisol, linewidth=2.0, label="Cortisol Setpoint")
    ax2a.plot(steps, data["dopamine_sp"], color=c_dopamine, linewidth=2.0, label="Dopamine Setpoint")
    ax2a.plot(steps, data["oxytocin_sp"], color=c_oxytocin, linewidth=2.0, label="Oxytocin Setpoint")
    ax2a.axhline(0.3, color=c_cortisol, linestyle=":", alpha=0.3)
    ax2a.axhline(0.4, color=c_dopamine, linestyle=":", alpha=0.3)
    ax2a.axhline(0.1, color=c_oxytocin, linestyle=":", alpha=0.3)
    ax2a.set_title("2a. EHD: Dynamic Setpoint Recalibration", fontsize=11)
    ax2a.legend(loc="upper right", fontsize=9)
    ax2a.set_ylim(-0.05, 0.85)
    ax2a.set_ylabel("Setpoint")

    # Subplot 2b: Actual hormone levels vs setpoints
    ax2b = axes[2]
    ax2b.plot(steps, data["cortisol"], color=c_cortisol, linewidth=1.5, label="Cortisol")
    ax2b.plot(steps, data["dopamine"], color=c_dopamine, linewidth=1.5, label="Dopamine")
    ax2b.plot(steps, data["oxytocin"], color=c_oxytocin, linewidth=1.5, label="Oxytocin")
    ax2b.plot(steps, data["cortisol_sp"], color=c_cortisol, linestyle="--", alpha=0.4, linewidth=1.0)
    ax2b.plot(steps, data["dopamine_sp"], color=c_dopamine, linestyle="--", alpha=0.4, linewidth=1.0)
    ax2b.plot(steps, data["oxytocin_sp"], color=c_oxytocin, linestyle="--", alpha=0.4, linewidth=1.0)
    ax2b.set_title("2b. Actual Hormone Levels (solid) vs Setpoints (dashed)", fontsize=11)
    ax2b.legend(loc="upper right", fontsize=9)
    ax2b.set_ylim(-0.05, 1.1)
    ax2b.set_ylabel("Level")

    # Subplot 3: Individual neuron firing
    ax3 = axes[3]
    fired = data["fired_matrix"]
    for idx in range(data["n_neurons"]):
        fire_times = steps[fired[:, idx]]
        ax3.eventplot([fire_times], lineoffsets=idx, linelengths=0.5, label=f"Neuron {idx}")
    ax3.set_yticks(range(data["n_neurons"]))
    ax3.set_yticklabels([f"N{i}" for i in range(data["n_neurons"])])
    ax3.set_title("3. Individual Neuron Firing Patterns (Differentiated weights)", fontsize=11)

    # Subplot 4: Distributed inhibition
    ax4 = axes[4]
    cluster_firing_rate = fired.mean(axis=1)
    window = 10
    smoothed = np.convolve(cluster_firing_rate, np.ones(window) / window, mode="same")
    ax4.plot(steps, smoothed, color="#ff7f0e", linewidth=2.0, label=f"Cluster Firing Rate ({window}-step avg)")
    ax4.axvspan(100, 150, alpha=0.2, color="red")
    ax4.set_title("4. Distributed Inhibition (Note the drop during localized threat at step 100-150)", fontsize=11)
    ax4.legend(loc="upper right", fontsize=9)
    ax4.set_ylim(-0.05, 1.05)
    ax4.set_xlabel("Simulation Step")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_validation_tests(data: dict) -> dict:
    results = {}
    
    # Test 1: Backward Compatibility (Unit-level)
    # Verify that TernaryNeuron.forward() produces identical results to a v0
    # neuron given the same weights, inputs, and endocrine state. This confirms
    # the core neuron logic was not altered — only the contribute() method was added.
    from endocrine_neuron import TernaryNeuron as TN
    from endocrine_system import EndocrineState as ES
    test_neuron = TN(n_inputs=8, seed=42)
    ref_rng = np.random.default_rng(99)
    test_states = [
        ES(cortisol=0.2, dopamine=0.4, oxytocin=0.1, cortisol_setpoint=0.3, dopamine_setpoint=0.4, oxytocin_setpoint=0.1, step=0),
        ES(cortisol=0.8, dopamine=0.1, oxytocin=0.05, cortisol_setpoint=0.7, dopamine_setpoint=0.2, oxytocin_setpoint=0.05, step=1),
        ES(cortisol=0.1, dopamine=0.8, oxytocin=0.5, cortisol_setpoint=0.1, dopamine_setpoint=0.7, oxytocin_setpoint=0.4, step=2),
    ]
    all_match = True
    for es in test_states:
        inp = ref_rng.choice([-1.0, 0.0, 1.0], size=8)
        fired, act, theta = test_neuron.forward(inp, es)
        # Recompute v0 logic manually: activation = dot(W, x), theta = base * (1 + cg*c) * (1 - dg*d)
        expected_act = float(np.dot(test_neuron.weights, inp))
        expected_theta = 1.0 * (1.0 + 2.0 * es.cortisol) * (1.0 - 0.8 * es.dopamine)
        if abs(act - expected_act) > 1e-10 or abs(theta - expected_theta) > 1e-10:
            all_match = False
    results["Test 1 (Backward Compatibility)"] = {
        "pass": bool(all_match),
        "value": 1.0 if all_match else 0.0,
        "note": "Unit-level: forward() produces identical output to v0 formula",
    }

    # Test 2: Distributed Inhibition
    # Baseline before threat
    baseline_rate = data["fired_matrix"][50:100].mean()
    # Rate during localized threat to N0
    threat_rate = data["fired_matrix"][100:150].mean()
    drop_pct = (baseline_rate - threat_rate) / max(baseline_rate, 0.01)
    test2 = drop_pct > 0.20
    results["Test 2 (Distributed Inhibition)"] = {"pass": bool(test2), "value": float(drop_pct)}

    # Test 3: EHD at Scale
    initial_cortisol_sp = 0.3
    crisis_end_sp = data["cortisol_sp"][540]
    test3 = crisis_end_sp > initial_cortisol_sp * 1.5
    results["Test 3 (EHD at Scale)"] = {"pass": bool(test3), "value": float(crisis_end_sp)}

    # Test 4: Neuron Differentiation
    neuron_rates = data["fired_matrix"].mean(axis=0)
    rate_std = np.std(neuron_rates)
    test4 = rate_std > 0.05
    results["Test 4 (Neuron Differentiation)"] = {"pass": bool(test4), "value": float(rate_std)}

    # Test 5: Oxytocin Tracking
    # Correlation between coordination (# firing neurons >= 2) and oxytocin levels
    coordination_metric = (data["coordination"] >= 2).astype(float)
    # shift oxytocin slightly because effect is delayed
    if len(coordination_metric) > 1:
        corr = np.corrcoef(coordination_metric[:-1], data["oxytocin"][1:])[0, 1]
    else:
        corr = 0
    test5 = corr > 0.3
    results["Test 5 (Oxytocin Tracking)"] = {"pass": bool(test5), "value": float(corr)}

    return results

def main():
    print("Running Micro-Network Simulation...")
    data = run_simulation(n_steps=1000, n_neurons=4, seed=42)
    generate_plot(data, output_path="cluster_simulation_output.png")
    test_results = run_validation_tests(data)
    
    with open("test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
        
    print("Outputs generated")

if __name__ == "__main__":
    main()
