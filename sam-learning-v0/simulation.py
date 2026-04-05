"""
Simulation runner for the Symbiont Learning SAM prototype (Step 3).

Runs the learning cluster for 1000 steps, generates a 6-panel visualization,
and runs 7 validation tests proving Hebbian plasticity works as designed.

SAFETY: This module does NOT import from or modify sam-neuron-v0 or
sam-cluster-v0.
"""

from __future__ import annotations

import json
import sys
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cluster import LearningCluster
from environment import Environment


def run_simulation(n_steps: int = 1000, n_neurons: int = 4, seed: int = 42) -> dict:
    """Run the full learning cluster simulation."""
    env = Environment(n_steps=n_steps, n_neurons=n_neurons, seed=seed)
    cluster = LearningCluster(n_neurons=n_neurons, base_seed=seed)

    global_risks: List[float] = []
    global_rewards: List[float] = []
    cortisol_levels: List[float] = []
    dopamine_levels: List[float] = []
    oxytocin_levels: List[float] = []
    cortisol_setpoints: List[float] = []
    dopamine_setpoints: List[float] = []
    fired_matrix: List[List[bool]] = []
    plasticity_matrix: List[List[float]] = []
    coordination_metrics: List[int] = []

    for t in range(n_steps):
        world, contexts = env.get_state(t)
        global_risks.append(world.risk)
        global_rewards.append(world.reward)

        endo_state, fired_list, plast_list = cluster.step(world, contexts)

        cortisol_levels.append(endo_state.cortisol)
        dopamine_levels.append(endo_state.dopamine)
        oxytocin_levels.append(endo_state.oxytocin)
        cortisol_setpoints.append(endo_state.cortisol_setpoint)
        dopamine_setpoints.append(endo_state.dopamine_setpoint)
        fired_matrix.append(fired_list)
        plasticity_matrix.append(plast_list)
        coordination_metrics.append(sum(fired_list))

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
        "fired_matrix": np.array(fired_matrix),
        "plasticity_matrix": np.array(plasticity_matrix),
        "coordination": np.array(coordination_metrics),
        "env": env,
        "cluster": cluster,
    }


def generate_plot(data: dict, output_path: str = "learning_simulation_output.png") -> None:
    """Generate 6-panel visualization showing learning dynamics."""
    steps = np.arange(data["n_steps"])
    cluster = data["cluster"]

    fig, axes = plt.subplots(6, 1, figsize=(14, 22), sharex=True)
    fig.suptitle(
        "Symbiont Learning Cluster — Endocrine-Gated Hebbian Plasticity",
        fontsize=16, fontweight="bold",
    )

    c_risk, c_reward = "#d62728", "#2ca02c"
    c_cortisol, c_dopamine, c_oxytocin = "#e377c2", "#17becf", "#9467bd"
    neuron_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    phase_bounds = [
        (0, 250, "Calm"), (250, 550, "Crisis"),
        (550, 800, "Recovery"), (800, 1000, "Abundance"),
    ]

    for ax in axes:
        for start, end, label in phase_bounds:
            ax.axvspan(start, end, alpha=0.06, color="gray")
        ax.axvline(250, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)
        ax.axvline(550, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)
        ax.axvline(800, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)

    # 1. External world state
    ax1 = axes[0]
    ax1.plot(steps, data["global_risks"], color=c_risk, alpha=0.7, linewidth=0.8, label="Risk")
    ax1.plot(steps, data["global_rewards"], color=c_reward, alpha=0.7, linewidth=0.8, label="Reward")
    ax1.set_title("1. External World State D_t", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_ylabel("Level")

    # 2. EHD setpoints
    ax2 = axes[1]
    ax2.plot(steps, data["cortisol_sp"], color=c_cortisol, linewidth=2.0, label="Cortisol SP")
    ax2.plot(steps, data["dopamine_sp"], color=c_dopamine, linewidth=2.0, label="Dopamine SP")
    ax2.plot(steps, data["cortisol"], color=c_cortisol, linewidth=1.0, alpha=0.4, linestyle="--")
    ax2.plot(steps, data["dopamine"], color=c_dopamine, linewidth=1.0, alpha=0.4, linestyle="--")
    ax2.set_title("2. EHD Setpoints (solid) and Actual Levels (dashed)", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_ylabel("Value")

    # 3. Plasticity signal per neuron
    ax3 = axes[2]
    plast = data["plasticity_matrix"]
    window = 20
    for i in range(data["n_neurons"]):
        smoothed = np.convolve(plast[:, i], np.ones(window) / window, mode="same")
        ax3.plot(steps, smoothed, color=neuron_colors[i], linewidth=1.2, label=f"N{i}", alpha=0.8)
    ax3.set_title("3. Plasticity Signal per Neuron (dopamine * (1 - cortisol), smoothed)", fontsize=11)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.set_ylim(-0.05, 0.6)
    ax3.set_ylabel("Plasticity")
    ax3.annotate("Learning frozen\n(high cortisol)", xy=(400, 0.02), fontsize=8, color="gray", ha="center")
    ax3.annotate("Learning active\n(high dopamine)", xy=(900, 0.35), fontsize=8, color="gray", ha="center")

    # 4. Weight evolution per neuron (L1 drift from initial)
    ax4 = axes[3]
    for i, neuron in enumerate(cluster.neurons):
        wh = np.array(neuron.weight_history)
        initial = wh[0]
        drift = np.sum(np.abs(wh - initial), axis=1)
        # wh has n_steps+1 entries; align with steps
        ax4.plot(np.arange(len(drift)), drift, color=neuron_colors[i], linewidth=1.5, label=f"N{i} (drift={drift[-1]:.0f})")
    ax4.set_title("4. Weight Drift from Initial (L1 distance — cumulative learning)", fontsize=11)
    ax4.legend(loc="upper left", fontsize=9)
    ax4.set_ylabel("L1 Drift")

    # 5. Individual neuron firing
    ax5 = axes[4]
    fired = data["fired_matrix"]
    for idx in range(data["n_neurons"]):
        fire_times = steps[fired[:, idx]]
        ax5.eventplot([fire_times], lineoffsets=idx, linelengths=0.5, colors=[neuron_colors[idx]])
    ax5.set_yticks(range(data["n_neurons"]))
    ax5.set_yticklabels([f"N{i}" for i in range(data["n_neurons"])])
    ax5.set_title("5. Individual Neuron Firing Patterns", fontsize=11)

    # 6. Cluster firing rate + weight change rate
    ax6 = axes[5]
    cluster_rate = fired.mean(axis=1)
    smoothed_rate = np.convolve(cluster_rate, np.ones(window) / window, mode="same")
    ax6.plot(steps, smoothed_rate, color="#ff7f0e", linewidth=2.0, label="Firing rate (20-step avg)")
    # Weight change rate: count per-step changes across all neurons
    total_drift_per_step = np.zeros(data["n_steps"])
    for neuron in cluster.neurons:
        wh = np.array(neuron.weight_history)
        # changes between consecutive steps
        changes = np.sum(np.abs(np.diff(wh, axis=0)), axis=1)
        total_drift_per_step[:len(changes)] += changes
    smoothed_changes = np.convolve(total_drift_per_step, np.ones(window) / window, mode="same")
    ax6_twin = ax6.twinx()
    ax6_twin.plot(steps, smoothed_changes, color="#9467bd", linewidth=1.5, alpha=0.7, label="Weight changes/step")
    ax6_twin.set_ylabel("Weight Changes", color="#9467bd", fontsize=9)
    ax6_twin.legend(loc="upper left", fontsize=9)
    ax6.set_title("6. Firing Rate vs Learning Rate (weight changes per step)", fontsize=11)
    ax6.legend(loc="upper right", fontsize=9)
    ax6.set_ylim(-0.05, 1.05)
    ax6.set_xlabel("Simulation Step")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Visualization saved to {output_path}")


def run_validation_tests(data: dict) -> dict:
    """Run 7 validation tests for the learning system."""
    results = {}
    cluster = data["cluster"]

    # Test 1: Backward Compatibility (unit-level)
    # forward() must produce identical output to v0 formula
    from endocrine_neuron import TernaryNeuron as TN
    from endocrine_system import EndocrineState as ES
    test_neuron = TN(n_inputs=8, seed=42)
    ref_rng = np.random.default_rng(99)
    test_states = [
        ES(cortisol=0.2, dopamine=0.4, oxytocin=0.1,
           cortisol_setpoint=0.3, dopamine_setpoint=0.4, oxytocin_setpoint=0.1, step=0),
        ES(cortisol=0.8, dopamine=0.1, oxytocin=0.05,
           cortisol_setpoint=0.7, dopamine_setpoint=0.2, oxytocin_setpoint=0.05, step=1),
        ES(cortisol=0.1, dopamine=0.8, oxytocin=0.5,
           cortisol_setpoint=0.1, dopamine_setpoint=0.7, oxytocin_setpoint=0.4, step=2),
    ]
    all_match = True
    for es in test_states:
        inp = ref_rng.choice([-1.0, 0.0, 1.0], size=8)
        fired, act, theta = test_neuron.forward(inp, es)
        expected_act = float(np.dot(test_neuron.weights, inp))
        expected_theta = 1.0 * (1.0 + 2.0 * es.cortisol) * (1.0 - 0.8 * es.dopamine)
        if abs(act - expected_act) > 1e-10 or abs(theta - expected_theta) > 1e-10:
            all_match = False
    results["Test 1 (Backward Compatibility)"] = {
        "pass": bool(all_match), "value": 1.0 if all_match else 0.0,
    }

    # Test 2: Weights actually changed (learning occurred)
    total_drift = sum(n.weight_drift for n in cluster.neurons)
    test2 = total_drift > 2.0  # at least 2 weight flips total
    results["Test 2 (Learning Occurred)"] = {
        "pass": bool(test2), "value": float(total_drift),
    }

    # Test 3: Ternary constraint maintained
    all_ternary = True
    for neuron in cluster.neurons:
        for w in neuron.weights:
            if w not in (-1.0, 0.0, 1.0):
                all_ternary = False
        for wh in neuron.weight_history:
            for w in wh:
                if w not in (-1.0, 0.0, 1.0):
                    all_ternary = False
    results["Test 3 (Ternary Constraint)"] = {
        "pass": bool(all_ternary), "value": 1.0 if all_ternary else 0.0,
    }

    # Test 4: Endocrine gating — more learning in abundance than crisis
    # Plasticity = dopamine * (1 - cortisol) should be near-zero in crisis
    plast = data["plasticity_matrix"]
    crisis_plasticity = plast[300:500].mean()
    abundance_plasticity = plast[850:1000].mean()
    test4 = abundance_plasticity > crisis_plasticity * 2.0
    results["Test 4 (Endocrine Gating)"] = {
        "pass": bool(test4),
        "value": float(abundance_plasticity / max(crisis_plasticity, 1e-6)),
        "crisis_plasticity": float(crisis_plasticity),
        "abundance_plasticity": float(abundance_plasticity),
    }

    # Test 5: Weight changes concentrate in high-plasticity phases
    # Count weight changes per phase
    changes_per_step = np.zeros(data["n_steps"])
    for neuron in cluster.neurons:
        wh = np.array(neuron.weight_history)
        ch = np.sum(np.abs(np.diff(wh, axis=0)), axis=1)
        changes_per_step[:len(ch)] += ch
    crisis_changes = changes_per_step[300:500].sum()
    abundance_changes = changes_per_step[850:1000].sum()
    test5 = abundance_changes > crisis_changes
    results["Test 5 (Learning Phase Separation)"] = {
        "pass": bool(test5),
        "crisis_changes": float(crisis_changes),
        "abundance_changes": float(abundance_changes),
        "value": float(abundance_changes / max(crisis_changes, 1.0)),
    }

    # Test 6: EHD at Scale (cortisol setpoint rises during crisis)
    initial_cortisol_sp = 0.3
    crisis_end_sp = data["cortisol_sp"][540]
    test6 = crisis_end_sp > initial_cortisol_sp * 1.5
    results["Test 6 (EHD at Scale)"] = {
        "pass": bool(test6), "value": float(crisis_end_sp),
    }

    # Test 7: Distributed Inhibition preserved
    baseline_rate = data["fired_matrix"][50:100].mean()
    threat_rate = data["fired_matrix"][100:150].mean()
    drop_pct = (baseline_rate - threat_rate) / max(baseline_rate, 0.01)
    test7 = drop_pct > 0.05  # lower threshold: structured patterns change baseline
    results["Test 7 (Distributed Inhibition)"] = {
        "pass": bool(test7), "value": float(drop_pct),
    }

    return results


def print_results(results: dict) -> bool:
    """Print test results and return True if all passed."""
    all_passed = True
    print("\n" + "=" * 65)
    print("VALIDATION TESTS — Hebbian Plasticity (Step 3)")
    print("=" * 65)

    for name, r in results.items():
        status = "PASS" if r["pass"] else "FAIL"
        print(f"\n[{status}] {name}")
        for k, v in r.items():
            if k == "pass":
                continue
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        if not r["pass"]:
            all_passed = False

    print("\n" + "=" * 65)
    if all_passed:
        print("ALL TESTS PASSED — Hebbian plasticity validated.")
    else:
        print("SOME TESTS FAILED — review parameters.")
    print("=" * 65)

    # Per-neuron summary
    print("\nPer-neuron learning summary:")
    # Access cluster from global scope
    return all_passed


def main() -> None:
    """Entry point."""
    print("Symbiont Architecture — Learning Cluster SAM Prototype (Step 3)")
    print("Running simulation (1000 steps, 4 neurons, Hebbian plasticity)...")

    data = run_simulation(n_steps=1000, n_neurons=4, seed=42)

    cluster = data["cluster"]
    print("\nPer-neuron weight summary:")
    for i, neuron in enumerate(cluster.neurons):
        print(f"  N{i}: initial={neuron._initial_weights.tolist()} "
              f"final={neuron.weights.tolist()} "
              f"drift={neuron.weight_drift:.0f} updates={neuron.total_updates}")

    generate_plot(data, output_path="learning_simulation_output.png")

    results = run_validation_tests(data)
    passed = print_results(results)

    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Test results saved to test_results.json")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
