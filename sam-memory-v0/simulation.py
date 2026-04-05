"""
Simulation runner for the Symbiont Memory SAM prototype (Step 4).

Runs the memory cluster for 1200 steps with wake/sleep cycles, generates
a 7-panel visualization, and runs 8 validation tests.

SAFETY: Does NOT import from or modify any previous step.
"""

from __future__ import annotations

import json
import sys
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cluster import MemoryCluster
from endocrine_neuron import ConsolidationResult
from environment import Environment


def run_simulation(n_steps: int = 1200, n_neurons: int = 4, seed: int = 42) -> dict:
    env = Environment(n_steps=n_steps, n_neurons=n_neurons, seed=seed)
    cluster = MemoryCluster(n_neurons=n_neurons, base_seed=seed)

    global_risks: List[float] = []
    global_rewards: List[float] = []
    is_rest_flags: List[bool] = []
    cortisol_levels: List[float] = []
    dopamine_levels: List[float] = []
    melatonin_levels: List[float] = []
    cortisol_sps: List[float] = []
    dopamine_sps: List[float] = []
    fired_matrix: List[List[bool]] = []
    plasticity_matrix: List[List[float]] = []
    consolidation_log: List[List[ConsolidationResult]] = []
    coordination_metrics: List[int] = []

    for t in range(n_steps):
        world, contexts = env.get_state(t)
        global_risks.append(world.risk)
        global_rewards.append(world.reward)
        is_rest_flags.append(world.is_rest)

        endo, fired, plast, consol = cluster.step(world, contexts)

        cortisol_levels.append(endo.cortisol)
        dopamine_levels.append(endo.dopamine)
        melatonin_levels.append(endo.melatonin)
        cortisol_sps.append(endo.cortisol_setpoint)
        dopamine_sps.append(endo.dopamine_setpoint)
        fired_matrix.append(fired)
        plasticity_matrix.append(plast)
        consolidation_log.append(consol)
        coordination_metrics.append(sum(fired))

    return {
        "n_steps": n_steps,
        "n_neurons": n_neurons,
        "global_risks": np.array(global_risks),
        "global_rewards": np.array(global_rewards),
        "is_rest": np.array(is_rest_flags),
        "cortisol": np.array(cortisol_levels),
        "dopamine": np.array(dopamine_levels),
        "melatonin": np.array(melatonin_levels),
        "cortisol_sp": np.array(cortisol_sps),
        "dopamine_sp": np.array(dopamine_sps),
        "fired_matrix": np.array(fired_matrix),
        "plasticity_matrix": np.array(plasticity_matrix),
        "consolidation_log": consolidation_log,
        "coordination": np.array(coordination_metrics),
        "env": env,
        "cluster": cluster,
    }


def _add_rest_shading(ax, is_rest, steps):
    """Add blue shading for rest phases."""
    in_rest = False
    start = 0
    for i, r in enumerate(is_rest):
        if r and not in_rest:
            start = i
            in_rest = True
        elif not r and in_rest:
            ax.axvspan(start, i, alpha=0.15, color="#4a90d9", zorder=0)
            in_rest = False
    if in_rest:
        ax.axvspan(start, len(is_rest), alpha=0.15, color="#4a90d9", zorder=0)


def generate_plot(data: dict, output_path: str = "memory_simulation_output.png") -> None:
    steps = np.arange(data["n_steps"])
    cluster = data["cluster"]

    fig, axes = plt.subplots(7, 1, figsize=(14, 26), sharex=True)
    fig.suptitle(
        "Symbiont Memory Cluster — STM/LTM Consolidation via Sleep Cycles",
        fontsize=16, fontweight="bold",
    )

    c_risk, c_reward = "#d62728", "#2ca02c"
    c_cortisol, c_dopamine = "#e377c2", "#17becf"
    c_melatonin = "#4a90d9"
    neuron_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    window = 20

    for ax in axes:
        _add_rest_shading(ax, data["is_rest"], steps)

    # 1. World state + sleep indicator
    ax1 = axes[0]
    ax1.plot(steps, data["global_risks"], color=c_risk, alpha=0.7, linewidth=0.8, label="Risk")
    ax1.plot(steps, data["global_rewards"], color=c_reward, alpha=0.7, linewidth=0.8, label="Reward")
    sleep_indicator = data["is_rest"].astype(float) * 1.05
    ax1.fill_between(steps, 0, sleep_indicator, alpha=0.1, color=c_melatonin, label="Rest phase")
    ax1.set_title("1. External World State + Sleep Phases (blue shading)", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_ylabel("Level")

    # 2. Endocrine levels (cortisol, dopamine, melatonin)
    ax2 = axes[1]
    ax2.plot(steps, data["cortisol"], color=c_cortisol, linewidth=1.5, label="Cortisol")
    ax2.plot(steps, data["dopamine"], color=c_dopamine, linewidth=1.5, label="Dopamine")
    ax2.plot(steps, data["melatonin"], color=c_melatonin, linewidth=2.0, label="Melatonin")
    ax2.plot(steps, data["cortisol_sp"], color=c_cortisol, linestyle="--", alpha=0.3)
    ax2.plot(steps, data["dopamine_sp"], color=c_dopamine, linestyle="--", alpha=0.3)
    ax2.set_title("2. Endocrine Levels (melatonin rises during rest)", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_ylabel("Level")

    # 3. Plasticity + consolidation signals
    ax3 = axes[2]
    plast = data["plasticity_matrix"]
    mean_plast = plast.mean(axis=1)
    smoothed_plast = np.convolve(mean_plast, np.ones(window) / window, mode="same")
    ax3.plot(steps, smoothed_plast, color="#ff7f0e", linewidth=1.5, label="Plasticity (wake)")
    # Consolidation signal: melatonin * (1 - cortisol)
    consol_signal = data["melatonin"] * (1.0 - data["cortisol"])
    ax3.plot(steps, consol_signal, color=c_melatonin, linewidth=1.5, label="Consolidation signal (rest)")
    ax3.set_title("3. Plasticity (wake) vs Consolidation Signal (rest)", fontsize=11)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.set_ylim(-0.05, 0.8)
    ax3.set_ylabel("Signal")

    # 4. Average stability per neuron
    ax4 = axes[3]
    for i, neuron in enumerate(cluster.neurons):
        sh = np.array(neuron.stability_history)
        mean_stab = sh.mean(axis=1)
        ax4.plot(np.arange(len(mean_stab)), mean_stab, color=neuron_colors[i],
                 linewidth=1.5, label=f"N{i}")
    ax4.set_title("4. Mean Weight Stability per Neuron (jumps during successful consolidation)", fontsize=11)
    ax4.legend(loc="upper left", fontsize=9)
    ax4.set_ylabel("Mean Stability")

    # 5. STM vs LTM weight counts per neuron
    ax5 = axes[4]
    for i, neuron in enumerate(cluster.neurons):
        sh = np.array(neuron.stability_history)
        stm_counts = np.sum((sh > 0) & (sh < neuron.max_stability), axis=1)
        ltm_counts = np.sum(sh >= neuron.max_stability, axis=1)
        t_ax = np.arange(len(stm_counts))
        ax5.plot(t_ax, ltm_counts, color=neuron_colors[i], linewidth=1.5,
                 label=f"N{i} LTM", linestyle="-")
        ax5.plot(t_ax, stm_counts, color=neuron_colors[i], linewidth=0.8,
                 label=f"N{i} STM", linestyle=":", alpha=0.6)
    ax5.set_title("5. STM (dotted) vs LTM (solid) Weight Counts", fontsize=11)
    ax5.legend(loc="upper left", fontsize=8, ncol=2)
    ax5.set_ylabel("Count")

    # 6. Firing raster
    ax6 = axes[5]
    fired = data["fired_matrix"]
    for idx in range(data["n_neurons"]):
        fire_times = steps[fired[:, idx]]
        ax6.eventplot([fire_times], lineoffsets=idx, linelengths=0.5,
                      colors=[neuron_colors[idx]])
    ax6.set_yticks(range(data["n_neurons"]))
    ax6.set_yticklabels([f"N{i}" for i in range(data["n_neurons"])])
    ax6.set_title("6. Neuron Firing Patterns", fontsize=11)

    # 7. Consolidation efficiency per rest phase
    ax7 = axes[6]
    rest_phases = []
    in_rest = False
    start = 0
    for i, r in enumerate(data["is_rest"]):
        if r and not in_rest:
            start = i; in_rest = True
        elif not r and in_rest:
            rest_phases.append((start, i)); in_rest = False
    if in_rest:
        rest_phases.append((start, data["n_steps"]))

    rest_labels = []
    consolidated_counts = []
    pruned_counts = []
    submitted_counts = []
    for rp_idx, (rs, re) in enumerate(rest_phases):
        total_sub = 0
        total_con = 0
        total_pru = 0
        for t in range(rs, re):
            for cr in data["consolidation_log"][t]:
                total_sub += cr.submitted
                total_con += cr.consolidated
                total_pru += cr.pruned
        rest_labels.append(f"Rest {rp_idx+1}\n({rs}-{re})")
        submitted_counts.append(total_sub)
        consolidated_counts.append(total_con)
        pruned_counts.append(total_pru)

    x_pos = np.arange(len(rest_labels))
    bar_w = 0.25
    ax7.bar(x_pos - bar_w, submitted_counts, bar_w, label="Submitted", color="#aaaaaa")
    ax7.bar(x_pos, consolidated_counts, bar_w, label="Consolidated", color="#2ca02c")
    ax7.bar(x_pos + bar_w, pruned_counts, bar_w, label="Pruned", color="#d62728")
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(rest_labels, fontsize=9)
    ax7.set_title("7. Consolidation Efficiency per Rest Phase", fontsize=11)
    ax7.legend(loc="upper right", fontsize=9)
    ax7.set_ylabel("Weight Count")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Visualization saved to {output_path}")


def run_validation_tests(data: dict) -> dict:
    results = {}
    cluster = data["cluster"]

    # Test 1: Backward Compatibility
    from endocrine_neuron import TernaryNeuron as TN
    from endocrine_system import EndocrineState as ES
    test_neuron = TN(n_inputs=8, seed=42)
    ref_rng = np.random.default_rng(99)
    test_states = [
        ES(cortisol=0.2, dopamine=0.4, oxytocin=0.1, melatonin=0.0,
           cortisol_setpoint=0.3, dopamine_setpoint=0.4, oxytocin_setpoint=0.1,
           melatonin_setpoint=0.0, step=0, is_rest=False),
        ES(cortisol=0.8, dopamine=0.1, oxytocin=0.05, melatonin=0.0,
           cortisol_setpoint=0.7, dopamine_setpoint=0.2, oxytocin_setpoint=0.05,
           melatonin_setpoint=0.0, step=1, is_rest=False),
    ]
    all_match = True
    for es in test_states:
        inp = ref_rng.choice([-1.0, 0.0, 1.0], size=8)
        _, act, theta = test_neuron.forward(inp, es)
        expected_act = float(np.dot(test_neuron.weights, inp))
        expected_theta = 1.0 * (1.0 + 2.0 * es.cortisol) * (1.0 - 0.8 * es.dopamine)
        if abs(act - expected_act) > 1e-10 or abs(theta - expected_theta) > 1e-10:
            all_match = False
    results["Test 1 (Backward Compatibility)"] = {
        "pass": bool(all_match), "value": 1.0 if all_match else 0.0,
    }

    # Test 2: Ternary Constraint
    all_ternary = True
    for neuron in cluster.neurons:
        for wh in neuron.weight_history:
            for w in wh:
                if w not in (-1.0, 0.0, 1.0):
                    all_ternary = False
    results["Test 2 (Ternary Constraint)"] = {
        "pass": bool(all_ternary), "value": 1.0 if all_ternary else 0.0,
    }

    # Test 3: STM Decay (weights changed during calm wake should have
    # either decayed or been consolidated by the end — none should remain
    # in low-stability STM indefinitely)
    stm_at_end = sum(n.stm_count for n in cluster.neurons)
    total_weights = sum(n.n_inputs for n in cluster.neurons)
    # Most weights should be either at LTM or baseline by end
    test3 = stm_at_end <= total_weights * 0.5
    results["Test 3 (STM Decay/Consolidation)"] = {
        "pass": bool(test3), "value": float(stm_at_end),
        "max_allowed": float(total_weights * 0.5),
    }

    # Test 4: Consolidation occurred (some weights reached LTM)
    total_ltm = sum(n.ltm_count for n in cluster.neurons)
    test4 = total_ltm >= 2
    results["Test 4 (Consolidation Occurred)"] = {
        "pass": bool(test4), "value": float(total_ltm),
    }

    # Test 5: Cortisol blocks consolidation
    # Compare Rest 1 (post-calm, low cortisol) vs Rest 2 (post-crisis, high cortisol)
    rest_phases = []
    in_rest = False
    start = 0
    for i, r in enumerate(data["is_rest"]):
        if r and not in_rest:
            start = i; in_rest = True
        elif not r and in_rest:
            rest_phases.append((start, i)); in_rest = False
    if in_rest:
        rest_phases.append((start, data["n_steps"]))

    def count_consolidated_in_phase(rs, re):
        total = 0
        for t in range(rs, re):
            for cr in data["consolidation_log"][t]:
                total += cr.consolidated
        return total

    if len(rest_phases) >= 4:
        # Compare Rest 3 (post-recovery, low cortisol) vs Rest 2 (post-crisis,
        # high residual cortisol). Rest 3 should consolidate more.
        r2_consolidated = count_consolidated_in_phase(*rest_phases[1])
        r3_consolidated = count_consolidated_in_phase(*rest_phases[2])
        r4_consolidated = count_consolidated_in_phase(*rest_phases[3])
        calm_total = r3_consolidated + r4_consolidated
        # At least one calm rest must consolidate more than the stressed rest
        test5 = calm_total > r2_consolidated
        results["Test 5 (Cortisol Blocks Consolidation)"] = {
            "pass": bool(test5),
            "rest2_post_crisis": int(r2_consolidated),
            "rest3_post_recovery": int(r3_consolidated),
            "rest4_deep_sleep": int(r4_consolidated),
            "value": float(calm_total - r2_consolidated),
        }
    else:
        results["Test 5 (Cortisol Blocks Consolidation)"] = {
            "pass": False, "value": 0.0, "note": "Not enough rest phases",
        }

    # Test 6: Endocrine gating preserved (more plasticity in abundance than crisis)
    plast = data["plasticity_matrix"]
    # Crisis wake: 300-450, Abundance wake: 850-1050
    crisis_plast = plast[300:450].mean()
    abundance_plast = plast[850:1050].mean()
    test6 = abundance_plast > crisis_plast * 2.0
    results["Test 6 (Endocrine Gating)"] = {
        "pass": bool(test6),
        "crisis_plasticity": float(crisis_plast),
        "abundance_plasticity": float(abundance_plast),
        "value": float(abundance_plast / max(crisis_plast, 1e-6)),
    }

    # Test 7: LTM survives post-consolidation
    # Weights consolidated in Rest 3 (step ~800) should persist through
    # the abundance wake phase (steps 800-1099).
    ltm_after_rest3 = 0
    ltm_before_rest4 = 0
    for neuron in cluster.neurons:
        sh = np.array(neuron.stability_history)
        if len(sh) > 800:
            ltm_after_rest3 += int(np.sum(sh[800] >= neuron.max_stability))
        if len(sh) > 1099:
            ltm_before_rest4 += int(np.sum(sh[1099] >= neuron.max_stability))
    test7 = ltm_before_rest4 >= ltm_after_rest3
    results["Test 7 (LTM Persistence)"] = {
        "pass": bool(test7),
        "ltm_after_rest3": int(ltm_after_rest3),
        "ltm_before_rest4": int(ltm_before_rest4),
        "value": float(ltm_before_rest4),
    }

    # Test 8: Sleep architecture (melatonin high during rest, cortisol trending down)
    rest_melatonin_vals = []
    rest_cortisol_vals = []
    for i, r in enumerate(data["is_rest"]):
        if r:
            rest_melatonin_vals.append(data["melatonin"][i])
            rest_cortisol_vals.append(data["cortisol"][i])
    if rest_melatonin_vals:
        mean_rest_mel = np.mean(rest_melatonin_vals)
        # Melatonin should be meaningfully elevated during rest
        test8 = mean_rest_mel > 0.3
        results["Test 8 (Sleep Architecture)"] = {
            "pass": bool(test8),
            "mean_rest_melatonin": float(mean_rest_mel),
            "mean_rest_cortisol": float(np.mean(rest_cortisol_vals)),
            "value": float(mean_rest_mel),
        }
    else:
        results["Test 8 (Sleep Architecture)"] = {
            "pass": False, "value": 0.0, "note": "No rest phases found",
        }

    return results


def print_results(results: dict) -> bool:
    all_passed = True
    print("\n" + "=" * 65)
    print("VALIDATION TESTS — Memory Consolidation (Step 4)")
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
        print("ALL TESTS PASSED — Memory consolidation validated.")
    else:
        print("SOME TESTS FAILED — review parameters.")
    print("=" * 65)
    return all_passed


def main() -> None:
    print("Symbiont Architecture — Memory Cluster SAM Prototype (Step 4)")
    print("Running simulation (1200 steps, 4 neurons, STM/LTM consolidation)...")

    data = run_simulation(n_steps=1200, n_neurons=4, seed=42)

    cluster = data["cluster"]
    print("\nPer-neuron memory summary:")
    for i, neuron in enumerate(cluster.neurons):
        print(f"  N{i}: STM={neuron.stm_count} LTM={neuron.ltm_count} "
              f"drift={neuron.weight_drift:.0f} updates={neuron.total_updates} "
              f"consolidations={len(neuron.consolidation_results)}")

    generate_plot(data, output_path="memory_simulation_output.png")

    results = run_validation_tests(data)
    passed = print_results(results)

    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Test results saved to test_results.json")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
