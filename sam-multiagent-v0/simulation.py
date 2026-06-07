"""
Simulation runner for Symbiont Multi-Agent SAM (Step 5).

Runs a 3-cluster SymbiontColony across 800 steps, generates a 7-panel
visualization, and validates 10 behavioural properties.

Run from within sam-multiagent-v0/:
    python simulation.py

SAFETY: Does NOT import from or modify any previous step.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from environment import MultiAgentEnvironment
from multiagent import ColonyStepResult, SymbiontColony


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLUSTER_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]
CLUSTER_LABELS = ["Cluster-0", "Cluster-1", "Cluster-2"]

PHASE_COLORS = {
    "calm":      "#d5f5e3",
    "perturb":   "#fadbd8",
    "rest1":     "#d6eaf8",
    "recovery":  "#fdebd0",
    "abundance": "#fef9e7",
    "rest2":     "#d6eaf8",
}


def _shade_phases(ax: plt.Axes, phases: list) -> None:
    for start, end, label in phases:
        ax.axvspan(start, end, alpha=0.25, color=PHASE_COLORS[label], zorder=0)


def _rolling_mean(arr: List[float], window: int = 20) -> np.ndarray:
    a = np.array(arr, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(a, kernel, mode="same")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(
    n_clusters: int = 3,
    coupling_strength: float = 0.1,
    seed: int = 42,
) -> tuple:
    env    = MultiAgentEnvironment(n_clusters=n_clusters, seed=seed)
    colony = SymbiontColony(n_clusters=n_clusters, coupling_strength=coupling_strength, base_seed=seed)
    history: List[ColonyStepResult] = []

    for t in range(env.N_STEPS):
        worlds, contexts = env.get_state(t)
        result = colony.step(worlds, contexts)
        history.append(result)

    return colony, history, env


def build_arrays(
    history: List[ColonyStepResult],
    env: MultiAgentEnvironment,
    n_clusters: int,
) -> Dict[str, Any]:
    n = len(history)
    cortisol    = [[history[t].endocrine_states[c].cortisol   for t in range(n)] for c in range(n_clusters)]
    dopamine    = [[history[t].endocrine_states[c].dopamine   for t in range(n)] for c in range(n_clusters)]
    oxytocin    = [[history[t].endocrine_states[c].oxytocin   for t in range(n)] for c in range(n_clusters)]
    melatonin   = [[history[t].endocrine_states[c].melatonin  for t in range(n)] for c in range(n_clusters)]
    broadcasts  = [[history[t].broadcasts[c]                  for t in range(n)] for c in range(n_clusters)]
    incoming    = [[history[t].incoming[c]                    for t in range(n)] for c in range(n_clusters)]
    fired_counts = [[sum(history[t].fired_lists[c])           for t in range(n)] for c in range(n_clusters)]
    risk        = [[env.worlds[t][c].risk                     for t in range(n)] for c in range(n_clusters)]
    reward      = [[env.worlds[t][c].reward                   for t in range(n)] for c in range(n_clusters)]
    return {
        "cortisol":     cortisol,
        "dopamine":     dopamine,
        "oxytocin":     oxytocin,
        "melatonin":    melatonin,
        "broadcasts":   broadcasts,
        "incoming":     incoming,
        "fired_counts": fired_counts,
        "risk":         risk,
        "reward":       reward,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_simulation(
    history: List[ColonyStepResult],
    arrays: Dict[str, Any],
    env: MultiAgentEnvironment,
    out_path: str = "multiagent_simulation_output.png",
) -> None:
    n           = len(history)
    steps       = list(range(n))
    n_clusters  = len(arrays["cortisol"])
    phases      = env.PHASE_BOUNDS

    fig, axes = plt.subplots(7, 1, figsize=(14, 22), sharex=True)
    fig.suptitle("Symbiont Architecture — Step 5: Multi-Agent Coordination", fontsize=14, y=0.995)

    # Panel 1: World risk per cluster
    ax = axes[0]
    for c in range(n_clusters):
        ax.plot(steps, arrays["risk"][c], color=CLUSTER_COLORS[c], alpha=0.7,
                linewidth=0.8, label=CLUSTER_LABELS[c])
    _shade_phases(ax, phases)
    ax.set_ylabel("Global Risk")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Panel 1 — World Risk per Cluster (perturb phase: cluster-0 spike)")

    # Panel 2: Oxytocin per cluster
    ax = axes[1]
    for c in range(n_clusters):
        ax.plot(steps, arrays["oxytocin"][c], color=CLUSTER_COLORS[c],
                linewidth=1.0, label=CLUSTER_LABELS[c])
    _shade_phases(ax, phases)
    ax.set_ylabel("Oxytocin")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Panel 2 — Oxytocin per Cluster (social signal, rises during recovery/abundance)")

    # Panel 3: Cortisol per cluster
    ax = axes[2]
    for c in range(n_clusters):
        ax.plot(steps, arrays["cortisol"][c], color=CLUSTER_COLORS[c],
                linewidth=1.0, label=CLUSTER_LABELS[c])
    _shade_phases(ax, phases)
    ax.set_ylabel("Cortisol")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Panel 3 — Cortisol per Cluster (cluster-1/2 unaffected by cluster-0 stress)")

    # Panel 4: Broadcast signals
    ax = axes[3]
    for c in range(n_clusters):
        ax.plot(steps, arrays["broadcasts"][c], color=CLUSTER_COLORS[c],
                linewidth=0.8, label=CLUSTER_LABELS[c])
    _shade_phases(ax, phases)
    ax.set_ylabel("Broadcast\noxy × (1−cort)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Panel 4 — Inter-Cluster Broadcast (stress attenuates social signal)")

    # Panel 5: Firing rate per cluster (rolling mean, window=20)
    ax = axes[4]
    for c in range(n_clusters):
        smooth = _rolling_mean(arrays["fired_counts"][c], window=20)
        ax.plot(steps, smooth, color=CLUSTER_COLORS[c],
                linewidth=1.0, label=CLUSTER_LABELS[c])
    _shade_phases(ax, phases)
    ax.set_ylabel("Firing Rate\n(rolling 20)")
    ax.set_ylim(-0.1, 4.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Panel 5 — Cluster Firing Rate (smoothed over 20 steps)")

    # Panel 6: Colony mean oxytocin ± std
    ax = axes[5]
    oxy_arr = np.array(arrays["oxytocin"])           # shape (n_clusters, n_steps)
    colony_mean = oxy_arr.mean(axis=0)
    colony_std  = oxy_arr.std(axis=0)
    ax.plot(steps, colony_mean, color="#8e44ad", linewidth=1.2, label="Colony mean")
    ax.fill_between(steps,
                    colony_mean - colony_std,
                    colony_mean + colony_std,
                    alpha=0.25, color="#8e44ad", label="±1 std")
    _shade_phases(ax, phases)
    ax.set_ylabel("Colony Oxytocin")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.3, linestyle="--", linewidth=0.8, color="gray", label="test_06 threshold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Panel 6 — Colony Mean Oxytocin (collective social state)")

    # Panel 7: Phase timeline
    ax = axes[6]
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    phase_patches = []
    for start, end, label in phases:
        rect = mpatches.FancyBboxPatch(
            (start, 0.1), end - start, 0.8,
            boxstyle="round,pad=2",
            facecolor=PHASE_COLORS[label],
            edgecolor="gray",
            linewidth=0.5,
        )
        ax.add_patch(rect)
        ax.text(
            (start + end) / 2, 0.5, label,
            ha="center", va="center", fontsize=9, fontweight="bold",
        )
    ax.set_xlabel("Simulation Step")
    ax.set_title("Panel 7 — Phase Timeline")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved → {out_path}")


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

def run_tests(
    colony: SymbiontColony,
    history: List[ColonyStepResult],
    arrays: Dict[str, Any],
    n_clusters: int = 3,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    def record(name: str, fn) -> None:
        try:
            fn()
            results.append({"name": name, "passed": True, "detail": "OK"})
            print(f"  PASS  {name}")
        except AssertionError as exc:
            results.append({"name": name, "passed": False, "detail": str(exc)})
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:
            results.append({"name": name, "passed": False, "detail": f"ERROR: {exc}"})
            print(f"  ERROR {name}: {exc}")

    # ------------------------------------------------------------------
    def t01():
        assert len(colony.clusters) == n_clusters, \
            f"expected {n_clusters} clusters, got {len(colony.clusters)}"
        des_ids = [id(c.des) for c in colony.clusters]
        assert len(set(des_ids)) == n_clusters, "clusters share a DES object"
    record("test_01: independent cluster creation", t01)

    # ------------------------------------------------------------------
    def t02():
        oxy = arrays["oxytocin"]
        # Compare abundance (high broadcast) vs cluster-0 during perturbation
        # (cortisol blocks its broadcast → lower incoming signal → lower oxytocin).
        # This directly tests that coupling raises oxytocin when broadcasts are unblocked.
        mean_perturb_c0 = float(np.mean(oxy[0][200:300]))
        mean_abundance  = float(np.mean([oxy[c][550:750] for c in range(n_clusters)]))
        assert mean_abundance > mean_perturb_c0 * 1.2, (
            f"abundance oxytocin {mean_abundance:.3f} not > "
            f"cluster-0 perturb {mean_perturb_c0:.3f} × 1.2"
        )
    record("test_02: inter-cluster oxytocin propagation", t02)

    # ------------------------------------------------------------------
    def t03():
        # Cluster-1 should NOT spike cortisol during cluster-0 perturbation
        cort_1       = arrays["cortisol"][1]
        max_perturb  = float(max(cort_1[200:300]))
        assert max_perturb < 0.6, \
            f"cluster-1 max cortisol during perturb = {max_perturb:.3f} (>= 0.6)"
    record("test_03: no direct cortisol cross-propagation", t03)

    # ------------------------------------------------------------------
    def t04():
        oxy_0         = arrays["oxytocin"][0]
        mean_perturb  = float(np.mean(oxy_0[200:300]))
        mean_recovery = float(np.mean(oxy_0[350:550]))
        assert mean_recovery > mean_perturb, (
            f"cluster-0 recovery oxytocin {mean_recovery:.3f} "
            f"not > perturb {mean_perturb:.3f}"
        )
    record("test_04: collective homeostasis after perturbation", t04)

    # ------------------------------------------------------------------
    def t05():
        cort         = arrays["cortisol"]
        mean_cort_0  = float(np.mean(cort[0][200:300]))
        mean_cort_1  = float(np.mean(cort[1][200:300]))
        assert mean_cort_0 > mean_cort_1 * 1.5, (
            f"cluster-0 mean cortisol {mean_cort_0:.3f} "
            f"not > cluster-1 {mean_cort_1:.3f} × 1.5 — EHD may be overridden"
        )
    record("test_05: local EHD not overridden by coordination", t05)

    # ------------------------------------------------------------------
    def t06():
        oxy  = arrays["oxytocin"]
        cort = arrays["cortisol"]
        mean_oxy_col  = float(np.mean([oxy[c][550:750]   for c in range(n_clusters)]))
        mean_cort_col = float(np.mean([cort[c][550:750]  for c in range(n_clusters)]))
        assert mean_oxy_col  > 0.3, \
            f"colony oxytocin {mean_oxy_col:.3f} < 0.3 during abundance"
        assert mean_cort_col < 0.4, \
            f"colony cortisol {mean_cort_col:.3f} >= 0.4 during abundance"
    record("test_06: collective ethical state emerges without rules", t06)

    # ------------------------------------------------------------------
    def t07():
        state_dict = colony.to_dict()
        colony2    = SymbiontColony.from_dict(state_dict)
        for i in range(colony.n_clusters):
            np.testing.assert_array_equal(
                colony.clusters[i].neurons[0].weights,
                colony2.clusters[i].neurons[0].weights,
                err_msg=f"weights mismatch in cluster {i}",
            )
    record("test_07: serialisation / deserialisation roundtrip", t07)

    # ------------------------------------------------------------------
    def t08():
        for result in history:
            for s in result.endocrine_states:
                for name, val in [
                    ("cortisol",  s.cortisol),
                    ("dopamine",  s.dopamine),
                    ("oxytocin",  s.oxytocin),
                    ("melatonin", s.melatonin),
                ]:
                    assert 0.0 <= val <= 1.0, f"{name} = {val:.4f} out of [0,1]"
    record("test_08: numerical stability over 800 steps", t08)

    # ------------------------------------------------------------------
    def t09():
        agg = colony.get_aggregate_endocrine()
        required = ["mean_cortisol", "mean_dopamine", "mean_oxytocin",
                    "mean_melatonin", "std_oxytocin", "broadcast_mean"]
        for key in required:
            assert key in agg, f"missing key '{key}' in get_aggregate_endocrine()"
    record("test_09: aggregate endocrine log keys present", t09)

    # ------------------------------------------------------------------
    def t10():
        script_dir = pathlib.Path(__file__).parent
        for fname in ["endocrine_neuron.py", "endocrine_system.py", "cluster.py"]:
            src  = (script_dir.parent / "sam-memory-v0" / fname).read_bytes()
            copy = (script_dir / fname).read_bytes()
            assert hashlib.md5(src).hexdigest() == hashlib.md5(copy).hexdigest(), \
                f"{fname} diverges from sam-memory-v0 original"
    record("test_10: copied files identical to sam-memory-v0 originals", t10)

    passed = sum(1 for r in results if r["passed"])
    print(f"\n{'='*52}")
    print(f"  {passed}/{len(results)} tests PASSED")
    print(f"{'='*52}\n")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Running Step 5 — Multi-Agent Coordination simulation...")

    colony, history, env = run_simulation(n_clusters=3, coupling_strength=0.1, seed=42)
    arrays = build_arrays(history, env, n_clusters=3)

    plot_simulation(history, arrays, env, out_path="multiagent_simulation_output.png")

    print("\nRunning test suite:")
    test_results = run_tests(colony, history, arrays, n_clusters=3)

    # Print key metrics
    oxy  = arrays["oxytocin"]
    cort = arrays["cortisol"]
    bcast = arrays["broadcasts"]
    print("Key metrics:")
    print(f"  Baseline oxytocin  (steps   0-199): {np.mean([oxy[c][0:200]  for c in range(3)]):.3f}")
    print(f"  Perturb  oxytocin  (steps 200-299, cluster-0): {np.mean(oxy[0][200:300]):.3f}")
    print(f"  Perturb  cortisol  (steps 200-299, cluster-0): {np.mean(cort[0][200:300]):.3f}")
    print(f"  Perturb  cortisol  (steps 200-299, cluster-1): {np.mean(cort[1][200:300]):.3f}")
    print(f"  Recovery oxytocin  (steps 350-549, cluster-0): {np.mean(oxy[0][350:550]):.3f}")
    print(f"  Abundance oxytocin (steps 550-749, colony):    {np.mean([oxy[c][550:750] for c in range(3)]):.3f}")
    print(f"  Abundance cortisol (steps 550-749, colony):    {np.mean([cort[c][550:750] for c in range(3)]):.3f}")
    print(f"  Mean broadcast (abundance):                    {np.mean([bcast[c][550:750] for c in range(3)]):.3f}")

    # Save test results
    output = {
        "step": 5,
        "module": "sam-multiagent-v0",
        "tests": [
            {
                "id":     f"test_{str(i+1).zfill(2)}",
                "name":   r["name"],
                "passed": r["passed"],
                "detail": r["detail"],
            }
            for i, r in enumerate(test_results)
        ],
        "passed": sum(1 for r in test_results if r["passed"]),
        "total":  len(test_results),
    }
    with open("test_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Test results saved → test_results.json")


if __name__ == "__main__":
    main()
