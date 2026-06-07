"""
benchmark_homeostasis_vs_reward.py
===================================
Confronto falsificabile: regolazione omeostatica (DES) vs policy a reward
esterno (Q-learning tabellare) su un compito 1D non-stazionario.

=============================================================================
PREREG — Pre-registration (fissata prima di eseguire; NON modificare dopo)
=============================================================================

IPOTESI:
  L'agente omeostatico (MemoryCluster di Symbiont Architecture, Step 4-5)
  recupera l'equilibrio più velocemente dell'agente Q-learning dopo uno shock
  fuori distribuzione, grazie al meccanismo EHD di smorzamento automatico
  del guadagno tramite cortisolo.

METRICA PRINCIPALE:
  recovery_time = numero di step dopo lo shift fino al primo step in cui
  |x - setpoint| < EPSILON (0.1), capped a T_POST se mai raggiunto.

SOGLIA DI FALSIFICAZIONE:
  Se il recovery_time medio dell'omeostatico NON è significativamente
  inferiore a quello del Q-baseline (one-tailed Welch's t-test, p >= 0.05)
  su N_SEEDS=30 seed, l'ipotesi è FALSIFICATA.

MECCANISMO ATTESO:
  Dinamica pre-shift: x_{t+1} = x_t + 1.0 * a_t + noise
  Dinamica post-shift: x_{t+1} = x_t + 2.0 * a_t + noise   (guadagno raddoppia)
  Con gain=2, le azioni calibrate per gain=1 causano overshoot.
  L'EHD: error aumenta → risk = |error|/2 → cortisol_setpoint cresce →
  cortisolo alto inibisce la dopamina → il termine (1 - cortisol) nell'azione
  riduce automaticamente il guadagno effettivo → convergenza più rapida.
  Il Q-agent deve invece re-esplorare e aggiornare la Q-table per scoprire
  che azioni più piccole sono ora ottimali.

ONESTÀ:
  Riportiamo il risultato così come esce. Un risultato negativo è valido.
=============================================================================

Run:
    cd symbiont-architecture
    python benchmark_homeostasis_vs_reward.py
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Import del cluster Symbiont (interfaccia reale — nessuna reimplementazione)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, "sam-multiagent-v0"))

from cluster import MemoryCluster               # MemoryCluster.step() → (EndocrineState, ...)
from environment import GlobalWorldState, NeuronContext

# ---------------------------------------------------------------------------
# Costanti (comuni ai due agenti)
# ---------------------------------------------------------------------------
T_PRE          = 200          # step pre-shift (fase stazionaria)
T_POST         = 200          # step post-shift (da misurare)
T_STEPS        = T_PRE + T_POST
SHIFT_T        = T_PRE

GAIN_PRE       = 1.0
GAIN_POST      = 2.0          # raddoppia il guadagno dell'azione

SETPOINT       = 0.0
EPSILON        = 0.1          # soglia di recupero per |x - setpoint|
NOISE_STD      = 0.05
N_SEEDS        = 30

# ---------------------------------------------------------------------------
# Ambiente 1D
# ---------------------------------------------------------------------------
class Env1D:
    """
    x_{t+1} = x_t + gain_t * clip(action, -1, 1) + noise
    gain_t = GAIN_PRE per t < SHIFT_T, poi GAIN_POST.
    Nessuno dei due agenti viene notificato dello shift.
    """

    def __init__(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.x     = float(rng.uniform(-1.0, 1.0))
        self._noise = rng.standard_normal(T_STEPS) * NOISE_STD
        self._t     = 0

    @property
    def error(self) -> float:
        return self.x - SETPOINT

    def step(self, action: float) -> Tuple[float, float]:
        """Applica azione, restituisce (x_new, error_new)."""
        gain    = GAIN_PRE if self._t < SHIFT_T else GAIN_POST
        action  = float(np.clip(action, -1.0, 1.0))
        self.x  = float(np.clip(
            self.x + gain * action + self._noise[self._t],
            -6.0, 6.0,
        ))
        self._t += 1
        return self.x, self.error


# ---------------------------------------------------------------------------
# Passo 0 — Agente omeostatico (usa MemoryCluster esistente)
# ---------------------------------------------------------------------------
#
# Interfaccia trovata:
#   MemoryCluster.step(world: GlobalWorldState, contexts: List[NeuronContext])
#   → (EndocrineState, List[bool], List[float], List[ConsolidationResult])
#
#   EndocrineState.cortisol: float [0,1]  — cresce con |error| via EHD
#   EndocrineState.dopamine: float [0,1]  — inibita dal cortisolo
#
# Mapping error → world state:
#   risk   = min(|error| / 2, 1.0)   →  cortisol_setpoint = 0.1 + 0.7*risk
#   reward = 1 - risk
#
# Formula azione (meccanismo testato):
#   k = 1 - 0.5 * cortisol          (EHD automatic gain damping)
#   action = -tanh(error * k)
#
#   Quando gain raddoppia → overshoot → error cresce → cortisolo sale →
#   k scende → azione più smorzata → convergenza.
# ---------------------------------------------------------------------------

class SymbiontAgent:
    """Agente omeostatico basato su MemoryCluster (Symbiont Step 4/5)."""

    N_NEURONS = 4
    N_INPUTS  = 8

    def __init__(self, seed: int = 0) -> None:
        self.cluster   = MemoryCluster(
            n_neurons=self.N_NEURONS,
            n_inputs=self.N_INPUTS,
            base_seed=seed,
        )
        self._endo     = self.cluster.current_state
        self._step_idx = 0

    def _make_contexts(self, error: float, risk: float, reward: float) -> List[NeuronContext]:
        """Codifica lo stato in ingresso al cluster."""
        inp          = np.zeros(self.N_INPUTS)
        inp[0]       = float(np.sign(error))                       # direzione errore
        inp[1]       = 1.0 if abs(error) > 0.5 else 0.0            # errore grande
        inp[2]       = 1.0 if abs(error) < 0.1 else 0.0            # vicino al setpoint
        inp[3]       = -float(np.sign(error))                      # segnale correttivo
        return [
            NeuronContext(inputs=inp.copy(), local_risk=risk, local_reward=reward)
            for _ in range(self.N_NEURONS)
        ]

    def act(self, error: float) -> float:
        """Aggiorna DES e restituisce l'azione."""
        risk   = float(min(abs(error) / 2.0, 1.0))
        reward = 1.0 - risk

        world    = GlobalWorldState(risk=risk, reward=reward, step=self._step_idx, is_rest=False)
        contexts = self._make_contexts(error, risk, reward)

        self._endo, _, _, _ = self.cluster.step(world, contexts)
        self._step_idx += 1

        # EHD automatic gain damping (meccanismo centrale del test)
        k      = 1.0 - 0.5 * self._endo.cortisol
        action = -math.tanh(error * k)
        return float(np.clip(action, -1.0, 1.0))

    @property
    def cortisol(self) -> float:
        return self._endo.cortisol

    @property
    def dopamine(self) -> float:
        return self._endo.dopamine


# ---------------------------------------------------------------------------
# Passo 3 — Baseline Q-learning (competente, non azzoppato)
# ---------------------------------------------------------------------------
#
# Scelte di progetto per fairness:
#  - 30 bin di stato sull'intervallo [-3, 3] (copre sia regime pre che post)
#  - 7 azioni discrete in [-1, 1] (stesso range dell'omeostatico)
#  - epsilon=0.20: alto per permettere rapida re-esplorazione dopo lo shift
#  - alpha=0.30: learning rate ragionevole per Q-tabellare
#  - Reward = -|error|: stesso segnale che misura la metrica del benchmark
# ---------------------------------------------------------------------------

class QLearningAgent:
    N_BINS  = 30
    X_MIN   = -3.0
    X_MAX   =  3.0
    ACTIONS = np.array([-1.0, -0.67, -0.33, 0.0, 0.33, 0.67, 1.0])
    ALPHA   = 0.30
    GAMMA   = 0.95
    EPSILON = 0.20

    def __init__(self, seed: int = 0) -> None:
        self.Q    = np.zeros((self.N_BINS, len(self.ACTIONS)))
        self.rng  = np.random.default_rng(seed + 777)
        self._s   : int | None = None
        self._a   : int | None = None

    def _bin(self, error: float) -> int:
        c = np.clip(error, self.X_MIN, self.X_MAX)
        return int(min((c - self.X_MIN) / (self.X_MAX - self.X_MIN) * self.N_BINS,
                       self.N_BINS - 1))

    def act(self, error: float) -> float:
        s = self._bin(error)
        if self.rng.random() < self.EPSILON:
            a = int(self.rng.integers(len(self.ACTIONS)))
        else:
            a = int(np.argmax(self.Q[s]))
        self._s, self._a = s, a
        return float(self.ACTIONS[a])

    def update(self, error_next: float, reward: float) -> None:
        if self._s is None:
            return
        s_next = self._bin(error_next)
        target = reward + self.GAMMA * np.max(self.Q[s_next])
        self._s
        self.Q[self._s, self._a] += self.ALPHA * (target - self.Q[self._s, self._a])


# ---------------------------------------------------------------------------
# Passo 4 — Esecuzione di un episodio e calcolo delle metriche
# ---------------------------------------------------------------------------

Metrics = Dict[str, float]


def run_episode(env_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ritorna (traj_s, traj_q) — shape (T_STEPS, 2): colonne = (x, |error|).
    I due agenti girano su ambienti con lo stesso seed (stessa traiettoria di rumore).
    """
    env_s = Env1D(seed=env_seed)
    env_q = Env1D(seed=env_seed)
    agent_s = SymbiontAgent(seed=0)   # seed fisso per i pesi iniziali
    agent_q = QLearningAgent(seed=0)

    traj_s = np.zeros((T_STEPS, 2))
    traj_q = np.zeros((T_STEPS, 2))

    for t in range(T_STEPS):
        # --- Symbiont ---
        err_s  = env_s.error
        act_s  = agent_s.act(err_s)
        x_s, e_s = env_s.step(act_s)
        traj_s[t] = (x_s, abs(e_s))

        # --- Q-learning ---
        err_q  = env_q.error
        act_q  = agent_q.act(err_q)
        x_q, e_q = env_q.step(act_q)
        agent_q.update(e_q, -abs(e_q))
        traj_q[t] = (x_q, abs(e_q))

    return traj_s, traj_q


def compute_metrics(traj: np.ndarray) -> Metrics:
    """traj: (T_STEPS, 2), colonna 1 = |error|."""
    errors     = traj[:, 1]
    pre        = errors[:SHIFT_T]
    post       = errors[SHIFT_T:]

    # Tempo di recupero: primo step post-shift con |error| < EPSILON
    rec_time = T_POST
    for i, e in enumerate(post):
        if e < EPSILON:
            rec_time = i + 1
            break

    return {
        "pre_error":      float(np.mean(pre)),
        "recovery_time":  float(rec_time),
        "post_cum_error": float(np.sum(post)),
        "final_error":    float(np.mean(errors[-20:])),
    }


# ---------------------------------------------------------------------------
# Test di significatività (solo numpy + math — no scipy)
# ---------------------------------------------------------------------------

def welch_t_one_tailed(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """
    H1: mean(a) < mean(b)   (l'omeostatico recupera più in fretta).
    Approssimazione normale (valida per n=30).
    Restituisce (t_stat, p_value).
    """
    n1, n2 = len(a), len(b)
    m1, m2 = np.mean(a), np.mean(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    se     = math.sqrt(s1**2 / n1 + s2**2 / n2)
    if se < 1e-12:
        return 0.0, (0.0 if m1 < m2 else 1.0)
    t_stat = (m1 - m2) / se
    p_val  = 0.5 * (1.0 + math.erf(t_stat / math.sqrt(2.0)))  # CDF normale
    return t_stat, p_val


# ---------------------------------------------------------------------------
# Benchmark principale
# ---------------------------------------------------------------------------

def run_benchmark() -> Tuple[List[Metrics], List[Metrics], np.ndarray, np.ndarray]:
    all_s: List[Metrics] = []
    all_q: List[Metrics] = []
    # Conserviamo tutte le traiettorie per il grafico delle medie
    trajs_s = np.zeros((N_SEEDS, T_STEPS, 2))
    trajs_q = np.zeros((N_SEEDS, T_STEPS, 2))

    print(f"Benchmark: {N_SEEDS} seed, T={T_STEPS} (shift a t={SHIFT_T})")
    print(f"  Gain: {GAIN_PRE} → {GAIN_POST}  |  epsilon_recovery = {EPSILON}")
    print(f"  Agente omeostatico: MemoryCluster (EHD cortisol-damping)")
    print(f"  Baseline: Q-learning tabellare (epsilon={QLearningAgent.EPSILON}, alpha={QLearningAgent.ALPHA})")
    print()

    for seed in range(N_SEEDS):
        ts, tq = run_episode(env_seed=seed)
        all_s.append(compute_metrics(ts))
        all_q.append(compute_metrics(tq))
        trajs_s[seed] = ts
        trajs_q[seed] = tq
        if (seed + 1) % 10 == 0:
            print(f"  {seed+1}/{N_SEEDS} seed completati...")

    return all_s, all_q, trajs_s, trajs_q


# ---------------------------------------------------------------------------
# Stampa risultati
# ---------------------------------------------------------------------------

def print_results(all_s: List[Metrics], all_q: List[Metrics]) -> None:
    metric_keys = ["pre_error", "recovery_time", "post_cum_error", "final_error"]
    metric_names = {
        "pre_error":      "Errore medio pre-shift",
        "recovery_time":  "Tempo di recupero (step)  ←",
        "post_cum_error": "Errore cumulato post-shift",
        "final_error":    "Errore medio ultimi 20 step",
    }

    print("\n" + "=" * 74)
    print(f"{'Metrica':<36} {'Omeostatico':>16} {'Q-baseline':>16}")
    print("-" * 74)
    for k in metric_keys:
        vs = np.array([m[k] for m in all_s])
        vq = np.array([m[k] for m in all_q])
        print(f"{metric_names[k]:<36} {np.mean(vs):>8.2f} ±{np.std(vs):>5.2f}"
              f"   {np.mean(vq):>8.2f} ±{np.std(vq):>5.2f}")
    print("=" * 74)

    # Test sulla metrica primaria
    rt_s = np.array([m["recovery_time"] for m in all_s])
    rt_q = np.array([m["recovery_time"] for m in all_q])
    t_stat, p_val = welch_t_one_tailed(rt_s, rt_q)
    sig = "SIGNIFICATIVA (p < 0.05)" if p_val < 0.05 else "non significativa (p >= 0.05)"
    verdict = "CONFERMATA" if (p_val < 0.05 and np.mean(rt_s) < np.mean(rt_q)) else "FALSIFICATA"

    print(f"\nTest Welch one-tailed (H1: omeostatico recupera prima):")
    print(f"  t = {t_stat:.3f}   p = {p_val:.4f}   ({sig})")
    print()
    print(
        f"VERDETTO: omeostato recupera in {np.mean(rt_s):.1f}±{np.std(rt_s):.1f} step "
        f"vs baseline {np.mean(rt_q):.1f}±{np.std(rt_q):.1f}, "
        f"differenza {sig}, ipotesi {verdict}"
    )
    print()


# ---------------------------------------------------------------------------
# Grafico
# ---------------------------------------------------------------------------

def plot_results(
    trajs_s: np.ndarray,
    trajs_q: np.ndarray,
    all_s: List[Metrics],
    all_q: List[Metrics],
    out_path: str = "benchmark_output.png",
) -> None:
    steps  = np.arange(T_STEPS)
    mean_s = trajs_s[:, :, 1].mean(axis=0)   # media |error| su tutti i seed
    std_s  = trajs_s[:, :, 1].std(axis=0)
    mean_q = trajs_q[:, :, 1].mean(axis=0)
    std_q  = trajs_q[:, :, 1].std(axis=0)

    rt_s = np.array([m["recovery_time"] for m in all_s])
    rt_q = np.array([m["recovery_time"] for m in all_q])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Benchmark: EHD (Symbiont) vs Q-Learning — gain {GAIN_PRE}→{GAIN_POST} a t={SHIFT_T}",
        fontsize=12,
    )

    # --- Pannello 1: |error| medio nel tempo ---
    ax = axes[0]
    ax.plot(steps, mean_s, color="#e74c3c", linewidth=1.2, label="Omeostatico (DES/EHD)")
    ax.fill_between(steps, mean_s - std_s, mean_s + std_s, alpha=0.2, color="#e74c3c")
    ax.plot(steps, mean_q, color="#3498db", linewidth=1.2, label="Q-learning baseline")
    ax.fill_between(steps, mean_q - std_q, mean_q + std_q, alpha=0.2, color="#3498db")
    ax.axhline(EPSILON, linestyle="--", color="gray", linewidth=0.8, label=f"ε={EPSILON}")
    ax.axvline(SHIFT_T, linestyle=":", color="black", linewidth=1.2, label="shift")
    ax.axvspan(SHIFT_T, T_STEPS, alpha=0.07, color="orange")
    ax.set_xlabel("Step")
    ax.set_ylabel("|error| medio (30 seed)")
    ax.set_title("|error| medio ± std")
    ax.legend(fontsize=8)

    # --- Pannello 2: Traiettoria x (ultimo seed) ---
    ax = axes[1]
    ax.plot(steps, trajs_s[-1, :, 0], color="#e74c3c", alpha=0.8, linewidth=0.9, label="Omeostatico")
    ax.plot(steps, trajs_q[-1, :, 0], color="#3498db", alpha=0.8, linewidth=0.9, label="Q-learning")
    ax.axhline(SETPOINT, linestyle="--", color="gray", linewidth=0.8)
    ax.axhline(SETPOINT + EPSILON, linestyle=":", color="gray", linewidth=0.6)
    ax.axhline(SETPOINT - EPSILON, linestyle=":", color="gray", linewidth=0.6)
    ax.axvline(SHIFT_T, linestyle=":", color="black", linewidth=1.2, label="shift")
    ax.axvspan(SHIFT_T, T_STEPS, alpha=0.07, color="orange")
    ax.set_xlabel("Step")
    ax.set_ylabel("x")
    ax.set_title(f"Traiettoria x (seed {N_SEEDS-1})")
    ax.legend(fontsize=8)

    # --- Pannello 3: Distribuzione tempi di recupero ---
    ax = axes[2]
    bins = np.linspace(0, T_POST + 1, 22)
    ax.hist(rt_s, bins=bins, alpha=0.6, color="#e74c3c",
            label=f"Omeostatico (μ={np.mean(rt_s):.1f})")
    ax.hist(rt_q, bins=bins, alpha=0.6, color="#3498db",
            label=f"Q-learning (μ={np.mean(rt_q):.1f})")
    ax.set_xlabel("Recovery time (step post-shift)")
    ax.set_ylabel("Conteggio seed")
    ax.set_title(f"Distribuzione recovery time ({N_SEEDS} seed)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafico salvato → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    all_s, all_q, trajs_s, trajs_q = run_benchmark()
    print_results(all_s, all_q)
    plot_results(trajs_s, trajs_q, all_s, all_q)


if __name__ == "__main__":
    main()
