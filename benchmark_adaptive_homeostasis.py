"""
benchmark_adaptive_homeostasis.py  (v1 — omeostasi adattiva vs fissa vs Q)
===========================================================================
Testa se l'EHD con stima online del guadagno ripara il fallimento specifico
documentato in benchmark_homeostasis_vs_reward.py (v4, CHIUSO, FALSIFICATO).

CONTESTO — non riaprire il v4:
  v4 ha dimostrato che l'agente EHD a legge fissa recupera affidabilmente
  solo in 20/30 seed dopo gain×4 (10/30 mai_recuperato). Il meccanismo del
  fallimento: la legge -tanh a segno fisso + plasticità bloccata dallo stress
  = nessun percorso di adattamento. Il v4 resta agli atti così com'è.

=============================================================================
PREREG — Pre-registration (fissata prima di eseguire; NON modificare dopo)
=============================================================================

IPOTESI:
  Un omeostato che aggiorna online la propria stima del guadagno del mondo
  (ĝ, derivata dall'errore di predizione) recupera dalla stessa perturbazione
  OOD in modo affidabile quanto il Q-learning (mai_recuperato ~0/30)
  MANTENENDO una precisione a regime significativamente migliore di Q.

CONDIZIONE A — Affidabilità (fragilità riparata?):
  H1A: mai_recuperato_adaptive < mai_recuperato_fixed
  Riferimento v4: mai_recuperato_fixed = 10/30. Target: mai_recuperato_adaptive = 0/30.
  FALSIFICATA se: mai_recuperato_adaptive >= mai_recuperato_fixed (10/30)

CONDIZIONE B — Precisione preservata (il guadagno non annulla il vantaggio EHD?):
  H1B: pre_baseline_adaptive < pre_baseline_q   [Welch one-tailed, p < 0.05]
  FALSIFICATA se: p >= 0.05

  Motivazione di B: se l'agente adattivo ha perso la precisione EHD (0.04 in v4),
  ha comprato robustezza al prezzo del vantaggio distintivo — non è un progresso.

VINCOLO DI EQUITÀ:
  L'agente adattivo NON viene avvisato dello shift e NON riceve il nuovo guadagno.
  Lo scopre dall'errore di predizione (x_actual - x_prev - ĝ * action_prev),
  esattamente come Q scopre il cambiamento dal reward. Barare invalida il test.

AGENTE ADATTIVO — minimo kernel di inferenza attiva:
  Aggiunge un solo parametro ĝ (gain_hat) all'agente EHD fisso.
  Aggiornamento ogni step:
    x_predicted = x_prev + ĝ * action_prev
    pred_err    = x_current - x_predicted
    ĝ          += LR_GAIN * pred_err * action_prev   [gradient descent su MSE]
    ĝ           = clip(ĝ, GAIN_HAT_MIN, GAIN_HAT_MAX)
  Azione:
    k      = 1 - 0.5 * cortisol   (EHD identico al fisso)
    action = clip(-tanh(error * k) / max(ĝ, GAIN_HAT_MIN), -1, 1)

STESSO AMBIENTE del v4 (nessuna modifica):
  Gain 1.0 → 4.0 a t=1000. 30 seed. Stesse metriche. Stesso gate plateau.
  run_episode() usa tre ambienti identici (stesso seed → stessa traiettoria rumore).

ONESTÀ:
  Tre colonne. Riporta quello che esce. Se l'adattivo perde, scrivilo chiaro.
=============================================================================

Run:
    cd symbiont-architecture
    python benchmark_adaptive_homeostasis.py
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
# Import del cluster Symbiont
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, "sam-multiagent-v0"))

from cluster import MemoryCluster
from environment import GlobalWorldState, NeuronContext

# ---------------------------------------------------------------------------
# Costanti — identiche al v4 salvo le tre nuove righe LR/GAIN_HAT
# ---------------------------------------------------------------------------
T_PRE             = 1000
T_POST            = 200
T_STEPS           = T_PRE + T_POST
SHIFT_T           = T_PRE

GAIN_PRE          = 1.0
GAIN_POST         = 4.0

SETPOINT          = 0.0
EPSILON           = 0.1
NOISE_STD         = 0.05
N_SEEDS           = 30

BASELINE_WINDOW   = 50
PLATEAU_WINDOW    = 200
PLATEAU_MIN_RATIO = 0.85

# Parametri agente adattivo
LR_GAIN           = 0.05    # tasso di apprendimento per ĝ
GAIN_HAT_MIN      = 0.5     # clip inferiore: evita divisione per quasi-zero
GAIN_HAT_MAX      = 10.0    # clip superiore: evita deriva esplosiva
GAIN_HAT_INIT     = GAIN_PRE  # prior equo = guadagno pre-shift noto

# ---------------------------------------------------------------------------
# Ambiente 1D — identico al v4
# ---------------------------------------------------------------------------
class Env1D:
    """x_{t+1} = x_t + gain_t * clip(action,-1,1) + noise. Nessuno viene notificato."""

    def __init__(self, seed: int) -> None:
        rng         = np.random.default_rng(seed)
        self.x      = float(rng.uniform(-1.0, 1.0))
        self._noise = rng.standard_normal(T_STEPS) * NOISE_STD
        self._t     = 0

    @property
    def error(self) -> float:
        return self.x - SETPOINT

    def step(self, action: float) -> Tuple[float, float]:
        gain   = GAIN_PRE if self._t < SHIFT_T else GAIN_POST
        action = float(np.clip(action, -1.0, 1.0))
        self.x = float(np.clip(self.x + gain * action + self._noise[self._t], -6.0, 6.0))
        self._t += 1
        return self.x, self.error


# ---------------------------------------------------------------------------
# Agente omeostatico FISSO — identico al v4 (il baseline fallito)
# ---------------------------------------------------------------------------
class SymbiontAgent:
    N_NEURONS = 4
    N_INPUTS  = 8

    def __init__(self, seed: int = 0) -> None:
        self.cluster   = MemoryCluster(n_neurons=self.N_NEURONS, n_inputs=self.N_INPUTS, base_seed=seed)
        self._endo     = self.cluster.current_state
        self._step_idx = 0

    def _make_contexts(self, error: float, risk: float, reward: float) -> List[NeuronContext]:
        inp    = np.zeros(self.N_INPUTS)
        inp[0] = float(np.sign(error))
        inp[1] = 1.0 if abs(error) > 0.5 else 0.0
        inp[2] = 1.0 if abs(error) < 0.1 else 0.0
        inp[3] = -float(np.sign(error))
        return [NeuronContext(inputs=inp.copy(), local_risk=risk, local_reward=reward)
                for _ in range(self.N_NEURONS)]

    def act(self, error: float) -> float:
        risk   = float(min(abs(error) / 2.0, 1.0))
        reward = 1.0 - risk
        world  = GlobalWorldState(risk=risk, reward=reward, step=self._step_idx, is_rest=False)
        self._endo, _, _, _ = self.cluster.step(world, self._make_contexts(error, risk, reward))
        self._step_idx += 1
        k = 1.0 - 0.5 * self._endo.cortisol
        return float(np.clip(-math.tanh(error * k), -1.0, 1.0))

    @property
    def cortisol(self) -> float:
        return self._endo.cortisol


# ---------------------------------------------------------------------------
# Agente omeostatico ADATTIVO — modifica minima: aggiunge ĝ online
# ---------------------------------------------------------------------------
class AdaptiveSymbiontAgent:
    """
    EHD con stima online del guadagno.

    Rispetto all'agente fisso aggiunge un solo parametro (ĝ = gain_hat)
    aggiornato dall'errore di predizione:
      pred_err = x_current - (x_prev + ĝ * action_prev)
      ĝ       += LR_GAIN * pred_err * action_prev   [discesa del gradiente su MSE]

    Azione:
      action = clip(-tanh(error * k) / max(ĝ, GAIN_HAT_MIN), -1, 1)

    Il cortisolo gata la plasticità Hebbiana (invariato dal fisso) ma NON
    gata l'aggiornamento di ĝ — la stima del guadagno è un'inferenza sul
    modello del mondo, non un apprendimento sinaptico, e deve aggiornare
    anche in condizioni di stress.
    """
    N_NEURONS = 4
    N_INPUTS  = 8

    def __init__(self, seed: int = 0) -> None:
        self.cluster      = MemoryCluster(n_neurons=self.N_NEURONS, n_inputs=self.N_INPUTS, base_seed=seed)
        self._endo        = self.cluster.current_state
        self._step_idx    = 0
        self._gain_hat    = float(GAIN_HAT_INIT)
        self._x_prev      : float | None = None
        self._action_prev : float | None = None

    def _make_contexts(self, error: float, risk: float, reward: float) -> List[NeuronContext]:
        inp    = np.zeros(self.N_INPUTS)
        inp[0] = float(np.sign(error))
        inp[1] = 1.0 if abs(error) > 0.5 else 0.0
        inp[2] = 1.0 if abs(error) < 0.1 else 0.0
        inp[3] = -float(np.sign(error))
        return [NeuronContext(inputs=inp.copy(), local_risk=risk, local_reward=reward)
                for _ in range(self.N_NEURONS)]

    def act(self, error: float) -> float:
        x_current = error  # SETPOINT = 0 → x = error

        # Aggiorna ĝ dall'errore di predizione (vincolo di equità: solo da pred_err)
        if self._x_prev is not None and self._action_prev is not None:
            if abs(self._action_prev) > 1e-6:
                x_predicted     = self._x_prev + self._gain_hat * self._action_prev
                pred_err        = x_current - x_predicted
                new_gain        = self._gain_hat + LR_GAIN * pred_err * self._action_prev
                self._gain_hat  = float(np.clip(new_gain, GAIN_HAT_MIN, GAIN_HAT_MAX))

        # EHD step — identico all'agente fisso
        risk   = float(min(abs(error) / 2.0, 1.0))
        reward = 1.0 - risk
        world  = GlobalWorldState(risk=risk, reward=reward, step=self._step_idx, is_rest=False)
        self._endo, _, _, _ = self.cluster.step(world, self._make_contexts(error, risk, reward))
        self._step_idx += 1

        # Azione corretta per il guadagno stimato
        k      = 1.0 - 0.5 * self._endo.cortisol
        action = float(np.clip(-math.tanh(error * k) / max(self._gain_hat, GAIN_HAT_MIN), -1.0, 1.0))

        self._x_prev      = x_current
        self._action_prev = action
        return action

    @property
    def cortisol(self) -> float:
        return self._endo.cortisol

    @property
    def gain_hat(self) -> float:
        return self._gain_hat


# ---------------------------------------------------------------------------
# Q-learning — identico al v4
# ---------------------------------------------------------------------------
class QLearningAgent:
    N_BINS    = 30
    X_MIN     = -3.0
    X_MAX     =  3.0
    ACTIONS   = np.array([-1.0, -0.67, -0.33, 0.0, 0.33, 0.67, 1.0])
    ALPHA     = 0.30
    GAMMA     = 0.95
    EPS_START = 0.20
    EPS_END   = 0.05

    def __init__(self, seed: int = 0) -> None:
        self.Q       = np.zeros((self.N_BINS, len(self.ACTIONS)))
        self.rng     = np.random.default_rng(seed + 777)
        self._s      : int | None = None
        self._a      : int | None = None
        self._t_step = 0

    def _epsilon(self) -> float:
        frac = min(self._t_step / T_PRE, 1.0)
        return self.EPS_START + frac * (self.EPS_END - self.EPS_START)

    def _bin(self, error: float) -> int:
        c = np.clip(error, self.X_MIN, self.X_MAX)
        return int(min((c - self.X_MIN) / (self.X_MAX - self.X_MIN) * self.N_BINS, self.N_BINS - 1))

    def act(self, error: float) -> float:
        s   = self._bin(error)
        eps = self._epsilon()
        a   = int(self.rng.integers(len(self.ACTIONS))) if self.rng.random() < eps else int(np.argmax(self.Q[s]))
        self._s, self._a = s, a
        self._t_step += 1
        return float(self.ACTIONS[a])

    def update(self, error_next: float, reward: float) -> None:
        if self._s is None:
            return
        s_next = self._bin(error_next)
        target = reward + self.GAMMA * np.max(self.Q[s_next])
        self.Q[self._s, self._a] += self.ALPHA * (target - self.Q[self._s, self._a])


# ---------------------------------------------------------------------------
# Episodio — tre ambienti indipendenti, stesso seed (stessa traiettoria rumore)
# ---------------------------------------------------------------------------
Metrics = Dict[str, float]


def run_episode(env_seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ritorna (traj_fixed, traj_adaptive, traj_q, gain_hat_traj).
    traj_*: shape (T_STEPS, 2), colonne = (x, |error|).
    gain_hat_traj: shape (T_STEPS,), ĝ dell'adattivo passo-passo.
    """
    env_f   = Env1D(seed=env_seed)
    env_a   = Env1D(seed=env_seed)
    env_q   = Env1D(seed=env_seed)
    agent_f = SymbiontAgent(seed=0)
    agent_a = AdaptiveSymbiontAgent(seed=0)
    agent_q = QLearningAgent(seed=env_seed)

    traj_f       = np.zeros((T_STEPS, 2))
    traj_a       = np.zeros((T_STEPS, 2))
    traj_q       = np.zeros((T_STEPS, 2))
    gain_hat_traj = np.zeros(T_STEPS)

    for t in range(T_STEPS):
        err_f = env_f.error
        act_f = agent_f.act(err_f)
        x_f, e_f = env_f.step(act_f)
        traj_f[t] = (x_f, abs(e_f))

        err_a = env_a.error
        act_a = agent_a.act(err_a)
        x_a, e_a = env_a.step(act_a)
        traj_a[t] = (x_a, abs(e_a))
        gain_hat_traj[t] = agent_a.gain_hat

        err_q = env_q.error
        act_q = agent_q.act(err_q)
        x_q, e_q = env_q.step(act_q)
        agent_q.update(e_q, -abs(e_q))
        traj_q[t] = (x_q, abs(e_q))

    return traj_f, traj_a, traj_q, gain_hat_traj


# ---------------------------------------------------------------------------
# Metriche — identiche al v4
# ---------------------------------------------------------------------------
def compute_metrics(traj: np.ndarray) -> Metrics:
    errors = traj[:, 1]
    pre    = errors[:SHIFT_T]
    post   = errors[SHIFT_T:]

    pre_baseline  = float(np.mean(pre[-BASELINE_WINDOW:]))
    rec_threshold = pre_baseline + EPSILON

    rec_time = T_POST
    for i, e in enumerate(post):
        if e < rec_threshold:
            rec_time = i + 1
            break

    w             = PLATEAU_WINDOW
    early_mean    = float(np.mean(pre[-2 * w : -w]))
    late_mean     = float(np.mean(pre[-w:]))
    plateau_ratio = late_mean / (early_mean + 1e-9)

    return {
        "pre_error":       float(np.mean(pre)),
        "pre_baseline":    pre_baseline,
        "rec_threshold":   rec_threshold,
        "plateau_ratio":   plateau_ratio,
        "initial_shock":   float(post[0]),
        "recovery_time":   float(rec_time),
        "post_cum_error":  float(np.sum(post)),
        "final_error":     float(np.mean(errors[-20:])),
        "never_recovered": float(rec_time == T_POST),
    }


# ---------------------------------------------------------------------------
# Welch t one-tailed — identico al v4
# ---------------------------------------------------------------------------
def welch_t_one_tailed(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """H1: mean(a) < mean(b). Approssimazione normale (n=30). Restituisce (t, p)."""
    n1, n2 = len(a), len(b)
    m1, m2 = np.mean(a), np.mean(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    se     = math.sqrt(s1**2 / n1 + s2**2 / n2)
    if se < 1e-12:
        return 0.0, (0.0 if m1 < m2 else 1.0)
    t_stat = (m1 - m2) / se
    p_val  = 0.5 * (1.0 + math.erf(t_stat / math.sqrt(2.0)))
    return t_stat, p_val


# ---------------------------------------------------------------------------
# Benchmark principale — tre agenti
# ---------------------------------------------------------------------------
def run_benchmark():
    all_f: List[Metrics] = []
    all_a: List[Metrics] = []
    all_q: List[Metrics] = []
    trajs_f        = np.zeros((N_SEEDS, T_STEPS, 2))
    trajs_a        = np.zeros((N_SEEDS, T_STEPS, 2))
    trajs_q        = np.zeros((N_SEEDS, T_STEPS, 2))
    gain_hat_trajs = np.zeros((N_SEEDS, T_STEPS))

    print(f"Benchmark adattivo: {N_SEEDS} seed, T={T_STEPS} (shift a t={SHIFT_T})")
    print(f"  Gain: {GAIN_PRE} → {GAIN_POST}  |  recovery = baseline_propria + {EPSILON}")
    print(f"  Fisso:    MemoryCluster (EHD cortisol-damping, legge fissa)")
    print(f"  Adattivo: MemoryCluster + ĝ online (LR={LR_GAIN}, init={GAIN_HAT_INIT})")
    print(f"  Q:        Q-learning tabellare (epsilon {QLearningAgent.EPS_START}→"
          f"{QLearningAgent.EPS_END}, alpha={QLearningAgent.ALPHA})")
    print()

    for seed in range(N_SEEDS):
        tf, ta, tq, gh = run_episode(env_seed=seed)
        all_f.append(compute_metrics(tf))
        all_a.append(compute_metrics(ta))
        all_q.append(compute_metrics(tq))
        trajs_f[seed]        = tf
        trajs_a[seed]        = ta
        trajs_q[seed]        = tq
        gain_hat_trajs[seed] = gh
        if (seed + 1) % 10 == 0:
            print(f"  {seed+1}/{N_SEEDS} seed completati...")

    return all_f, all_a, all_q, trajs_f, trajs_a, trajs_q, gain_hat_trajs


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _esito_label(p: float, mean_a: float, mean_b: float,
                 label_a_wins: str = "A VINCE",
                 label_b_wins: str = "B VINCE") -> str:
    """H1: mean_a < mean_b. p<0.05 → A vince. p>0.95 e A>B → B vince."""
    if p < 0.05 and mean_a < mean_b:
        return f"{label_a_wins}  (p={p:.4f})"
    elif p > 0.95 and mean_a > mean_b:
        return f"{label_b_wins}  (p_inverso={1-p:.4f}) — H1 falsificata, opposta confermata"
    else:
        return f"PARI / NON CONCLUSIVO  (p={p:.4f})"


def print_results(all_f: List[Metrics], all_a: List[Metrics], all_q: List[Metrics]) -> None:
    # ------------------------------------------------------------------
    # Tabella riassuntiva — tre colonne
    # ------------------------------------------------------------------
    metric_keys = [
        "pre_error", "pre_baseline", "rec_threshold", "plateau_ratio",
        "initial_shock", "recovery_time", "post_cum_error", "final_error",
    ]
    metric_names = {
        "pre_error":      "Errore medio pre-shift",
        "pre_baseline":   "Baseline propria (ult. 50 step)",
        "rec_threshold":  "Soglia recupero (baseline+ε)",
        "plateau_ratio":  "Plateau ratio (convergenza)  *",
        "initial_shock":  "Shock iniziale post-shift  [C]",
        "recovery_time":  "Tempo di recupero (step)",
        "post_cum_error": "Errore cumulato post-shift",
        "final_error":    "Errore medio ultimi 20 step",
    }

    COL = 14
    print("\n" + "=" * 82)
    print(f"{'Metrica':<34} {'Fisso (EHD)':>{COL}} {'Adattivo':>{COL}} {'Q-learning':>{COL}}")
    print("-" * 82)
    for k in metric_keys:
        vf = np.array([m[k] for m in all_f])
        va = np.array([m[k] for m in all_a])
        vq = np.array([m[k] for m in all_q])
        cf = f"{np.mean(vf):>6.2f}±{np.std(vf):<5.2f}"
        ca = f"{np.mean(va):>6.2f}±{np.std(va):<5.2f}"
        cq = f"{np.mean(vq):>6.2f}±{np.std(vq):<5.2f}"
        print(f"{metric_names[k]:<34} {cf:>{COL}} {ca:>{COL}} {cq:>{COL}}")
    print("=" * 82)
    print(f"  * plateau_ratio ≥ {PLATEAU_MIN_RATIO} → convergito")

    # ------------------------------------------------------------------
    # Gate di validità — tutti e tre devono convergere
    # ------------------------------------------------------------------
    plateau_f = float(np.mean([m["plateau_ratio"] for m in all_f]))
    plateau_a = float(np.mean([m["plateau_ratio"] for m in all_a]))
    plateau_q = float(np.mean([m["plateau_ratio"] for m in all_q]))

    parity_ok   = True
    fail_reasons = []
    for label, val in [("Fisso", plateau_f), ("Adattivo", plateau_a), ("Q", plateau_q)]:
        if val < PLATEAU_MIN_RATIO:
            fail_reasons.append(f"  {label}: plateau_ratio={val:.3f} < {PLATEAU_MIN_RATIO}")
            parity_ok = False

    if not parity_ok:
        print(f"\n{'!'*82}")
        print("WARNING — Gate convergenza FALLITO:")
        for r in fail_reasons:
            print(r)
        print("  → Nessun esito emesso. Aumentare T_PRE.")
        print(f"{'!'*82}\n")
        return

    print(f"\n  Gate convergenza OK: Fisso={plateau_f:.3f} | Adattivo={plateau_a:.3f} | Q={plateau_q:.3f}")

    # Array
    rt_f   = np.array([m["recovery_time"] for m in all_f])
    rt_a   = np.array([m["recovery_time"] for m in all_a])
    rt_q   = np.array([m["recovery_time"] for m in all_q])
    nr_f   = int(np.sum([m["never_recovered"] for m in all_f]))
    nr_a   = int(np.sum([m["never_recovered"] for m in all_a]))
    nr_q   = int(np.sum([m["never_recovered"] for m in all_q]))
    shk_f  = np.array([m["initial_shock"] for m in all_f])
    shk_a  = np.array([m["initial_shock"] for m in all_a])
    shk_q  = np.array([m["initial_shock"] for m in all_q])
    prec_f = np.array([m["pre_baseline"] for m in all_f])
    prec_a = np.array([m["pre_baseline"] for m in all_a])
    prec_q = np.array([m["pre_baseline"] for m in all_q])

    SEP = "─" * 82

    # ------------------------------------------------------------------
    # BLOCCO 1 — PRECISIONE a regime
    # ------------------------------------------------------------------
    t_af, p_af = welch_t_one_tailed(prec_a, prec_f)   # adattivo vs fisso (contesto)
    t_aq, p_aq = welch_t_one_tailed(prec_a, prec_q)   # adattivo vs Q (PREREG B)
    t_fq, p_fq = welch_t_one_tailed(prec_f, prec_q)   # fisso vs Q (riferimento v4)

    print(f"\n{SEP}")
    print("BLOCCO 1 — PRECISIONE a regime (baseline propria pre-shift)")
    print(SEP)
    print(f"  Fisso:    {np.mean(prec_f):.4f} ± {np.std(prec_f):.4f}")
    print(f"  Adattivo: {np.mean(prec_a):.4f} ± {np.std(prec_a):.4f}")
    print(f"  Q:        {np.mean(prec_q):.4f} ± {np.std(prec_q):.4f}")
    print(f"  Adattivo vs Fisso: Welch t={t_af:.3f}  "
          f"{_esito_label(p_af, np.mean(prec_a), np.mean(prec_f), 'ADT≈FSS', 'FSS < ADT')}")
    print(f"  Adattivo vs Q   : Welch t={t_aq:.3f}  "
          f"{_esito_label(p_aq, np.mean(prec_a), np.mean(prec_q), 'ADT PIÙ PRECISO', 'Q PIÙ PRECISO')}")
    print(f"  Fisso    vs Q   : Welch t={t_fq:.3f}  "
          f"{_esito_label(p_fq, np.mean(prec_f), np.mean(prec_q), 'FSS PIÙ PRECISO', 'Q PIÙ PRECISO')}"
          f"  [rif. v4]")

    # ------------------------------------------------------------------
    # BLOCCO 2 — AFFIDABILITÀ del recupero
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("BLOCCO 2 — AFFIDABILITÀ del recupero (distribuzione recovery_time)")
    print(SEP)
    rows = [
        ("mai_recuperato",   f"{nr_f}/{N_SEEDS}",          f"{nr_a}/{N_SEEDS}",          f"{nr_q}/{N_SEEDS}"),
        ("mediana (step)",   f"{np.median(rt_f):.1f}",     f"{np.median(rt_a):.1f}",     f"{np.median(rt_q):.1f}"),
        ("min (step)",       f"{np.min(rt_f):.1f}",        f"{np.min(rt_a):.1f}",        f"{np.min(rt_q):.1f}"),
        ("max (step)",       f"{np.max(rt_f):.1f}",        f"{np.max(rt_a):.1f}",        f"{np.max(rt_q):.1f}"),
        ("std (step)",       f"{np.std(rt_f):.1f}",        f"{np.std(rt_a):.1f}",        f"{np.std(rt_q):.1f}"),
        ("mean (step)",      f"{np.mean(rt_f):.1f}",       f"{np.mean(rt_a):.1f}",       f"{np.mean(rt_q):.1f}"),
    ]
    print(f"  {'':24} {'Fisso':>14} {'Adattivo':>14} {'Q':>14}")
    for label, vf, va, vq in rows:
        print(f"  {label:<24} {vf:>14} {va:>14} {vq:>14}")

    # ------------------------------------------------------------------
    # BLOCCO 3 — VELOCITÀ del recupero
    # ------------------------------------------------------------------
    shock_note_fa = ("comparabili ✓"
                     if abs(np.mean(shk_f) - np.mean(shk_a)) < 1.0
                     else "DIVERGONO")
    shock_note_aq = ("comparabili ✓"
                     if abs(np.mean(shk_a) - np.mean(shk_q)) < 1.0
                     else "DIVERGONO — velocità va letta con cautela")
    t_aq3, p_aq3 = welch_t_one_tailed(rt_a, rt_q)

    print(f"\n{SEP}")
    print("BLOCCO 3 — VELOCITÀ del recupero (central tendency)")
    print(SEP)
    print(f"  Shock post-shift:")
    print(f"    Fisso    = {np.mean(shk_f):.2f}±{np.std(shk_f):.2f}")
    print(f"    Adattivo = {np.mean(shk_a):.2f}±{np.std(shk_a):.2f}   Fisso vs Adattivo: {shock_note_fa}")
    print(f"    Q        = {np.mean(shk_q):.2f}±{np.std(shk_q):.2f}   Adattivo vs Q:     {shock_note_aq}")
    print(f"  Recovery Adattivo: media={np.mean(rt_a):.1f}  mediana={np.median(rt_a):.1f}")
    print(f"  Recovery Q:        media={np.mean(rt_q):.1f}  mediana={np.median(rt_q):.1f}")
    print(f"  Recovery Fisso:    media={np.mean(rt_f):.1f}  mediana={np.median(rt_f):.1f}  [rif. v4]")
    print(f"  Welch (Adattivo vs Q) t={t_aq3:.3f}  H1: Adattivo < Q (Adattivo più veloce)")
    print(f"  ESITO: {_esito_label(p_aq3, np.mean(rt_a), np.mean(rt_q), 'ADT PIÙ VELOCE', 'Q PIÙ VELOCE')}")

    # ------------------------------------------------------------------
    # VERDETTO PREREG
    # ------------------------------------------------------------------
    print(f"\n{'═'*82}")
    print("VERDETTO PREREG")
    print(f"{'═'*82}")

    # Condizione A — affidabilità
    print(f"\n  CONDIZIONE A — Affidabilità (H1A: mai_recuperato_adaptive < fisso):")
    print(f"    Fisso:    {nr_f}/{N_SEEDS}  [v4 chiuso = 10/30]")
    print(f"    Adattivo: {nr_a}/{N_SEEDS}")
    print(f"    Q:        {nr_q}/{N_SEEDS}")
    if nr_a < nr_f:
        esito_a = f"H1A NON FALSIFICATA — fragilità PARZIALMENTE RIPARATA ({nr_a} < {nr_f})"
        if nr_a == 0:
            esito_a = f"H1A NON FALSIFICATA — fragilità RIPARATA ({nr_a}/{N_SEEDS}, pari a Q={nr_q}/{N_SEEDS})"
    else:
        esito_a = f"H1A FALSIFICATA — fragilità NON riparata (mai_recuperato_adaptive={nr_a} >= fisso={nr_f})"
    print(f"    ESITO A: {esito_a}")

    # Condizione B — precisione
    print(f"\n  CONDIZIONE B — Precisione preservata (H1B: Adattivo < Q, Welch one-tailed):")
    print(f"    Adattivo pre_baseline: {np.mean(prec_a):.4f} ± {np.std(prec_a):.4f}")
    print(f"    Q        pre_baseline: {np.mean(prec_q):.4f} ± {np.std(prec_q):.4f}")
    print(f"    Welch t={t_aq:.3f}  p={p_aq:.4f}")
    if p_aq < 0.05 and np.mean(prec_a) < np.mean(prec_q):
        esito_b = "H1B NON FALSIFICATA — precisione EHD PRESERVATA (p < 0.05)"
    else:
        esito_b = f"H1B FALSIFICATA — precisione non significativamente migliore di Q (p={p_aq:.4f})"
    print(f"    ESITO B: {esito_b}")

    # Verdetto globale
    print()
    a_ok = nr_a < nr_f
    b_ok = p_aq < 0.05 and np.mean(prec_a) < np.mean(prec_q)
    if a_ok and b_ok:
        verd = "IPOTESI NON FALSIFICATA — omeostasi adattiva ripara la fragilità e preserva la precisione"
    elif a_ok and not b_ok:
        verd = "PARZIALMENTE NON FALSIFICATA — fragilità riparata ma precisione non dimostrata"
    elif not a_ok and b_ok:
        verd = "PARZIALMENTE FALSIFICATA — precisione preservata ma fragilità non riparata"
    else:
        verd = "IPOTESI FALSIFICATA — né affidabilità né precisione soddisfatte"
    print(f"  → {verd}")
    print(f"{'═'*82}\n")


# ---------------------------------------------------------------------------
# Grafico — 2×2
# ---------------------------------------------------------------------------
def plot_results(
    trajs_f: np.ndarray,
    trajs_a: np.ndarray,
    trajs_q: np.ndarray,
    gain_hat_trajs: np.ndarray,
    all_f: List[Metrics],
    all_a: List[Metrics],
    all_q: List[Metrics],
    out_path: str = "benchmark_adaptive_output.png",
) -> None:
    steps  = np.arange(T_STEPS)
    mean_f = trajs_f[:, :, 1].mean(axis=0)
    std_f  = trajs_f[:, :, 1].std(axis=0)
    mean_a = trajs_a[:, :, 1].mean(axis=0)
    std_a  = trajs_a[:, :, 1].std(axis=0)
    mean_q = trajs_q[:, :, 1].mean(axis=0)
    std_q  = trajs_q[:, :, 1].std(axis=0)
    mean_gh = gain_hat_trajs.mean(axis=0)
    std_gh  = gain_hat_trajs.std(axis=0)

    rt_f = np.array([m["recovery_time"] for m in all_f])
    rt_a = np.array([m["recovery_time"] for m in all_a])
    rt_q = np.array([m["recovery_time"] for m in all_q])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Benchmark Omeostasi Adattiva — gain {GAIN_PRE}→{GAIN_POST} a t={SHIFT_T}"
        f"  [LR_GAIN={LR_GAIN}, {N_SEEDS} seed]",
        fontsize=12,
    )

    CLR_F = "#e74c3c"   # rosso — fisso
    CLR_A = "#27ae60"   # verde — adattivo
    CLR_Q = "#3498db"   # blu   — Q-learning

    # (0,0) |error| medio nel tempo
    ax = axes[0, 0]
    ax.plot(steps, mean_f, color=CLR_F, linewidth=1.0, label="Fisso (EHD)", alpha=0.9)
    ax.fill_between(steps, mean_f - std_f, mean_f + std_f, alpha=0.15, color=CLR_F)
    ax.plot(steps, mean_a, color=CLR_A, linewidth=1.2, label="Adattivo (EHD+ĝ)")
    ax.fill_between(steps, mean_a - std_a, mean_a + std_a, alpha=0.15, color=CLR_A)
    ax.plot(steps, mean_q, color=CLR_Q, linewidth=1.0, label="Q-learning", alpha=0.9)
    ax.fill_between(steps, mean_q - std_q, mean_q + std_q, alpha=0.15, color=CLR_Q)
    ax.axvline(SHIFT_T, linestyle=":", color="black", linewidth=1.2, label="shift gain×4")
    ax.axvspan(SHIFT_T, T_STEPS, alpha=0.06, color="orange")
    ax.set_xlabel("Step")
    ax.set_ylabel("|error| medio (30 seed)")
    ax.set_title("|error| medio ± std")
    ax.legend(fontsize=8)

    # (0,1) Traiettoria x (ultimo seed)
    ax = axes[0, 1]
    ax.plot(steps, trajs_f[-1, :, 0], color=CLR_F, alpha=0.8, linewidth=0.9, label="Fisso")
    ax.plot(steps, trajs_a[-1, :, 0], color=CLR_A, alpha=0.8, linewidth=1.1, label="Adattivo")
    ax.plot(steps, trajs_q[-1, :, 0], color=CLR_Q, alpha=0.8, linewidth=0.9, label="Q-learning")
    ax.axhline(SETPOINT, linestyle="--", color="gray", linewidth=0.8, label="setpoint=0")
    ax.axvline(SHIFT_T, linestyle=":", color="black", linewidth=1.2, label="shift")
    ax.axvspan(SHIFT_T, T_STEPS, alpha=0.06, color="orange")
    ax.set_xlabel("Step")
    ax.set_ylabel("x")
    ax.set_title(f"Traiettoria x (seed {N_SEEDS-1})")
    ax.legend(fontsize=8)

    # (1,0) Distribuzione recovery time
    ax = axes[1, 0]
    bins = np.linspace(0, T_POST + 1, 22)
    ax.hist(rt_f, bins=bins, alpha=0.55, color=CLR_F, label=f"Fisso (μ={np.mean(rt_f):.1f})")
    ax.hist(rt_a, bins=bins, alpha=0.55, color=CLR_A, label=f"Adattivo (μ={np.mean(rt_a):.1f})")
    ax.hist(rt_q, bins=bins, alpha=0.55, color=CLR_Q, label=f"Q (μ={np.mean(rt_q):.1f})")
    ax.set_xlabel("Recovery time (step post-shift)")
    ax.set_ylabel("Conteggio seed")
    ax.set_title(f"Distribuzione recovery time ({N_SEEDS} seed)")
    ax.legend(fontsize=8)

    # (1,1) Convergenza ĝ
    ax = axes[1, 1]
    ax.plot(steps, mean_gh, color=CLR_A, linewidth=1.2, label="ĝ medio (30 seed)")
    ax.fill_between(steps, mean_gh - std_gh, mean_gh + std_gh, alpha=0.2, color=CLR_A)
    ax.axhline(GAIN_PRE,  linestyle="--", color="gray",  linewidth=0.8, label=f"gain_pre={GAIN_PRE}")
    ax.axhline(GAIN_POST, linestyle="--", color="orange", linewidth=0.8, label=f"gain_post={GAIN_POST}")
    ax.axvline(SHIFT_T, linestyle=":", color="black", linewidth=1.2, label="shift")
    ax.axvspan(SHIFT_T, T_STEPS, alpha=0.06, color="orange")
    ax.set_xlabel("Step")
    ax.set_ylabel("ĝ (gain stimato)")
    ax.set_title("Convergenza stima guadagno (adattivo)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafico salvato → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    all_f, all_a, all_q, trajs_f, trajs_a, trajs_q, gain_hat_trajs = run_benchmark()
    print_results(all_f, all_a, all_q)
    plot_results(trajs_f, trajs_a, trajs_q, gain_hat_trajs, all_f, all_a, all_q)


if __name__ == "__main__":
    main()
