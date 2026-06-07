"""
benchmark_homeostasis_vs_reward.py  (v4 — perturbazione severa, annealing, shock, tre esiti)
==============================================================================================
Confronto falsificabile: regolazione omeostatica (DES) vs policy a reward
esterno (Q-learning tabellare) su un compito 1D non-stazionario.

=============================================================================
PREREG — Pre-registration (fissata prima di eseguire; NON modificare dopo)
=============================================================================

IPOTESI GENERALE:
  L'agente omeostatico (MemoryCluster, Symbiont Architecture Step 4-5)
  si comporta meglio del Q-learning dopo uno shock fuori distribuzione, grazie
  al meccanismo EHD di smorzamento automatico del guadagno tramite cortisolo.
  "Meglio" è scomposto in tre esiti separati (vedi BLOCCHI 1-2-3).

PERTURBAZIONE (Fix A):
  GAIN_POST = 4.0 (quadruplica il guadagno al posto di raddoppiarlo).
  Motivazione: con GAIN_POST = -1.0 (inversione di segno, candidato primario),
  la formula -tanh(error * k) dell'agente EHD produce azione sempre nel verso
  sbagliato → divergenza strutturale al clip-boundary, recupero impossibile.
  Questo non testa il meccanismo cortisolo-damping ma una limitazione fissa
  del segno della legge di controllo — confronto non pertinente all'ipotesi.
  Con gain=4.0 entrambi gli agenti sviluppano oscillazioni misurabili e il
  cortisolo EHD può effettivamente smorzare.

SCHEDULE EPSILON Q-LEARNING (Fix B):
  epsilon decresce linearmente da EPS_START=0.20 a EPS_END=0.05 nei primi
  T_PRE=1000 step (pre-shift), poi rimane a 0.05 per tutta la fase post-shift.
  Dichiarazione: epsilon=0.05 post-shift riduce l'esplorazione; compensato dal
  fatto che Q-table è ora ben convergita. L'effetto netto sul confronto dipende
  dai dati.

METRICHE (tre esiti separati — Fix D):
  BLOCCO 1 — PRECISIONE a regime:
    pre_baseline = media |error| ultimi BASELINE_WINDOW step pre-shift.
    H1: S più preciso (Welch one-tailed p < 0.05).

  BLOCCO 2 — AFFIDABILITÀ del recupero:
    Distribuzione recovery_time: mai_recuperato, mediana, min, max, std.
    Nessun test formale — descrittivo.

  BLOCCO 3 — VELOCITÀ del recupero:
    recovery_time = primo step post-shift con |error| < pre_baseline_i + EPSILON.
    Contesto: initial_shock = |error| al primo step post-shift (Fix C),
    riportato per entrambi gli agenti — se divergono, leggere velocità con cautela.
    H1: S più veloce (Welch one-tailed p < 0.05).

GATE DI VALIDITÀ (plateau convergence check — invariato da v3):
  plateau_ratio = mean(errors[T_PRE-W:]) / mean(errors[T_PRE-2W:T_PRE-W])
  Entrambi >= PLATEAU_MIN_RATIO (0.85) → verdetti emessi.
  Altrimenti: WARNING + nessun verdetto.

ONESTÀ:
  Tre esiti separati: uno può vincere e gli altri no. Risultato onesto
  qualunque esca. "S più preciso e affidabile ma non più veloce" è un
  risultato valido e difendibile.
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

from cluster import MemoryCluster
from environment import GlobalWorldState, NeuronContext

# ---------------------------------------------------------------------------
# Costanti (comuni ai due agenti)
# ---------------------------------------------------------------------------
T_PRE           = 1000        # step pre-shift — sufficiente per plateau Q-table
T_POST          = 200         # step post-shift (finestra di misura)
T_STEPS         = T_PRE + T_POST
SHIFT_T         = T_PRE

GAIN_PRE        = 1.0
GAIN_POST       = 4.0         # quadruplica — severa ma recuperabile (Fix A)

SETPOINT        = 0.0
EPSILON         = 0.1         # margine sopra la baseline propria per "recuperato"
NOISE_STD       = 0.05
N_SEEDS         = 30

BASELINE_WINDOW  = 50          # ultimi N step pre-shift → baseline propria per agente
PLATEAU_WINDOW   = 200         # finestra per check convergenza (20% di T_PRE)
PLATEAU_MIN_RATIO = 0.85       # late/early >= 0.85 → converged (plateau ≤ ±15%)

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
        self.x = float(np.clip(
            self.x + gain * action + self._noise[self._t],
            -6.0, 6.0,
        ))
        self._t += 1
        return self.x, self.error


# ---------------------------------------------------------------------------
# Agente omeostatico (usa MemoryCluster esistente — nessuna reimplementazione)
# ---------------------------------------------------------------------------
#
# Mapping error → world state:
#   risk   = min(|error| / 2, 1.0)   → cortisol_setpoint = 0.1 + 0.7*risk
#   reward = 1 - risk
#
# Formula azione (meccanismo testato):
#   k = 1 - 0.5 * cortisol          (EHD automatic gain damping)
#   action = -tanh(error * k)
#
#   Quando gain raddoppia → error cresce → cortisolo sale →
#   k scende → azione più smorzata → convergenza più rapida.
# ---------------------------------------------------------------------------

class SymbiontAgent:
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
        inp    = np.zeros(self.N_INPUTS)
        inp[0] = float(np.sign(error))
        inp[1] = 1.0 if abs(error) > 0.5 else 0.0
        inp[2] = 1.0 if abs(error) < 0.1 else 0.0
        inp[3] = -float(np.sign(error))
        return [
            NeuronContext(inputs=inp.copy(), local_risk=risk, local_reward=reward)
            for _ in range(self.N_NEURONS)
        ]

    def act(self, error: float) -> float:
        risk   = float(min(abs(error) / 2.0, 1.0))
        reward = 1.0 - risk

        world    = GlobalWorldState(risk=risk, reward=reward, step=self._step_idx, is_rest=False)
        contexts = self._make_contexts(error, risk, reward)

        self._endo, _, _, _ = self.cluster.step(world, contexts)
        self._step_idx += 1

        k = 1.0 - 0.5 * self._endo.cortisol
        return float(np.clip(-math.tanh(error * k), -1.0, 1.0))

    @property
    def cortisol(self) -> float:
        return self._endo.cortisol


# ---------------------------------------------------------------------------
# Baseline Q-learning (competente, non azzoppato)
# ---------------------------------------------------------------------------
#
# Scelte di progetto per fairness:
#  - 30 bin sull'intervallo [-3, 3] (stati |x|>3 → bin boundary, accettabile)
#  - 7 azioni discrete in [-1, 1] (stesso range dell'omeostatico)
#  - epsilon annealing 0.20→0.05 durante T_PRE (Fix B): convergenza Q più
#    pulita; epsilon finale 0.05 mantenuto post-shift. Dichiarato in PREREG.
#  - alpha=0.30, gamma=0.95
#  - Reward = -|error|: stesso segnale che misura la metrica del benchmark
# ---------------------------------------------------------------------------

class QLearningAgent:
    N_BINS    = 30
    X_MIN     = -3.0
    X_MAX     =  3.0
    ACTIONS   = np.array([-1.0, -0.67, -0.33, 0.0, 0.33, 0.67, 1.0])
    ALPHA     = 0.30
    GAMMA     = 0.95
    EPS_START = 0.20   # epsilon iniziale (fix B: annealing)
    EPS_END   = 0.05   # epsilon finale mantenuto anche post-shift

    def __init__(self, seed: int = 0) -> None:
        self.Q       = np.zeros((self.N_BINS, len(self.ACTIONS)))
        self.rng     = np.random.default_rng(seed + 777)
        self._s      : int | None = None
        self._a      : int | None = None
        self._t_step = 0

    def _epsilon(self) -> float:
        """Anneal lineare da EPS_START a EPS_END su T_PRE step, poi costante."""
        frac = min(self._t_step / T_PRE, 1.0)
        return self.EPS_START + frac * (self.EPS_END - self.EPS_START)

    def _bin(self, error: float) -> int:
        c = np.clip(error, self.X_MIN, self.X_MAX)
        return int(min((c - self.X_MIN) / (self.X_MAX - self.X_MIN) * self.N_BINS,
                       self.N_BINS - 1))

    def act(self, error: float) -> float:
        s   = self._bin(error)
        eps = self._epsilon()
        if self.rng.random() < eps:
            a = int(self.rng.integers(len(self.ACTIONS)))
        else:
            a = int(np.argmax(self.Q[s]))
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
# Episodio
# ---------------------------------------------------------------------------

Metrics = Dict[str, float]


def run_episode(env_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ritorna (traj_s, traj_q) — shape (T_STEPS, 2): colonne = (x, |error|).
    Stesso noise di ambiente, agenti partono da zero su ogni episodio.
    """
    env_s   = Env1D(seed=env_seed)
    env_q   = Env1D(seed=env_seed)
    agent_s = SymbiontAgent(seed=0)         # seed fisso per i pesi iniziali
    agent_q = QLearningAgent(seed=env_seed) # seed varia: esplorazione indipendente

    traj_s = np.zeros((T_STEPS, 2))
    traj_q = np.zeros((T_STEPS, 2))

    for t in range(T_STEPS):
        err_s     = env_s.error
        act_s     = agent_s.act(err_s)
        x_s, e_s  = env_s.step(act_s)
        traj_s[t] = (x_s, abs(e_s))

        err_q     = env_q.error
        act_q     = agent_q.act(err_q)
        x_q, e_q  = env_q.step(act_q)
        agent_q.update(e_q, -abs(e_q))
        traj_q[t] = (x_q, abs(e_q))

    return traj_s, traj_q


def compute_metrics(traj: np.ndarray) -> Metrics:
    """
    traj: (T_STEPS, 2), colonna 1 = |error|.

    Recovery relativo alla baseline PROPRIA dell'agente:
      pre_baseline  = media |error| negli ultimi BASELINE_WINDOW step pre-shift
      rec_threshold = pre_baseline + EPSILON
      recovery_time = primo step post-shift con |error| < rec_threshold
                      (capped a T_POST se mai raggiunto)

    Convergenza:
      plateau_ratio = mean(errors[T_PRE-WINDOW:]) / mean(errors[T_PRE-2W:T_PRE-W])
      >= PLATEAU_MIN_RATIO → converged (curva piatta)
      <  PLATEAU_MIN_RATIO → ancora in discesa → non convergito
    """
    errors = traj[:, 1]
    pre    = errors[:SHIFT_T]
    post   = errors[SHIFT_T:]

    # Baseline propria e soglia di recupero
    pre_baseline  = float(np.mean(pre[-BASELINE_WINDOW:]))
    rec_threshold = pre_baseline + EPSILON

    rec_time = T_POST
    for i, e in enumerate(post):
        if e < rec_threshold:
            rec_time = i + 1
            break

    # Plateau convergence: late window vs early window
    w           = PLATEAU_WINDOW
    early_mean  = float(np.mean(pre[-2 * w : -w]))
    late_mean   = float(np.mean(pre[-w:]))
    plateau_ratio = late_mean / (early_mean + 1e-9)

    return {
        "pre_error":       float(np.mean(pre)),
        "pre_baseline":    pre_baseline,
        "rec_threshold":   rec_threshold,
        "plateau_ratio":   plateau_ratio,
        "initial_shock":   float(post[0]),        # |error| al 1° step post-shift (Fix C)
        "recovery_time":   float(rec_time),
        "post_cum_error":  float(np.sum(post)),
        "final_error":     float(np.mean(errors[-20:])),
        "never_recovered": float(rec_time == T_POST),
    }


# ---------------------------------------------------------------------------
# Test di significatività (solo numpy + math — no scipy)
# ---------------------------------------------------------------------------

def welch_t_one_tailed(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """
    H1: mean(a) < mean(b)  (omeostatico recupera prima).
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
    p_val  = 0.5 * (1.0 + math.erf(t_stat / math.sqrt(2.0)))
    return t_stat, p_val


# ---------------------------------------------------------------------------
# Benchmark principale
# ---------------------------------------------------------------------------

def run_benchmark() -> Tuple[List[Metrics], List[Metrics], np.ndarray, np.ndarray]:
    all_s: List[Metrics] = []
    all_q: List[Metrics] = []
    trajs_s = np.zeros((N_SEEDS, T_STEPS, 2))
    trajs_q = np.zeros((N_SEEDS, T_STEPS, 2))

    print(f"Benchmark: {N_SEEDS} seed, T={T_STEPS} (shift a t={SHIFT_T})")
    print(f"  Gain: {GAIN_PRE} → {GAIN_POST}  |  recovery = baseline_propria + {EPSILON}")
    print(f"  Agente omeostatico: MemoryCluster (EHD cortisol-damping)")
    print(f"  Baseline: Q-learning tabellare "
          f"(epsilon {QLearningAgent.EPS_START}→{QLearningAgent.EPS_END} su {T_PRE} step, "
          f"poi costante; alpha={QLearningAgent.ALPHA})")
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

def _dist_extended(arr: np.ndarray, cap: float, label: str) -> None:
    """Stampa stats estese quando std > media (distribuzione degenere)."""
    n_never = int(np.sum(arr == cap))
    print(f"\n  [!] {label}: std({np.std(arr):.2f}) > media({np.mean(arr):.2f}) "
          f"— distribuzione degenere")
    print(f"      mediana={np.median(arr):.1f}  min={np.min(arr):.1f}  "
          f"max={np.max(arr):.1f}  mai_recuperato={n_never}/{len(arr)}")


def _esito_label(p: float, mean_s: float, mean_q: float,
                 label_s_wins: str = "S VINCE",
                 label_q_wins: str = "Q VINCE") -> str:
    """
    p = Welch one-tailed (H1: mean_s < mean_q).
    p < 0.05 → H1 confermata (S vince).
    p > 0.95 → H1 fortemente rigettata (Q vince nell'altra direzione).
    Altrimenti pari / non conclusivo.
    """
    if p < 0.05 and mean_s < mean_q:
        return f"{label_s_wins}  (p={p:.4f})"
    elif p > 0.95 and mean_s > mean_q:
        return f"{label_q_wins}  (p_inverso={1-p:.4f}) — H1 falsificata, opposta confermata"
    else:
        return f"PARI / NON CONCLUSIVO  (p={p:.4f})"


def print_results(all_s: List[Metrics], all_q: List[Metrics]) -> None:
    # Tabella riepilogativa
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

    print("\n" + "=" * 74)
    print(f"{'Metrica':<36} {'Omeostatico':>16} {'Q-baseline':>16}")
    print("-" * 74)
    for k in metric_keys:
        vs = np.array([m[k] for m in all_s])
        vq = np.array([m[k] for m in all_q])
        print(f"{metric_names[k]:<36} {np.mean(vs):>8.2f} ±{np.std(vs):>5.2f}"
              f"   {np.mean(vq):>8.2f} ±{np.std(vq):>5.2f}")
    print("=" * 74)
    print(f"  * plateau_ratio ≥ {PLATEAU_MIN_RATIO} → convergito  "
          f"[C] Fix C: shock comparabili = confronto velocità valido")

    # -----------------------------------------------------------------------
    # Gate di validità: convergenza via plateau
    # -----------------------------------------------------------------------
    mean_plateau_s = float(np.mean([m["plateau_ratio"] for m in all_s]))
    mean_plateau_q = float(np.mean([m["plateau_ratio"] for m in all_q]))

    parity_ok = True
    fail_reasons = []
    if mean_plateau_s < PLATEAU_MIN_RATIO:
        fail_reasons.append(
            f"  Omeostatico: plateau_ratio={mean_plateau_s:.3f} < {PLATEAU_MIN_RATIO}"
        )
        parity_ok = False
    if mean_plateau_q < PLATEAU_MIN_RATIO:
        fail_reasons.append(
            f"  Q-baseline: plateau_ratio={mean_plateau_q:.3f} < {PLATEAU_MIN_RATIO}"
        )
        parity_ok = False

    if not parity_ok:
        print(f"\n{'!'*74}")
        print("WARNING — Gate convergenza FALLITO (almeno un agente non ha raggiunto il plateau):")
        for r in fail_reasons:
            print(r)
        print("  → Nessun esito emesso. Aumentare T_PRE.")
        print(f"{'!'*74}\n")
        return

    print(f"\n  Gate convergenza OK: S plateau={mean_plateau_s:.3f} | "
          f"Q plateau={mean_plateau_q:.3f}")

    # Array principali
    rt_s   = np.array([m["recovery_time"] for m in all_s])
    rt_q   = np.array([m["recovery_time"] for m in all_q])
    nr_s   = int(np.sum([m["never_recovered"] for m in all_s]))
    nr_q   = int(np.sum([m["never_recovered"] for m in all_q]))
    shk_s  = np.array([m["initial_shock"] for m in all_s])
    shk_q  = np.array([m["initial_shock"] for m in all_q])
    prec_s = np.array([m["pre_baseline"] for m in all_s])
    prec_q = np.array([m["pre_baseline"] for m in all_q])

    SEP = "─" * 74

    # -----------------------------------------------------------------------
    # BLOCCO 1 — PRECISIONE a regime
    # -----------------------------------------------------------------------
    t1, p1 = welch_t_one_tailed(prec_s, prec_q)
    print(f"\n{SEP}")
    print("BLOCCO 1 — PRECISIONE a regime (baseline propria, post-annealing)")
    print(SEP)
    print(f"  S: {np.mean(prec_s):.3f} ± {np.std(prec_s):.3f}   "
          f"Q: {np.mean(prec_q):.3f} ± {np.std(prec_q):.3f}")
    print(f"  Welch t={t1:.3f}  H1: S < Q (S più preciso)")
    print(f"  ESITO: {_esito_label(p1, np.mean(prec_s), np.mean(prec_q), 'S PIÙ PRECISO', 'Q PIÙ PRECISO')}")

    # -----------------------------------------------------------------------
    # BLOCCO 2 — AFFIDABILITÀ del recupero
    # -----------------------------------------------------------------------
    print(f"\n{SEP}")
    print("BLOCCO 2 — AFFIDABILITÀ del recupero (distribuzione recovery_time)")
    print(SEP)
    rows = [
        ("mai_recuperato",    f"{nr_s}/{N_SEEDS}",               f"{nr_q}/{N_SEEDS}"),
        ("mediana (step)",    f"{np.median(rt_s):.1f}",          f"{np.median(rt_q):.1f}"),
        ("min (step)",        f"{np.min(rt_s):.1f}",             f"{np.min(rt_q):.1f}"),
        ("max (step)",        f"{np.max(rt_s):.1f}",             f"{np.max(rt_q):.1f}"),
        ("std (step)",        f"{np.std(rt_s):.1f}",             f"{np.std(rt_q):.1f}"),
    ]
    for label, vs, vq in rows:
        print(f"  {label:<24} {vs:>14} {vq:>14}")
    if np.std(rt_s) > np.mean(rt_s):
        _dist_extended(rt_s, T_POST, "Omeostatico")
    if np.std(rt_q) > np.mean(rt_q):
        _dist_extended(rt_q, T_POST, "Q-baseline")
    # Esito: chi ha meno mai_recuperato e coda più corta
    if nr_s < nr_q:
        esito_r = "S PIÙ AFFIDABILE (meno mai_recuperato)"
    elif nr_s > nr_q:
        esito_r = "Q PIÙ AFFIDABILE (meno mai_recuperato)"
    elif np.std(rt_s) < np.std(rt_q):
        esito_r = "S PIÙ AFFIDABILE (stessa frequenza, coda S più corta)"
    elif np.std(rt_s) > np.std(rt_q):
        esito_r = "Q PIÙ AFFIDABILE (stessa frequenza, coda Q più corta)"
    else:
        esito_r = "PARI IN AFFIDABILITÀ"
    print(f"  ESITO: {esito_r}")

    # -----------------------------------------------------------------------
    # BLOCCO 3 — VELOCITÀ del recupero
    # -----------------------------------------------------------------------
    shock_ok = abs(np.mean(shk_s) - np.mean(shk_q)) < 1.0
    shock_note = "comparabili ✓" if shock_ok else "DIVERGONO — velocità va letta con cautela"
    t3, p3 = welch_t_one_tailed(rt_s, rt_q)
    print(f"\n{SEP}")
    print("BLOCCO 3 — VELOCITÀ del recupero (central tendency)")
    print(SEP)
    print(f"  Shock post-shift [Fix C]: "
          f"S={np.mean(shk_s):.2f}±{np.std(shk_s):.2f}  "
          f"Q={np.mean(shk_q):.2f}±{np.std(shk_q):.2f}  → {shock_note}")
    print(f"  Recovery S: media={np.mean(rt_s):.1f}  mediana={np.median(rt_s):.1f}")
    print(f"  Recovery Q: media={np.mean(rt_q):.1f}  mediana={np.median(rt_q):.1f}")
    print(f"  Welch t={t3:.3f}  H1: S < Q (S più veloce)")
    print(f"  ESITO: {_esito_label(p3, np.mean(rt_s), np.mean(rt_q), 'S PIÙ VELOCE', 'Q PIÙ VELOCE')}")
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
    mean_s = trajs_s[:, :, 1].mean(axis=0)
    std_s  = trajs_s[:, :, 1].std(axis=0)
    mean_q = trajs_q[:, :, 1].mean(axis=0)
    std_q  = trajs_q[:, :, 1].std(axis=0)

    rt_s = np.array([m["recovery_time"] for m in all_s])
    rt_q = np.array([m["recovery_time"] for m in all_q])

    mean_thr_s = float(np.mean([m["rec_threshold"] for m in all_s]))
    mean_thr_q = float(np.mean([m["rec_threshold"] for m in all_q]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Benchmark EHD vs Q-Learning — gain {GAIN_PRE}→{GAIN_POST} a t={SHIFT_T}"
        f"  [T_PRE={T_PRE}, recovery = baseline_propria + {EPSILON}]",
        fontsize=11,
    )

    # Pannello 1: |error| medio nel tempo
    ax = axes[0]
    ax.plot(steps, mean_s, color="#e74c3c", linewidth=1.2, label="Omeostatico (DES/EHD)")
    ax.fill_between(steps, mean_s - std_s, mean_s + std_s, alpha=0.2, color="#e74c3c")
    ax.plot(steps, mean_q, color="#3498db", linewidth=1.2, label="Q-learning baseline")
    ax.fill_between(steps, mean_q - std_q, mean_q + std_q, alpha=0.2, color="#3498db")
    ax.axhline(mean_thr_s, linestyle="--", color="#e74c3c", linewidth=0.8,
               label=f"rec.thr. S={mean_thr_s:.2f}")
    ax.axhline(mean_thr_q, linestyle="--", color="#3498db", linewidth=0.8,
               label=f"rec.thr. Q={mean_thr_q:.2f}")
    ax.axvline(SHIFT_T, linestyle=":", color="black", linewidth=1.2, label="shift")
    ax.axvspan(SHIFT_T, T_STEPS, alpha=0.07, color="orange")
    ax.set_xlabel("Step")
    ax.set_ylabel("|error| medio (30 seed)")
    ax.set_title("|error| medio ± std")
    ax.legend(fontsize=7)

    # Pannello 2: Traiettoria x (ultimo seed)
    ax = axes[1]
    ax.plot(steps, trajs_s[-1, :, 0], color="#e74c3c", alpha=0.8,
            linewidth=0.9, label="Omeostatico")
    ax.plot(steps, trajs_q[-1, :, 0], color="#3498db", alpha=0.8,
            linewidth=0.9, label="Q-learning")
    ax.axhline(SETPOINT, linestyle="--", color="gray", linewidth=0.8, label="setpoint=0")
    ax.axvline(SHIFT_T, linestyle=":", color="black", linewidth=1.2, label="shift")
    ax.axvspan(SHIFT_T, T_STEPS, alpha=0.07, color="orange")
    ax.set_xlabel("Step")
    ax.set_ylabel("x")
    ax.set_title(f"Traiettoria x (seed {N_SEEDS-1})")
    ax.legend(fontsize=7)

    # Pannello 3: Distribuzione recovery time
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
