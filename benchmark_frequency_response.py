"""
benchmark_frequency_response.py — risposta in frequenza / disturbance rejection
================================================================================
Banco di risposta in frequenza per i tre agenti (EHD fisso, EHD adattivo+ĝ,
Q-learning). Ripara il difetto di metrica del benchmark sul ritardo: invece di
"tempo di recupero da un gradino" (contaminato dalle oscillazioni indotte dal
ritardo), si misura: per ogni frequenza di disturbo sinusoidale, quanto il
sistema oscilla A REGIME attorno a un setpoint FISSO.

CONTESTO — benchmark chiusi, non riaprire:
  v4 (gain×4): legge fissa fragile → 10/30 mai_recuperato.
  Adattivo: ĝ ripara gain×4 → 0/30.
  Setpoint shift: ĝ irrilevante → benchmark falsificato per condizione A.
  Ritardo: metrica baseline-propria contaminata da D≥1 → non determinabile.

PUNTO DI INNESTO DEL DISTURBO:
  Setpoint = 0.0 costante per tutta la run (nessun gradino).
  A ogni step t: disturbance = DIST_AMP * sin(2π * f * t) viene addizionato
  alla transizione di stato PRIMA che l'errore venga osservato dall'agente.
  L'agente non sa che il disturbo è sinusoidale.
  Lo stesso disturbo (stessa A, stessa f, stesso seed) viene applicato a tutti
  e tre gli agenti: confronto ad armi pari.

METRICA:
  Guadagno di risposta = RMS(error_stazionario) / DIST_AMP.
  >1 = il sistema AMPLIFICA il disturbo (male).
  <1 = il sistema ATTENUA il disturbo (bene).
  La finestra stazionaria è [T_TRANS, T_TOTAL] (ultimi 40% della run).
  Il transitorio (incluso il training di Q) è scartato.

================================================================================
PREREG — Pre-registration (fissata prima di eseguire; NON modificare dopo)
================================================================================

IPOTESI DESCRITTIVA:
  Ogni agente ha una curva di risposta in frequenza caratteristica.
  Non si ipotizza un verdetto unico CONFERMATA/FALSIFICATA: il deliverable
  è la CURVA + lettura delle tre domande.

DOMANDA (a) — Legge fissa:
  La legge fissa ha una risonanza f* dove il guadagno esplode?
  O la curva è monotona? Un P-controller discreto con polo a (1-k) ≈ 0.5
  attenua disturbanze ad alta frequenza e amplifica quelle lente (DC gain = 1/k ≈ 2).
  Non ha risonanza nel senso classico, ma la curva è informativa.

DOMANDA (b) — ĝ rispetto alla legge fissa:
  ĝ sposta la curva in meglio (curva più bassa = più attenuazione),
  in peggio (curva più alta = meno attenuazione), o è inerte (curve identiche)?
  PREVISIONE PRE-REGISTRATA: ĝ è plausibilmente INERTE O DANNOSO.
  Meccanismo: il disturbo sinusoidale introduce nel pred_err una componente
  (-disturbance) non correlata al guadagno vero. L'update
  Δĝ = LR * pred_err * action ≈ LR * (-A*sin(2πft)) * (-k*sin(2πft))
     = LR * k * A * sin²(2πft) < 0 per ogni t
  → ĝ deriva sistematicamente verso GAIN_HAT_MIN=0.5 → action amplificata
  → la curva del adattivo diverge dalla legge fissa.

DOMANDA (c) — Q:
  La curva di Q è più piatta (robusta su banda larga) o ha risonanze proprie?
  Q ha T_TRANS step di training con il disturbo presente — può adattare la
  policy alla frequenza specifica, ma solo con risoluzione discreta dei bin.

SANITY CHECK (a priori):
  Guadagno finito (≤ SANITY_GAIN_MAX=5.0) a TUTTE le frequenze per tutti
  gli agenti. Un P-controller ha DC gain ≈ 2.0: guadagno > 5 indica
  instabilità o bug nell'harness.

STAZIONARIETÀ:
  Se nella finestra stazionaria il RMS cresce oltre STATIONARITY_RATIO,
  il transitorio non è finito → segnalare e suggerire T_TOTAL più lungo.

ONESTÀ:
  "ĝ inerte" = adattamento parametrico non tocca la robustezza in frequenza.
  "ĝ peggiora" = adattamento controproducente sotto disturbo dinamico
                 (il risultato più informativo e più probabile).
  "nessun agente attenua" = serve cambio di struttura, non di parametro.
================================================================================

Run:
    cd symbiont-architecture
    python benchmark_frequency_response.py
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

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, "sam-multiagent-v0"))

from cluster import MemoryCluster
from environment import GlobalWorldState, NeuronContext

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
T_TOTAL    = 5000
T_TRANS    = 3000    # step scartati: transitorio + training Q
T_STEADY   = T_TOTAL - T_TRANS   # 2000 step di finestra stazionaria

DIST_AMP   = 0.3    # ampiezza disturbo sinusoidale — identica per tutti gli agenti
GAIN       = 1.0    # guadagno azione→effetto (costante, nessun guasto)
SETPOINT   = 0.0    # setpoint fisso per tutta la run
NOISE_STD  = 0.05
N_SEEDS    = 30

LR_GAIN       = 0.05
GAIN_HAT_MIN  = 0.5
GAIN_HAT_MAX  = 10.0
GAIN_HAT_INIT = 1.0   # prior equo = guadagno vero

# Frequenze — scala approssimativamente logaritmica, 0.005 → 0.45 (quasi-Nyquist)
FREQS = [0.005, 0.010, 0.020, 0.035, 0.060, 0.100, 0.150,
         0.200, 0.250, 0.300, 0.350, 0.400, 0.450]

STATIONARITY_RATIO = 1.25   # RMS(2°metà) / RMS(1°metà) > questo → segnalare
SANITY_GAIN_MAX    = 5.0    # guadagno massimo ammissibile (P-controller DC ≈ 2.0)
GAIN_EQUIV_TOL     = 0.05   # soglia |delta| per "pari" vs "migliore/peggiore"


# ---------------------------------------------------------------------------
# Ambiente 1D — setpoint fisso, disturbo come parametro di step()
# ---------------------------------------------------------------------------
class Env1D:
    """
    x_{t+1} = x_t + GAIN * clip(action,-1,1) + noise_t + disturbance
    setpoint = SETPOINT costante. error = x - SETPOINT.
    """
    def __init__(self, seed: int) -> None:
        rng         = np.random.default_rng(seed)
        self.x      = float(rng.uniform(-0.5, 0.5))
        self._noise = rng.standard_normal(T_TOTAL) * NOISE_STD
        self._t     = 0

    @property
    def error(self) -> float:
        return self.x - SETPOINT

    def step(self, action: float, disturbance: float = 0.0) -> Tuple[float, float]:
        action  = float(np.clip(action, -1.0, 1.0))
        self.x  = float(np.clip(
            self.x + GAIN * action + self._noise[self._t] + disturbance,
            -8.0, 8.0,
        ))
        self._t += 1
        return self.x, self.error


# ---------------------------------------------------------------------------
# Agenti — identici agli harness precedenti
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


class AdaptiveSymbiontAgent:
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
        x_current = error
        if self._x_prev is not None and self._action_prev is not None:
            if abs(self._action_prev) > 1e-6:
                x_predicted    = self._x_prev + self._gain_hat * self._action_prev
                pred_err       = x_current - x_predicted
                new_gain       = self._gain_hat + LR_GAIN * pred_err * self._action_prev
                self._gain_hat = float(np.clip(new_gain, GAIN_HAT_MIN, GAIN_HAT_MAX))
        risk   = float(min(abs(error) / 2.0, 1.0))
        reward = 1.0 - risk
        world  = GlobalWorldState(risk=risk, reward=reward, step=self._step_idx, is_rest=False)
        self._endo, _, _, _ = self.cluster.step(world, self._make_contexts(error, risk, reward))
        self._step_idx += 1
        k      = 1.0 - 0.5 * self._endo.cortisol
        action = float(np.clip(-math.tanh(error * k) / max(self._gain_hat, GAIN_HAT_MIN), -1.0, 1.0))
        self._x_prev      = x_current
        self._action_prev = action
        return action

    @property
    def gain_hat(self) -> float:
        return self._gain_hat


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
        # Annealing su T_TRANS step: Q converge prima della finestra stazionaria
        frac = min(self._t_step / T_TRANS, 1.0)
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
# Episodio con disturbo sinusoidale a frequenza f
# ---------------------------------------------------------------------------
def run_episode_freq(
    env_seed: int,
    freq: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tre ambienti con stesso seed (stesso rumore), disturbo sinusoidale a frequenza f.
    Ritorna: (errors_f, errors_a, errors_q, gain_hat_traj) ciascuno shape (T_TOTAL,).
    errors_* = errore signed (x - SETPOINT).
    """
    env_f   = Env1D(seed=env_seed)
    env_a   = Env1D(seed=env_seed)
    env_q   = Env1D(seed=env_seed)
    agent_f = SymbiontAgent(seed=0)
    agent_a = AdaptiveSymbiontAgent(seed=0)
    agent_q = QLearningAgent(seed=env_seed)

    errors_f = np.zeros(T_TOTAL)
    errors_a = np.zeros(T_TOTAL)
    errors_q = np.zeros(T_TOTAL)
    gain_hat = np.zeros(T_TOTAL)

    for t in range(T_TOTAL):
        dist = DIST_AMP * math.sin(2.0 * math.pi * freq * t)

        err_f = env_f.error
        act_f = agent_f.act(err_f)
        _, e_f = env_f.step(act_f, dist)
        errors_f[t] = e_f

        err_a = env_a.error
        act_a = agent_a.act(err_a)
        _, e_a = env_a.step(act_a, dist)
        errors_a[t] = e_a
        gain_hat[t] = agent_a.gain_hat

        err_q = env_q.error
        act_q = agent_q.act(err_q)
        _, e_q = env_q.step(act_q, dist)
        agent_q.update(e_q, -abs(e_q))
        errors_q[t] = e_q

    return errors_f, errors_a, errors_q, gain_hat


# ---------------------------------------------------------------------------
# Metriche di frequenza
# ---------------------------------------------------------------------------
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def compute_gain(errors_steady: np.ndarray) -> float:
    """Guadagno = RMS(error_stazionario) / DIST_AMP. >1 = amplifica, <1 = attenua."""
    return rms(errors_steady) / DIST_AMP


def is_nonstationary(errors_steady: np.ndarray) -> bool:
    """True se il RMS nella seconda metà supera STATIONARITY_RATIO × prima metà."""
    h  = len(errors_steady) // 2
    r1 = rms(errors_steady[:h])
    r2 = rms(errors_steady[h:])
    return r2 / (r1 + 1e-9) > STATIONARITY_RATIO


# ---------------------------------------------------------------------------
# Sweep su tutte le frequenze
# ---------------------------------------------------------------------------
def run_sweep() -> dict:
    """
    results[freq] = {
        'f': {'gains': np.ndarray(N_SEEDS,), 'ns_count': int},
        'a': {'gains': np.ndarray(N_SEEDS,), 'ns_count': int,
              'gain_hat_steady_mean': np.ndarray(N_SEEDS,)},
        'q': {'gains': np.ndarray(N_SEEDS,), 'ns_count': int},
    }
    """
    print("Benchmark risposta in frequenza — disturbance rejection, setpoint fisso")
    print(f"  DIST_AMP={DIST_AMP}  GAIN={GAIN}  SETPOINT={SETPOINT}  NOISE_STD={NOISE_STD}")
    print(f"  T_TOTAL={T_TOTAL}  T_TRANS={T_TRANS}  T_STEADY={T_STEADY}")
    print(f"  N_SEEDS={N_SEEDS}  |FREQS|={len(FREQS)}: {FREQS[0]:.3f} → {FREQS[-1]:.3f}")
    print(f"  SANITY_GAIN_MAX={SANITY_GAIN_MAX}  STATIONARITY_RATIO={STATIONARITY_RATIO}")
    print()

    results: dict = {}

    for freq in FREQS:
        gains_f  = np.zeros(N_SEEDS)
        gains_a  = np.zeros(N_SEEDS)
        gains_q  = np.zeros(N_SEEDS)
        gh_mean  = np.zeros(N_SEEDS)
        ns_f = ns_a = ns_q = 0

        for seed in range(N_SEEDS):
            ef, ea, eq, gh = run_episode_freq(seed, freq)
            ef_s = ef[T_TRANS:]
            ea_s = ea[T_TRANS:]
            eq_s = eq[T_TRANS:]
            gh_s = gh[T_TRANS:]

            gains_f[seed] = compute_gain(ef_s)
            gains_a[seed] = compute_gain(ea_s)
            gains_q[seed] = compute_gain(eq_s)
            gh_mean[seed] = float(np.mean(gh_s))

            if is_nonstationary(ef_s): ns_f += 1
            if is_nonstationary(ea_s): ns_a += 1
            if is_nonstationary(eq_s): ns_q += 1

        results[freq] = {
            'f': {'gains': gains_f, 'ns_count': ns_f},
            'a': {'gains': gains_a, 'ns_count': ns_a, 'gain_hat_steady_mean': gh_mean},
            'q': {'gains': gains_q, 'ns_count': ns_q},
        }

        print(
            f"  f={freq:.3f}  F:{np.mean(gains_f):.3f}(med {np.median(gains_f):.3f})"
            f"  A:{np.mean(gains_a):.3f}(med {np.median(gains_a):.3f})"
            f"  Q:{np.mean(gains_q):.3f}(med {np.median(gains_q):.3f})"
            f"  ns:F={ns_f} A={ns_a} Q={ns_q}"
            f"  ĝ_mean={np.mean(gh_mean):.3f}"
        )

    return results


# ---------------------------------------------------------------------------
# Output testuale
# ---------------------------------------------------------------------------
def print_results(results: dict) -> None:
    SEP  = "─" * 94
    SEP2 = "═" * 94

    # ------- SANITY CHECK -------
    print(f"\n{SEP}")
    print(f"SANITY CHECK — guadagno ≤ {SANITY_GAIN_MAX} a tutte le frequenze (P-ctrl DC ≈ 2.0)")
    print(SEP)
    sanity_ok = True
    for freq in FREQS:
        gf = float(np.mean(results[freq]['f']['gains']))
        ga = float(np.mean(results[freq]['a']['gains']))
        gq = float(np.mean(results[freq]['q']['gains']))
        worst = max(gf, ga, gq)
        flag = "  ← ATTENZIONE: guadagno eccessivo" if worst > SANITY_GAIN_MAX else ""
        if worst > SANITY_GAIN_MAX:
            sanity_ok = False
        print(f"  f={freq:.3f}: Fisso={gf:.3f}  Adattivo={ga:.3f}  Q={gq:.3f}{flag}")
    if sanity_ok:
        print("  → PASSA. Guadagni finiti su tutto lo spettro testato.")
    else:
        print("\n  SANITY CHECK FALLITO. Leggere i risultati con cautela.")

    # ------- STAZIONARIETÀ -------
    print(f"\n{SEP}")
    print("CHECK STAZIONARIETÀ — [!] se ns > 3/30 su qualsiasi (agente, freq)")
    print(SEP)
    any_ns = False
    for freq in FREQS:
        ns_f = results[freq]['f']['ns_count']
        ns_a = results[freq]['a']['ns_count']
        ns_q = results[freq]['q']['ns_count']
        if max(ns_f, ns_a, ns_q) > 3:
            print(f"  [!] f={freq:.3f}: Fisso={ns_f}/30  Adattivo={ns_a}/30  Q={ns_q}/30"
                  f"  ← finestra non stazionaria, aumentare T_TOTAL")
            any_ns = True
    if not any_ns:
        print("  → Tutte le finestre stazionarie (ns ≤ 3/30 ovunque).")

    # ------- TABELLA PRINCIPALE -------
    print(f"\n{SEP}")
    print("TABELLA — Guadagno di risposta (RMS_errore_stazionario / DIST_AMP)  [30 seed]")
    print(f"  > 1 = AMPLIFICA il disturbo (peggio)   < 1 = ATTENUA (meglio)")
    print(SEP)
    hdr = (f"  {'freq':>6}  {'F_mean':>7} {'F_med':>7}  "
           f"{'A_mean':>7} {'A_med':>7}  "
           f"{'Q_mean':>7} {'Q_med':>7}  "
           f"{'A vs F':>10}  {'div M-m F':>9}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for freq in FREQS:
        gf = results[freq]['f']['gains']
        ga = results[freq]['a']['gains']
        gq = results[freq]['q']['gains']
        delta = float(np.mean(ga) - np.mean(gf))
        if delta < -GAIN_EQUIV_TOL:
            avf = "A MEGLIO"
        elif delta > +GAIN_EQUIV_TOL:
            avf = "A PEGGIO"
        else:
            avf = "PARI"
        # divergenza media/mediana come indicatore di distribuzione degenere
        div_mm_f = abs(float(np.mean(gf)) - float(np.median(gf)))
        div_flag = "  [!]" if div_mm_f > 0.20 else ""
        print(f"  {freq:.3f}  {np.mean(gf):>7.3f} {np.median(gf):>7.3f}  "
              f"{np.mean(ga):>7.3f} {np.median(ga):>7.3f}  "
              f"{np.mean(gq):>7.3f} {np.median(gq):>7.3f}  "
              f"{avf:>10}  {div_mm_f:>7.3f}{div_flag}")

    # ------- DERIVA DI ĝ -------
    print(f"\n{SEP}")
    print("DERIVA DI ĝ — media nella finestra stazionaria [T_TRANS, T_TOTAL]")
    print(f"  Atteso se INERTE: ĝ ≈ {GAIN_HAT_INIT:.1f} (gain vero)")
    print(f"  Atteso se DERIVA: ĝ → {GAIN_HAT_MIN} (GAIN_HAT_MIN) per la previsione pre-reg")
    print(SEP)
    for freq in FREQS:
        gh = results[freq]['a']['gain_hat_steady_mean']
        print(f"  f={freq:.3f}: ĝ  mean={np.mean(gh):.3f}  std={np.std(gh):.3f}"
              f"  min={np.min(gh):.3f}  max={np.max(gh):.3f}")

    # ------- VERDETTI PREREG -------
    gains_f_arr = np.array([np.mean(results[f]['f']['gains']) for f in FREQS])
    gains_a_arr = np.array([np.mean(results[f]['a']['gains']) for f in FREQS])
    gains_q_arr = np.array([np.mean(results[f]['q']['gains']) for f in FREQS])
    delta_af    = gains_a_arr - gains_f_arr

    print(f"\n{SEP2}")
    print("VERDETTI PREREG — lettura della curva di risposta in frequenza")
    print(SEP2)

    # (a) Legge fissa
    print(f"\n  (a) LEGGE FISSA:")
    f_star_f     = FREQS[int(np.argmax(gains_f_arr))]
    f_min_f      = FREQS[int(np.argmin(gains_f_arr))]
    mono_up      = bool(np.all(np.diff(gains_f_arr) > 0))
    mono_dn      = bool(np.all(np.diff(gains_f_arr) < 0))
    n_above_1_f  = int(np.sum(gains_f_arr > 1.0))
    print(f"    Guadagno min = {np.min(gains_f_arr):.3f} a f = {f_min_f:.3f}")
    print(f"    Guadagno max = {np.max(gains_f_arr):.3f} a f = {f_star_f:.3f}")
    print(f"    Frequenze con guadagno > 1 (amplificazione): {n_above_1_f}/{len(FREQS)}")
    if mono_dn:
        print("    PROFILO: monotono DECRESCENTE — attenua meglio ad alta frequenza (P-ctrl classico)")
    elif mono_up:
        print("    PROFILO: monotono CRESCENTE — amplifica sempre di più ad alta frequenza")
    else:
        print("    PROFILO: non monotono — presenza di picco intermedio")

    # (b) ĝ vs fisso
    print(f"\n  (b) ĝ (ADATTIVO) vs LEGGE FISSA [soglia GAIN_EQUIV_TOL={GAIN_EQUIV_TOL}]:")
    n_better = int(np.sum(delta_af < -GAIN_EQUIV_TOL))
    n_worse  = int(np.sum(delta_af > +GAIN_EQUIV_TOL))
    n_equal  = len(FREQS) - n_better - n_worse
    worst_freq = FREQS[int(np.argmax(delta_af))]
    best_freq  = FREQS[int(np.argmin(delta_af))]
    print(f"    A < F (adattivo meglio): {n_better}/{len(FREQS)} freq")
    print(f"    A > F (adattivo peggio): {n_worse}/{len(FREQS)} freq  ← freq peggiore: {worst_freq:.3f}")
    print(f"    A ≈ F (inerte):          {n_equal}/{len(FREQS)} freq  ← freq migliore: {best_freq:.3f}")
    gh_all = np.concatenate([results[f]['a']['gain_hat_steady_mean'] for f in FREQS])
    mean_gh_global = float(np.mean(gh_all))
    print(f"    ĝ medio globale (tutti freq, tutti seed): {mean_gh_global:.3f}"
          f"  (GAIN_HAT_INIT={GAIN_HAT_INIT:.1f}, MIN={GAIN_HAT_MIN})")
    if n_worse > n_better and n_worse > n_equal:
        print("    → PREVISIONE PRE-REG CONFERMATA: ĝ PEGGIORA la risposta in frequenza")
        print("      Meccanismo: disturbance inquina la stima → ĝ deriva → action scalata male")
    elif n_equal >= n_better + n_worse:
        print("    → PREVISIONE PRE-REG CONFERMATA (caso inerte): curve ≈ identiche")
        print("      ĝ non aiuta né peggiora su questo tipo di disturbo")
    elif n_better > n_worse and n_better > n_equal:
        print("    → PREVISIONE PRE-REG FALSIFICATA: ĝ MIGLIORA la risposta in frequenza")
        print("      Inatteso — riportare con cautela, verificare il meccanismo")
    else:
        print("    → RISULTATO MISTO: ĝ meglio in alcune bande, peggio in altre")
        if mean_gh_global < 0.6:
            print(f"      ĝ tende al minimo ({GAIN_HAT_MIN}) → action amplificata → coerente con previsione")

    # (c) Q
    print(f"\n  (c) Q-LEARNING:")
    q_better = int(np.sum(gains_q_arr < gains_f_arr - GAIN_EQUIV_TOL))
    q_worse  = int(np.sum(gains_q_arr > gains_f_arr + GAIN_EQUIV_TOL))
    q_equal  = len(FREQS) - q_better - q_worse
    f_star_q = FREQS[int(np.argmax(gains_q_arr))]
    f_min_q  = FREQS[int(np.argmin(gains_q_arr))]
    mono_q_dn = bool(np.all(np.diff(gains_q_arr) < 0))
    mono_q_up = bool(np.all(np.diff(gains_q_arr) > 0))
    print(f"    Q < F (meglio del fisso): {q_better}/{len(FREQS)} freq")
    print(f"    Q > F (peggio del fisso): {q_worse}/{len(FREQS)} freq")
    print(f"    Q ≈ F (pari):             {q_equal}/{len(FREQS)} freq")
    print(f"    Guadagno max Q = {np.max(gains_q_arr):.3f} a f = {f_star_q:.3f}")
    print(f"    Guadagno min Q = {np.min(gains_q_arr):.3f} a f = {f_min_q:.3f}")
    if mono_q_dn:
        print("    PROFILO Q: monotono decrescente (più robusto ad alta frequenza)")
    elif mono_q_up:
        print("    PROFILO Q: monotono crescente (meno robusto ad alta frequenza)")
    else:
        print("    PROFILO Q: non monotono — Q ha addestramento localizzato su quella freq")
    n_above_1_q = int(np.sum(gains_q_arr > 1.0))
    print(f"    Frequenze con guadagno > 1: {n_above_1_q}/{len(FREQS)}")

    print(f"\n{'═' * 94}\n")


# ---------------------------------------------------------------------------
# Grafico — curva di risposta in frequenza
# ---------------------------------------------------------------------------
def plot_results(results: dict, out_path: str = "benchmark_frequency_response_output.png") -> None:
    freq_arr = np.array(FREQS)
    CLR_F = "#e74c3c"
    CLR_A = "#27ae60"
    CLR_Q = "#3498db"

    mean_f  = np.array([np.mean(results[f]['f']['gains']) for f in FREQS])
    std_f   = np.array([np.std(results[f]['f']['gains'])  for f in FREQS])
    mean_a  = np.array([np.mean(results[f]['a']['gains']) for f in FREQS])
    std_a   = np.array([np.std(results[f]['a']['gains'])  for f in FREQS])
    mean_q  = np.array([np.mean(results[f]['q']['gains']) for f in FREQS])
    std_q   = np.array([np.std(results[f]['q']['gains'])  for f in FREQS])
    mean_gh = np.array([np.mean(results[f]['a']['gain_hat_steady_mean']) for f in FREQS])
    std_gh  = np.array([np.std(results[f]['a']['gain_hat_steady_mean'])  for f in FREQS])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Risposta in Frequenza — disturbance rejection, setpoint fisso\n"
        f"DIST_AMP={DIST_AMP}  GAIN={GAIN}  T_TOTAL={T_TOTAL}  [{N_SEEDS} seed]",
        fontsize=11,
    )

    # (0) Curva di risposta in frequenza — OUTPUT PRIMARIO
    ax = axes[0]
    ax.semilogx(freq_arr, mean_f, "o-", color=CLR_F, linewidth=2.0, markersize=6,
                label="Fisso (EHD)", zorder=3)
    ax.fill_between(freq_arr, mean_f - std_f, mean_f + std_f, alpha=0.18, color=CLR_F)
    ax.semilogx(freq_arr, mean_a, "s-", color=CLR_A, linewidth=2.0, markersize=6,
                label="Adattivo (EHD+ĝ)", zorder=3)
    ax.fill_between(freq_arr, mean_a - std_a, mean_a + std_a, alpha=0.18, color=CLR_A)
    ax.semilogx(freq_arr, mean_q, "^-", color=CLR_Q, linewidth=2.0, markersize=6,
                label="Q-learning", zorder=3)
    ax.fill_between(freq_arr, mean_q - std_q, mean_q + std_q, alpha=0.18, color=CLR_Q)
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1.0,
               label="guadagno = 1 (soglia amplificazione)")
    ax.set_xlabel("Frequenza f (cicli/step)  [scala log]", fontsize=10)
    ax.set_ylabel(f"Guadagno di risposta  RMS(error) / {DIST_AMP}", fontsize=10)
    ax.set_title("Curva di risposta in frequenza\nmedia ± std su 30 seed", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(freq_arr[0] * 0.8, freq_arr[-1] * 1.2)

    # (1) Deriva di ĝ vs frequenza
    ax = axes[1]
    ax.semilogx(freq_arr, mean_gh, "s-", color=CLR_A, linewidth=2.0, markersize=6,
                label="ĝ medio (finestra stazionaria)")
    ax.fill_between(freq_arr, mean_gh - std_gh, mean_gh + std_gh, alpha=0.2, color=CLR_A)
    ax.axhline(GAIN_HAT_INIT, linestyle="--", color="gray", linewidth=1.0,
               label=f"ĝ = {GAIN_HAT_INIT:.1f} (gain vero, nessuna deriva)")
    ax.axhline(GAIN_HAT_MIN, linestyle=":", color="#e74c3c", linewidth=1.0,
               label=f"GAIN_HAT_MIN = {GAIN_HAT_MIN}")
    ax.set_xlabel("Frequenza f (cicli/step)  [scala log]", fontsize=10)
    ax.set_ylabel("ĝ (gain stimato)", fontsize=10)
    ax.set_title("Deriva di ĝ sotto disturbo sinusoidale\n(previsione: ĝ → MIN per correlazione spurio)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(freq_arr[0] * 0.8, freq_arr[-1] * 1.2)
    ax.set_ylim(GAIN_HAT_MIN - 0.2, GAIN_HAT_INIT + 0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafico salvato → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    results = run_sweep()
    print_results(results)
    plot_results(results)


if __name__ == "__main__":
    main()
