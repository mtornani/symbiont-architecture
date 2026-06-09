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
HARDENING — Quarta curva: Fisso×2
================================================================================
Quando ĝ si clampaa a GAIN_HAT_MIN=0.5, l'azione dell'adattivo diventa:
  action_adaptive = clip(-tanh(error*k) / 0.5, -1, 1) = clip(2*(-tanh(error*k)), -1, 1)
Questo è esattamente un EHD fisso con guadagno raddoppiato (Fisso×2).
La quarta curva verifica se il "vantaggio" dell'adattivo a f < 0.2 sia
interamente spiegato dall'action amplificata (nessun adattamento reale)
o se vi sia una differenza residua.

DOMANDA HARDENING (d):
  La curva di Fisso×2 coincide con quella dell'adattivo a f < 0.2?
  Se sì: il vantaggio a bassa frequenza è solo guadagno più alto, non adattamento.
  Se no: c'è un contributo del meccanismo adattivo oltre il semplice raddoppio.

SFORZO DI CONTROLLO:
  Metrica aggiuntiva = RMS(azione) nella finestra stazionaria.
  Se Adattivo e Fisso×2 ottengono meno errore spendendo più azione rispetto
  al Fisso, il vantaggio è un trade sforzo↔errore, non superiorità strutturale.
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

# Soglia per dichiarare "coincidenza" tra F×2 e Adattivo (hardening)
F2_MATCH_TOL = 0.03   # |gain_f2 - gain_a| < questa → curve coincidono


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


class SymbiontAgentX2:
    """EHD fisso con azione raddoppiata (clip(2×action,-1,1)).

    Equivalente esatto dell'adattivo quando ĝ è clampato a GAIN_HAT_MIN=0.5:
      action_adaptive = clip(-tanh(error*k) / 0.5, -1, 1)
                      = clip(2 * (-tanh(error*k)), -1, 1)
    Usato come controllo causale: se la curva di F×2 coincide con quella
    dell'adattivo a f < 0.2, il vantaggio è interamente spiegato dall'action
    amplificata, non da alcun adattamento strutturale.
    """
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
        return float(np.clip(-2.0 * math.tanh(error * k), -1.0, 1.0))


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
# Episodio con disturbo sinusoidale a frequenza f — quattro agenti
# ---------------------------------------------------------------------------
def run_episode_freq(env_seed: int, freq: float) -> dict:
    """
    Quattro ambienti con stesso seed (stesso rumore), disturbo sinusoidale a freq f.
    Ritorna dict con chiavi 'f', 'f2', 'a', 'q', ciascuna con:
      'errors':   np.ndarray(T_TOTAL,) — errore signed (x - SETPOINT)
      'actions':  np.ndarray(T_TOTAL,) — azione prodotta (pre-clip dell'env)
    Più 'a'['gain_hat']: np.ndarray(T_TOTAL,).
    """
    env_f   = Env1D(seed=env_seed)
    env_f2  = Env1D(seed=env_seed)
    env_a   = Env1D(seed=env_seed)
    env_q   = Env1D(seed=env_seed)

    agent_f  = SymbiontAgent(seed=0)
    agent_f2 = SymbiontAgentX2(seed=0)
    agent_a  = AdaptiveSymbiontAgent(seed=0)
    agent_q  = QLearningAgent(seed=env_seed)

    errors_f  = np.zeros(T_TOTAL)
    errors_f2 = np.zeros(T_TOTAL)
    errors_a  = np.zeros(T_TOTAL)
    errors_q  = np.zeros(T_TOTAL)
    actions_f  = np.zeros(T_TOTAL)
    actions_f2 = np.zeros(T_TOTAL)
    actions_a  = np.zeros(T_TOTAL)
    actions_q  = np.zeros(T_TOTAL)
    gain_hat   = np.zeros(T_TOTAL)

    for t in range(T_TOTAL):
        dist = DIST_AMP * math.sin(2.0 * math.pi * freq * t)

        err_f  = env_f.error
        act_f  = agent_f.act(err_f)
        _, e_f = env_f.step(act_f, dist)
        errors_f[t]  = e_f
        actions_f[t] = act_f

        err_f2  = env_f2.error
        act_f2  = agent_f2.act(err_f2)
        _, e_f2 = env_f2.step(act_f2, dist)
        errors_f2[t]  = e_f2
        actions_f2[t] = act_f2

        err_a  = env_a.error
        act_a  = agent_a.act(err_a)
        _, e_a = env_a.step(act_a, dist)
        errors_a[t]  = e_a
        actions_a[t] = act_a
        gain_hat[t]  = agent_a.gain_hat

        err_q  = env_q.error
        act_q  = agent_q.act(err_q)
        _, e_q = env_q.step(act_q, dist)
        agent_q.update(e_q, -abs(e_q))
        errors_q[t]  = e_q
        actions_q[t] = act_q

    return {
        'f':  {'errors': errors_f,  'actions': actions_f},
        'f2': {'errors': errors_f2, 'actions': actions_f2},
        'a':  {'errors': errors_a,  'actions': actions_a, 'gain_hat': gain_hat},
        'q':  {'errors': errors_q,  'actions': actions_q},
    }


# ---------------------------------------------------------------------------
# Metriche di frequenza
# ---------------------------------------------------------------------------
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def compute_gain(errors_steady: np.ndarray) -> float:
    """Guadagno = RMS(error_stazionario) / DIST_AMP."""
    return rms(errors_steady) / DIST_AMP


def compute_effort(actions_steady: np.ndarray) -> float:
    """Sforzo = RMS(azione_stazionaria)."""
    return rms(actions_steady)


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
        'f':  {'gains': ndarray(N_SEEDS), 'efforts': ndarray(N_SEEDS), 'ns_count': int},
        'f2': {'gains': ndarray(N_SEEDS), 'efforts': ndarray(N_SEEDS), 'ns_count': int},
        'a':  {'gains': ndarray(N_SEEDS), 'efforts': ndarray(N_SEEDS), 'ns_count': int,
               'gain_hat_steady_mean': ndarray(N_SEEDS)},
        'q':  {'gains': ndarray(N_SEEDS), 'efforts': ndarray(N_SEEDS), 'ns_count': int},
    }
    """
    print("Benchmark risposta in frequenza — disturbance rejection, setpoint fisso")
    print("  [Hardening: quarta curva Fisso×2 + sforzo di controllo RMS(azione)]")
    print(f"  DIST_AMP={DIST_AMP}  GAIN={GAIN}  SETPOINT={SETPOINT}  NOISE_STD={NOISE_STD}")
    print(f"  T_TOTAL={T_TOTAL}  T_TRANS={T_TRANS}  T_STEADY={T_STEADY}")
    print(f"  N_SEEDS={N_SEEDS}  |FREQS|={len(FREQS)}: {FREQS[0]:.3f} → {FREQS[-1]:.3f}")
    print()

    results: dict = {}

    for freq in FREQS:
        gains_f  = np.zeros(N_SEEDS)
        gains_f2 = np.zeros(N_SEEDS)
        gains_a  = np.zeros(N_SEEDS)
        gains_q  = np.zeros(N_SEEDS)
        effs_f   = np.zeros(N_SEEDS)
        effs_f2  = np.zeros(N_SEEDS)
        effs_a   = np.zeros(N_SEEDS)
        effs_q   = np.zeros(N_SEEDS)
        gh_mean  = np.zeros(N_SEEDS)
        ns_f = ns_f2 = ns_a = ns_q = 0

        for seed in range(N_SEEDS):
            ep = run_episode_freq(seed, freq)

            ef_s   = ep['f']['errors'][T_TRANS:]
            ef2_s  = ep['f2']['errors'][T_TRANS:]
            ea_s   = ep['a']['errors'][T_TRANS:]
            eq_s   = ep['q']['errors'][T_TRANS:]
            af_s   = ep['f']['actions'][T_TRANS:]
            af2_s  = ep['f2']['actions'][T_TRANS:]
            aa_s   = ep['a']['actions'][T_TRANS:]
            aq_s   = ep['q']['actions'][T_TRANS:]
            gh_s   = ep['a']['gain_hat'][T_TRANS:]

            gains_f[seed]  = compute_gain(ef_s)
            gains_f2[seed] = compute_gain(ef2_s)
            gains_a[seed]  = compute_gain(ea_s)
            gains_q[seed]  = compute_gain(eq_s)
            effs_f[seed]   = compute_effort(af_s)
            effs_f2[seed]  = compute_effort(af2_s)
            effs_a[seed]   = compute_effort(aa_s)
            effs_q[seed]   = compute_effort(aq_s)
            gh_mean[seed]  = float(np.mean(gh_s))

            if is_nonstationary(ef_s):  ns_f  += 1
            if is_nonstationary(ef2_s): ns_f2 += 1
            if is_nonstationary(ea_s):  ns_a  += 1
            if is_nonstationary(eq_s):  ns_q  += 1

        results[freq] = {
            'f':  {'gains': gains_f,  'efforts': effs_f,  'ns_count': ns_f},
            'f2': {'gains': gains_f2, 'efforts': effs_f2, 'ns_count': ns_f2},
            'a':  {'gains': gains_a,  'efforts': effs_a,  'ns_count': ns_a,
                   'gain_hat_steady_mean': gh_mean},
            'q':  {'gains': gains_q,  'efforts': effs_q,  'ns_count': ns_q},
        }

        print(
            f"  f={freq:.3f}  "
            f"F:{np.mean(gains_f):.3f}/{np.mean(effs_f):.3f}  "
            f"F2:{np.mean(gains_f2):.3f}/{np.mean(effs_f2):.3f}  "
            f"A:{np.mean(gains_a):.3f}/{np.mean(effs_a):.3f}  "
            f"Q:{np.mean(gains_q):.3f}/{np.mean(effs_q):.3f}  "
            f"ĝ:{np.mean(gh_mean):.3f}  "
            f"(gain/effort)"
        )

    return results


# ---------------------------------------------------------------------------
# Output testuale
# ---------------------------------------------------------------------------
def print_results(results: dict) -> None:
    SEP  = "─" * 100
    SEP2 = "═" * 100

    agents    = ['f', 'f2', 'a', 'q']
    ag_labels = {'f': 'Fisso', 'f2': 'Fisso×2', 'a': 'Adattivo', 'q': 'Q-learning'}

    # ------- SANITY CHECK -------
    print(f"\n{SEP}")
    print(f"SANITY CHECK — guadagno ≤ {SANITY_GAIN_MAX} a tutte le frequenze")
    print(SEP)
    sanity_ok = True
    for freq in FREQS:
        vals = {ag: float(np.mean(results[freq][ag]['gains'])) for ag in agents}
        worst = max(vals.values())
        flag = "  ← ATTENZIONE" if worst > SANITY_GAIN_MAX else ""
        if worst > SANITY_GAIN_MAX:
            sanity_ok = False
        print(f"  f={freq:.3f}: " +
              "  ".join(f"{ag_labels[ag]}={vals[ag]:.3f}" for ag in agents) + flag)
    print("  → PASSA." if sanity_ok else "\n  SANITY CHECK FALLITO.")

    # ------- STAZIONARIETÀ -------
    print(f"\n{SEP}")
    print("CHECK STAZIONARIETÀ — [!] se ns > 3/30")
    print(SEP)
    any_ns = False
    for freq in FREQS:
        ns = {ag: results[freq][ag]['ns_count'] for ag in agents}
        if max(ns.values()) > 3:
            print(f"  [!] f={freq:.3f}: " + "  ".join(f"{ag_labels[ag]}={ns[ag]}/30" for ag in agents))
            any_ns = True
    if not any_ns:
        print("  → Tutte le finestre stazionarie.")

    # ------- TABELLA GUADAGNO -------
    print(f"\n{SEP}")
    print("TABELLA GUADAGNO — RMS(error) / DIST_AMP  [30 seed, media]")
    print(f"  > 1 = amplifica   < 1 = attenua   |F2-A| < {F2_MATCH_TOL} = curve coincidono [≡]")
    print(SEP)
    hdr = f"  {'freq':>6}  {'Fisso':>7}  {'Fisso×2':>8}  {'Adattivo':>9}  {'Q':>7}  {'|F2-A|':>7}  {'match?':>7}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for freq in FREQS:
        gf  = float(np.mean(results[freq]['f']['gains']))
        gf2 = float(np.mean(results[freq]['f2']['gains']))
        ga  = float(np.mean(results[freq]['a']['gains']))
        gq  = float(np.mean(results[freq]['q']['gains']))
        delta_f2a = abs(gf2 - ga)
        match = "[≡]" if delta_f2a < F2_MATCH_TOL else ""
        print(f"  {freq:.3f}  {gf:>7.3f}  {gf2:>8.3f}  {ga:>9.3f}  {gq:>7.3f}  "
              f"{delta_f2a:>7.3f}  {match:>7}")

    # ------- TABELLA SFORZO -------
    print(f"\n{SEP}")
    print("TABELLA SFORZO — RMS(azione) nella finestra stazionaria  [30 seed, media]")
    print("  Sforzo maggiore → azione più aggressiva → consuma attuatore")
    print(SEP)
    hdr2 = f"  {'freq':>6}  {'Fisso':>7}  {'Fisso×2':>8}  {'Adattivo':>9}  {'Q':>7}  {'F2/F':>6}  {'A/F':>6}"
    print(hdr2)
    print("  " + "─" * (len(hdr2) - 2))
    for freq in FREQS:
        ef  = float(np.mean(results[freq]['f']['efforts']))
        ef2 = float(np.mean(results[freq]['f2']['efforts']))
        ea  = float(np.mean(results[freq]['a']['efforts']))
        eq  = float(np.mean(results[freq]['q']['efforts']))
        r_f2f = ef2 / (ef + 1e-9)
        r_af  = ea  / (ef + 1e-9)
        print(f"  {freq:.3f}  {ef:>7.3f}  {ef2:>8.3f}  {ea:>9.3f}  {eq:>7.3f}  "
              f"{r_f2f:>6.2f}  {r_af:>6.2f}")

    # ------- DERIVA DI ĝ -------
    print(f"\n{SEP}")
    print("DERIVA DI ĝ — media nella finestra stazionaria [T_TRANS, T_TOTAL]")
    print(f"  Atteso: ĝ ≈ {GAIN_HAT_INIT:.1f} (inerte); ĝ → {GAIN_HAT_MIN} a bassa f; ĝ → >1 ad alta f")
    print(SEP)
    for freq in FREQS:
        gh = results[freq]['a']['gain_hat_steady_mean']
        print(f"  f={freq:.3f}: ĝ  mean={np.mean(gh):.3f}  std={np.std(gh):.3f}"
              f"  min={np.min(gh):.3f}  max={np.max(gh):.3f}")

    # ------- VERDETTI PREREG -------
    gains_f_arr  = np.array([np.mean(results[fr]['f']['gains'])  for fr in FREQS])
    gains_f2_arr = np.array([np.mean(results[fr]['f2']['gains']) for fr in FREQS])
    gains_a_arr  = np.array([np.mean(results[fr]['a']['gains'])  for fr in FREQS])
    gains_q_arr  = np.array([np.mean(results[fr]['q']['gains'])  for fr in FREQS])
    effs_f_arr   = np.array([np.mean(results[fr]['f']['efforts'])  for fr in FREQS])
    effs_f2_arr  = np.array([np.mean(results[fr]['f2']['efforts']) for fr in FREQS])
    effs_a_arr   = np.array([np.mean(results[fr]['a']['efforts'])  for fr in FREQS])
    delta_af     = gains_a_arr - gains_f_arr

    print(f"\n{SEP2}")
    print("VERDETTI PREREG — lettura della curva di risposta in frequenza")
    print(SEP2)

    # (a) Legge fissa
    print(f"\n  (a) LEGGE FISSA:")
    mono_dn = bool(np.all(np.diff(gains_f_arr) < 0))
    mono_up = bool(np.all(np.diff(gains_f_arr) > 0))
    n_above_1_f = int(np.sum(gains_f_arr > 1.0))
    print(f"    Guadagno min = {np.min(gains_f_arr):.3f} a f = {FREQS[int(np.argmin(gains_f_arr))]:.3f}")
    print(f"    Guadagno max = {np.max(gains_f_arr):.3f} a f = {FREQS[int(np.argmax(gains_f_arr))]:.3f}")
    print(f"    Freq con guadagno > 1: {n_above_1_f}/{len(FREQS)}")
    if mono_dn:
        print("    PROFILO: monotono DECRESCENTE — attenua meglio ad alta frequenza (P-ctrl classico)")
    elif mono_up:
        print("    PROFILO: monotono CRESCENTE")
    else:
        print("    PROFILO: non monotono")

    # (b) ĝ vs fisso
    print(f"\n  (b) ĝ (ADATTIVO) vs LEGGE FISSA:")
    n_better = int(np.sum(delta_af < -GAIN_EQUIV_TOL))
    n_worse  = int(np.sum(delta_af > +GAIN_EQUIV_TOL))
    n_equal  = len(FREQS) - n_better - n_worse
    gh_all   = np.concatenate([results[fr]['a']['gain_hat_steady_mean'] for fr in FREQS])
    print(f"    A < F (meglio): {n_better}/{len(FREQS)}  A > F (peggio): {n_worse}/{len(FREQS)}"
          f"  A≈F: {n_equal}/{len(FREQS)}")
    print(f"    ĝ medio globale: {float(np.mean(gh_all)):.3f}")
    if n_better > n_worse and n_better > n_equal:
        print("    → PREVISIONE PRE-REG FALSIFICATA: ĝ MIGLIORA (inatteso — vedere hardening)")
    elif n_equal >= n_better + n_worse:
        print("    → PREVISIONE PRE-REG CONFERMATA (inerte)")
    elif n_worse > n_better:
        print("    → PREVISIONE PRE-REG CONFERMATA: ĝ peggiora")
    else:
        print("    → RISULTATO MISTO")

    # (c) Q
    print(f"\n  (c) Q-LEARNING:")
    q_worse = int(np.sum(gains_q_arr > gains_f_arr + GAIN_EQUIV_TOL))
    n_above_1_q = int(np.sum(gains_q_arr > 1.0))
    f_star_q = FREQS[int(np.argmax(gains_q_arr))]
    print(f"    Q > F (peggio del fisso): {q_worse}/{len(FREQS)} freq")
    print(f"    Guadagno max Q = {np.max(gains_q_arr):.3f} a f = {f_star_q:.3f}")
    print(f"    Freq con guadagno > 1: {n_above_1_q}/{len(FREQS)}")

    # ------- VERDETTO HARDENING -------
    print(f"\n{SEP2}")
    print("VERDETTO HARDENING — Fisso×2 come controllo causale")
    print(SEP2)

    # Selezione freq < 0.2 per la domanda principale
    low_freqs = [fr for fr in FREQS if fr < 0.20]
    n_match_low = sum(
        1 for fr in low_freqs
        if abs(np.mean(results[fr]['f2']['gains']) - np.mean(results[fr]['a']['gains'])) < F2_MATCH_TOL
    )
    n_total_low = len(low_freqs)

    print(f"\n  (d) F×2 vs Adattivo a f < 0.20 ({n_total_low} punti, soglia |F2-A| < {F2_MATCH_TOL}):")
    for fr in low_freqs:
        gf2 = float(np.mean(results[fr]['f2']['gains']))
        ga  = float(np.mean(results[fr]['a']['gains']))
        delta = abs(gf2 - ga)
        tag = "[≡ coincide]" if delta < F2_MATCH_TOL else f"[Δ={delta:.3f}]"
        print(f"    f={fr:.3f}: F×2={gf2:.3f}  A={ga:.3f}  {tag}")

    if n_match_low == n_total_low:
        concl_d = (
            f"F×2 COINCIDE con Adattivo a f < 0.2 ({n_match_low}/{n_total_low} punti).\n"
            f"    Il vantaggio dell'adattivo a bassa frequenza è INTERAMENTE spiegato\n"
            f"    dall'action amplificata (ĝ → 0.5 → azione ×2) — non dall'adattamento strutturale."
        )
    elif n_match_low == 0:
        concl_d = (
            f"F×2 NON coincide con Adattivo a f < 0.2 (0/{n_total_low} punti).\n"
            f"    Esiste un contributo dell'adattamento strutturale oltre il semplice raddoppio."
        )
    else:
        concl_d = (
            f"F×2 coincide parzialmente: {n_match_low}/{n_total_low} punti sotto soglia.\n"
            f"    Risultato misto — leggere i delta riga per riga."
        )
    print(f"    → {concl_d}")

    # Trade sforzo↔errore
    print(f"\n  (e) TRADE SFORZO↔ERRORE:")
    print(f"    [ratio sforzo = RMS(azione agente) / RMS(azione Fisso)]")
    print(f"    [ratio errore = guadagno agente / guadagno Fisso]")
    print(f"    {'freq':>6}  {'sforzo F×2/F':>13}  {'errore F×2/F':>13}  "
          f"{'sforzo A/F':>11}  {'errore A/F':>11}")
    for fr in FREQS:
        gf   = float(np.mean(results[fr]['f']['gains']))
        gf2  = float(np.mean(results[fr]['f2']['gains']))
        ga   = float(np.mean(results[fr]['a']['gains']))
        ef_  = float(np.mean(results[fr]['f']['efforts']))
        ef2_ = float(np.mean(results[fr]['f2']['efforts']))
        ea_  = float(np.mean(results[fr]['a']['efforts']))
        print(f"    {fr:.3f}  {ef2_/ef_:>13.3f}  {gf2/gf:>13.3f}  "
              f"{ea_/ef_:>11.3f}  {ga/gf:>11.3f}")

    # Riassunto trade
    mean_eff_ratio_f2 = float(np.mean(effs_f2_arr / (effs_f_arr + 1e-9)))
    mean_gain_ratio_f2 = float(np.mean(gains_f2_arr / (gains_f_arr + 1e-9)))
    mean_eff_ratio_a  = float(np.mean(effs_a_arr  / (effs_f_arr + 1e-9)))
    mean_gain_ratio_a = float(np.mean(gains_a_arr  / (gains_f_arr + 1e-9)))
    print(f"\n    Medie su tutte le frequenze:")
    print(f"    Fisso×2: sforzo={mean_eff_ratio_f2:.3f}× Fisso  errore={mean_gain_ratio_f2:.3f}× Fisso")
    print(f"    Adattivo: sforzo={mean_eff_ratio_a:.3f}× Fisso  errore={mean_gain_ratio_a:.3f}× Fisso")
    if mean_gain_ratio_f2 < 1.0 and mean_eff_ratio_f2 > 1.0:
        print("    → F×2: meno errore AL COSTO di più azione. Trade sforzo↔errore CONFERMATO.")
    if mean_gain_ratio_a < 1.0 and mean_eff_ratio_a > 1.0:
        print("    → Adattivo: meno errore AL COSTO di più azione. Trade sforzo↔errore CONFERMATO.")
    if abs(mean_gain_ratio_f2 - mean_gain_ratio_a) < 0.05 and abs(mean_eff_ratio_f2 - mean_eff_ratio_a) < 0.05:
        print("    → Adattivo e F×2 equivalenti su entrambe le dimensioni. Il vantaggio è solo gain.")
    elif mean_gain_ratio_a < mean_gain_ratio_f2 and mean_eff_ratio_a <= mean_eff_ratio_f2:
        print("    → Adattivo MIGLIORE di F×2 a parità o minor sforzo: c'è un contributo strutturale.")
    elif mean_gain_ratio_a > mean_gain_ratio_f2:
        print("    → F×2 MIGLIORE di Adattivo: ĝ introduce inefficienza rispetto al puro raddoppio.")

    print(f"\n{'═' * 100}\n")


# ---------------------------------------------------------------------------
# Grafico — tre pannelli: guadagno, sforzo, ĝ
# ---------------------------------------------------------------------------
def plot_results(results: dict, out_path: str = "benchmark_frequency_response_output.png") -> None:
    freq_arr = np.array(FREQS)
    CLR_F  = "#e74c3c"   # rosso — Fisso
    CLR_F2 = "#f39c12"   # arancio — Fisso×2
    CLR_A  = "#27ae60"   # verde — Adattivo
    CLR_Q  = "#3498db"   # blu — Q-learning

    def arr(ag: str, key: str) -> np.ndarray:
        return np.array([np.mean(results[f][ag][key]) for f in FREQS])

    def arr_std(ag: str, key: str) -> np.ndarray:
        return np.array([np.std(results[f][ag][key]) for f in FREQS])

    mean_f   = arr('f',  'gains');   std_f   = arr_std('f',  'gains')
    mean_f2  = arr('f2', 'gains');   std_f2  = arr_std('f2', 'gains')
    mean_a   = arr('a',  'gains');   std_a   = arr_std('a',  'gains')
    mean_q   = arr('q',  'gains');   std_q   = arr_std('q',  'gains')

    eff_f    = arr('f',  'efforts'); eff_std_f  = arr_std('f',  'efforts')
    eff_f2   = arr('f2', 'efforts'); eff_std_f2 = arr_std('f2', 'efforts')
    eff_a    = arr('a',  'efforts'); eff_std_a  = arr_std('a',  'efforts')
    eff_q    = arr('q',  'efforts'); eff_std_q  = arr_std('q',  'efforts')

    mean_gh  = np.array([np.mean(results[f]['a']['gain_hat_steady_mean']) for f in FREQS])
    std_gh   = np.array([np.std( results[f]['a']['gain_hat_steady_mean']) for f in FREQS])

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(
        f"Risposta in Frequenza — disturbance rejection, setpoint fisso\n"
        f"DIST_AMP={DIST_AMP}  GAIN={GAIN}  T_TOTAL={T_TOTAL}  [{N_SEEDS} seed]",
        fontsize=11,
    )

    # --- Pannello 0: Curva di risposta in frequenza (OUTPUT PRIMARIO) ---
    ax = axes[0]
    for mean, std, clr, lbl, mrk in [
        (mean_f,  std_f,  CLR_F,  "Fisso (EHD)",    "o"),
        (mean_f2, std_f2, CLR_F2, "Fisso×2",         "D"),
        (mean_a,  std_a,  CLR_A,  "Adattivo (EHD+ĝ)","s"),
        (mean_q,  std_q,  CLR_Q,  "Q-learning",      "^"),
    ]:
        ax.semilogx(freq_arr, mean, f"{mrk}-", color=clr, linewidth=2.0, markersize=6,
                    label=lbl, zorder=3)
        ax.fill_between(freq_arr, mean - std, mean + std, alpha=0.15, color=clr)
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1.0, label="guadagno = 1")
    ax.set_xlabel("Frequenza f (cicli/step)  [scala log]", fontsize=10)
    ax.set_ylabel(f"Guadagno  RMS(error) / {DIST_AMP}", fontsize=10)
    ax.set_title("Curva di risposta in frequenza\nmedia ± std su 30 seed", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(freq_arr[0] * 0.8, freq_arr[-1] * 1.2)

    # --- Pannello 1: Sforzo di controllo ---
    ax = axes[1]
    for eff, std, clr, lbl, mrk in [
        (eff_f,  eff_std_f,  CLR_F,  "Fisso",    "o"),
        (eff_f2, eff_std_f2, CLR_F2, "Fisso×2",  "D"),
        (eff_a,  eff_std_a,  CLR_A,  "Adattivo", "s"),
        (eff_q,  eff_std_q,  CLR_Q,  "Q",        "^"),
    ]:
        ax.semilogx(freq_arr, eff, f"{mrk}-", color=clr, linewidth=2.0, markersize=6,
                    label=lbl, zorder=3)
        ax.fill_between(freq_arr, eff - std, eff + std, alpha=0.15, color=clr)
    ax.set_xlabel("Frequenza f (cicli/step)  [scala log]", fontsize=10)
    ax.set_ylabel("Sforzo  RMS(azione)", fontsize=10)
    ax.set_title("Sforzo di controllo\n(più alto = più azione applicata)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(freq_arr[0] * 0.8, freq_arr[-1] * 1.2)

    # --- Pannello 2: Deriva di ĝ ---
    ax = axes[2]
    ax.semilogx(freq_arr, mean_gh, "s-", color=CLR_A, linewidth=2.0, markersize=6,
                label="ĝ medio (finestra stazionaria)")
    ax.fill_between(freq_arr, mean_gh - std_gh, mean_gh + std_gh, alpha=0.2, color=CLR_A)
    ax.axhline(GAIN_HAT_INIT, linestyle="--", color="gray", linewidth=1.0,
               label=f"ĝ = {GAIN_HAT_INIT:.1f} (gain vero)")
    ax.axhline(GAIN_HAT_MIN, linestyle=":", color=CLR_F, linewidth=1.2,
               label=f"GAIN_HAT_MIN = {GAIN_HAT_MIN}  (azione ×2)")
    ax.set_xlabel("Frequenza f (cicli/step)  [scala log]", fontsize=10)
    ax.set_ylabel("ĝ (gain stimato)", fontsize=10)
    ax.set_title("Deriva di ĝ sotto disturbo sinusoidale\n(< 0.5 clampato → azione raddoppiata)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(freq_arr[0] * 0.8, freq_arr[-1] * 1.2)
    gh_max = max(float(np.max(mean_gh + std_gh)), GAIN_HAT_INIT + 0.5)
    ax.set_ylim(GAIN_HAT_MIN - 0.3, gh_max + 0.3)

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
