"""
benchmark_frequency_robustness.py — robustezza dell'effetto ad alta frequenza
================================================================================
STESSO banco di risposta in frequenza di benchmark_frequency_response.py,
rigirato variando DUE condizioni:
  A. Ampiezza disturbo: DIST_AMP ∈ {0.1, 0.3, 0.6}
  B. Range frequenze esteso fino a 0.480 (più vicino a Nyquist = 0.5)

Domanda: l'effetto "ĝ ad alta frequenza usa MENO azione E fa MENO errore del
Fisso×2" osservato nella run di riferimento (DIST_AMP=0.3, f≥0.25) è ROBUSTO
(sopravvive a tutte le ampiezze, non crolla avvicinandosi a Nyquist) o è un
ARTEFATTO di quella singola ampiezza e di quel range?

NON toccare gli esperimenti chiusi. Onesto qualunque esca.

================================================================================
PREREG — Pre-registration (fissata prima di eseguire; NON modificare dopo)
================================================================================

CONTESTO DELL'EFFETTO DA STRESSARE (run chiusa, non modificabile):
  A DIST_AMP=0.3, f≥0.25: Adattivo (ĝ) usa ~28% dell'azione del Fisso E
  fa meno errore (gain_A=0.484 vs gain_F=0.650 a f=0.450).
  A bassa frequenza (f<0.20): Adattivo ≡ Fisso×2 (7/7 punti, Δ<0.03) —
  il vantaggio lì è solo action raddoppiata, non adattamento strutturale.

DOMANDA ROBUSTEZZA (A): Ampiezza del disturbo
  L'effetto "Adattivo domina a f≥0.25 su errore E sforzo" persiste
  a DIST_AMP piccola (0.1) e grande (0.6), o scompare/si inverte?
  DEFINIZIONE ROBUSTO: gain_A < gain_F×2 E effort_A < effort_F×2
    per almeno 4 punti a f ≥ 0.25 su CIASCUNA ampiezza.
  DEFINIZIONE ARTEFATTO: su almeno una ampiezza, A ≥ F×2 su gain OPPURE
    A ≥ F×2 su effort per la maggioranza dei punti a f ≥ 0.25.

DOMANDA ROBUSTEZZA (B): Avvicinamento a Nyquist (f ∈ [0.460, 0.480])
  La dominanza di ĝ sopravvive fino a f = 0.480, o crolla vicino a Nyquist?
  DEFINIZIONE ROBUSTO: gain_A < gain_F E gain_A < gain_F×2 per tutte le
    nuove frequenze (0.460, 0.470, 0.480) alla DIST_AMP di riferimento (0.3).
  DEFINIZIONE CROLLA: gain_A > gain_F OPPURE gain_A > gain_F×2 per almeno
    una delle nuove frequenze.

CHECK BODE/WATERBED (empirico — il waterbed in senso stretto non si applica
  a sistemi nonlineari):
  Se la legge fissa attenua meglio ad alta f, guadagna a bassa f?
  Verifica: il guadagno fisso è monotono DECRESCENTE su [0.005, 0.480]?

SANITY CHECK:
  A DIST_AMP=0.3 (condizione di riferimento) e f=0.005, ordinamento:
    gain_A < gain_F < 1.0  e  gain_Q > 1.5
  Compatibile con la run chiusa. Se fallisce → bug nell'harness.

================================================================================

Run:
    cd symbiont-architecture
    python benchmark_frequency_robustness.py
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, "sam-multiagent-v0"))

from cluster import MemoryCluster
from environment import GlobalWorldState, NeuronContext

# ---------------------------------------------------------------------------
# Costanti fisse (identiche a benchmark_frequency_response.py)
# ---------------------------------------------------------------------------
T_TOTAL   = 5000
T_TRANS   = 3000
T_STEADY  = T_TOTAL - T_TRANS

GAIN      = 1.0
SETPOINT  = 0.0
NOISE_STD = 0.05
N_SEEDS   = 30

LR_GAIN       = 0.05
GAIN_HAT_MIN  = 0.5
GAIN_HAT_MAX  = 10.0
GAIN_HAT_INIT = 1.0

# ---------------------------------------------------------------------------
# Variazioni sperimentali
# ---------------------------------------------------------------------------
DIST_AMPS    = [0.1, 0.3, 0.6]
DIST_AMP_REF = 0.3                 # ampiezza di riferimento (benchmark chiuso)

FREQS_BASE = [0.005, 0.010, 0.020, 0.035, 0.060, 0.100, 0.150,
              0.200, 0.250, 0.300, 0.350, 0.400, 0.450]
FREQS_EXT  = [0.460, 0.470, 0.480]
FREQS      = FREQS_BASE + FREQS_EXT   # 16 frequenze

STATIONARITY_RATIO = 1.25
SANITY_GAIN_MAX    = 5.0
GAIN_EQUIV_TOL     = 0.05
F2_MATCH_TOL       = 0.03

HIGH_FREQ_THRESHOLD = 0.25   # soglia "alta frequenza" per la PREREG
ROBUST_MIN_POINTS   = 4      # min punti f≥0.25 con A dom. → ROBUSTO per ampiezza


# ---------------------------------------------------------------------------
# Ambiente 1D (identico a benchmark_frequency_response.py)
# ---------------------------------------------------------------------------
class Env1D:
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
# Agenti (identici a benchmark_frequency_response.py)
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
    """EHD fisso con azione raddoppiata — controllo causale per hardening."""
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
# Episodio — dist_amp come parametro (non più globale)
# ---------------------------------------------------------------------------
def run_episode_freq(env_seed: int, freq: float, dist_amp: float) -> dict:
    """Quattro ambienti con stesso seed, disturbo sinusoidale parametrizzato."""
    env_f   = Env1D(seed=env_seed)
    env_f2  = Env1D(seed=env_seed)
    env_a   = Env1D(seed=env_seed)
    env_q   = Env1D(seed=env_seed)

    agent_f  = SymbiontAgent(seed=0)
    agent_f2 = SymbiontAgentX2(seed=0)
    agent_a  = AdaptiveSymbiontAgent(seed=0)
    agent_q  = QLearningAgent(seed=env_seed)

    errors_f   = np.zeros(T_TOTAL)
    errors_f2  = np.zeros(T_TOTAL)
    errors_a   = np.zeros(T_TOTAL)
    errors_q   = np.zeros(T_TOTAL)
    actions_f  = np.zeros(T_TOTAL)
    actions_f2 = np.zeros(T_TOTAL)
    actions_a  = np.zeros(T_TOTAL)
    actions_q  = np.zeros(T_TOTAL)
    gain_hat   = np.zeros(T_TOTAL)

    for t in range(T_TOTAL):
        dist = dist_amp * math.sin(2.0 * math.pi * freq * t)

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
# Metriche
# ---------------------------------------------------------------------------
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def compute_gain(errors_steady: np.ndarray, dist_amp: float) -> float:
    return rms(errors_steady) / dist_amp


def compute_effort(actions_steady: np.ndarray) -> float:
    return rms(actions_steady)


def is_nonstationary(errors_steady: np.ndarray) -> bool:
    h  = len(errors_steady) // 2
    r1 = rms(errors_steady[:h])
    r2 = rms(errors_steady[h:])
    return r2 / (r1 + 1e-9) > STATIONARITY_RATIO


# ---------------------------------------------------------------------------
# Sweep multi-ampiezza
# ---------------------------------------------------------------------------
def run_sweep(dist_amps: list, freqs: list) -> dict:
    """
    results[dist_amp][freq] = {
        'f':  {'gains': ndarray(N_SEEDS), 'efforts': ndarray(N_SEEDS), 'ns_count': int},
        'f2': {'gains': ndarray(N_SEEDS), 'efforts': ndarray(N_SEEDS), 'ns_count': int},
        'a':  {'gains': ndarray(N_SEEDS), 'efforts': ndarray(N_SEEDS), 'ns_count': int,
               'gain_hat_steady_mean': ndarray(N_SEEDS)},
        'q':  {'gains': ndarray(N_SEEDS), 'efforts': ndarray(N_SEEDS), 'ns_count': int},
    }
    """
    print("Benchmark robustezza risposta in frequenza — disturbance rejection, setpoint fisso")
    print(f"  DIST_AMPS={dist_amps}  GAIN={GAIN}  SETPOINT={SETPOINT}  NOISE_STD={NOISE_STD}")
    print(f"  T_TOTAL={T_TOTAL}  T_TRANS={T_TRANS}  T_STEADY={T_STEADY}")
    print(f"  N_SEEDS={N_SEEDS}  |FREQS|={len(freqs)}: {freqs[0]:.3f} → {freqs[-1]:.3f}")
    print(f"  Frequenze nuove (variazione B): {FREQS_EXT}")
    print()

    results: dict = {}

    for dist_amp in dist_amps:
        print(f"  ── DIST_AMP={dist_amp} ──────────────────────────────────────────────────────────")
        results[dist_amp] = {}

        for freq in freqs:
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
                ep = run_episode_freq(seed, freq, dist_amp)

                ef_s   = ep['f']['errors'][T_TRANS:]
                ef2_s  = ep['f2']['errors'][T_TRANS:]
                ea_s   = ep['a']['errors'][T_TRANS:]
                eq_s   = ep['q']['errors'][T_TRANS:]
                af_s   = ep['f']['actions'][T_TRANS:]
                af2_s  = ep['f2']['actions'][T_TRANS:]
                aa_s   = ep['a']['actions'][T_TRANS:]
                aq_s   = ep['q']['actions'][T_TRANS:]
                gh_s   = ep['a']['gain_hat'][T_TRANS:]

                gains_f[seed]  = compute_gain(ef_s,  dist_amp)
                gains_f2[seed] = compute_gain(ef2_s, dist_amp)
                gains_a[seed]  = compute_gain(ea_s,  dist_amp)
                gains_q[seed]  = compute_gain(eq_s,  dist_amp)
                effs_f[seed]   = compute_effort(af_s)
                effs_f2[seed]  = compute_effort(af2_s)
                effs_a[seed]   = compute_effort(aa_s)
                effs_q[seed]   = compute_effort(aq_s)
                gh_mean[seed]  = float(np.mean(gh_s))

                if is_nonstationary(ef_s):  ns_f  += 1
                if is_nonstationary(ef2_s): ns_f2 += 1
                if is_nonstationary(ea_s):  ns_a  += 1
                if is_nonstationary(eq_s):  ns_q  += 1

            results[dist_amp][freq] = {
                'f':  {'gains': gains_f,  'efforts': effs_f,  'ns_count': ns_f},
                'f2': {'gains': gains_f2, 'efforts': effs_f2, 'ns_count': ns_f2},
                'a':  {'gains': gains_a,  'efforts': effs_a,  'ns_count': ns_a,
                       'gain_hat_steady_mean': gh_mean},
                'q':  {'gains': gains_q,  'efforts': effs_q,  'ns_count': ns_q},
            }

            new_tag = " [NEW]" if freq in FREQS_EXT else ""
            print(
                f"    f={freq:.3f}{new_tag}  "
                f"F:{np.mean(gains_f):.3f}/{np.mean(effs_f):.3f}  "
                f"F2:{np.mean(gains_f2):.3f}/{np.mean(effs_f2):.3f}  "
                f"A:{np.mean(gains_a):.3f}/{np.mean(effs_a):.3f}  "
                f"Q:{np.mean(gains_q):.3f}/{np.mean(effs_q):.3f}  "
                f"ĝ:{np.mean(gh_mean):.3f}"
            )

    return results


# ---------------------------------------------------------------------------
# Output testuale
# ---------------------------------------------------------------------------
def print_results(results: dict, dist_amps: list, freqs: list) -> None:
    SEP  = "─" * 100
    SEP2 = "═" * 100
    agents    = ['f', 'f2', 'a', 'q']
    ag_labels = {'f': 'Fisso', 'f2': 'Fisso×2', 'a': 'Adattivo', 'q': 'Q-learning'}
    high_freqs = [f for f in freqs if f >= HIGH_FREQ_THRESHOLD]

    # ── SANITY CHECK ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"SANITY CHECK — guadagno ≤ {SANITY_GAIN_MAX} + ordinamento a DIST_AMP={DIST_AMP_REF}, f={freqs[0]:.3f}")
    print(SEP)
    sanity_ok = True
    for dist_amp in dist_amps:
        for freq in freqs:
            vals = {ag: float(np.mean(results[dist_amp][freq][ag]['gains'])) for ag in agents}
            worst = max(vals.values())
            if worst > SANITY_GAIN_MAX:
                sanity_ok = False
                print(f"  FAIL: DIST_AMP={dist_amp} f={freq:.3f} guadagno max={worst:.3f} > {SANITY_GAIN_MAX}")

    ref_f0 = freqs[0]
    ga0  = float(np.mean(results[DIST_AMP_REF][ref_f0]['a']['gains']))
    gf0  = float(np.mean(results[DIST_AMP_REF][ref_f0]['f']['gains']))
    gq0  = float(np.mean(results[DIST_AMP_REF][ref_f0]['q']['gains']))
    order_ok = (ga0 < gf0 < 1.0) and (gq0 > 1.5)
    print(f"  Ordinamento a DIST_AMP={DIST_AMP_REF}, f={ref_f0:.3f}:")
    print(f"    gain_A={ga0:.3f}  gain_F={gf0:.3f}  gain_Q={gq0:.3f}")
    print(f"    Atteso: A < F < 1.0 e Q > 1.5 → {'OK' if order_ok else 'FAIL (possibile bug)'}")
    if sanity_ok and order_ok:
        print("  → SANITY PASSA.")
    else:
        print("  → SANITY FALLITO — verificare harness prima di interpretare i risultati.")

    # ── STAZIONARIETÀ ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("CHECK STAZIONARIETÀ — [!] se ns > 3/30")
    print(SEP)
    any_ns = False
    for dist_amp in dist_amps:
        for freq in freqs:
            ns = {ag: results[dist_amp][freq][ag]['ns_count'] for ag in agents}
            if max(ns.values()) > 3:
                print(f"  [!] DIST_AMP={dist_amp} f={freq:.3f}: " +
                      "  ".join(f"{ag_labels[ag]}={ns[ag]}/30" for ag in agents))
                any_ns = True
    if not any_ns:
        print("  → Tutte le finestre stazionarie.")

    # ── TABELLE PER AMPIEZZA ─────────────────────────────────────────────────
    for dist_amp in dist_amps:
        print(f"\n{SEP}")
        print(f"TABELLA GUADAGNO — DIST_AMP={dist_amp}  [N_SEEDS={N_SEEDS}, media]")
        print(f"  > 1 = amplifica   < 1 = attenua   [NEW] = frequenza aggiunta (variazione B)")
        print(SEP)
        hdr = f"  {'freq':>6}  {'Fisso':>7}  {'Fisso×2':>8}  {'Adattivo':>9}  {'Q':>7}  {'A/F':>6}  {'A/F×2':>7}"
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for freq in freqs:
            gf  = float(np.mean(results[dist_amp][freq]['f']['gains']))
            gf2 = float(np.mean(results[dist_amp][freq]['f2']['gains']))
            ga  = float(np.mean(results[dist_amp][freq]['a']['gains']))
            gq  = float(np.mean(results[dist_amp][freq]['q']['gains']))
            r_af  = ga / (gf  + 1e-9)
            r_af2 = ga / (gf2 + 1e-9)
            new_tag = " [NEW]" if freq in FREQS_EXT else ""
            print(f"  {freq:.3f}  {gf:>7.3f}  {gf2:>8.3f}  {ga:>9.3f}  {gq:>7.3f}  "
                  f"{r_af:>6.3f}  {r_af2:>7.3f}{new_tag}")

        print(f"\n  SFORZO — DIST_AMP={dist_amp}  [N_SEEDS={N_SEEDS}, media]")
        hdr2 = f"  {'freq':>6}  {'Fisso':>7}  {'Fisso×2':>8}  {'Adattivo':>9}  {'Q':>7}  {'A/F':>6}  {'A/F×2':>7}"
        print(hdr2)
        print("  " + "─" * (len(hdr2) - 2))
        for freq in freqs:
            ef  = float(np.mean(results[dist_amp][freq]['f']['efforts']))
            ef2 = float(np.mean(results[dist_amp][freq]['f2']['efforts']))
            ea  = float(np.mean(results[dist_amp][freq]['a']['efforts']))
            eq  = float(np.mean(results[dist_amp][freq]['q']['efforts']))
            r_af  = ea / (ef  + 1e-9)
            r_af2 = ea / (ef2 + 1e-9)
            new_tag = " [NEW]" if freq in FREQS_EXT else ""
            print(f"  {freq:.3f}  {ef:>7.3f}  {ef2:>8.3f}  {ea:>9.3f}  {eq:>7.3f}  "
                  f"{r_af:>6.3f}  {r_af2:>7.3f}{new_tag}")

        # Deriva di ĝ
        print(f"\n  DERIVA ĝ — DIST_AMP={dist_amp}")
        for freq in freqs:
            gh = results[dist_amp][freq]['a']['gain_hat_steady_mean']
            new_tag = " [NEW]" if freq in FREQS_EXT else ""
            print(f"    f={freq:.3f}: ĝ mean={np.mean(gh):.3f}  std={np.std(gh):.3f}"
                  f"  min={np.min(gh):.3f}  max={np.max(gh):.3f}{new_tag}")

    # ── VERDETTI ROBUSTEZZA ───────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("VERDETTI ROBUSTEZZA PREREG")
    print(SEP2)

    # DOMANDA A
    print(f"\n  [A] ROBUSTEZZA ALL'AMPIEZZA del disturbo")
    print(f"      Effetto da stressare: A < F×2 su gain E effort per f≥{HIGH_FREQ_THRESHOLD}")
    print(f"      ROBUSTO se ≥{ROBUST_MIN_POINTS} punti a f≥{HIGH_FREQ_THRESHOLD} su OGNI ampiezza")
    robust_A = True
    for dist_amp in dist_amps:
        dom_points = 0
        print(f"\n    DIST_AMP={dist_amp}  ({len(high_freqs)} punti a f≥{HIGH_FREQ_THRESHOLD}):")
        for freq in high_freqs:
            ga_g  = float(np.mean(results[dist_amp][freq]['a']['gains']))
            gf2_g = float(np.mean(results[dist_amp][freq]['f2']['gains']))
            ea_e  = float(np.mean(results[dist_amp][freq]['a']['efforts']))
            ef2_e = float(np.mean(results[dist_amp][freq]['f2']['efforts']))
            dom_gain   = ga_g  < gf2_g
            dom_effort = ea_e  < ef2_e
            dom        = dom_gain and dom_effort
            if dom:
                dom_points += 1
            tick = "OK" if dom else ("gain-NOK" if not dom_gain else "effort-NOK")
            print(f"      f={freq:.3f}: gain_A={ga_g:.3f}/F2={gf2_g:.3f} "
                  f"eff_A={ea_e:.3f}/F2={ef2_e:.3f}  → {tick}")
        if dom_points < ROBUST_MIN_POINTS:
            robust_A = False
            print(f"      → ARTEFATTO a questa ampiezza ({dom_points}/{len(high_freqs)} < {ROBUST_MIN_POINTS})")
        else:
            print(f"      → ROBUSTO a questa ampiezza ({dom_points}/{len(high_freqs)} ≥ {ROBUST_MIN_POINTS})")

    print(f"\n    VERDETTO DOMANDA A: "
          f"{'EFFETTO ROBUSTO alle variazioni di ampiezza' if robust_A else 'EFFETTO NON ROBUSTO / ARTEFATTO DI AMPIEZZA'}")

    # DOMANDA B
    print(f"\n  [B] ROBUSTEZZA A NYQUIST — nuove frequenze {FREQS_EXT}  (DIST_AMP={DIST_AMP_REF})")
    print(f"      ROBUSTO se gain_A < gain_F E gain_A < gain_F×2 su tutte le f nuove")
    robust_B = True
    for freq in FREQS_EXT:
        gf  = float(np.mean(results[DIST_AMP_REF][freq]['f']['gains']))
        gf2 = float(np.mean(results[DIST_AMP_REF][freq]['f2']['gains']))
        ga  = float(np.mean(results[DIST_AMP_REF][freq]['a']['gains']))
        ea  = float(np.mean(results[DIST_AMP_REF][freq]['a']['efforts']))
        ef  = float(np.mean(results[DIST_AMP_REF][freq]['f']['efforts']))
        ok_f  = ga < gf
        ok_f2 = ga < gf2
        if not (ok_f and ok_f2):
            robust_B = False
        tag = "OK" if (ok_f and ok_f2) else ("A>F" if not ok_f else "A>F2")
        print(f"    f={freq:.3f}: gain_F={gf:.3f}  gain_F2={gf2:.3f}  gain_A={ga:.3f}  "
              f"eff_A={ea:.3f}  eff_F={ef:.3f}  → {tag}")

    print(f"\n    VERDETTO DOMANDA B: "
          f"{'EFFETTO ROBUSTO fino a f=0.480 (quasi-Nyquist)' if robust_B else 'EFFETTO CROLLA vicino a Nyquist'}")

    # CHECK BODE
    print(f"\n  [C] CHECK BODE/WATERBED EMPIRICO — curva Fisso su [0.005, 0.480]")
    print(f"      (Il waterbed non si applica ai sistemi non lineari — solo check descrittivo)")
    gains_f_all = [float(np.mean(results[DIST_AMP_REF][freq]['f']['gains'])) for freq in freqs]
    diffs = np.diff(gains_f_all)
    mono_dn = bool(np.all(diffs < 0))
    mono_up = bool(np.all(diffs > 0))
    n_above_1 = sum(1 for g in gains_f_all if g > 1.0)
    print(f"    Guadagno fisso: min={min(gains_f_all):.3f}  max={max(gains_f_all):.3f}"
          f"  freq con gain>1: {n_above_1}/{len(freqs)}")
    if mono_dn:
        print("    PROFILO: monotono DECRESCENTE — no waterfall (P-controller classico).")
        print("    La legge fissa attenua meglio ad alta f senza compensazione a bassa f.")
    elif mono_up:
        print("    PROFILO: monotono CRESCENTE.")
    else:
        n_ups = int(np.sum(diffs > 0))
        print(f"    PROFILO: NON monotono — {n_ups}/{len(diffs)} salti positivi.")
    if n_above_1 == 0:
        print("    → Legge fissa ATTENUA a tutte le 16 frequenze (gain < 1.0). Nessun waterbed osservato.")
    else:
        print(f"    → Legge fissa AMPLIFICA in {n_above_1}/{len(freqs)} frequenze (gain > 1.0).")

    print(f"\n{'═' * 100}\n")


# ---------------------------------------------------------------------------
# Grafico — 2 righe × 3 colonne (gain/effort × ampiezza)
# ---------------------------------------------------------------------------
def plot_results(results: dict, dist_amps: list, freqs: list,
                 out_path: str = "benchmark_frequency_robustness_output.png") -> None:
    freq_arr = np.array(freqs)
    CLR_F  = "#e74c3c"
    CLR_F2 = "#f39c12"
    CLR_A  = "#27ae60"
    CLR_Q  = "#3498db"

    n_amps = len(dist_amps)
    fig, axes = plt.subplots(2, n_amps, figsize=(7 * n_amps, 12))
    fig.suptitle(
        f"Robustezza risposta in frequenza — disturbance rejection, setpoint fisso\n"
        f"GAIN={GAIN}  T_TOTAL={T_TOTAL}  T_TRANS={T_TRANS}  [{N_SEEDS} seed]  "
        f"Linee verticali tratteggiate = frequenze nuove {FREQS_EXT}",
        fontsize=11,
    )

    for col, dist_amp in enumerate(dist_amps):
        def arr_m(ag, key):
            return np.array([np.mean(results[dist_amp][f][ag][key]) for f in freqs])
        def arr_s(ag, key):
            return np.array([np.std(results[dist_amp][f][ag][key]) for f in freqs])

        mean_f  = arr_m('f',  'gains');  std_f  = arr_s('f',  'gains')
        mean_f2 = arr_m('f2', 'gains');  std_f2 = arr_s('f2', 'gains')
        mean_a  = arr_m('a',  'gains');  std_a  = arr_s('a',  'gains')
        mean_q  = arr_m('q',  'gains');  std_q  = arr_s('q',  'gains')
        eff_f   = arr_m('f',  'efforts'); eff_sf = arr_s('f',  'efforts')
        eff_f2  = arr_m('f2', 'efforts'); eff_sf2 = arr_s('f2', 'efforts')
        eff_a   = arr_m('a',  'efforts'); eff_sa = arr_s('a',  'efforts')
        eff_q   = arr_m('q',  'efforts'); eff_sq = arr_s('q',  'efforts')

        # Riga 0: Guadagno
        ax = axes[0, col]
        for mean, std, clr, lbl, mrk in [
            (mean_f,  std_f,  CLR_F,  "Fisso",    "o"),
            (mean_f2, std_f2, CLR_F2, "Fisso×2",  "D"),
            (mean_a,  std_a,  CLR_A,  "Adattivo", "s"),
            (mean_q,  std_q,  CLR_Q,  "Q",        "^"),
        ]:
            ax.semilogx(freq_arr, mean, f"{mrk}-", color=clr, linewidth=2.0,
                        markersize=5, label=lbl, zorder=3)
            ax.fill_between(freq_arr, mean - std, mean + std, alpha=0.12, color=clr)
        ax.axhline(1.0, linestyle="--", color="black", linewidth=1.0, label="gain=1")
        for nf in FREQS_EXT:
            ax.axvline(nf, linestyle=":", color="purple", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("f (cicli/step) [scala log]", fontsize=9)
        ax.set_ylabel(f"Guadagno  RMS(err)/{dist_amp}", fontsize=9)
        ax.set_title(f"Guadagno  DIST_AMP={dist_amp}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(freq_arr[0] * 0.8, freq_arr[-1] * 1.2)

        # Riga 1: Sforzo
        ax = axes[1, col]
        for eff, std, clr, lbl, mrk in [
            (eff_f,  eff_sf,  CLR_F,  "Fisso",    "o"),
            (eff_f2, eff_sf2, CLR_F2, "Fisso×2",  "D"),
            (eff_a,  eff_sa,  CLR_A,  "Adattivo", "s"),
            (eff_q,  eff_sq,  CLR_Q,  "Q",        "^"),
        ]:
            ax.semilogx(freq_arr, eff, f"{mrk}-", color=clr, linewidth=2.0,
                        markersize=5, label=lbl, zorder=3)
            ax.fill_between(freq_arr, eff - std, eff + std, alpha=0.12, color=clr)
        for nf in FREQS_EXT:
            ax.axvline(nf, linestyle=":", color="purple", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("f (cicli/step) [scala log]", fontsize=9)
        ax.set_ylabel("Sforzo  RMS(azione)", fontsize=9)
        ax.set_title(f"Sforzo  DIST_AMP={dist_amp}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(freq_arr[0] * 0.8, freq_arr[-1] * 1.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafico salvato → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    results = run_sweep(DIST_AMPS, FREQS)
    print_results(results, DIST_AMPS, FREQS)
    plot_results(results, DIST_AMPS, FREQS)


if __name__ == "__main__":
    main()
