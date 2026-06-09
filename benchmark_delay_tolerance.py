"""
benchmark_delay_tolerance.py — caratterizzazione della tolleranza al ritardo
=============================================================================
Caratterizza il comportamento dei tre agenti sotto ritardo costante D sul
canale azione→effetto. Questo NON è un test di generalizzazione di ĝ: è la
caratterizzazione del confine del meccanismo su un asse nuovo (tempo, non forza).

CONTESTO — benchmark chiusi, non riaprire:
  v4: legge fissa fragile su gain×4 → 10/30 mai_recuperato.
  Adattivo: ĝ ripara gain×4 → 0/30, precisione 0.04 vs Q 0.71.
  Setpoint shift: ĝ irrilevante (fisso già recuperava 0/30).

PUNTO DI INNESTO DEL RITARDO:
  In run_episode_with_delay() un deque di dimensione D sostituisce il passaggio
  diretto dell'azione all'ambiente. L'azione al passo t produce effetto a t+D.
  Il buffer è inizializzato con D zeri (azione neutra). Il ritardo è costante
  per TUTTA la run: sia durante il pre-training Q (T_PRE=1000 step) sia dopo il
  gradino di setpoint. Questo è equo: ogni agente convive col ritardo dall'inizio.
  Il gradino di setpoint (0→2.0 a t=T_PRE) rimane invariato come stimolo.

=============================================================================
PREREG — Pre-registration (fissata prima di eseguire; NON modificare dopo)
=============================================================================

H_fix: esiste un ritardo critico D*_fix oltre il quale la legge fissa EHD oscilla
  e non recupera più (mai_recuperato sale oltre 0/30). Un P-controller con ritardo
  D >~2 su un sistema discreto di ordine 1 entra in oscillazione — ci aspettiamo
  D*_fix ∈ [2, 6].

H_ĝ (CENTRALE): ĝ NON alza il ritardo critico, e plausibilmente lo ABBASSA.
  Meccanismo: il ritardo D introduce una correlazione incrociata tra action_{t-D}
  e error_{t+1}, ma il modello di ĝ la attribuisce ad action_t. Per D>0,
  l'aggiornamento ĝ += LR * pred_err * action_t corre su un segnale mal-correlato
  → ĝ deriva verso GAIN_HAT_MIN=0.5 → azione amplificata del doppio → instabilità
  aggravata. FALSIFICATA se D*_ĝ > D*_fix in modo significativo.

H_Q (riferimento): Q-learning, avendo 1000 step di training CON il ritardo presente,
  può adattare la sua policy alla dinamica ritardata. D*_Q > D*_fix e D*_Q > D*_ĝ.

SANITY CHECK obbligatorio: a D=0, tutti e tre gli agenti devono riprodurre
  benchmark_setpoint_shift (mai_recuperato=0/30, recovery mediana ~3 step).
  Se non lo fa → harness alterato → FERMATI.

SCANSIONE: D ∈ [0, 1, 2, 3, 4, 6, 8, 12]. Se a D=12 la legge fissa non si è
  ancora rotta, il codice lo segnala e suggerisce di estendere.

INDICATORE DI OSCILLAZIONE: numero di cambi di segno dell'errore nel T_POST
  post-shift. Stabile: 0-2. Oscillante: >5.

ONESTÀ: esiti attesi e informativi:
  "ĝ si rompe prima della legge fissa" → adattamento di parametro controproducente
  contro guasti di tempo.
  "Nessun agente regge oltre D*" → serve un cambio di struttura.
  Riporta quello che esce.
=============================================================================

Run:
    cd symbiont-architecture
    python benchmark_delay_tolerance.py
"""

from __future__ import annotations

import math
import os
import sys
from collections import deque
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
# Costanti — identiche al benchmark setpoint shift
# ---------------------------------------------------------------------------
T_PRE             = 1000
T_POST            = 200
T_STEPS           = T_PRE + T_POST
SHIFT_T           = T_PRE

GAIN              = 1.0
SETPOINT_PRE      = 0.0
SETPOINT_POST     = 2.0

EPSILON           = 0.1
NOISE_STD         = 0.05
N_SEEDS           = 30

BASELINE_WINDOW   = 50
PLATEAU_WINDOW    = 200
PLATEAU_MIN_RATIO = 0.85

LR_GAIN           = 0.05
GAIN_HAT_MIN      = 0.5
GAIN_HAT_MAX      = 10.0
GAIN_HAT_INIT     = GAIN

# Scansione ritardi — estesa se a D=12 la legge fissa non si è ancora rotta
D_VALUES          = [0, 1, 2, 3, 4, 6, 8, 12, 20, 30, 50]

# D* threshold: mai_recuperato >= questa soglia → agente "si rompe" per quel D
D_STAR_THRESHOLD  = 2   # ≥ 2/30 → instabilità confermata

# Soglia confound: se pre_baseline > CONFOUND_FACTOR × baseline a D=0,
# la metrica baseline-propria è inflazionata dalle oscillazioni di ritardo
# → recovery threshold troppo largo → H_fix non determinabile con questa metrica
CONFOUND_FACTOR   = 3.0

# Soglia oscillazione: sign_changes > questa → agente oscilla
OSC_THRESHOLD     = 5

# ---------------------------------------------------------------------------
# Ambiente 1D — identico a benchmark_setpoint_shift (setpoint shift, gain fisso)
# ---------------------------------------------------------------------------
class Env1D:
    def __init__(self, seed: int) -> None:
        rng         = np.random.default_rng(seed)
        self.x      = float(rng.uniform(-1.0, 1.0))
        self._noise = rng.standard_normal(T_STEPS) * NOISE_STD
        self._t     = 0

    @property
    def setpoint(self) -> float:
        return SETPOINT_PRE if self._t < SHIFT_T else SETPOINT_POST

    @property
    def error(self) -> float:
        return self.x - self.setpoint

    def step(self, action: float) -> Tuple[float, float]:
        action = float(np.clip(action, -1.0, 1.0))
        self.x = float(np.clip(self.x + GAIN * action + self._noise[self._t], -6.0, 6.0))
        self._t += 1
        return self.x, self.error


# ---------------------------------------------------------------------------
# Agenti — identici agli esperimenti precedenti
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
# Metriche
# ---------------------------------------------------------------------------
Metrics = Dict[str, float]


def compute_metrics(traj: np.ndarray) -> Metrics:
    """traj: (T_STEPS, 2), col 1 = |error|."""
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


def compute_sign_changes(signed_errors: np.ndarray) -> int:
    """Count zero-crossings in signed post-shift error (oscillation indicator)."""
    post  = signed_errors[SHIFT_T:]
    signs = np.sign(post)
    # Only count transitions between non-negligible values
    mask  = np.abs(post) > 0.05
    out   = 0
    prev  = 0
    for i in range(len(signs)):
        if mask[i]:
            s = int(signs[i])
            if prev != 0 and s != prev:
                out += 1
            prev = s
    return out


# ---------------------------------------------------------------------------
# Episodio con ritardo D — punto di innesto della perturbazione temporale
# ---------------------------------------------------------------------------
def run_episode_with_delay(
    env_seed: int,
    delay_D: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stessa struttura di benchmark_setpoint_shift, con ritardo D sul canale azione→effetto.

    Il buffer è inizializzato con D zeri (azione neutra = 0.0).
    L'agente osserva sempre l'errore CORRENTE — non sa del ritardo.
    L'effettiva azione applicata all'env è quella di D step fa.

    Ritorna: (traj_f, traj_a, traj_q, gain_hat_traj, serr_f, serr_a, serr_q)
      traj_*:       (T_STEPS, 2) — col 0=x, col 1=|error|
      gain_hat_traj: (T_STEPS,)
      serr_*:       (T_STEPS,) — errore con segno (per sign_changes)
    """
    env_f   = Env1D(seed=env_seed)
    env_a   = Env1D(seed=env_seed)
    env_q   = Env1D(seed=env_seed)
    agent_f = SymbiontAgent(seed=0)
    agent_a = AdaptiveSymbiontAgent(seed=0)
    agent_q = QLearningAgent(seed=env_seed)

    # Deque buffer (dimensione D, inizializzato con azione neutra)
    buf_f = deque([0.0] * delay_D) if delay_D > 0 else None
    buf_a = deque([0.0] * delay_D) if delay_D > 0 else None
    buf_q = deque([0.0] * delay_D) if delay_D > 0 else None

    traj_f        = np.zeros((T_STEPS, 2))
    traj_a        = np.zeros((T_STEPS, 2))
    traj_q        = np.zeros((T_STEPS, 2))
    gain_hat_traj = np.zeros(T_STEPS)
    serr_f        = np.zeros(T_STEPS)
    serr_a        = np.zeros(T_STEPS)
    serr_q        = np.zeros(T_STEPS)

    for t in range(T_STEPS):
        # Fixed EHD
        err_f  = env_f.error
        act_f  = agent_f.act(err_f)
        if delay_D == 0:
            eff_f = act_f
        else:
            eff_f = buf_f.popleft()
            buf_f.append(act_f)
        x_f, e_f     = env_f.step(eff_f)
        traj_f[t]    = (x_f, abs(e_f))
        serr_f[t]    = e_f

        # Adaptive EHD
        err_a  = env_a.error
        act_a  = agent_a.act(err_a)
        if delay_D == 0:
            eff_a = act_a
        else:
            eff_a = buf_a.popleft()
            buf_a.append(act_a)
        x_a, e_a          = env_a.step(eff_a)
        traj_a[t]         = (x_a, abs(e_a))
        serr_a[t]         = e_a
        gain_hat_traj[t]  = agent_a.gain_hat

        # Q-learning
        err_q  = env_q.error
        act_q  = agent_q.act(err_q)
        if delay_D == 0:
            eff_q = act_q
        else:
            eff_q = buf_q.popleft()
            buf_q.append(act_q)
        x_q, e_q  = env_q.step(eff_q)
        agent_q.update(e_q, -abs(e_q))
        traj_q[t] = (x_q, abs(e_q))
        serr_q[t] = e_q

    return traj_f, traj_a, traj_q, gain_hat_traj, serr_f, serr_a, serr_q


# ---------------------------------------------------------------------------
# Sweep su tutti i valori di D
# ---------------------------------------------------------------------------
def run_sweep():
    """
    Ritorna results[D] = {
      'f': {'metrics': List[Metrics], 'sign_changes': List[int], 'gain_hat': np.ndarray},
      'a': {'metrics': List[Metrics], 'sign_changes': List[int], 'gain_hat': np.ndarray},
      'q': {'metrics': List[Metrics], 'sign_changes': List[int]},
    }
    """
    print(f"Sweep ritardo D ∈ {D_VALUES}")
    print(f"  N_SEEDS={N_SEEDS}  T={T_STEPS}  shift setpoint {SETPOINT_PRE}→{SETPOINT_POST} a t={SHIFT_T}")
    print(f"  Gain costante={GAIN}  recovery=baseline+{EPSILON}")
    print()

    results = {}

    for D in D_VALUES:
        all_mf, all_ma, all_mq     = [], [], []
        sc_f,   sc_a,   sc_q       = [], [], []
        gh_trajs = np.zeros((N_SEEDS, T_STEPS))

        for seed in range(N_SEEDS):
            tf, ta, tq, gh, sf, sa, sq = run_episode_with_delay(seed, D)
            all_mf.append(compute_metrics(tf))
            all_ma.append(compute_metrics(ta))
            all_mq.append(compute_metrics(tq))
            sc_f.append(compute_sign_changes(sf))
            sc_a.append(compute_sign_changes(sa))
            sc_q.append(compute_sign_changes(sq))
            gh_trajs[seed] = gh

        results[D] = {
            'f': {'metrics': all_mf, 'sign_changes': sc_f},
            'a': {'metrics': all_ma, 'sign_changes': sc_a, 'gain_hat_trajs': gh_trajs.copy()},
            'q': {'metrics': all_mq, 'sign_changes': sc_q},
        }
        nr_f = int(np.sum([m['never_recovered'] for m in all_mf]))
        nr_a = int(np.sum([m['never_recovered'] for m in all_ma]))
        nr_q = int(np.sum([m['never_recovered'] for m in all_mq]))
        print(f"  D={D:2d}  mai_recuperato: Fisso={nr_f:2d}/30  Adattivo={nr_a:2d}/30  Q={nr_q:2d}/30"
              f"  |  sign_changes: F={np.mean(sc_f):.1f}  A={np.mean(sc_a):.1f}  Q={np.mean(sc_q):.1f}")

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def welch_t_one_tailed(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    n1, n2 = len(a), len(b)
    m1, m2 = np.mean(a), np.mean(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    se     = math.sqrt(s1**2 / n1 + s2**2 / n2)
    if se < 1e-12:
        return 0.0, (0.0 if m1 < m2 else 1.0)
    t_stat = (m1 - m2) / se
    p_val  = 0.5 * (1.0 + math.erf(t_stat / math.sqrt(2.0)))
    return t_stat, p_val


def find_d_star(results: dict, agent: str) -> int | None:
    """Primo D dove mai_recuperato >= D_STAR_THRESHOLD."""
    for D in D_VALUES:
        nr = int(np.sum([m['never_recovered'] for m in results[D][agent]['metrics']]))
        if nr >= D_STAR_THRESHOLD:
            return D
    return None  # non trovato nell'intervallo testato


def print_results(results: dict) -> None:
    SEP    = "─" * 90
    SEP2   = "═" * 90

    # ------------------------------------------------------------------
    # SANITY CHECK a D=0
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("SANITY CHECK — D=0 deve riprodurre benchmark_setpoint_shift")
    print(SEP)
    r0 = results[0]
    nr_f0 = int(np.sum([m['never_recovered'] for m in r0['f']['metrics']]))
    nr_a0 = int(np.sum([m['never_recovered'] for m in r0['a']['metrics']]))
    nr_q0 = int(np.sum([m['never_recovered'] for m in r0['q']['metrics']]))
    med_f0 = float(np.median([m['recovery_time'] for m in r0['f']['metrics']]))
    med_a0 = float(np.median([m['recovery_time'] for m in r0['a']['metrics']]))
    med_q0 = float(np.median([m['recovery_time'] for m in r0['q']['metrics']]))
    shk_f0 = float(np.mean([m['initial_shock'] for m in r0['f']['metrics']]))
    shk_a0 = float(np.mean([m['initial_shock'] for m in r0['a']['metrics']]))
    shk_q0 = float(np.mean([m['initial_shock'] for m in r0['q']['metrics']]))

    ok_f = (nr_f0 == 0 and 2.0 <= med_f0 <= 5.0)
    ok_a = (nr_a0 == 0 and 2.0 <= med_a0 <= 5.0)
    ok_q = (nr_q0 == 0 and 1.0 <= med_q0 <= 6.0)

    print(f"  Fisso:    mai_rec={nr_f0}/30  mediana={med_f0:.1f}  shock={shk_f0:.2f}  {'✓' if ok_f else '✗ ATTENZIONE'}")
    print(f"  Adattivo: mai_rec={nr_a0}/30  mediana={med_a0:.1f}  shock={shk_a0:.2f}  {'✓' if ok_a else '✗ ATTENZIONE'}")
    print(f"  Q:        mai_rec={nr_q0}/30  mediana={med_q0:.1f}  shock={shk_q0:.2f}  {'✓' if ok_q else '✗ ATTENZIONE'}")

    if not (ok_f and ok_a and ok_q):
        print(f"\n  {'!'*86}")
        print("  SANITY CHECK FALLITO — l'harness non riproduce benchmark_setpoint_shift a D=0.")
        print("  Non emettere verdetti. Verifica la struttura.")
        print(f"  {'!'*86}\n")
        return

    print("  → Sanity check PASSA. Procedo.")

    # Pre-baseline a D=0 per calcolo confound ratio
    pb0_f = float(np.mean([m['pre_baseline'] for m in results[0]['f']['metrics']]))
    pb0_a = float(np.mean([m['pre_baseline'] for m in results[0]['a']['metrics']]))
    pb0_q = float(np.mean([m['pre_baseline'] for m in results[0]['q']['metrics']]))

    # ------------------------------------------------------------------
    # TABELLA PRINCIPALE — mai_recuperato, mediana, sign_changes, pre_baseline vs D
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("TABELLA — affidabilità vs ritardo D (30 seed)")
    print(f"  Baseline D=0: F={pb0_f:.3f}  A={pb0_a:.3f}  Q={pb0_q:.3f}")
    print(f"  [C] indica confound: pre_baseline > {CONFOUND_FACTOR:.0f}× baseline D=0 → rec_threshold inflazionato")
    print(SEP)
    H = (f"{'D':>3}  {'F mai_r':>7} {'F baseln':>8} {'F med':>6}  |  "
         f"{'A mai_r':>7} {'A baseln':>8} {'A med':>6}  |  "
         f"{'Q mai_r':>7} {'Q baseln':>8} {'Q med':>6}")
    print(H)
    print("─" * len(H))

    confound_d = {}  # D → (confound_f, confound_a, confound_q)
    for D in D_VALUES:
        r    = results[D]
        nr_f = int(np.sum([m['never_recovered'] for m in r['f']['metrics']]))
        nr_a = int(np.sum([m['never_recovered'] for m in r['a']['metrics']]))
        nr_q = int(np.sum([m['never_recovered'] for m in r['q']['metrics']]))
        med_f = float(np.median([m['recovery_time'] for m in r['f']['metrics']]))
        med_a = float(np.median([m['recovery_time'] for m in r['a']['metrics']]))
        med_q = float(np.median([m['recovery_time'] for m in r['q']['metrics']]))
        pb_f  = float(np.mean([m['pre_baseline'] for m in r['f']['metrics']]))
        pb_a  = float(np.mean([m['pre_baseline'] for m in r['a']['metrics']]))
        pb_q  = float(np.mean([m['pre_baseline'] for m in r['q']['metrics']]))

        cf_f = pb_f > CONFOUND_FACTOR * pb0_f
        cf_a = pb_a > CONFOUND_FACTOR * pb0_a
        cf_q = pb_q > CONFOUND_FACTOR * pb0_q
        confound_d[D] = (cf_f, cf_a, cf_q)
        any_confound = cf_f or cf_a or cf_q

        cf_tag = " [C]" if any_confound else ""
        tag_f  = " ←D*?" if (D_STAR_THRESHOLD <= nr_f < N_SEEDS and D_VALUES.index(D) > 0 and
                               int(np.sum([m['never_recovered'] for m in results[D_VALUES[D_VALUES.index(D)-1]]['f']['metrics']])) < D_STAR_THRESHOLD) else ""
        tag_a  = " ←D*?" if (D_STAR_THRESHOLD <= nr_a < N_SEEDS and D_VALUES.index(D) > 0 and
                               int(np.sum([m['never_recovered'] for m in results[D_VALUES[D_VALUES.index(D)-1]]['a']['metrics']])) < D_STAR_THRESHOLD) else ""
        tag_q  = " ←D*?" if (D_STAR_THRESHOLD <= nr_q < N_SEEDS and D_VALUES.index(D) > 0 and
                               int(np.sum([m['never_recovered'] for m in results[D_VALUES[D_VALUES.index(D)-1]]['q']['metrics']])) < D_STAR_THRESHOLD) else ""

        print(f"{D:>3}{cf_tag:<4}  {nr_f:>3}/{N_SEEDS}{tag_f:<5} {pb_f:>7.3f}  {med_f:>6.1f}  |  "
              f"{nr_a:>3}/{N_SEEDS}{tag_a:<5} {pb_a:>7.3f}  {med_a:>6.1f}  |  "
              f"{nr_q:>3}/{N_SEEDS}{tag_q:<5} {pb_q:>7.3f}  {med_q:>6.1f}")

    # Check se l'ultimo D non ha rotto niente
    last_D = D_VALUES[-1]
    nr_last_f = int(np.sum([m['never_recovered'] for m in results[last_D]['f']['metrics']]))
    if nr_last_f < D_STAR_THRESHOLD:
        print(f"\n  [!] A D={last_D} la legge fissa ha ancora mai_recuperato={nr_last_f}/{N_SEEDS} < {D_STAR_THRESHOLD}.")
        print("      D*_fix non trovato nell'intervallo testato. Estendere D_VALUES verso l'alto.")

    # ------------------------------------------------------------------
    # BLOCCO DETTAGLIATO PER OGNI D CON DISTRIBUZIONE ESTESA
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("DETTAGLIO — distribuzione recovery_time per valori chiave di D")
    print(SEP)
    for D in D_VALUES:
        r    = results[D]
        rt_f = np.array([m['recovery_time'] for m in r['f']['metrics']])
        rt_a = np.array([m['recovery_time'] for m in r['a']['metrics']])
        rt_q = np.array([m['recovery_time'] for m in r['q']['metrics']])
        nr_f = int(np.sum(rt_f == T_POST))
        nr_a = int(np.sum(rt_a == T_POST))
        nr_q = int(np.sum(rt_q == T_POST))
        shk_f = float(np.mean([m['initial_shock'] for m in r['f']['metrics']]))
        shk_a = float(np.mean([m['initial_shock'] for m in r['a']['metrics']]))
        shk_q = float(np.mean([m['initial_shock'] for m in r['q']['metrics']]))

        any_issue = (nr_f >= D_STAR_THRESHOLD or nr_a >= D_STAR_THRESHOLD or nr_q >= D_STAR_THRESHOLD
                     or np.std(rt_f) > np.mean(rt_f) + 0.1
                     or np.std(rt_a) > np.mean(rt_a) + 0.1)
        if any_issue or D in [0, D_VALUES[-1]]:
            shock_diverg = abs(shk_f - shk_q) >= 1.0
            print(f"\n  D={D}")
            for name, rt, nr, shk in [("Fisso   ", rt_f, nr_f, shk_f),
                                       ("Adattivo", rt_a, nr_a, shk_a),
                                       ("Q       ", rt_q, nr_q, shk_q)]:
                print(f"    {name}: mai_rec={nr:2d}/30  med={np.median(rt):.1f}  "
                      f"mean={np.mean(rt):.1f}  std={np.std(rt):.1f}  "
                      f"max={np.max(rt):.0f}  shock={shk:.2f}")
            if shock_diverg:
                print(f"    [!] Shock Fisso={shk_f:.2f} vs Q={shk_q:.2f}: DIVERGONO — velocità non comparabile")
            if np.std(rt_f) > np.mean(rt_f) + 0.1:
                print(f"    [!] Fisso: std({np.std(rt_f):.1f}) > media({np.mean(rt_f):.1f}) — distribuzione degenere")
            if np.std(rt_a) > np.mean(rt_a) + 0.1:
                print(f"    [!] Adattivo: std({np.std(rt_a):.1f}) > media({np.mean(rt_a):.1f}) — distribuzione degenere")

    # ------------------------------------------------------------------
    # ĝ drift sotto ritardo — valore medio finale per ogni D
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("DERIVA DI ĝ — media di ĝ all'ultimo step (post-shift) per ogni D")
    print(SEP)
    print(f"  (atteso: ĝ ≈ 1.0 per D=0; sotto ritardo ĝ può derivare verso GAIN_HAT_MIN={GAIN_HAT_MIN})")
    for D in D_VALUES:
        if 'gain_hat_trajs' in results[D]['a']:
            gh_end = results[D]['a']['gain_hat_trajs'][:, -1]
            print(f"  D={D:2d}: ĝ_finale  media={np.mean(gh_end):.3f}  std={np.std(gh_end):.3f}  "
                  f"min={np.min(gh_end):.3f}  max={np.max(gh_end):.3f}")

    # ------------------------------------------------------------------
    # CONFOUND CHECK — pre_baseline cresce con D?
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("CONFOUND CHECK — crescita di pre_baseline con D")
    print(SEP)
    print("  Se pre_baseline > 3× baseline D=0, la metrica baseline-propria è inflazionata")
    print("  dalle oscillazioni indotte dal ritardo. Recovery sembra 'facile' ma è l'asticella")
    print("  che si è abbassata, non l'agente che ha migliorato. H_fix non determinabile.")
    first_confound_f = None
    first_confound_a = None
    first_confound_q = None
    for D in D_VALUES:
        cf_f, cf_a, cf_q = confound_d[D]
        pb_f = float(np.mean([m['pre_baseline'] for m in results[D]['f']['metrics']]))
        pb_a = float(np.mean([m['pre_baseline'] for m in results[D]['a']['metrics']]))
        pb_q = float(np.mean([m['pre_baseline'] for m in results[D]['q']['metrics']]))
        if cf_f and first_confound_f is None:
            first_confound_f = D
            print(f"  [C] Fisso confound inizia a D={D}: pre_baseline={pb_f:.3f} > {CONFOUND_FACTOR:.0f}× {pb0_f:.3f}")
        if cf_a and first_confound_a is None:
            first_confound_a = D
            print(f"  [C] Adattivo confound inizia a D={D}: pre_baseline={pb_a:.3f} > {CONFOUND_FACTOR:.0f}× {pb0_a:.3f}")
        if cf_q and first_confound_q is None:
            first_confound_q = D
            print(f"  [C] Q confound inizia a D={D}: pre_baseline={pb_q:.3f} > {CONFOUND_FACTOR:.0f}× {pb0_q:.3f}")
    if first_confound_f is None and first_confound_a is None and first_confound_q is None:
        print(f"  Nessun confound rilevato (pre_baseline < {CONFOUND_FACTOR:.0f}× baseline D=0 per tutti i D).")

    # ------------------------------------------------------------------
    # VERDETTO PREREG
    # ------------------------------------------------------------------
    print(f"\n{SEP2}")
    print("VERDETTO PREREG — CARATTERIZZAZIONE RITARDO CRITICO D*")
    print(SEP2)

    # D* solo su D < primo confound (dati non contaminati)
    valid_D = [D for D in D_VALUES if not any(confound_d.get(D, (False, False, False)))]
    print(f"\n  D validi (pre_baseline non confounded): {valid_D}")

    d_star_f = find_d_star(results, 'f')
    d_star_a = find_d_star(results, 'a')
    d_star_q = find_d_star(results, 'q')

    max_D = max(D_VALUES)

    def d_str(d: int | None) -> str:
        return f"D={d}" if d is not None else f">D={max_D} (non trovato)"

    print(f"\n  Ritardi critici D* (mai_recuperato ≥ {D_STAR_THRESHOLD}/{N_SEEDS}):")
    print(f"    Fisso:    {d_str(d_star_f)}")
    print(f"    Adattivo: {d_str(d_star_a)}")
    print(f"    Q:        {d_str(d_star_q)}")

    print(f"\n  H_fix: esiste D* dove la legge fissa si rompe?")
    if d_star_f is not None:
        print(f"    CONFERMATA — D*_fix = {d_star_f}")
    else:
        print(f"    NON CONFERMATA nell'intervallo testato (D ≤ {max_D}). Estendere.")

    print(f"\n  H_ĝ: ĝ non alza D* (e plausibilmente lo abbassa)?")
    if d_star_f is None and d_star_a is None:
        print("    NON DETERMINABILE — nessuno dei due si è rotto. Estendere D.")
    elif d_star_f is None and d_star_a is not None:
        print(f"    CONFERMATA — ĝ si rompe prima ({d_str(d_star_a)}) della legge fissa (>D={max_D})")
        print(f"    ĝ è CONTROPRODUCENTE sotto ritardo: abbassa il ritardo critico.")
    elif d_star_f is not None and d_star_a is None:
        print(f"    FALSIFICATA — ĝ non si rompe (>D={max_D}) mentre la legge fissa si rompe a D={d_star_f}")
        print(f"    ĝ ALZA il ritardo critico — non era atteso.")
    elif d_star_a <= d_star_f:
        if d_star_a < d_star_f:
            print(f"    CONFERMATA — ĝ si rompe prima (D*_ĝ={d_star_a}) della legge fissa (D*_fix={d_star_f})")
            print(f"    ĝ è CONTROPRODUCENTE sotto ritardo.")
        else:
            print(f"    CONFERMATA (inerte) — ĝ si rompe allo stesso D* della legge fissa (D={d_star_a})")
            print(f"    ĝ non aiuta né peggiora sotto ritardo.")
    else:
        print(f"    FALSIFICATA — D*_ĝ={d_star_a} > D*_fix={d_star_f}")

    print(f"\n  H_Q: Q tollera ritardi più alti?")
    if d_star_q is None:
        if d_star_f is not None:
            print(f"    CONFERMATA — Q non si rompe (>D={max_D}) mentre la legge fissa si rompe a D={d_star_f}")
        else:
            print(f"    NON DETERMINABILE — nessuno si rompe nell'intervallo testato.")
    elif d_star_f is not None and d_star_q > d_star_f:
        print(f"    CONFERMATA — D*_Q={d_star_q} > D*_fix={d_star_f}")
    elif d_star_f is not None and d_star_q <= d_star_f:
        print(f"    FALSIFICATA — D*_Q={d_star_q} ≤ D*_fix={d_star_f}")
    else:
        print(f"    NON DETERMINABILE")

    print(f"\n{'═'*90}\n")


# ---------------------------------------------------------------------------
# Grafico — sweep ritardo
# ---------------------------------------------------------------------------
def plot_results(results: dict, out_path: str = "benchmark_delay_output.png") -> None:
    D_arr   = np.array(D_VALUES, dtype=float)
    CLR_F   = "#e74c3c"
    CLR_A   = "#27ae60"
    CLR_Q   = "#3498db"

    nr_f_arr  = np.array([int(np.sum([m['never_recovered'] for m in results[D]['f']['metrics']])) for D in D_VALUES], dtype=float)
    nr_a_arr  = np.array([int(np.sum([m['never_recovered'] for m in results[D]['a']['metrics']])) for D in D_VALUES], dtype=float)
    nr_q_arr  = np.array([int(np.sum([m['never_recovered'] for m in results[D]['q']['metrics']])) for D in D_VALUES], dtype=float)

    med_f_arr = np.array([float(np.median([m['recovery_time'] for m in results[D]['f']['metrics']])) for D in D_VALUES])
    med_a_arr = np.array([float(np.median([m['recovery_time'] for m in results[D]['a']['metrics']])) for D in D_VALUES])
    med_q_arr = np.array([float(np.median([m['recovery_time'] for m in results[D]['q']['metrics']])) for D in D_VALUES])

    osc_f_arr = np.array([float(np.mean(results[D]['f']['sign_changes'])) for D in D_VALUES])
    osc_a_arr = np.array([float(np.mean(results[D]['a']['sign_changes'])) for D in D_VALUES])
    osc_q_arr = np.array([float(np.mean(results[D]['q']['sign_changes'])) for D in D_VALUES])

    pb_f_arr  = np.array([float(np.mean([m['pre_baseline'] for m in results[D]['f']['metrics']])) for D in D_VALUES])
    pb_a_arr  = np.array([float(np.mean([m['pre_baseline'] for m in results[D]['a']['metrics']])) for D in D_VALUES])
    pb_q_arr  = np.array([float(np.mean([m['pre_baseline'] for m in results[D]['q']['metrics']])) for D in D_VALUES])

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"Tolleranza al Ritardo — setpoint shift {SETPOINT_PRE}→{SETPOINT_POST}, "
        f"gain costante={GAIN}  [{N_SEEDS} seed]",
        fontsize=12,
    )

    # (0) mai_recuperato vs D
    ax = axes[0]
    ax.plot(D_arr, nr_f_arr, "o-", color=CLR_F, linewidth=1.5, markersize=6, label="Fisso (EHD)")
    ax.plot(D_arr, nr_a_arr, "s-", color=CLR_A, linewidth=1.5, markersize=6, label="Adattivo (EHD+ĝ)")
    ax.plot(D_arr, nr_q_arr, "^-", color=CLR_Q, linewidth=1.5, markersize=6, label="Q-learning")
    ax.axhline(D_STAR_THRESHOLD, linestyle="--", color="gray", linewidth=0.8,
               label=f"soglia D* ({D_STAR_THRESHOLD}/{N_SEEDS})")
    ax.set_xlabel("Ritardo D (step)")
    ax.set_ylabel("mai_recuperato (/ 30 seed)")
    ax.set_title("Affidabilità vs ritardo")
    ax.set_xticks(D_arr)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.5, N_SEEDS + 0.5)

    # (1) mediana recovery time vs D
    ax = axes[1]
    ax.plot(D_arr, med_f_arr, "o-", color=CLR_F, linewidth=1.5, markersize=6, label="Fisso")
    ax.plot(D_arr, med_a_arr, "s-", color=CLR_A, linewidth=1.5, markersize=6, label="Adattivo")
    ax.plot(D_arr, med_q_arr, "^-", color=CLR_Q, linewidth=1.5, markersize=6, label="Q-learning")
    ax.set_xlabel("Ritardo D (step)")
    ax.set_ylabel("Mediana recovery time (step)")
    ax.set_title("Velocità di recupero (mediana) vs ritardo")
    ax.set_xticks(D_arr)
    ax.legend(fontsize=8)

    # (2) oscillazione vs D
    ax = axes[2]
    ax.plot(D_arr, osc_f_arr, "o-", color=CLR_F, linewidth=1.5, markersize=6, label="Fisso")
    ax.plot(D_arr, osc_a_arr, "s-", color=CLR_A, linewidth=1.5, markersize=6, label="Adattivo")
    ax.plot(D_arr, osc_q_arr, "^-", color=CLR_Q, linewidth=1.5, markersize=6, label="Q-learning")
    ax.axhline(OSC_THRESHOLD, linestyle="--", color="gray", linewidth=0.8,
               label=f"soglia oscillazione ({OSC_THRESHOLD})")
    ax.set_xlabel("Ritardo D (step)")
    ax.set_ylabel("Sign changes medi (post-shift)")
    ax.set_title("Oscillazione vs ritardo")
    ax.set_xticks(D_arr)
    ax.legend(fontsize=8)

    # (3) pre_baseline vs D — confound check
    ax = axes[3]
    ax.plot(D_arr, pb_f_arr, "o-", color=CLR_F, linewidth=1.5, markersize=6, label="Fisso")
    ax.plot(D_arr, pb_a_arr, "s-", color=CLR_A, linewidth=1.5, markersize=6, label="Adattivo")
    ax.plot(D_arr, pb_q_arr, "^-", color=CLR_Q, linewidth=1.5, markersize=6, label="Q-learning")
    ax.axhline(pb_f_arr[0] * CONFOUND_FACTOR, linestyle="--", color="#e74c3c", linewidth=0.8,
               label=f"soglia confound F ({CONFOUND_FACTOR:.0f}× D=0)")
    ax.axhline(pb_q_arr[0] * CONFOUND_FACTOR, linestyle="--", color="#3498db", linewidth=0.8,
               label=f"soglia confound Q ({CONFOUND_FACTOR:.0f}× D=0)")
    ax.set_xlabel("Ritardo D (step)")
    ax.set_ylabel("pre_baseline medio (|error| pre-shift)")
    ax.set_title("Confound check: pre_baseline vs D\n(sale → rec_threshold inflazionato)")
    ax.set_xticks(D_arr)
    ax.legend(fontsize=7)

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
