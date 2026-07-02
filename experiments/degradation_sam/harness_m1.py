"""
M1 — Replica R1-R3 su classi SAM reali. Harness esterno: ZERO modifiche ai moduli sam-*.
Prereg: experiments/degradation_sam/prereg_M1.md (congelata al Gate 1b, commit 23eb698).
Riferimento strutturale: docs/experiments/2026-07-02_M0/symbiont_degradation.py (b8bdc18).

Agenti:
  R   regolazione nativa (SymbiontAgent IMPORTATO — integrazione 1b-a)
  A1  R + feedforward dal segnale, azione scalata da g_hat (parametri congelati dopo tuning a sigma=0)
  A2  A1 + fiducia w-regressione online (dieta controllata — soggetto a bias da loop chiuso)
  A3  A1 + fiducia w-varianza calibrata (V_cal da fase di calibrazione a segnale pulito)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)

from benchmark_adaptive_homeostasis import (            # noqa: E402
    SymbiontAgent, AdaptiveSymbiontAgent,
    LR_GAIN, GAIN_HAT_MIN, GAIN_HAT_MAX, GAIN_HAT_INIT,
)

# ---------------- Parametri (da prereg — NON modificare) ----------------
RHO_X, G_PLANT, C_ACT = 0.9, 1.0, 0.10
THETA_HARM = 1.0
EPS_STD = 0.05
P_BURST, BURST_MEAN = 0.02, 2.0
T, BURN, LEAD = 3000, 200, 3
N_TEST, N_TUNE = 30, 8
SIGMAS = np.array([0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
ALPHA_EMA, W_WARMUP = 0.01, 200
X_CLIP = 6.0
X0 = 0.0          # stato iniziale: convenzione M0 (h0=0); Env1D canonico usa U(-1,1), qui
                  # x0=0 per determinismo pieno dell'appaiamento (riferimento strutturale App. B)
SEED_D_TEST, SEED_EPS_TEST, SEED_ETA_TEST = 5000, 5300, 7000
SEED_D_TUNE, SEED_EPS_TUNE, SEED_ETA_TUNE = 1000, 1300, 2000
TWO_DIET_SIGMAS = [0.0, 0.5, 1.0, 2.0]


# ---------------- g_hat: replica verbatim (vincolo Gate 1a-3a) ----------------
class GainEstimator:
    """Replica verbatim di benchmark_adaptive_homeostasis.py:215-219
    (incluso guard |action_prev| > 1e-6) con costanti dalle righe 106-109.
    Organo SEPARATO da w (vincolo Gate 1a-3b): stato disgiunto, nessuna variabile condivisa."""

    def __init__(self) -> None:
        self._gain_hat = float(GAIN_HAT_INIT)
        self._x_prev: float | None = None
        self._action_prev: float | None = None

    def observe(self, x_current: float) -> None:
        if self._x_prev is not None and self._action_prev is not None:
            if abs(self._action_prev) > 1e-6:
                x_predicted     = self._x_prev + self._gain_hat * self._action_prev
                pred_err        = x_current - x_predicted
                new_gain        = self._gain_hat + LR_GAIN * pred_err * self._action_prev
                self._gain_hat  = float(np.clip(new_gain, GAIN_HAT_MIN, GAIN_HAT_MAX))

    def record(self, x_current: float, action: float) -> None:
        self._x_prev = x_current
        self._action_prev = action

    @property
    def value(self) -> float:
        return self._gain_hat


def test_gain_estimator_equivalence(n_steps: int = 500, seed: int = 0) -> tuple[bool, int]:
    """Unit test di equivalenza (prereg, vincolo 3a): stesso vettore di input ->
    stessa traiettoria g_hat passo-passo vs AdaptiveSymbiontAgent canonico.
    Uguaglianza ESATTA (==), nessuna tolleranza. Fallimento = STOP."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-2.0, 2.0, n_steps)
    canon = AdaptiveSymbiontAgent(seed=0)
    ge = GainEstimator()
    for t in range(n_steps):
        x = float(xs[t])
        ge.observe(x)
        a = canon.act(x)          # il canonico aggiorna g_hat internamente PRIMA dell'azione
        if ge.value != canon.gain_hat:
            return False, t
        ge.record(x, a)           # stesse coppie (x_t, a_t) del canonico
    return True, n_steps


# ---------------- Generazione ambiente (appaiamento: integrazione 1b-c) ----------------
def gen_block(n: int, seed_d: int, seed_eps: int) -> tuple[np.ndarray, np.ndarray]:
    """d, eps [T, n] generati UNA volta per blocco di seed, condivisi tra R/A1/A2/A3."""
    rng_d = np.random.default_rng(seed_d)
    hits = rng_d.random((T, n)) < P_BURST
    mags = rng_d.exponential(BURST_MEAN, (T, n))
    rng_e = np.random.default_rng(seed_eps)
    eps = rng_e.normal(0.0, EPS_STD, (T, n))
    return hits * mags, eps


def open_loop_state(d: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """X_open: stessa ricorsione dell'impianto, azione = 0. Esogena, precomputata."""
    X = np.zeros_like(d)
    for t in range(T - 1):
        X[t + 1] = np.clip(RHO_X * X[t] + d[t + 1] + eps[t + 1], -X_CLIP, X_CLIP)
    return X


def make_signal(X_open: np.ndarray, sigma: float, seed0: int) -> np.ndarray:
    """s[t] = X_open[t+LEAD] + eta. Iniezione rumore = QUI (assunzione-oracolo: sigma=0)."""
    rng = np.random.default_rng(seed0)
    s = np.empty_like(X_open)
    s[:T - LEAD] = X_open[LEAD:]
    s[T - LEAD:] = X_open[-1]
    return s + rng.normal(0.0, sigma, X_open.shape)


# ---------------- Agenti ----------------
class AgentR:
    """Regolazione nativa: SymbiontAgent importato, nessun parametro libero (SCARTO-3)."""

    def __init__(self, seed: int) -> None:
        self.native = SymbiontAgent(seed=seed)

    def act(self, x: float, sig: float) -> tuple[float, float]:
        return float(self.native.act(x)), 1.0

    def update_post(self, x_new: float, sp: float | None, sig: float) -> None:
        pass


class AgentAnticipatory:
    """A1 (trust='none'), A2 (trust='reg'), A3 (trust='var').
    a = clip(a_R + w * a_ff, -1, 1);  a_ff = -kf * max(0, s - thf) / max(g_hat, GAIN_HAT_MIN).
    Il floor max(g_hat, GAIN_HAT_MIN) e' costante di harness (integrazione 1b-b, verbatim
    riga 230 del canonico, anti-blowup). Clamp di g_hat: SOLO quelli canonici."""

    def __init__(self, seed: int, kf: float, thf: float,
                 trust: str = "none", V_cal: float | None = None) -> None:
        self.native = SymbiontAgent(seed=seed)
        self.ge = GainEstimator()
        self.kf, self.thf, self.trust, self.V_cal = kf, thf, trust, V_cal
        # stato EMA w-regressione (A2) — disgiunto da g_hat
        self.m_s = 0.0; self.m_h = 0.0; self.c_sh = 0.0; self.v_s = 1e-6
        # stato EMA w-varianza (A3)
        self.m_sig = 0.0; self.v_sig = 1e-6
        self.t = 0
        self.w_sum = 0.0; self.w_n = 0
        self.g_sum = 0.0; self.floor_hits = 0; self.g_n = 0

    def _trust_weight(self) -> float:
        if self.trust == "reg":
            return float(np.clip(self.c_sh / (self.v_s + 1e-8), 0, 1)) if self.t >= W_WARMUP else 0.5
        if self.trust == "var":
            return float(np.clip(self.V_cal / (self.v_sig + 1e-8), 0, 1)) if self.t >= W_WARMUP else 0.5
        return 1.0

    def act(self, x: float, sig: float) -> tuple[float, float]:
        self.ge.observe(x)
        g_eff = max(self.ge.value, GAIN_HAT_MIN)
        w = self._trust_weight()
        a_R = float(self.native.act(x))
        a_ff = -self.kf * max(0.0, sig - self.thf) / g_eff
        a = float(np.clip(a_R + w * a_ff, -1.0, 1.0))
        self.ge.record(x, a)
        self.g_sum += self.ge.value; self.g_n += 1
        if self.ge.value <= GAIN_HAT_MIN + 1e-12:
            self.floor_hits += 1
        if self.t >= W_WARMUP:
            self.w_sum += w; self.w_n += 1
        return a, w

    def update_post(self, x_new: float, sp: float | None, sig: float) -> None:
        # Ordine M0 (App. B): aggiornamento stimatori DOPO la dinamica.
        # A2 accoppia s[t-LEAD] con la x realizzata post-dinamica (fedele al rollout M0).
        if self.trust == "reg" and sp is not None:
            self.m_s += ALPHA_EMA * (sp - self.m_s)
            self.m_h += ALPHA_EMA * (x_new - self.m_h)
            self.c_sh += ALPHA_EMA * ((sp - self.m_s) * (x_new - self.m_h) - self.c_sh)
            self.v_s += ALPHA_EMA * ((sp - self.m_s) ** 2 - self.v_s)
        if self.trust == "var":
            self.m_sig += ALPHA_EMA * (sig - self.m_sig)
            self.v_sig += ALPHA_EMA * ((sig - self.m_sig) ** 2 - self.v_sig)
        self.t += 1


# ---------------- Rollout (scalare per seed: MemoryCluster e' stateful) ----------------
def rollout(seed: int, d: np.ndarray, eps: np.ndarray, s: np.ndarray,
            kind: str, kf: float = 0.0, thf: float = 1.0,
            V_cal: float | None = None, collect_traj: bool = False) -> dict:
    if kind == "R":
        agent: AgentR | AgentAnticipatory = AgentR(seed)
    else:
        trust = {"A1": "none", "A2": "reg", "A3": "var"}[kind]
        agent = AgentAnticipatory(seed, kf, thf, trust=trust, V_cal=V_cal)

    x = X0
    harm = 0.0; eff = 0.0
    traj = np.empty(T) if collect_traj else None
    for t in range(T):
        a, _w = agent.act(x, float(s[t]))
        if t >= BURN:
            harm += max(0.0, abs(x) - THETA_HARM)   # danno calcolato dall'AMBIENTE
            eff += abs(a)
        if traj is not None:
            traj[t] = x
        if t + 1 < T:
            x_new = float(np.clip(RHO_X * x + G_PLANT * a + d[t + 1] + eps[t + 1],
                                  -X_CLIP, X_CLIP))
        else:
            x_new = x
        sp = float(s[t - LEAD]) if t >= LEAD else None
        agent.update_post(x_new, sp, float(s[t]))
        x = x_new

    steps = T - BURN
    out = dict(cost=harm / steps + C_ACT * eff / steps,
               harm=harm / steps, effort=eff / steps)
    if isinstance(agent, AgentAnticipatory):
        out["w_mean"] = agent.w_sum / max(agent.w_n, 1)
        out["g_mean"] = agent.g_sum / max(agent.g_n, 1)
        out["floor_frac"] = agent.floor_hits / max(agent.g_n, 1)
    else:
        out["w_mean"] = float("nan"); out["g_mean"] = float("nan"); out["floor_frac"] = float("nan")
    out["traj"] = traj
    return out


# ---------------- Verifica causale a due diete (integrazione 1b-d) ----------------
def w_regression_offline(s_mat: np.ndarray, h_mat: np.ndarray) -> float:
    """Stessa regressione EMA di A2, verbatim da verify_closedloop_bias.py, su serie
    h arbitraria. Unica variabile tra le diete: la serie realizzata h_mat."""
    n = s_mat.shape[1]
    m_s = np.zeros(n); m_h = np.zeros(n)
    c_sh = np.zeros(n); v_s = np.full(n, 1e-6)
    for t in range(LEAD, T):
        sp = s_mat[t - LEAD]
        m_s += ALPHA_EMA * (sp - m_s)
        m_h += ALPHA_EMA * (h_mat[t] - m_h)
        c_sh += ALPHA_EMA * ((sp - m_s) * (h_mat[t] - m_h) - c_sh)
        v_s += ALPHA_EMA * ((sp - m_s) ** 2 - v_s)
    return float(np.clip(c_sh / (v_s + 1e-8), 0, 1).mean())


# ---------------- Main ----------------
def main() -> None:
    t_start = time.time()
    log: dict = {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "errors": [],
                 "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                              "matplotlib": matplotlib.__version__}}

    # 0. Unit test di equivalenza g_hat — fallimento = STOP (prereg)
    ok, n = test_gain_estimator_equivalence()
    log["equivalence_test"] = {"passed": ok, "steps": n}
    print(f"[equivalenza g_hat] {'PASS' if ok else 'FAIL'} ({n} step, uguaglianza esatta)")
    if not ok:
        log["errors"].append(f"equivalence test FAILED at step {n}")
        _dump_log(log); sys.exit("STOP: test di equivalenza g_hat fallito")

    # 1. Calibrazione su seed di TUNING (separati)
    d_tune, eps_tune = gen_block(N_TUNE, SEED_D_TUNE, SEED_EPS_TUNE)
    X_open_tune = open_loop_state(d_tune, eps_tune)
    V_CAL = float(X_open_tune[BURN:].var())
    std_open = float(np.sqrt(V_CAL))
    log["V_cal"] = V_CAL; log["std_open"] = std_open
    print(f"Calibrazione: Var(X_open) = {V_CAL:.4f}, std = {std_open:.4f}")
    if std_open > 4.0 / 3.0:
        log["errors"].append(f"std_open={std_open:.3f} > 4/3: requisito sigma_max >= 3*std violato")
        _dump_log(log); sys.exit("STOP: requisito sweep violato — segnalare al gate")

    # 2. Tuning A1 a sigma=0 (griglia M0), poi parametri CONGELATI
    s0_tune = make_signal(X_open_tune, 0.0, SEED_ETA_TUNE)
    best = (np.inf, None)
    for kf in [0.25, 0.5, 1.0, 2.0]:
        for thf in [0.4, 0.7, 1.0]:
            costs = [rollout(k, d_tune[:, k], eps_tune[:, k], s0_tune[:, k],
                             "A1", kf, thf)["cost"] for k in range(N_TUNE)]
            m = float(np.mean(costs))
            if m < best[0]:
                best = (m, (kf, thf))
    KF, THF = best[1]
    log["tuning"] = {"kf": KF, "thf": THF, "cost": best[0]}
    print(f"Tuning A1 (sigma=0, congelato): kf={KF}, thf={THF} | R: nessun parametro (SCARTO-3)")

    # 3. Blocco di TEST appaiato
    d_test, eps_test = gen_block(N_TEST, SEED_D_TEST, SEED_EPS_TEST)
    X_open_test = open_loop_state(d_test, eps_test)

    # R e' indipendente da sigma (non vede il segnale): 30 rollout, riusati nello sweep
    zeros = np.zeros(T)
    res_R = [rollout(k, d_test[:, k], eps_test[:, k], zeros, "R") for k in range(N_TEST)]
    cost_R = np.array([r["cost"] for r in res_R])
    mean_abs_a_R = float(np.mean([r["effort"] for r in res_R]))
    log["null_check_R"] = {"mean_abs_action": mean_abs_a_R}
    print(f"Nullita' R: |azione| media = {mean_abs_a_R:.4f} (degenere se <0.001 o >0.999)")
    if not (0.001 < mean_abs_a_R < 0.999):
        log["errors"].append("R degenere: esperimento NULLO")
        _dump_log(log); sys.exit("STOP: criterio di nullita' scattato (R degenere)")

    # 4. Sweep
    rows_csv: list[dict] = []
    sweep: list[dict] = []
    a1_trajs: dict[float, np.ndarray] = {}
    for i, sg in enumerate(SIGMAS):
        s = make_signal(X_open_test, float(sg), SEED_ETA_TEST + i)  # appaiato tra A1/A2/A3
        res = {"R": res_R}
        for kind in ("A1", "A2", "A3"):
            collect = kind == "A1" and float(sg) in TWO_DIET_SIGMAS
            rs = [rollout(k, d_test[:, k], eps_test[:, k], s[:, k], kind,
                          KF, THF, V_cal=V_CAL, collect_traj=collect) for k in range(N_TEST)]
            res[kind] = rs
            if collect:
                a1_trajs[float(sg)] = np.column_stack([r["traj"] for r in rs])
        for kind in ("R", "A1", "A2", "A3"):
            for k, r in enumerate(res[kind]):
                rows_csv.append(dict(sigma=float(sg), seed=k, agent=kind,
                                     cost=r["cost"], harm=r["harm"], effort=r["effort"],
                                     w_mean=r["w_mean"], g_mean=r["g_mean"],
                                     floor_frac=r["floor_frac"]))
        c = {kind: np.array([r["cost"] for r in res[kind]]) for kind in res}
        ci = {kind: 1.96 * c[kind].std() / np.sqrt(N_TEST) for kind in c}
        w2 = float(np.mean([r["w_mean"] for r in res["A2"]]))
        w3 = float(np.mean([r["w_mean"] for r in res["A3"]]))
        g1 = float(np.mean([r["g_mean"] for r in res["A1"]]))
        ff = {kind: float(np.mean([r["floor_frac"] for r in res[kind]]))
              for kind in ("A1", "A2", "A3")}
        sweep.append(dict(sigma=float(sg), cost=c, ci=ci, w2=w2, w3=w3, g1=g1, floor=ff,
                          theory=V_CAL / (V_CAL + float(sg) ** 2)))
        print(f"sigma={sg:4.2f} | R={c['R'].mean():.4f}±{ci['R']:.4f}  "
              f"A1={c['A1'].mean():.4f}±{ci['A1']:.4f}  A2={c['A2'].mean():.4f}±{ci['A2']:.4f}  "
              f"A3={c['A3'].mean():.4f}±{ci['A3']:.4f}  w2={w2:.2f} w3={w3:.2f} "
              f"(teoria={sweep[-1]['theory']:.2f})  g_hat_A1={g1:.2f}  floorA1={ff['A1']:.3f}")

    # 5. H1: crossover sigma* (interpolazione lineare) + IC bootstrap appaiato
    mR = np.array([sw["cost"]["R"].mean() for sw in sweep])
    m1 = np.array([sw["cost"]["A1"].mean() for sw in sweep])
    m2 = np.array([sw["cost"]["A2"].mean() for sw in sweep])
    m3 = np.array([sw["cost"]["A3"].mean() for sw in sweep])
    sigma_star = _crossing(SIGMAS, m1 - mR)
    boot_rng = np.random.default_rng(42)
    boot_stars = []
    cR_mat = np.column_stack([sw["cost"]["R"] for sw in sweep])   # [seed, sigma]
    c1_mat = np.column_stack([sw["cost"]["A1"] for sw in sweep])
    for _ in range(1000):
        idx = boot_rng.integers(0, N_TEST, N_TEST)
        boot_stars.append(_crossing(SIGMAS, c1_mat[idx].mean(0) - cR_mat[idx].mean(0)))
    found = [b for b in boot_stars if b is not None]
    ci_star = (float(np.percentile(found, 2.5)), float(np.percentile(found, 97.5))) if found else None
    log["H1"] = {"sigma_star": sigma_star, "boot_ci95": ci_star,
                 "boot_found_frac": len(found) / 1000}
    print(f"\nH1 — crossover sigma*: "
          f"{('%.2f' % sigma_star) if sigma_star else 'MAI (criterio di fallimento prereg)'}"
          f"  IC95 bootstrap: {ci_star}  (incroci trovati: {len(found)}/1000)"
          f"  [std segnale pulito = {std_open:.2f}]")

    # 6. H2: verifica causale a due diete
    two_diet = []
    for sg in TWO_DIET_SIGMAS:
        i = int(np.where(SIGMAS == sg)[0][0])
        s = make_signal(X_open_test, sg, SEED_ETA_TEST + i)
        w_ctrl = w_regression_offline(s, a1_trajs[sg])
        w_open = w_regression_offline(s, X_open_test)
        two_diet.append(dict(sigma=sg, w_ctrl=w_ctrl, w_open=w_open,
                             theory=V_CAL / (V_CAL + sg ** 2)))
        print(f"H2 dieta sigma={sg:4.2f} | w controllato={w_ctrl:.3f} | "
              f"w osservativo={w_open:.3f} | teoria={V_CAL/(V_CAL+sg**2):.2f}")
    h2_row = two_diet[0]
    h2_pass = h2_row["w_ctrl"] < 0.33 * h2_row["w_open"]
    log["H2"] = {"rows": two_diet, "pass": h2_pass}
    print(f"H2 — firma bias di loop a sigma=0: w_ctrl={h2_row['w_ctrl']:.3f} "
          f"{'<' if h2_pass else '>='} 0.33*w_open={0.33*h2_row['w_open']:.3f} -> "
          f"{'CONFERMATA' if h2_pass else 'FALSIFICATA'}")

    # 7. H3
    h3_a = bool(np.all(m3 <= 1.05 * mR))
    h3_b = bool(m3[0] <= 1.10 * m1[0])
    log["H3"] = {"floor_ok_all_sweep": h3_a, "sigma0_ok": h3_b,
                 "viol_sigmas": [float(s) for s in SIGMAS[m3 > 1.05 * mR]]}
    print(f"H3 — A3<=1.05*R su tutto lo sweep: {h3_a} "
          f"(violazioni: {log['H3']['viol_sigmas']}); A3<=1.10*A1 a sigma=0: {h3_b} "
          f"-> {'CONFERMATA' if (h3_a and h3_b) else 'FALSIFICATA'}")

    # 8. Flag floor >5% (integrazione 1b-b)
    floor_flags = [(sw["sigma"], k, sw["floor"][k]) for sw in sweep
                   for k in ("A1", "A2", "A3") if sw["floor"][k] > 0.05]
    log["floor_flags"] = floor_flags
    print(f"Floor g_hat=0.5 >5%: {floor_flags if floor_flags else 'nessuna cella'}")

    # 9. Artefatti
    with open(os.path.join(_HERE, "results_raw.csv"), "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        wcsv.writeheader(); wcsv.writerows(rows_csv)
    with open(os.path.join(_HERE, "two_diet_verification.csv"), "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["sigma", "w_ctrl", "w_open", "theory"])
        wcsv.writeheader(); wcsv.writerows(two_diet)
    _plot(sweep, mR, m1, m2, m3, sigma_star, V_CAL, std_open)

    log["duration_s"] = round(time.time() - t_start, 1)
    log["n_rollouts"] = 30 + 12 * N_TUNE + 3 * len(SIGMAS) * N_TEST
    log["seeds"] = {"test": [SEED_D_TEST, SEED_EPS_TEST, SEED_ETA_TEST],
                    "tune": [SEED_D_TUNE, SEED_EPS_TUNE, SEED_ETA_TUNE]}
    _dump_log(log)
    print(f"\nDurata: {log['duration_s']}s. Artefatti scritti in {_HERE}")


def _crossing(sig: np.ndarray, diff: np.ndarray) -> float | None:
    for i in range(1, len(sig)):
        if diff[i - 1] < 0 <= diff[i]:
            f = -diff[i - 1] / (diff[i] - diff[i - 1])
            return float(sig[i - 1] + f * (sig[i] - sig[i - 1]))
    return None


def _plot(sweep, mR, m1, m2, m3, sigma_star, V_CAL, std_open) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1d28")
        ax.tick_params(colors="#9a9ab0")
        for sp in ax.spines.values():
            sp.set_color("#2a2d3a")
        ax.grid(color="#2a2d3a", lw=0.5)

    def band(ax, m, key, color, label):
        cis = np.array([sw["ci"][key] for sw in sweep])
        ax.plot(SIGMAS, m, color=color, lw=2, label=label, marker="o", ms=4)
        ax.fill_between(SIGMAS, m - cis, m + cis, color=color, alpha=0.18)

    band(ax1, mR, "R", "#9a9ab0", "R nativo (SymbiontAgent)")
    band(ax1, m1, "A1", "#f87171", "A1 feedforward ingenuo")
    band(ax1, m2, "A2", "#667eea", "A2 w-regressione (prereg)")
    band(ax1, m3, "A3", "#34d399", "A3 w-varianza calibrato")
    if sigma_star:
        ax1.axvline(sigma_star, color="#fbbf24", ls="--", lw=1.5)
        ax1.text(sigma_star + 0.05, ax1.get_ylim()[1] * 0.95, f"sigma*={sigma_star:.2f}",
                 color="#fbbf24", fontsize=9, va="top")
    ax1.set_xlabel("Rumore del segnale sigma", color="#e8e8ed")
    ax1.set_ylabel("Costo medio/step (danno + sforzo)", color="#e8e8ed")
    ax1.set_title("M1: il vantaggio anticipatorio sopravvive? (classi SAM reali)", color="#e8e8ed")
    ax1.legend(facecolor="#242736", edgecolor="#2a2d3a", labelcolor="#e8e8ed", fontsize=8)

    th = V_CAL / (V_CAL + SIGMAS ** 2)
    ax2.plot(SIGMAS, th, color="#fbbf24", lw=2, ls="--", label="Teoria (shrinkage di Wiener)")
    ax2.plot(SIGMAS, [sw["w2"] for sw in sweep], color="#667eea", lw=2, marker="o", ms=4,
             label="w2 empirico (regressione)")
    ax2.plot(SIGMAS, [sw["w3"] for sw in sweep], color="#34d399", lw=2, marker="s", ms=4,
             label="w3 empirico (varianza)")
    ax2.set_xlabel("Rumore del segnale sigma", color="#e8e8ed")
    ax2.set_ylabel("Fiducia w", color="#e8e8ed")
    ax2.set_title("w segue la teoria?", color="#e8e8ed")
    ax2.legend(facecolor="#242736", edgecolor="#2a2d3a", labelcolor="#e8e8ed", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(_HERE, "degradation_sam_results.png"), dpi=150,
                facecolor="#0f1117")


def _dump_log(log: dict) -> None:
    log["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(_HERE, "run_log.json"), "w") as f:
        json.dump(log, f, indent=2, default=str)


if __name__ == "__main__":
    main()
