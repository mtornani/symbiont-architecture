"""
Esperimento pre-registrato: degradazione del segnale anticipatorio (M0).
Vedi prereg_degradazione.md. Modello giocattolo dichiarato — non il codebase SAM.

Agenti:
  R   reattivo puro
  A1  anticipatorio ingenuo (fiducia = 1, parametri congelati dopo tuning a sigma=0)
  A2  anticipatorio + w-regressione online (PRE-REGISTRATO, soggetto ad bias da loop chiuso)
  A3  anticipatorio + w-varianza calibrato (ESPLORATIVO, dichiarato fuori prereg)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Parametri ambiente (da prereg) ----------------
RHO, E_EFF, C_ACT = 0.9, 0.8, 0.10
THETA_HARM = 1.0
P_BURST, BURST_MEAN = 0.02, 2.0
T, BURN, LEAD = 3000, 200, 3
N_TEST, N_TUNE = 30, 8
SIGMAS = np.array([0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
ALPHA_EMA, W_WARMUP = 0.01, 200

rng_global = np.random.default_rng(42)

# ---------------- Generazione ambiente ----------------
def gen_shocks(n_seeds, seed0):
    """Burst esogeni b[t, seed] — identici per tutti gli agenti (design appaiato)."""
    rng = np.random.default_rng(seed0)
    hits = rng.random((T, n_seeds)) < P_BURST
    mags = rng.exponential(BURST_MEAN, (T, n_seeds))
    return hits * mags

def open_loop_hazard(b):
    """Hazard non controllato H_open — base esogena del segnale anticipatorio."""
    H = np.zeros_like(b)
    for t in range(1, T):
        H[t] = RHO * H[t - 1] + b[t]
    return H

def make_signal(H_open, sigma, seed0):
    """s[t] = H_open[t+LEAD] + rumore. Esogeno, precomputabile."""
    rng = np.random.default_rng(seed0)
    s = np.empty_like(H_open)
    s[:T - LEAD] = H_open[LEAD:]
    s[T - LEAD:] = H_open[-1]
    return s + rng.normal(0, sigma, H_open.shape)

# ---------------- Rollout vettorizzato sui seed ----------------
def rollout(b, s, kr, thr, kf=0.0, thf=1.0, trust="none", V_cal=None):
    """
    trust: 'none' (R/A1 con w=1 implicito via kf), 'reg' (A2), 'var' (A3).
    Ritorna costo medio/step, danno, azione, e traiettoria media di w.
    """
    n = b.shape[1]
    h = np.zeros(n)
    harm_acc = np.zeros(n); act_acc = np.zeros(n)
    # stato EMA per w-regressione (A2)
    m_s = np.zeros(n); m_h = np.zeros(n); c_sh = np.zeros(n); v_s = np.full(n, 1e-6)
    # stato EMA per w-varianza (A3)
    v_sig = np.full(n, 1e-6); m_sig = np.zeros(n)
    w_log = []

    for t in range(T):
        sig = s[t]
        # --- peso di fiducia ---
        if trust == "reg":
            w = np.clip(c_sh / (v_s + 1e-8), 0, 1) if t >= W_WARMUP else np.full(n, 0.5)
        elif trust == "var":
            w = np.clip(V_cal / (v_sig + 1e-8), 0, 1) if t >= W_WARMUP else np.full(n, 0.5)
        else:
            w = np.ones(n)
        # --- azione: termine reattivo + termine anticipatorio pesato ---
        a = np.clip(kr * np.maximum(0, h - thr) + w * kf * np.maximum(0, sig - thf), 0, 1)
        # --- costi (danno calcolato dall'ambiente, non dall'agente) ---
        harm_acc += (t >= BURN) * np.maximum(0, h - THETA_HARM)
        act_acc  += (t >= BURN) * a
        # --- dinamica ---
        if t + 1 < T:
            h = RHO * h * (1 - E_EFF * a) + b[t + 1] * (1 - E_EFF * a)
        # --- aggiornamento stimatori online ---
        if trust == "reg" and t >= LEAD:
            sp = s[t - LEAD]                      # segnale emesso LEAD step fa
            m_s += ALPHA_EMA * (sp - m_s)
            m_h += ALPHA_EMA * (h - m_h)
            c_sh += ALPHA_EMA * ((sp - m_s) * (h - m_h) - c_sh)
            v_s  += ALPHA_EMA * ((sp - m_s) ** 2 - v_s)
        if trust == "var":
            m_sig += ALPHA_EMA * (sig - m_sig)
            v_sig += ALPHA_EMA * ((sig - m_sig) ** 2 - v_sig)
        if trust in ("reg", "var") and t % 50 == 0 and t >= W_WARMUP:
            w_log.append(w.mean())

    steps = T - BURN
    harm, act = harm_acc / steps, act_acc / steps
    cost = harm + C_ACT * act
    return cost, harm, act, (np.mean(w_log) if w_log else 1.0)

# ---------------- Tuning (seed separati, come da prereg) ----------------
b_tune = gen_shocks(N_TUNE, seed0=1000)
H_tune = open_loop_hazard(b_tune)
V_CAL = H_tune[BURN:].var()          # varianza del segnale pulito (fase di calibrazione)

def tune_R():
    best = (np.inf, None)
    for kr in [0.5, 1, 2, 4]:
        for thr in [0.4, 0.7, 1.0]:
            c, *_ = rollout(b_tune, np.zeros_like(b_tune), kr, thr)
            if c.mean() < best[0]:
                best = (c.mean(), (kr, thr))
    return best[1]

def tune_A1(kr, thr):
    s0 = make_signal(H_tune, 0.0, seed0=2000)     # tuning a sigma=0, poi CONGELATO
    best = (np.inf, None)
    for kf in [0.25, 0.5, 1, 2]:
        for thf in [0.4, 0.7, 1.0]:
            c, *_ = rollout(b_tune, s0, kr, thr, kf, thf)
            if c.mean() < best[0]:
                best = (c.mean(), (kf, thf))
    return best[1]

KR, THR = tune_R()
KF, THF = tune_A1(KR, THR)
print(f"Tuning — R: kr={KR}, thr={THR} | A1 (a sigma=0, congelato): kf={KF}, thf={THF}")
print(f"Calibrazione: Var(H_open) = {V_CAL:.3f}, std = {np.sqrt(V_CAL):.3f}")

# Controllo criterio di nullita' (prereg): R non deve essere degenere
_, _, act_R_chk, _ = rollout(b_tune, np.zeros_like(b_tune), KR, THR)
assert 0.001 < act_R_chk.mean() < 0.999, "R degenere: esperimento nullo"

# ---------------- Test ----------------
b_test = gen_shocks(N_TEST, seed0=5000)
H_test = open_loop_hazard(b_test)

rows = []
for i, sg in enumerate(SIGMAS):
    s = make_signal(H_test, sg, seed0=7000 + i)   # stesso rumore per A1/A2/A3 (appaiato)
    cR, hR, aR, _  = rollout(b_test, np.zeros_like(b_test), KR, THR)
    c1, h1, a1, _  = rollout(b_test, s, KR, THR, KF, THF)
    c2, h2, a2, w2 = rollout(b_test, s, KR, THR, KF, THF, trust="reg")
    c3, h3, a3, w3 = rollout(b_test, s, KR, THR, KF, THF, trust="var", V_cal=V_CAL)
    rows.append(dict(sigma=sg,
                     R=cR, A1=c1, A2=c2, A3=c3,
                     hR=hR.mean(), h1=h1.mean(), h2=h2.mean(),
                     w2=w2, w3=w3))
    ci = lambda x: 1.96 * x.std() / np.sqrt(N_TEST)
    print(f"sigma={sg:4.2f} | R={cR.mean():.4f}±{ci(cR):.4f}  A1={c1.mean():.4f}±{ci(c1):.4f}  "
          f"A2={c2.mean():.4f}±{ci(c2):.4f}  A3={c3.mean():.4f}±{ci(c3):.4f}  "
          f"w2={w2:.2f} w3={w3:.2f} (teoria={V_CAL/(V_CAL+sg**2):.2f})")

# ---------------- Analisi: soglia di crossover sigma* ----------------
mR = np.array([r["R"].mean() for r in rows])
m1 = np.array([r["A1"].mean() for r in rows])
m2 = np.array([r["A2"].mean() for r in rows])
m3 = np.array([r["A3"].mean() for r in rows])

sigma_star = None
diff = m1 - mR
for i in range(1, len(SIGMAS)):
    if diff[i - 1] < 0 <= diff[i]:
        # interpolazione lineare del punto di incrocio
        f = -diff[i - 1] / (diff[i] - diff[i - 1])
        sigma_star = SIGMAS[i - 1] + f * (SIGMAS[i] - SIGMAS[i - 1])
        break
print(f"\nH1 — soglia di crossover sigma* (A1 incrocia R): "
      f"{'%.2f' % sigma_star if sigma_star else 'MAI (criterio di fallimento prereg)'}"
      f"  [std segnale pulito = {np.sqrt(V_CAL):.2f}]")

viol2 = SIGMAS[m2 > mR * 1.05]
viol3 = SIGMAS[m3 > mR * 1.05]
print(f"H2 — A2 (w-regressione) sotto il pavimento reattivo? "
      f"{'SI, viola a sigma=' + str(list(viol2)) if len(viol2) else 'mai violato: degradazione graduale'}")
print(f"     A3 (w-varianza, esplorativo) : "
      f"{'viola a sigma=' + str(list(viol3)) if len(viol3) else 'mai violato'}")

# ---------------- Grafico ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
fig.patch.set_facecolor("#0f1117")
for ax in (ax1, ax2):
    ax.set_facecolor("#1a1d28")
    ax.tick_params(colors="#9a9ab0")
    for sp in ax.spines.values(): sp.set_color("#2a2d3a")
    ax.grid(color="#2a2d3a", lw=0.5)

def band(ax, m, arrs, color, label):
    cis = np.array([1.96 * r.std() / np.sqrt(N_TEST) for r in arrs])
    ax.plot(SIGMAS, m, color=color, lw=2, label=label, marker="o", ms=4)
    ax.fill_between(SIGMAS, m - cis, m + cis, color=color, alpha=0.18)

band(ax1, mR, [r["R"] for r in rows], "#9a9ab0", "R reattivo")
band(ax1, m1, [r["A1"] for r in rows], "#f87171", "A1 ingenuo")
band(ax1, m2, [r["A2"] for r in rows], "#667eea", "A2 w-regressione (prereg)")
band(ax1, m3, [r["A3"] for r in rows], "#34d399", "A3 w-varianza (esplorativo)")
if sigma_star:
    ax1.axvline(sigma_star, color="#fbbf24", ls="--", lw=1.5)
    ax1.text(sigma_star + 0.05, ax1.get_ylim()[1] * 0.95, f"sigma*={sigma_star:.2f}",
             color="#fbbf24", fontsize=9, va="top")
ax1.set_xlabel("Rumore del segnale sigma", color="#e8e8ed")
ax1.set_ylabel("Costo medio/step (danno + sforzo)", color="#e8e8ed")
ax1.set_title("Il vantaggio anticipatorio sopravvive?", color="#e8e8ed")
ax1.legend(facecolor="#242736", edgecolor="#2a2d3a", labelcolor="#e8e8ed", fontsize=8)

th = V_CAL / (V_CAL + SIGMAS ** 2)
ax2.plot(SIGMAS, th, color="#fbbf24", lw=2, ls="--", label="Teoria (shrinkage di Wiener)")
ax2.plot(SIGMAS, [r["w2"] for r in rows], color="#667eea", lw=2, marker="o", ms=4, label="w2 empirico (regressione)")
ax2.plot(SIGMAS, [r["w3"] for r in rows], color="#34d399", lw=2, marker="s", ms=4, label="w3 empirico (varianza)")
ax2.set_xlabel("Rumore del segnale sigma", color="#e8e8ed")
ax2.set_ylabel("Fiducia w", color="#e8e8ed")
ax2.set_title("Predizione secondaria: w segue la teoria?", color="#e8e8ed")
ax2.legend(facecolor="#242736", edgecolor="#2a2d3a", labelcolor="#e8e8ed", fontsize=8)

plt.tight_layout()
plt.savefig("/home/claude/degradation_results.png", dpi=150, facecolor="#0f1117")
print("\nGrafico salvato: degradation_results.png")
