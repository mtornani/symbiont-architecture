"""Verifica causale: lo stimatore w-regressione e' rotto dal BIAS DA LOOP CHIUSO?
Stesso stimatore, stessi dati, due diete:
  (a) h CONTROLLATO dall'agente stesso (come in A2)
  (b) H_open NON controllato (osservazione pura, nessuna azione)
Se (b) recupera la teoria, la causa e' l'azione dell'agente che cancella la correlazione."""
import numpy as np
from symbiont_degradation import (gen_shocks, open_loop_hazard, make_signal,
                                   rollout, KR, THR, KF, THF, V_CAL, T, LEAD, ALPHA_EMA)

b = gen_shocks(30, seed0=5000)
H_open = open_loop_hazard(b)

def w_regression_offline(s, h):
    """Stessa regressione EMA di A2, ma su una serie h arbitraria."""
    m_s = np.zeros(s.shape[1]); m_h = np.zeros(s.shape[1])
    c_sh = np.zeros(s.shape[1]); v_s = np.full(s.shape[1], 1e-6)
    for t in range(LEAD, T):
        sp = s[t - LEAD]
        m_s += ALPHA_EMA * (sp - m_s)
        m_h += ALPHA_EMA * (h[t] - m_h)
        c_sh += ALPHA_EMA * ((sp - m_s) * (h[t] - m_h) - c_sh)
        v_s  += ALPHA_EMA * ((sp - m_s) ** 2 - v_s)
    return np.clip(c_sh / (v_s + 1e-8), 0, 1).mean()

print(f"{'sigma':>6} | {'w su h CONTROLLATO':>20} | {'w su H_open (osserva)':>22} | {'teoria':>7}")
for sg in [0.0, 0.5, 1.0, 2.0]:
    s = make_signal(H_open, sg, seed0=7000)
    # (a) traiettoria controllata: rollout A2 e ricalcolo w sulla SUA h — riuso rollout
    #     con trust='reg' che internamente fa esattamente questo; qui replico offline:
    #     simulo la h di un agente anticipatorio pieno (A1) e misuro w su quella.
    n = b.shape[1]; h = np.zeros(n); h_traj = np.zeros((T, n))
    for t in range(T):
        a = np.clip(KR * np.maximum(0, h - THR) + KF * np.maximum(0, s[t] - THF), 0, 1)
        h_traj[t] = h
        if t + 1 < T:
            h = 0.9 * h * (1 - 0.8 * a) + b[t + 1] * (1 - 0.8 * a)
    w_ctrl = w_regression_offline(s, h_traj)
    w_open = w_regression_offline(s, H_open)
    print(f"{sg:6.2f} | {w_ctrl:20.3f} | {w_open:22.3f} | {V_CAL/(V_CAL+sg**2):7.2f}")
