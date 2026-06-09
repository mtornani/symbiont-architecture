"""
benchmark_frequency_temporal.py — diagnosi temporale risposta in frequenza
===========================================================================
Registra e grafica le tracce passo-per-passo di singole run
per verificare l'IPOTESI FILTRO:

  "Ad alta frequenza ĝ agisce come filtro passa-basso: mantiene la media
   dell'azione anziché inseguire il disturbo veloce. A bassa frequenza invece
   la forza adattiva insegue il disturbo (ampiezza simile a quella del disturbo)."

DUE run a DIST_AMP=0.3, stesso seed (seed=42), due agenti soli:
  - Fisso (SymbiontAgent / EHD)
  - Adattivo (AdaptiveSymbiontAgent / ĝ)

Frequenze:
  - BASSA: f=0.020 (periodo = 50 step)
  - ALTA:  f=0.400 (periodo = 2.5 step)

Finestra analizzata: stazionaria [T_TRANS:T_TOTAL] = step 3000–5000.

DOMANDE:
  (1) Alta f: la forza dell'Adattivo è quasi PIATTA mentre il disturbo oscilla?
              Il Fisso invece oscilla inseguendo il disturbo?
  (2) Bassa f: la forza dell'Adattivo INSEGUE il disturbo (ampiezza grande)?
  (3) Quantifica: rapporto ampiezza_forza_adattivo / ampiezza_disturbo
                  a bassa f e ad alta f. Se diversi → filtro; se entrambi bassi → solo forza debole.

ONESTÀ: se i risultati non confermano l'ipotesi, viene detto esplicitamente.

Run:
    python benchmark_frequency_temporal.py
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, "sam-multiagent-v0"))

from cluster import MemoryCluster
from environment import GlobalWorldState, NeuronContext

# ---------------------------------------------------------------------------
# Costanti — identiche al banco chiuso
# ---------------------------------------------------------------------------
T_TOTAL    = 5000
T_TRANS    = 3000
T_STEADY   = T_TOTAL - T_TRANS   # 2000

DIST_AMP   = 0.3
GAIN       = 1.0
SETPOINT   = 0.0
NOISE_STD  = 0.05

LR_GAIN       = 0.05
GAIN_HAT_MIN  = 0.5
GAIN_HAT_MAX  = 10.0
GAIN_HAT_INIT = 1.0

SEED = 42

FREQ_LOW  = 0.020   # periodo 50 step
FREQ_HIGH = 0.400   # periodo 2.5 step

# Quanti step mostrare nel grafico (ultimi N della finestra stazionaria)
# A bassa f: mostrare 4 cicli completi → 4 × (1/0.02) = 200 step
# Ad alta f: mostrare 20 cicli → 20 × (1/0.4) = 50 step; usiamo un blocco comune
PLOT_LOW_STEPS  = 200   # 4 cicli a f=0.02
PLOT_HIGH_STEPS = 50    # 20 cicli a f=0.40


# ---------------------------------------------------------------------------
# Copie locali degli agenti (identiche al benchmark chiuso)
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

    def step(self, action: float, disturbance: float = 0.0):
        action  = float(np.clip(action, -1.0, 1.0))
        self.x  = float(np.clip(
            self.x + GAIN * action + self._noise[self._t] + disturbance,
            -8.0, 8.0,
        ))
        self._t += 1
        return self.x, self.error


class SymbiontAgent:
    N_NEURONS = 4
    N_INPUTS  = 8

    def __init__(self, seed: int = 0) -> None:
        self.cluster   = MemoryCluster(n_neurons=self.N_NEURONS, n_inputs=self.N_INPUTS, base_seed=seed)
        self._endo     = self.cluster.current_state
        self._step_idx = 0

    def _make_contexts(self, error: float, risk: float, reward: float):
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
        self._x_prev      = None
        self._action_prev = None

    def _make_contexts(self, error: float, risk: float, reward: float):
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


# ---------------------------------------------------------------------------
# Singola run con tracce complete
# ---------------------------------------------------------------------------
def run_episode(env_seed: int, freq: float) -> dict:
    """Ritorna le tracce complete [T_TOTAL,] per Fisso e Adattivo."""
    env_f = Env1D(seed=env_seed)
    env_a = Env1D(seed=env_seed)

    agent_f = SymbiontAgent(seed=0)
    agent_a = AdaptiveSymbiontAgent(seed=0)

    errors_f   = np.zeros(T_TOTAL)
    errors_a   = np.zeros(T_TOTAL)
    actions_f  = np.zeros(T_TOTAL)
    actions_a  = np.zeros(T_TOTAL)
    gain_hat   = np.zeros(T_TOTAL)
    disturbances = np.zeros(T_TOTAL)

    for t in range(T_TOTAL):
        dist = DIST_AMP * math.sin(2.0 * math.pi * freq * t)
        disturbances[t] = dist

        err_f  = env_f.error
        act_f  = agent_f.act(err_f)
        _, e_f = env_f.step(act_f, dist)
        errors_f[t]  = e_f
        actions_f[t] = act_f

        err_a  = env_a.error
        act_a  = agent_a.act(err_a)
        _, e_a = env_a.step(act_a, dist)
        errors_a[t]  = e_a
        actions_a[t] = act_a
        gain_hat[t]  = agent_a.gain_hat

    return {
        'disturbances': disturbances,
        'errors_f':    errors_f,
        'errors_a':    errors_a,
        'actions_f':   actions_f,
        'actions_a':   actions_a,
        'gain_hat':    gain_hat,
    }


def oscillation_amplitude(signal: np.ndarray) -> float:
    """Stima ampiezza oscillazione = std × sqrt(2) (approx peak per sinusoide pura)."""
    return float(np.std(signal)) * math.sqrt(2.0)


def peak_to_peak_half(signal: np.ndarray) -> float:
    """(max - min) / 2 — ampiezza semi-escursione."""
    return float((np.max(signal) - np.min(signal)) / 2.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("DIAGNOSI TEMPORALE — IPOTESI FILTRO ĝ")
    print("=" * 70)
    print(f"DIST_AMP={DIST_AMP}  SEED={SEED}  Agenti: Fisso, Adattivo(ĝ)")
    print(f"Finestra stazionaria: step {T_TRANS}–{T_TOTAL} ({T_STEADY} step)")
    print()

    print(f"Esecuzione run bassa frequenza (f={FREQ_LOW})...")
    data_low  = run_episode(SEED, FREQ_LOW)
    print(f"Esecuzione run alta frequenza  (f={FREQ_HIGH})...")
    data_high = run_episode(SEED, FREQ_HIGH)
    print()

    # Slices stazionari
    sl = slice(T_TRANS, T_TOTAL)
    for tag, data, freq in [("BASSA", data_low, FREQ_LOW), ("ALTA", data_high, FREQ_HIGH)]:
        d     = data['disturbances'][sl]
        af    = data['actions_f'][sl]
        aa    = data['actions_a'][sl]
        ef    = data['errors_f'][sl]
        ea    = data['errors_a'][sl]
        gh    = data['gain_hat'][sl]

        amp_d  = oscillation_amplitude(d)
        amp_af = oscillation_amplitude(af)
        amp_aa = oscillation_amplitude(aa)
        amp_ef = oscillation_amplitude(ef)
        amp_ea = oscillation_amplitude(ea)
        ratio_a  = amp_aa / amp_d if amp_d > 1e-9 else float('nan')
        ratio_f  = amp_af / amp_d if amp_d > 1e-9 else float('nan')
        gh_mean  = float(np.mean(gh))
        gh_std   = float(np.std(gh))

        print(f"--- f={freq:.3f} ({tag} FREQUENZA) ---")
        print(f"  Disturbo:         amp={amp_d:.4f}  (teoria={DIST_AMP/math.sqrt(2):.4f})")
        print(f"  Azione Fisso:     amp={amp_af:.4f}  ratio_vs_disturbo={ratio_f:.3f}")
        print(f"  Azione Adattivo:  amp={amp_aa:.4f}  ratio_vs_disturbo={ratio_a:.3f}")
        print(f"  Errore Fisso:     amp={amp_ef:.4f}")
        print(f"  Errore Adattivo:  amp={amp_ea:.4f}")
        print(f"  ĝ medio:  {gh_mean:.4f}  ĝ std: {gh_std:.4f}")
        print()

    # -----------------------------------------------------------------------
    # RISPOSTA ALLE TRE DOMANDE
    # -----------------------------------------------------------------------
    sl = slice(T_TRANS, T_TOTAL)

    d_low   = data_low['disturbances'][sl]
    af_low  = data_low['actions_f'][sl]
    aa_low  = data_low['actions_a'][sl]
    d_high  = data_high['disturbances'][sl]
    af_high = data_high['actions_f'][sl]
    aa_high = data_high['actions_a'][sl]

    amp_d_low   = oscillation_amplitude(d_low)
    amp_d_high  = oscillation_amplitude(d_high)
    amp_aa_low  = oscillation_amplitude(aa_low)
    amp_aa_high = oscillation_amplitude(aa_high)
    amp_af_low  = oscillation_amplitude(af_low)
    amp_af_high = oscillation_amplitude(af_high)

    ratio_a_low  = amp_aa_low  / amp_d_low  if amp_d_low  > 1e-9 else float('nan')
    ratio_a_high = amp_aa_high / amp_d_high if amp_d_high > 1e-9 else float('nan')
    ratio_f_low  = amp_af_low  / amp_d_low  if amp_d_low  > 1e-9 else float('nan')
    ratio_f_high = amp_af_high / amp_d_high if amp_d_high > 1e-9 else float('nan')

    print("=" * 70)
    print("RISPOSTA ALLE TRE DOMANDE")
    print("=" * 70)

    # Domanda 1
    FLAT_THRESHOLD = 0.15   # azione adattiva "quasi piatta" se rapporto < 15%
    is_flat_high   = ratio_a_high < FLAT_THRESHOLD
    print(f"\nDOMANDA (1) — Ad alta f={FREQ_HIGH}:")
    print(f"  ratio azione_Adattivo/disturbo = {ratio_a_high:.3f}  (soglia 'piatta' < {FLAT_THRESHOLD})")
    print(f"  ratio azione_Fisso/disturbo    = {ratio_f_high:.3f}")
    if is_flat_high:
        print(f"  RISPOSTA: Azione Adattivo QUASI PIATTA ad alta f.")
        print(f"  Il Fisso oscillates con rapporto {ratio_f_high:.3f}.")
    else:
        print(f"  RISPOSTA: Azione Adattivo NON è piatta ad alta f (rapporto={ratio_a_high:.3f}).")
        print(f"  L'ipotesi 'filtro' su questa dimensione non è confermata.")

    # Domanda 2
    FOLLOW_THRESHOLD = 0.40  # azione adattiva "segue" il disturbo se rapporto > 40%
    is_following_low  = ratio_a_low > FOLLOW_THRESHOLD
    print(f"\nDOMANDA (2) — A bassa f={FREQ_LOW}:")
    print(f"  ratio azione_Adattivo/disturbo = {ratio_a_low:.3f}  (soglia 'insegue' > {FOLLOW_THRESHOLD})")
    if is_following_low:
        print(f"  RISPOSTA: Azione Adattivo INSEGUE il disturbo a bassa f.")
    else:
        print(f"  RISPOSTA: Azione Adattivo NON insegue il disturbo a bassa f (rapporto={ratio_a_low:.3f}).")
        print(f"  L'ipotesi 'filtro' su questa dimensione non è confermata.")

    # Domanda 3
    print(f"\nDOMANDA (3) — Quantificazione rapporti:")
    print(f"  f={FREQ_LOW:.3f} (bassa): ratio_Adattivo = {ratio_a_low:.3f},  ratio_Fisso = {ratio_f_low:.3f}")
    print(f"  f={FREQ_HIGH:.3f} (alta):  ratio_Adattivo = {ratio_a_high:.3f},  ratio_Fisso = {ratio_f_high:.3f}")
    ratio_change = ratio_a_low / ratio_a_high if ratio_a_high > 1e-9 else float('nan')
    print(f"  Rapporto (bassa/alta) per Adattivo: {ratio_change:.2f}×")
    print()

    # Verdetto
    print("VERDETTO FILTRO:")
    filter_confirmed = is_flat_high and is_following_low and (ratio_change > 2.0)
    if filter_confirmed:
        print("  IPOTESI FILTRO CONFERMATA: l'Adattivo si comporta come filtro passa-basso.")
        print("  Ampiezza azione significativamente diversa tra bassa e alta frequenza.")
    elif is_flat_high and not is_following_low:
        print("  PARZIALE: Azione piatta ad alta f, ma NON segue il disturbo a bassa f.")
        print("  Non è un filtro classico: è semplicemente forza ridotta a qualunque frequenza.")
    elif not is_flat_high and is_following_low:
        print("  PARZIALE: Segue a bassa f, ma NON piatta ad alta f.")
    else:
        print("  IPOTESI FILTRO NON CONFERMATA: il comportamento è simile alle due frequenze.")
        print("  L'effetto ĝ non è frequenza-selettivo in questo modo.")

    print()

    # -----------------------------------------------------------------------
    # GRAFICO
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle(
        f"Diagnosi temporale — DIST_AMP={DIST_AMP}, seed={SEED}\n"
        f"Bassa f={FREQ_LOW} (sinistra) | Alta f={FREQ_HIGH} (destra)\n"
        f"Finestra: step {T_TRANS}–{T_TOTAL}",
        fontsize=12
    )

    # Slice per il plot: ultimi N step della finestra stazionaria
    n_low  = PLOT_LOW_STEPS
    n_high = PLOT_HIGH_STEPS
    t_low  = np.arange(T_STEADY - n_low,  T_STEADY)
    t_high = np.arange(T_STEADY - n_high, T_STEADY)

    configs = [
        (0, data_low,  FREQ_LOW,  t_low,  n_low),
        (1, data_high, FREQ_HIGH, t_high, n_high),
    ]

    for col, data, freq, t_idx, n_plot in configs:
        d   = data['disturbances'][T_TRANS:T_TOTAL][T_STEADY - n_plot:]
        af  = data['actions_f'][T_TRANS:T_TOTAL][T_STEADY - n_plot:]
        aa  = data['actions_a'][T_TRANS:T_TOTAL][T_STEADY - n_plot:]
        ef  = data['errors_f'][T_TRANS:T_TOTAL][T_STEADY - n_plot:]
        ea  = data['errors_a'][T_TRANS:T_TOTAL][T_STEADY - n_plot:]
        gh  = data['gain_hat'][T_TRANS:T_TOTAL][T_STEADY - n_plot:]
        steps = np.arange(len(d))

        # Row 0: Disturbo
        ax = axes[0, col]
        ax.plot(steps, d, color='black', lw=1.5, label=f"Disturbo (amp={DIST_AMP})")
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_ylabel("d(t)")
        ax.set_title(f"f={freq:.3f} — periodo≈{1/freq:.1f} step")
        ax.legend(fontsize=8)
        ax.set_ylim(-DIST_AMP * 1.3, DIST_AMP * 1.3)

        # Row 1: Azioni
        ax = axes[1, col]
        ax.plot(steps, af, color='steelblue', lw=1.2, alpha=0.8, label=f"Fisso (amp={oscillation_amplitude(af):.3f})")
        ax.plot(steps, aa, color='darkorange', lw=1.5, label=f"Adattivo (amp={oscillation_amplitude(aa):.3f})")
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_ylabel("azione(t)")
        ax.legend(fontsize=8)

        # Row 2: Errori
        ax = axes[2, col]
        ax.plot(steps, ef, color='steelblue', lw=1.2, alpha=0.8, label=f"Errore Fisso (rms={np.std(ef):.3f})")
        ax.plot(steps, ea, color='darkorange', lw=1.5, label=f"Errore Adattivo (rms={np.std(ea):.3f})")
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_ylabel("errore(t)")
        ax.legend(fontsize=8)

        # Row 3: ĝ
        ax = axes[3, col]
        ax.plot(steps, gh, color='forestgreen', lw=1.5)
        ax.axhline(float(np.mean(gh)), color='forestgreen', lw=0.8, ls='--', alpha=0.6,
                   label=f"ĝ medio={np.mean(gh):.3f}")
        ax.axhline(GAIN_HAT_MIN, color='red', lw=0.7, ls=':', label=f"min={GAIN_HAT_MIN}")
        ax.set_ylabel("ĝ(t)")
        ax.set_xlabel("step (finestra stazionaria)")
        ax.legend(fontsize=8)

    # Annotazione rapporti
    for col, freq, ratio_a, ratio_f_val in [
        (0, FREQ_LOW,  ratio_a_low,  ratio_f_low),
        (1, FREQ_HIGH, ratio_a_high, ratio_f_high),
    ]:
        axes[1, col].set_title(
            f"Azioni  |  ratio_A={ratio_a:.3f}  ratio_F={ratio_f_val:.3f}",
            fontsize=9
        )

    plt.tight_layout()
    out_path = os.path.join(_REPO_ROOT, "benchmark_frequency_temporal_output.png")
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"Figura salvata: {out_path}")

    # Figura separata: confronto azione vs disturbo (sovrapposizione diretta)
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 8))
    fig2.suptitle(
        "Confronto diretto: Azione Adattivo vs Disturbo\n"
        f"(scala normalizzata per disturbo = {DIST_AMP})",
        fontsize=11
    )

    configs2 = [
        (0, data_low,  FREQ_LOW,  n_low,  "BASSA FREQUENZA"),
        (1, data_high, FREQ_HIGH, n_high, "ALTA FREQUENZA"),
    ]

    for col, data, freq, n_plot, label in configs2:
        d  = data['disturbances'][T_TRANS:T_TOTAL][T_STEADY - n_plot:]
        aa = data['actions_a'][T_TRANS:T_TOTAL][T_STEADY - n_plot:]
        af = data['actions_f'][T_TRANS:T_TOTAL][T_STEADY - n_plot:]
        steps = np.arange(len(d))

        # Top row: Adattivo vs disturbo
        ax = axes2[0, col]
        ax.plot(steps, d / DIST_AMP, color='black', lw=1.5, ls='--', alpha=0.6,
                label=f"Disturbo normalizzato")
        ax.plot(steps, aa, color='darkorange', lw=1.5,
                label=f"Azione Adattivo (amp={oscillation_amplitude(aa):.3f})")
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.set_title(f"{label} f={freq:.3f} — Adattivo vs Disturbo")
        ax.set_ylabel("ampiezza")
        ax.legend(fontsize=8)

        # Bottom row: Fisso vs disturbo
        ax = axes2[1, col]
        ax.plot(steps, d / DIST_AMP, color='black', lw=1.5, ls='--', alpha=0.6,
                label=f"Disturbo normalizzato")
        ax.plot(steps, af, color='steelblue', lw=1.5,
                label=f"Azione Fisso (amp={oscillation_amplitude(af):.3f})")
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.set_title(f"{label} f={freq:.3f} — Fisso vs Disturbo")
        ax.set_ylabel("ampiezza")
        ax.set_xlabel("step (finestra stazionaria)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path2 = os.path.join(_REPO_ROOT, "benchmark_frequency_temporal_overlay.png")
    plt.savefig(out_path2, dpi=130, bbox_inches='tight')
    print(f"Figura overlay salvata: {out_path2}")


if __name__ == "__main__":
    main()
