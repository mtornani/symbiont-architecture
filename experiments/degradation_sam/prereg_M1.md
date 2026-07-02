# Prereg M1 — Degradazione segnale su SAM · 2026-07-02
Stato: BOZZA in attesa di Gate 1b. Le ipotesi si congelano all'OK di Mirko; dopo, nessuna modifica.
Base: piano SYMBIONT→STRUMENTO v1.3 §4.2 + vincoli Gate 1a (3a–3d). Riferimento strutturale: `docs/experiments/2026-07-02_M0/` (commit b8bdc18).

## Ambiente
Benchmark esocentrico: MAI COMMITTATO (verificato Gate 1a) → **ricostruzione** in `experiments/degradation_sam/harness_m1.py`.
Suite test verde: sì/33 (verificata al Gate 1a; ricontrollo pre-run — se rossa: STOP 4.a).

**Impianto ricostruito** (definito per intero; scarti dichiarati sotto):
```
x_{t+1} = clip( ρ_x·x_t + g·a_t + d_{t+1} + ε_t , −6, 6 )
ρ_x = 0.9        # SCARTO-1 dal canonico Env1D (integratore puro, benchmark_homeostasis_vs_reward.py)
g   = 1.0        # guadagno FISSO: M1 testa la degradazione del segnale, non lo shift di guadagno
ε_t ~ N(0, 0.05) # NOISE_STD canonico (benchmark_adaptive_homeostasis.py:98)
d_t = burst poissoniani: p=0.02/step, magnitudine ~Exp(2.0), segno positivo (M0)
```
SCARTO-1 motivato: la verifica causale a due diete richiede una traiettoria osservativa (senza azione) stazionaria; l'integratore puro con burst unilaterali satura al clip +6 e rende degenere la dieta osservativa. ρ_x=0.9 è il valore M0 (Appendice B).
Setpoint = 0; error = x (come nel canonico, `SETPOINT=0`).

**Traiettoria osservativa** (esogena, precomputata): `X_open_{t+1} = clip(ρ_x·X_open_t + d_{t+1} + ε_t, −6, 6)`.

**Segnale anticipatorio**: `s(t) = X_open(t+LEAD) + η`, `η ~ N(0, σ)`, LEAD=3 (M0). Esogeno, precomputato, identico per A1/A2/A3 (appaiamento).
Punto esatto di iniezione del rumore: generazione di `s` in `harness_m1.py::make_signal` — l'assunzione-oracolo da rompere è σ=0 in questa funzione. Nessun modulo sam-* toccato.

**Metrica (calcolata dall'AMBIENTE, mai dall'agente)**:
```
costo/step = danno + c·sforzo = max(0, |x_t| − 1.0) + 0.10·|a_t|,   media su t ≥ BURN
```
SCARTO-2: danno simmetrico `|x_t|` invece dell'unilaterale M0 `max(0, h−1)`. Motivo: nell'impianto additivo x esce dalla banda in ENTRAMBE le direzioni; danno unilaterale renderebbe gratuito (salvo sforzo) il pre-tuffo negativo illimitato del feedforward — pattern degenere non presente in M0 (dove h≥0 strutturalmente).

## Sweep
σ ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0} (9 livelli, incluso 0).
Requisito σ_max ≥ 3×std(X_open): std attesa ≈0.87 (analogo M0+ε) → 3× ≈ 2.6 ≤ 4.0. La std viene MISURATA in calibrazione prima dello sweep; se std > 4/3 (requisito violato): STOP, segnalazione al gate, nessuna estensione autonoma.

## Seed (pinnati nel codice)
Test: 30 appaiati — burst seed0=5000, segnale seed0=7000+i (per livello σ), stesso ambiente e stesso rumore per tutti gli agenti. `MemoryCluster(base_seed=seed)` identico tra agenti a parità di seed.
Tuning: 8 SEPARATI — burst seed0=1000, segnale seed0=2000.
T=3000, BURN=200, W_WARMUP=200.

## Agenti (mapping Gate 1a-3c)
- **R** — regolazione nativa, verbatim `benchmark_adaptive_homeostasis.py:138-167` (SymbiontAgent):
  `a = clip(−tanh(x·k), −1, 1)`, `k = 1 − 0.5·cortisol`, cortisolo da `MemoryCluster.step()` (4 neuroni, 8 input).
  SCARTO-3 dal template ("R ottimizzato per sé"): la legge nativa NON ha parametri liberi → tuning vacuo, R corre con la sua legge di progetto. La baseline non è svantaggiata: è esattamente l'agente reale.
- **A1** — R + feedforward dal segnale, azione scalata da ĝ:
  `a = clip( a_R + a_ff , −1, 1)`, `a_ff = −k_ff·max(0, s(t) − th_ff) / max(ĝ, GAIN_HAT_MIN)`.
  Tuning di (k_ff, th_ff) a σ=0 su griglia k_ff∈{0.25, 0.5, 1, 2} × th_ff∈{0.4, 0.7, 1.0} (griglia M0), 8 seed di tuning, poi CONGELATI. Feedforward unilaterale (burst positivi, come M0).
- **A2** — A1 + fiducia a regressione online: `a = clip(a_R + w·a_ff, −1, 1)`,
  `w = clip(cov_EMA(s(t−LEAD), x_t) / var_EMA(s(t−LEAD)), 0, 1)`, α=0.01, warmup 200 step a w=0.5.
  Variabile realizzata = x_t DELLA PROPRIA traiettoria (dieta controllata — replica esatta di A2 in M0, che regrediva su h propria). Eredita (k_ff, th_ff) da A1.
- **A3** — A1 + fiducia calibrata: `w = clip(V_cal / v̂_sig, 0, 1)`, `v̂_sig` = varianza EMA (α=0.01) del segnale ricevuto; `V_cal = Var(X_open[BURN:])` misurata sui seed di TUNING (fase di calibrazione a segnale pulito). Warmup 200 a w=0.5. Eredita (k_ff, th_ff) da A1.

## ĝ — provenienza e separazione (vincoli 3a, 3b)
Update rule CANONICA, replicata verbatim da `benchmark_adaptive_homeostasis.py:215-219` (incluso guard `|a_prev| > 1e-6`), costanti da righe 106-109: `LR_GAIN=0.05, GAIN_HAT_MIN=0.5, GAIN_HAT_MAX=10.0, GAIN_HAT_INIT=1.0`:
```
x_pred = x_prev + ĝ·a_prev ;  pred_err = x_t − x_pred
ĝ += LR_GAIN·pred_err·a_prev ;  ĝ = clip(ĝ, GAIN_HAT_MIN, GAIN_HAT_MAX)
```
Replica (classe `GainEstimator`) e non import diretto perché nel canonico l'update è inline in `act()` e accoppiato allo step del cluster; il vincolo 3b impone ĝ come organo separato. **Unit test di equivalenza (parte dell'harness, eseguito PRIMA dell'esperimento):** `AdaptiveSymbiontAgent` canonico (import con guard `__main__`, sicuro) pilotato su sequenza x fissa (rng seed 0, 500 step); stesse coppie (x_t, a_t) date a `GainEstimator`; assert di uguaglianza ESATTA (==, nessuna tolleranza) della traiettoria ĝ passo-passo. Fallimento del test = STOP.
**Bias dichiarato:** la rule canonica assume integratore puro; su impianto con leak, pred_err contiene (ρ_x−1)·x_t + d + ε → ĝ stimerà un valore ≠ g=1. La rule resta verbatim (provenienza vincolata); la traiettoria empirica di ĝ viene riportata nel report M1, non corretta.
**Separazione:** ĝ stima il guadagno dell'impianto dal canale pred_err/azione e scala `a_ff`; w stima l'affidabilità del segnale dai momenti EMA segnale/esito e scala la fiducia nel feedforward. Stati disgiunti, zero variabili condivise; unico punto di contatto: la formula di composizione `a = clip(a_R + w·a_ff(ĝ), −1, 1)`. ĝ attivo e identico in A1/A2/A3.

## Ipotesi (STRUTTURALI, FISSE all'OK)
- **H1:** esiste σ* finito nello sweep con costo(A1) ≥ costo(R) (incrocio, interpolazione lineare tra livelli adiacenti).
- **H2:** firma del bias di loop: a σ=0, `w_A2(dieta controllata) < 0.33 × w(stesso stimatore, dieta osservativa su X_open)`. Verifica causale a due diete OBBLIGATORIA: stessa regressione EMA offline su (a) traiettoria x di A1 in loop chiuso, (b) X_open — replica di `verify_closedloop_bias.py` sulle classi reali.
- **H3:** costo(A3) ≤ 1.05×costo(R) su TUTTO lo sweep, E costo(A3) ≤ 1.10×costo(A1) a σ=0.

## Criteri di nullità
R degenere (|azione| media < 0.001 o > 0.999 sui seed di test a σ=0); suite test rossa al pre-run; Var(X_open) non misurabile. Criterio di fallimento M0 ereditato: se A1 non incrocia mai R nemmeno a σ=4 → ambiente inadeguato, NON "tesi confermata".

## Ambiente di esecuzione (pinnato)
Python 3.11.15 · numpy 2.4.6 · matplotlib 3.11.0 · Linux (container Claude Code) · 1 core, nessun parallelismo.
Runtime stimato: ~5 min (misurato 81 µs/step per SymbiontAgent.act; 1176 rollout × T=3000).

## Artefatti e logging
`experiments/degradation_sam/`: `harness_m1.py` (ambiente+agenti+ĝ+test equivalenza), `results_raw.csv` (una riga per σ×seed×agente: costo, danno, sforzo, w medio, ĝ medio), `two_diet_verification.csv`, `degradation_sam_results.png` (due pannelli, convenzioni di `degradation_results.png`: costo vs σ con IC95%; w empirico vs teoria di Wiener `V_cal/(V_cal+σ²)`), `run_log.json` (PipelineStats-style: durata, seed, errori per run). PNG `*_simulation_output.png` MAI nei commit (regola Gate 1a-4).

## Attribuzione
Prereg compilata da Claude Code su template §4.2 (Claude chat) + vincoli 3a–3d decisi da Mirko (Gate 1a). Scarti 1–3 proposti da Claude Code, in attesa di decisione di Mirko al Gate 1b.
