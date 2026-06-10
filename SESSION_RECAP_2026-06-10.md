# Session Recap — Symbiont Architecture
> Data: 10 giugno 2026 — Arco di benchmark del 7–9 giugno + verifica finale

---

## 1. Cosa è cambiato dal recap del 28 maggio

Il recap precedente (`SESSION_RECAP.md`) chiudeva con Step 5 (multi-agent, 33/33
test) e una VISION per Step 6: "il DES smette di simulare il mondo — lo legge".

Tra il 7 e il 9 giugno è stato prodotto un arco di **9 benchmark
pre-registrati e falsificabili** (13 commit, ~6.000 righe), tutti con lo
stesso formato: ipotesi scritta PRIMA di vedere i dati, sanity check che
deve passare prima di interpretare i risultati, esito riportato anche
quando è negativo o parziale. Nessun test sul codice architetturale (Step
1–5) è stato toccato — questi benchmark sono harness esterni che usano
`MemoryCluster`/`EndocrineSystem` da `sam-multiagent-v0/` così come sono.

L'arco parte da una domanda pratica ("il cortisol-damping ripara una
perturbazione di guadagno meglio di Q-learning?") e finisce per testare
direttamente la VISION di Step 6 (segnale esocentrico vs endocentrico).

---

## 2. L'arco di benchmark, in ordine

| # | Script | Domanda | Esito |
|---|--------|---------|-------|
| 1 | `benchmark_homeostasis_vs_reward.py` (v1→v4) | Cortisol-damping ripara gain-shift ×4 meglio di Q-learning? | **3 esiti separati**: EHD più preciso (0.04 vs 0.71, p<0.0001), Q più affidabile (0/30 vs 10/30 mai-recuperato), velocità non determinabile (shock iniziali divergono 0.10 vs 1.47) |
| 2 | `benchmark_adaptive_homeostasis.py` | Una stima online ĝ (da prediction error) ripara la fragilità 10/30? | **Sì** — adaptive 0/30 mai-recuperato, precisione preservata (0.040 vs Q 0.706, p<0.0001). Q resta più veloce a convergere (4.0 vs 15.7 step) |
| 3 | `benchmark_setpoint_shift.py` | ĝ generalizza a perturbazioni di **setpoint** (non guadagno)? | **Falsificato per il caso A** — il fisso già recupera in 3 step deterministici; ĝ è irrilevante per questo tipo di perturbazione. Confine del meccanismo confermato: ĝ è uno stimatore di guadagno, non un correttore universale |
| 4 | `benchmark_delay_tolerance.py` | EHD/ĝ/Q sono fragili a ritardi di attuazione D∈[0,50]? | **Non determinabile** in gran parte — tanh+cortisol assorbono il ritardo (0/30 a ogni D), ĝ assorbe l'amplificazione via clip ±1. Confound nella metrica baseline-propria oltre D=0, dichiarato esplicitamente. Nota metodologica: serve un task di regolazione, non di inseguimento, per uno stress test reale |
| 5 | `benchmark_frequency_response.py` | ĝ è inerte o dannoso su disturbi sinusoidali a 13 frequenze? | **Falsificato nella direzione opposta** — ĝ-adattivo batte il fisso a TUTTE e 13 le frequenze (guadagno 0.475–0.668 vs 0.650–0.882). Q amplifica ovunque (1.968–2.643) |
| 6 | `benchmark_frequency_robustness.py` | La dominanza di ĝ ad alta frequenza regge ad ampiezze diverse e vicino a Nyquist? | **Sì** — confermata su DIST_AMP∈{0.1,0.3,0.6} e f fino a 0.480 |
| 7 | hardening (commit `5c51d2e`) | Il vantaggio di ĝ è solo "più azione" (Fisso×2)? | **Solo a bassa frequenza** (f<0.20, ĝ→0.5, equivalente a Fisso×2 con scarto <0.004). Ad alta frequenza (f≥0.25) ĝ→2.7, l'azione si RIDUCE a 0.28× del fisso e l'errore migliora comunque — non è un trade, è un gain-schedule emergente vicino al polo negativo |
| 8 | `benchmark_frequency_temporal.py` | ĝ agisce come filtro passa-basso sulla propria azione? | Diagnostica a tracce singole (f=0.02 e f=0.40), supporto qualitativo all'ipotesi del filtro emergente |
| 9 | `benchmark_exocentric_des.py` | Il DES fa qualcosa di non riducibile a una costante quando il rischio viene dal **mondo** invece che dal proprio errore? | **Sì, con riserva strutturale** — vedi sezione 3 |

---

## 3. Il risultato centrale: il DES esocentrico

Tutta la serie 1–8 condivide un fatto scomodo, isolato esplicitamente nel
benchmark 9: nei task di regolazione standard, `risk = min(|error|/2, 1)`
è una funzione monotona dell'errore già catturata dalla nonlinearità
`tanh`. Il canale endocrino (`k = 1 − 0.5·cortisol`) finisce per oscillare
in una banda strettissima e **non aggiunge nulla misurabile** rispetto a
una costante.

### Verifica di oggi (ablazione, riprodotta)

Ho rieseguito `/tmp/ablation_des.py` (lo script che ha generato i numeri
citati nel preregistro di `benchmark_exocentric_des.py`) per confermarli
prima di considerarli un fatto acquisito:

```
TEST 1 — varianza di k(t) durante regolazione (10 seed, agente EHD):
  f=0.02: k mean=0.8571  std=0.0219  min=0.7989  max=0.8903
  f=0.40: k mean=0.8803  std=0.0036  min=0.8679  max=0.8900

TEST 2 — EHD vs P-puro a k costante = media osservata:
  f=0.02: gain EHD=0.8634±0.0044  gain P-fisso=0.8611±0.0044  delta=0.0023
  f=0.40: gain EHD=0.6580±0.0027  gain P-fisso=0.6581±0.0027  delta=0.0001
```

**Confermato**: a parità di seed, il delta tra EHD e un controller-P con k
costante è sotto la deviazione standard del rumore (±0.0044). Il sistema
endocrino, per come è cablato oggi (rischio endocentrico), è
indistinguibile da una costante. Questo non è un bug isolato — è la
premessa esplicita su cui è stato costruito il benchmark 9.

### Cosa cambia con il segnale esocentrico

Il benchmark 9 ribalta la sorgente del rischio: invece di `|error|`, il
DES riceve un segnale di contesto **esterno** (raffiche di crisi a
incertezza moltiplicativa, dove la cautela preventiva è davvero ottimale).
6 contendenti, 30 seed, Welch one-tailed, RMS errore (più basso = meglio):

```
ORACOLO    0.0523  (upper bound teorico)
EHD-EXO-L  0.0621  DES esocentrico + anticipo
P-FISSO    0.0626  miglior guadagno fisso (compromesso ottimo)
EHD-EXO-0  0.0791  DES esocentrico simultaneo (nessun anticipo)
VARSCHED   0.2100  gain scheduling reattivo standard (SOTA)
EHD-ENDO   0.3109  DES endocentrico — l'architettura ATTUALE, la PEGGIORE
```

- **H2** (segnale esterno > reagire al proprio errore): NON falsificata — 5.0×
- **H3** (anticipo > simultaneo): NON falsificata
- **H4** (regge vs gain scheduling SOTA): NON falsificata — 3.4× meglio di VARSCHED
- **H1** (batte il fisso): tecnicamente falsificata — ma per un **limite
  strutturale dichiarato**: `k = 1 − 0.5·cortisol ∈ [0.5, 1.0]`, mentre
  l'ottimo dell'oracolo è k_crisis≈0.30. Il DES non può scendere lì per
  costruzione.
- **H5 (costruttiva)**: allargare il range (`cortisol_gain` 0.5→0.7, un
  solo parametro) porta RMS a 0.0546 — batte il fisso e sfiora l'oracolo.

**Lettura onesta**: il DES endocentrico (architettura attuale, Step 1–5)
non fa nulla che una costante non faccia — confermato due volte, in due
ambienti diversi. Ma il meccanismo cortisol→k *non è inerte di natura*: è
inerte perché alimentato da un segnale ridondante. Con un segnale esterno
indipendente e un range allargato (modifica di un parametro, non
un'architettura nuova), lo stesso meccanismo recupera quasi l'intero
margine teorico. Questo è esattamente ciò che VISION.md (Step 6) proponeva
— ed è la prima volta nell'intero arco che l'ipotesi "il DES serve a
qualcosa" non viene falsificata.

---

## 4. Cosa dice questo arco sul metodo, non solo sui risultati

Tre volte in tre giorni un sanity check pre-registrato ha bloccato un
risultato prima che diventasse un'interpretazione sbagliata:
- Benchmark 1: il gate di convergenza assoluto era confuso da oscillazioni
  pre-shift → sostituito con plateau_ratio (v3).
- Benchmark 4: confound nella baseline-propria oltre D=0 → dichiarato
  esplicitamente invece di essere nascosto in una media.
- Benchmark 9: la prima versione (rumore additivo) faceva perdere
  l'ORACOLO contro il fisso — sanity "l'oracolo deve vincere" fallito →
  ambiente corretto a incertezza moltiplicativa, ipotesi invariate.

In tutti e tre i casi l'esito non è stato "aggiustare finché non torna",
ma "il check ha rivelato un difetto nel banco, non nell'ipotesi" —
documentato nel codice e nei commit.

---

## 5. Stato del repository

```
9 nuovi benchmark + 2 blog post, 13 commit, 7-9 giugno 2026
sam-multiagent-v0/: invariato (33/33 test, nessuna modifica architetturale)
Branch: claude/project-status-recap-Yv5qv
```

---

## 6. Prossimi passi possibili

| Priorità | Cosa | Perché |
|----------|------|--------|
| Alta | Implementare H5 (cortisol_gain 0.5→0.7) come modifica reale a `endocrine_system.py` e ri-validare su benchmark 1–8 | È l'unico cambiamento che rende il DES misurabilmente utile, ed è un parametro, non un redesign |
| Alta | Blog post sul benchmark 9 (esocentrico) | È il primo risultato positivo dell'intero progetto — merita la stessa cura editoriale del post di falsificazione |
| Media | Step 6 reale: collegare `risk` a un segnale di contesto non derivato dall'errore (sensori, come da VISION.md) | Il benchmark 9 è una simulazione del segnale esocentrico; il prossimo passo è un segnale vero |
| Bassa | BitNet, adversarial testing, RAG-Cortex (invariati dal recap precedente) | Non bloccanti, non toccati da questo arco |

---

*Fine recap — 10 giugno 2026*
