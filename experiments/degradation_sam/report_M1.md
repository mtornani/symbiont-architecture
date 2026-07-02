# Report M1 — Replica R1–R3 su classi SAM reali
Mandata: 1 · Data: 2026-07-02 · Prereg congelata: `23eb698` · Harness: `791e626` · Suite test: 33/33 verde (pre-run)
Ambiente: Python 3.11.15, numpy 2.4.6, matplotlib 3.11.0 · Durata run: 225.8 s · Test equivalenza ĝ: PASS (500 step, uguaglianza esatta)
Calibrazione: Var(X_open)=0.6354, std=0.7971 (requisito σ_max≥3×std: 4.0 ≥ 2.39 ✓) · Tuning A1 (σ=0, congelato): kf=0.25, thf=0.4
Nullità R: |azione| media = 0.0640 ∈ (0.001, 0.999) → criterio NON scattato.

## Verdetti
| Ipotesi | Verdetto | Numeri |
|---|---|---|
| H1 — crossover σ* finito | **CONFERMATA** | σ* = 1.13, IC95 bootstrap appaiato [1.07, 1.19] (incrocio in 1000/1000 resample) |
| H2 — firma bias di loop | **CONFERMATA** | σ=0: w_ctrl=0.202 < 0.33×w_open=0.330 (w_open=1.000) |
| H3 — pavimento di A3 | **CONFERMATA** | A3≤1.05×R su tutto lo sweep (0 violazioni); a σ=0 A3/A1=0.0349/0.0321=1.087 ≤ 1.10 |

## Sweep (costo medio/step ± IC95%, 30 seed appaiati)
| σ | R | A1 | A2 | A3 | w2 | w3 | teoria | ĝ_A1 |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.0422±0.0030 | 0.0321±0.0023 | 0.0409±0.0029 | 0.0349±0.0027 | 0.15 | 0.89 | 1.00 | 1.02 |
| 0.25 | 0.0422±0.0030 | 0.0328±0.0023 | 0.0411±0.0029 | 0.0356±0.0027 | 0.13 | 0.87 | 0.91 | 1.02 |
| 0.50 | 0.0422±0.0030 | 0.0349±0.0023 | 0.0414±0.0029 | 0.0378±0.0028 | 0.10 | 0.81 | 0.72 | 1.02 |
| 0.75 | 0.0422±0.0030 | 0.0374±0.0024 | 0.0416±0.0029 | 0.0401±0.0027 | 0.08 | 0.65 | 0.53 | 1.02 |
| 1.00 | 0.0422±0.0030 | 0.0405±0.0024 | 0.0417±0.0029 | 0.0414±0.0028 | 0.06 | 0.45 | 0.39 | 1.03 |
| 1.50 | 0.0422±0.0030 | 0.0474±0.0025 | 0.0419±0.0029 | 0.0422±0.0028 | 0.03 | 0.24 | 0.22 | 1.03 |
| 2.00 | 0.0422±0.0030 | 0.0550±0.0029 | 0.0420±0.0029 | 0.0423±0.0029 | 0.02 | 0.15 | 0.14 | 1.03 |
| 3.00 | 0.0422±0.0030 | 0.0800±0.0035 | 0.0421±0.0030 | 0.0423±0.0029 | 0.01 | 0.07 | 0.07 | 1.03 |
| 4.00 | 0.0422±0.0030 | 0.1231±0.0051 | 0.0421±0.0030 | 0.0422±0.0029 | 0.01 | 0.04 | 0.04 | 1.03 |

σ* = 1.13 ≈ 1.42×std(segnale pulito) — su M0 era 1.79 ≈ 2.1×std. La STRUTTURA si trasferisce (incrocio finito,
IC stretto); il numero no, come dichiarato nel piano (§1.2-R1).

## Verifica causale a due diete (H2 — stessa regressione, unica variabile: la dieta)
| σ | w controllato (x di A1) | w osservativo (X_open) | teoria |
|---|---|---|---|
| 0.00 | 0.202 | 1.000 | 1.00 |
| 0.50 | 0.132 | 0.589 | 0.72 |
| 1.00 | 0.070 | 0.304 | 0.39 |
| 2.00 | 0.027 | 0.110 | 0.14 |

Legge replicata sulle classi reali: il monitor che interviene sopprime gli eventi che proverebbero che il suo
segnale funziona (a σ=0 la dieta osservativa recupera esattamente la teoria, quella controllata no).

## Floor e ĝ (integrazione 1b-b + osservazione esplorativa)
Frazione step con ĝ al floor 0.5: 0.000 in TUTTE le celle (σ, agente) → nessun flag (>5% mai raggiunto).
Osservazione esplorativa (etichetta da Gate 1b): ĝ medio ≈ 1.02–1.03 su tutto lo sweep, stabile: bias positivo
piccolo rispetto a g=1, coerente con il leak non modellato dichiarato in prereg ((ρ_x−1)·x correlato con a).
Riportato, non corretto.

## Osservazioni inattese
1. **H3 seconda clausola al margine:** A3/A1 a σ=0 = 1.087 con soglia 1.10 — conferma con margine sottile
   (0.0349 vs limite 0.0353). Il costo di calibrazione di A3 sulle classi reali è più alto che in M0.
2. **H2 con margine più stretto di M0:** w_ctrl=0.202 (M0: 0.113) contro soglia 0.330. Il bias di loop c'è ma
   è meno estremo: la legge nativa reale (tanh + cortisolo) sopprime meno della legge M0 accoppiata ai burst.
3. **w3 sopra la teoria a σ intermedi** (0.81 vs 0.72 a σ=0.5; 0.65 vs 0.53 a σ=0.75): la varianza EMA
   sottostima la varianza del segnale autocorrelato → w3 leggermente sovraconfidente; i costi non ne soffrono.
4. **A2 ≈ R su tutto lo sweep** (max |ΔA2−R| = 0.0013): il collasso di w rende A2 un reattivo de facto anche
   a segnale oracolo — perde quasi tutto il vantaggio anticipatorio (0.0409 vs 0.0321 di A1 a σ=0). È la
   legge R2 letta sui costi.

## Cosa NON è stato fatto e perché
- Nessuna modifica ai moduli sam-* né a file FREEZE (vincolo 4.3.1): tutto vive nell'harness esterno.
- Nessuna correzione del bias di ĝ (decisione Gate 1b: riportare, non correggere).
- Nessun tuning di R (SCARTO-3 approvato: legge nativa senza parametri liberi; A1 ⊇ R verbatim ⇒ parità di
  feedback per costruzione).
- Nessuna estensione dello sweep oltre σ=4 (requisito 3×std già soddisfatto: 2.39 < 4).
- Nessuna analisi oltre prereg salvo il bootstrap per l'IC di σ* (descrittivo, non inferenziale sui verdetti)
  e le osservazioni esplorative etichettate.
- Il costo per M0-vs-M1 del valore σ* non è confrontabile in assoluto (ambienti diversi): confrontata solo
  la struttura.

## Artefatti
`experiments/degradation_sam/`: `prereg_M1.md` (23eb698) · `harness_m1.py` (791e626) · `results_raw.csv`
(1080 righe: σ×seed×agente) · `two_diet_verification.csv` · `degradation_sam_results.png` (2 pannelli,
convenzioni M0) · `run_log.json` (PipelineStats-style: durata, seed, errori=0, versioni).

## Richieste per il Gate 1c
1. OK/KO sul report M1 (chiude la Mandata 1).
2. Se OK: la Mandata 2 richiede la risoluzione di `{PATH_OB1}` — repo non in scope in questa sessione;
   servono URL/accesso al gate M2.

## Attribuzione
Eseguito da Claude Code: harness, run, analisi, report. Deciso da Mirko: scarti 1–3, integrazioni a–d,
ipotesi congelate al Gate 1b. Proposto da Claude chat: template prereg §4.2, convenzioni M0 (Appendici A–C).
