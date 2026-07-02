# Pre-registrazione — Esperimento: degradazione del segnale anticipatorio
**Data:** 2026-07-02 · **Modello:** M0 (minimale, dichiaratamente giocattolo — NON il codebase SAM)
**Origine:** arco Symbiont 10/06/2026 — gap flaggato: il vantaggio 5× esocentrico assume segnale oracolo.

## Domanda
Il vantaggio del monitor anticipatorio sopravvive quando il segnale di rischio è imperfetto?
E un peso di fiducia adattivo stimato online (ŵ — cugino di ĝ: stessa famiglia, organo diverso)
garantisce degradazione graduale invece di collasso?

## Ambiente M0
- Hazard latente h(t): decadimento ρ=0.9 + burst poissoniani (p=0.02/step, magnitudine ~Exp(2.0)).
- Danno per step: max(0, h − 1.0). Intervento a∈[0,1] smorza hazard E burst in arrivo (efficacia e=0.8).
- Costo totale = danno + c·azione (errore + sforzo insieme, come nell'arco di giugno). c=0.10.
- Segnale anticipatorio: s(t) = h(t+3) + η, η~N(0, σ). Sweep σ = 0 → 4.0.

## Agenti
- **R** (reattivo): agisce su h(t) osservato. Tuning proprio via grid su seed di tuning separati.
- **A1** (anticipatorio ingenuo): agisce su s(t), fiducia piena. Tuning a σ=0, poi parametri CONGELATI.
- **A2** (anticipatorio + ŵ): blend a = ŵ·anticipatorio + (1−ŵ)·reattivo, con ŵ = pendenza di
  regressione online (EMA, α=0.01) tra segnale passato e hazard realizzato, clip [0,1].
  Dichiarato: A2 ha strutturalmente il fallback reattivo — la tesi testata è esattamente
  "fiducia adattiva + fallback ⇒ degradazione graduale", non "stessa architettura".

## Ipotesi (falsificabili, decise prima di eseguire)
- **H1:** il vantaggio di A1 su R decade con σ e attraversa lo zero a un σ* finito
  (soglia di crossover: oltre, la profezia rumorosa è PEGGIO di nessuna profezia).
- **H2:** A2 non scende mai sotto il pavimento reattivo: costo(A2) ≤ costo(R)·1.05 su tutto lo sweep,
  restando ≈A1 a σ→0.
- **Predizione secondaria (teoria):** ŵ empirico deve seguire la shrinkage di Wiener
  w(σ) ≈ Var(h)/(Var(h)+σ²). Se non la segue, lo stimatore è rotto anche se i costi tornano.

## Criteri di fallimento (dichiarati ora)
- Se A1 non incrocia mai R nemmeno a σ=4: l'ambiente M0 non contiene il meccanismo di
  crossover → il modello è inadeguato, NON "tesi confermata".
- Se il tuning di R degenera (azione sempre 0 o sempre 1): costi non comparabili, esperimento nullo.

## Metodo
30 seed di test appaiati (stessi burst e stesso rumore per i tre agenti), 8 seed di tuning separati,
T=3000 step, burn-in 200. Report: media ± IC95%, curve costo vs σ, σ* con incertezza, ŵ vs teoria.
