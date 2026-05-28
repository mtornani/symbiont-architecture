# Step 5 Brief — Symbiont Architecture: Multi-Agent Coordination

> Documento completo per Claude Code. Contiene contesto, analisi architetturale,
> spec di implementazione e suite di test. Non è necessario leggere altro.

---

## 1. Contesto del Progetto

**Repository:** `symbiont-architecture`
**Framework:** Small Action Models (SAMs) — etica emergente da homeostasi fisiologica interna,
non da regole esterne. Il meccanismo chiave è **Exocentric Homeostatic Deliberation (EHD)**:
i setpoint interni sono funzioni dinamiche dello stato del mondo esterno.

### Step completati (tutti i test passano)

| Step | Cartella | Cosa fa | Test |
|------|----------|---------|------|
| 1 | `sam-neuron-v0` | Singolo neurone ternario con cortisolo/dopamina e EHD | 3/3 |
| 2 | `sam-cluster-v0` | Cluster 4 neuroni, DES condiviso, ossitocina intra-cluster | 5/5 |
| 3 | `sam-learning-v0` | Plasticità Hebbiana modulata dagli ormoni | 7/7 |
| 4 | `sam-memory-v0` | Consolidamento STM→LTM via sonno (melatonina) | 8/8 |

**Totale: 23/23 test PASS.**

### Pattern architetturale invariante (rispettalo sempre)

Ogni step ha i propri file indipendenti. Il commento all'inizio di ogni modulo recita:

```
SAFETY: Does NOT import from or modify any previous step.
```

Lo Step 5 seguirà la stessa regola: la cartella `sam-multiagent-v0/` contiene
copie dei file necessari da `sam-memory-v0/`, non import cross-step.

---

## 2. Analisi dell'Ossitocina Attuale (Step 2–4)

Ho letto l'intera chain degli step. Ecco dove e come vive oggi l'ossitocina,
con riferimenti precisi alle righe.

### `sam-cluster-v0/endocrine_system.py`

**Setpoint EHD (righe 63–68):**
```python
safety = 1.0 - world.risk
target_oxytocin_sp = 0.05 + 0.45 * safety * world.reward
# Sale solo se rischio è basso AND reward è alto — richiede sicurezza + contesto positivo
self._oxytocin_setpoint += self.alpha * (target_oxytocin_sp - self._oxytocin_setpoint)
```

**Decay (riga 87):**
```python
self._oxytocin += self.tau * (self._oxytocin_setpoint - self._oxytocin)
# Nessuna cross-inibizione col cortisolo (a differenza della dopamina che ha:
# dopamine_target = dopamine_setpoint * (1 - inhibition * cortisol))
```

### `sam-cluster-v0/endocrine_neuron.py`

**Secrezione basale al firing (righe 64–74):**
```python
if fired:
    cort_delta = 0.05 * input_risk
    dopa_delta = 0.05 * input_reward
    oxy_delta  = 0.01   # base fissa per ogni firing
```

### `sam-cluster-v0/cluster.py`

**Bonus di coordinazione intra-cluster (righe 64–71):**
```python
n_fired = sum(fired_list)
if n_fired > 1:
    coordination_bonus = 0.05 * n_fired
    self.des.receive_contribution(0.0, 0.0, coordination_bonus)  # coordinazione → boost
elif n_fired == 1:
    self.des.receive_contribution(0.0, 0.0, -0.02)              # isolamento → penalità
```

### Gap critico

L'ossitocina è completamente **locale a un singolo DES**. Il canale sociale esiste
ma il segnale non esce mai dal cluster. Lo Step 5 aggiunge il layer che fa fluire
l'ossitocina **tra** cluster distinti.

---

## 3. Decisioni Architetturali (già prese — non riaprire)

**D1 — Base dei cluster: `MemoryCluster` da Step 4.**
Porta dentro plasticità + consolidazione → stack biologica completa.
I neuroni usano `TernaryNeuron` con STM/LTM e melatonina.

**D2 — Topologia: fully-connected con un unico parametro `coupling_strength`.**
Tutte le coppie di cluster si vedono. Nessuna adjacency matrix per ora.
`coupling_strength` è la sola knob configurabile per la forza del segnale sociale.

---

## 4. Meccanismo di Coordinazione Inter-Cluster

### Segnale: `oxytocin_broadcast`

Ogni cluster emette un segnale sociale proporzionale alla sua ossitocina,
attenuato dal proprio cortisolo. Un cluster stressato non contagia stress
agli altri — ma riduce la sua capacità di *trasmettere* ossitocina:

```
broadcast_i = oxytocin_i * (1 - cortisol_i)
```

I vicini ricevono la media dei broadcast degli altri:

```
incoming_i = coupling_strength * mean(broadcast_j  for j ≠ i)
cluster_i.des.receive_contribution(0.0, 0.0, incoming_i)
```

### Timing

Il metodo `cluster.step()` chiama `des.step()` internamente,
che flushes i delta accumulati. Il ciclo di coordinazione gira
**dopo** che tutti i cluster hanno completato il loro `step()`.
Le contribuzioni inter-cluster vengono depositate negli accumulatori
del DES e applicate al **ciclo successivo** — ritardo di 1 step,
biologicamente realistico (la segnalazione chimica ha latenza).

```python
# Pseudocodice SymbiontColony.step()
def step(self, worlds, contexts_per_cluster):
    # 1. Forward pass indipendente
    results = [
        cluster.step(world, ctxs)
        for cluster, world, ctxs in zip(self.clusters, worlds, contexts_per_cluster)
    ]

    # 2. Broadcast signals (post-step, valori già aggiornati)
    broadcasts = [
        c.des.oxytocin * (1.0 - c.des.cortisol)
        for c in self.clusters
    ]

    # 3. Distribuisce ai vicini (delta applicati al prossimo step)
    n = len(self.clusters)
    for i, cluster in enumerate(self.clusters):
        others = [b for j, b in enumerate(broadcasts) if j != i]
        incoming = (sum(others) / len(others)) * self.coupling_strength
        cluster.des.receive_contribution(0.0, 0.0, incoming)

    return results, broadcasts
```

### Proprietà del meccanismo

- **Cortisolo non si propaga direttamente** (test_03) — un cluster in stress
  riduce il suo broadcast, non inietta cortisolo negli altri.
- **EHD rimane locale** (test_05) — ogni cluster risponde al proprio `GlobalWorldState`.
  L'ossitocina inter-cluster è una contribuzione esterna, non una sostituzione dell'EHD.
- **Homeostasi collettiva** (test_04) — se un cluster è perturbato, gli altri
  continuano a trasmettere ossitocina stabile, che smorzata dall'EHD del cluster
  perturbato lo aiuta a convergere verso l'equilibrio.

---

## 5. Struttura della Cartella `sam-multiagent-v0/`

```
sam-multiagent-v0/
├── endocrine_neuron.py     ← copia ESATTA da sam-memory-v0/endocrine_neuron.py
│                              (TernaryNeuron con STM/LTM + ConsolidationResult)
├── endocrine_system.py     ← copia ESATTA da sam-memory-v0/endocrine_system.py
│                              (DES con melatonina — 4 ormoni)
├── cluster.py              ← copia ESATTA da sam-memory-v0/cluster.py
│                              (MemoryCluster — nessuna modifica)
├── multiagent.py           ← NUOVO — SymbiontColony
├── environment.py          ← NUOVO — MultiAgentEnvironment (N cluster, scene diverse)
├── simulation.py           ← NUOVO — runner + 10 test + visualizzazione
├── test_results.json       ← generato da simulation.py
└── README.md               ← generato dopo tutto il resto
```

---

## 6. Specifiche di Implementazione

### `multiagent.py` — `SymbiontColony`

```python
class SymbiontColony:
    def __init__(
        self,
        n_clusters: int = 3,
        n_neurons_per_cluster: int = 4,
        n_inputs: int = 8,
        coupling_strength: float = 0.1,
        base_seed: int = 42,
    ) -> None:
        ...

    def step(
        self,
        worlds: List[GlobalWorldState],
        contexts_per_cluster: List[List[NeuronContext]],
    ) -> ColonyStepResult:
        ...
        # Restituisce stato endocrino di ogni cluster + broadcasts

    def get_aggregate_endocrine(self) -> dict:
        # Media di cortisolo, dopamina, ossitocina su tutti i cluster
        ...

    def to_dict(self) -> dict:
        # Serializzazione completa (test_07)
        ...

    @classmethod
    def from_dict(cls, data: dict) -> "SymbiontColony":
        # Deserializzazione (test_07)
        ...
```

**`ColonyStepResult`** (dataclass):
```python
@dataclass
class ColonyStepResult:
    endocrine_states: List[EndocrineState]   # uno per cluster
    fired_lists: List[List[bool]]             # uno per cluster
    broadcasts: List[float]                   # ossitocina emessa da ogni cluster
    incoming: List[float]                     # ossitocina ricevuta da ogni cluster
    plasticity_lists: List[List[float]]
    consolidation_lists: List[List[ConsolidationResult]]
    step: int
```

### `environment.py` — `MultiAgentEnvironment`

Genera `GlobalWorldState` e `NeuronContext` per N cluster in parallelo.
Ogni cluster può vivere in un "bioma" leggermente diverso (risk/reward offset).

**Scenari da includere:**

```
Fase 0   (step   0-199): Calma — tutti i cluster in ambiente stabile
Fase 1   (step 200-299): Perturbazione cluster-0 — alto rischio localizzato
                          cluster-1 e cluster-2 rimangono in ambiente normale
Fase 2   (step 300-349): Rest — tutti dormono, consolidamento
Fase 3   (step 350-549): Recovery — cluster-0 si riprende, coordinazione cresce
Fase 4   (step 550-749): Abbondanza — reward alto su tutti, ossitocina collettiva
Fase 5   (step 750-799): Rest finale — deep sleep
```

Questo permette di osservare: propagazione ossitocina (fasi 3-4),
smorzamento del cluster stressato (fase 1-2), homeostasi collettiva (fase 3).

### `simulation.py`

Struttura identica agli step precedenti:
1. Istanzia `MultiAgentEnvironment` e `SymbiontColony`
2. Loop su tutti gli step, salva `ColonyStepResult` in una lista
3. Genera visualizzazione (vedi sezione 7)
4. Esegue suite di 10 test
5. Scrive `test_results.json`

---

## 7. Visualizzazione — 7 pannelli

```
Panel 1: World state per cluster (risk/reward) — linee colorate per cluster
Panel 2: Ossitocina per cluster nel tempo — convergenza visibile
Panel 3: Cortisolo per cluster nel tempo — perturbazione cluster-0 nella fase 1
Panel 4: Broadcast inter-cluster — quanto segnale sociale emette ognuno
Panel 5: Firing rate aggregato per cluster (rolling window 20 step)
Panel 6: Ossitocina media della colonia (homeostasi collettiva)
Panel 7: Fasi ambientali annotate (barre verticali colorate)
```

---

## 8. Suite di Test — 10/10 target

### test_01 — Creazione N cluster indipendenti
```python
colony = SymbiontColony(n_clusters=3)
assert len(colony.clusters) == 3
# Ogni cluster ha il proprio DES indipendente
des_ids = [id(c.des) for c in colony.clusters]
assert len(set(des_ids)) == 3
```

### test_02 — Propagazione ossitocina inter-cluster dopo evento positivo
```python
# Dopo fase abbondanza (step 550+), ossitocina media colonia > baseline
mean_oxy_abundance = mean(colony ossitocina nei step 550-749)
mean_oxy_baseline  = mean(colony ossitocina nei step 0-199)
assert mean_oxy_abundance > mean_oxy_baseline * 1.2
```

### test_03 — Nessuna propagazione cortisolo diretto
```python
# Durante perturbazione cluster-0 (step 200-299):
# cortisolo cluster-1 e cluster-2 NON supera quello della fase calma + soglia
max_cortisol_1_during_perturb = max(cluster-1 cortisol steps 200-299)
max_cortisol_1_baseline       = max(cluster-1 cortisol steps 0-199)
# I vicini non devono spikarsi proporzionalmente al cluster-0
# (il broadcast di cluster-0 è attenuato dal suo cortisolo alto)
assert max_cortisol_1_during_perturb < 0.6  # soglia ragionevole
```

### test_04 — Homeostasi collettiva dopo perturbazione
```python
# Cluster-0 durante perturbazione ha ossitocina bassa
mean_oxy_0_perturb   = mean(cluster-0 ossitocina step 200-299)
# Cluster-0 in recovery ha ossitocina che risale
mean_oxy_0_recovery  = mean(cluster-0 ossitocina step 350-549)
assert mean_oxy_0_recovery > mean_oxy_0_perturb
```

### test_05 — EHD locale non sovrascritta dalla coordinazione
```python
# Cluster-0 in perturbazione ha cortisolo alto (EHD risponde al suo world)
mean_cort_0_perturb = mean(cluster-0 cortisolo step 200-299)
mean_cort_1_perturb = mean(cluster-1 cortisolo step 200-299)
assert mean_cort_0_perturb > mean_cort_1_perturb * 1.5
# Se EHD fosse sovrascritta, i due sarebbero simili
```

### test_06 — Stato etico collettivo emerge senza regole esterne
```python
# In fase abbondanza (step 550-749): ossitocina collettiva alta,
# cortisolo collettivo basso — "stato cooperativo" senza regole hard-coded
mean_oxy_colony  = mean(mean ossitocina tutti cluster step 550-749)
mean_cort_colony = mean(mean cortisolo tutti cluster step 550-749)
assert mean_oxy_colony > 0.3
assert mean_cort_colony < 0.3
```

### test_07 — Serializzazione/deserializzazione stato multi-agent
```python
state_dict = colony.to_dict()
colony2 = SymbiontColony.from_dict(state_dict)
# Stesso numero cluster, stessi pesi
for i in range(colony.n_clusters):
    np.testing.assert_array_equal(
        colony.clusters[i].neurons[0].weights,
        colony2.clusters[i].neurons[0].weights,
    )
```

### test_08 — Stabilità su 100+ cicli sintetici
```python
# Nessun NaN/Inf nei valori ormonali dopo 800 step
for result in history:
    for state in result.endocrine_states:
        assert 0.0 <= state.cortisol  <= 1.0
        assert 0.0 <= state.dopamine  <= 1.0
        assert 0.0 <= state.oxytocin  <= 1.0
        assert 0.0 <= state.melatonin <= 1.0
```

### test_09 — Logging ormonale aggregato leggibile
```python
# get_aggregate_endocrine() restituisce un dict con chiavi attese
agg = colony.get_aggregate_endocrine()
assert "mean_cortisol"  in agg
assert "mean_dopamine"  in agg
assert "mean_oxytocin"  in agg
assert "mean_melatonin" in agg
assert "std_oxytocin"   in agg
assert "broadcast_mean" in agg
```

### test_10 — Nessuna regressione sugli step precedenti
```python
# I moduli copiati da sam-memory-v0 devono comportarsi identicamente.
# Crea un MemoryCluster standalone (come in step 4) e verifica che
# forward(), learn(), consolidate() producano gli stessi output.
import sys; sys.path.insert(0, "../sam-memory-v0")
from endocrine_neuron import TernaryNeuron as TN_v4
# confronta con TernaryNeuron locale
# (solo forward, senza learn/consolidate per semplicità)
neuron_v4  = TN_v4(seed=42)
from endocrine_neuron import TernaryNeuron as TN_v5
neuron_v5  = TN_v5(seed=42)
np.testing.assert_array_equal(neuron_v4.weights, neuron_v5.weights)
```

---

## 9. Vincoli da Rispettare

1. **Nessun import cross-step** — i file `endocrine_neuron.py`, `endocrine_system.py`,
   `cluster.py` in `sam-multiagent-v0/` sono copie indipendenti, non import relativi.
2. **Nessuna dipendenza nuova** — solo `numpy` e `matplotlib` (già presenti).
3. **Ternario soft mantenuto** — i pesi rimangono in `{-1, 0, +1}`.
4. **EHD locale** — la coordinazione inter-cluster agisce *tramite* il DES locale,
   non sostituendo l'EHD. `receive_contribution()` è il solo canale di ingresso esterno.
5. **Stesso stile di codice** — docstrings brevi, nessun commento ovvio,
   stesso pattern di naming degli step precedenti.

---

## 10. Ordine di Implementazione Consigliato

```
1. Copia i tre file da sam-memory-v0/ → sam-multiagent-v0/
   (endocrine_neuron.py, endocrine_system.py, cluster.py)
   Verifica che siano identici prima di modificare qualsiasi cosa.

2. Scrivi environment.py con MultiAgentEnvironment (6 fasi, N cluster).

3. Scrivi multiagent.py con SymbiontColony + ColonyStepResult.
   Implementa prima step(), poi to_dict()/from_dict(), poi get_aggregate_endocrine().

4. Scrivi simulation.py: loop + raccolta history.

5. Aggiungi la visualizzazione (7 pannelli).

6. Aggiungi i 10 test in fondo a simulation.py.

7. Esegui, verifica output, salva test_results.json.

8. Scrivi README.md (ultima cosa).
```

---

## 11. Primo Comando da Dare a Claude Code

Se vuoi partire step-by-step e mantenere il controllo:

> *"Leggi `sam-memory-v0/cluster.py` e `sam-memory-v0/endocrine_system.py`.
> Poi crea la cartella `sam-multiagent-v0/` e copiaci i tre file base da `sam-memory-v0/`
> senza modificarli. Mostrami la struttura della cartella prima di scrivere codice nuovo."*

Se vuoi che parta direttamente con l'implementazione completa:

> *"Implementa Step 5 seguendo esattamente le specifiche in `STEP5_BRIEF.md`.
> Rispetta l'ordine dei 10 passi nella sezione 10. Non modificare i file
> degli step precedenti. Torna a mostrarmi i risultati dei test prima di
> scrivere il README."*

---

*Fine del brief — tutto il necessario è qui sopra.*
