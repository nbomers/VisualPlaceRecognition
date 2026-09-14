# Nebenuntersuchungen

Einmalige Messungen, die **nicht** Teil der Pipeline sind. Sie beantworten je
eine Frage, laufen nicht bei `run.py` mit und schreiben nichts nach `results/`.

Hier liegen sie, damit die Zahlen auffindbar bleiben, ohne die Pipeline zu
belasten. Wer sie für eine Ausarbeitung oder Präsentation braucht, kopiert
sich heraus, was er zeigen will.

---

## Semantisches Re-Ranking mit Mapillary-Detections

**Frage:** Trägt Mapillarys Detection-Histogramm Information bei, die im
gelernten Deskriptor noch nicht steckt?

**Antwort: nein.** Gemessen am 2026-09-10, EigenPlaces, 2.000 Anfragen,
Top-10, 23.484 auswertbare Paare, 11.866 abgerufene Bilder.

| | alle Klassen | ohne Autos/Personen |
|---|---|---|
| AUC Detection-Histogramm | 0.5616 | 0.5586 |
| AUC Deskriptor (Messlatte) | 0.7347 | 0.7347 |

Umsortieren des Top-10, R@1 bei voller Abdeckung (1.286 Anfragen):

| Gewicht | R@1 |
|---|---|
| 0.00 | 0.7551 (Ausgangslage) |
| 0.05 | 0.7558 (bestes Ergebnis) |
| 0.50 | 0.7496 |
| 1.00 | 0.6998 |

Der beste Wert liegt 0,0007 über der Ausgangslage — bei 1.286 Anfragen ist
das eine einzige. Die R@1-Spalte ist bei jedem Gewicht flach oder fallend.

**Deutung:** Das Signal existiert (AUC 0,56 liegt messbar über 0,5), ist aber
**redundant**. Der gelernte Deskriptor kodiert die Szenensemantik bereits;
das Histogramm ist eine gröbere, handgemachte Projektion davon. Semantisches
Re-Ranking ist damit widerlegt, nicht bloß ungeprüft.

Nebenbefunde:

- Detection-Abdeckung 84,7 % der Anfragen, 90,1 % der Kandidaten
- Transiente Klassen auszublenden half **nicht** (0.5616 → 0.5586)
- Fehlende Detections müssen neutral gewertet werden, nicht als Ähnlichkeit
  0 — sonst bestraft die Messung fehlende Daten statt falscher Orte

### `detection_rerank.py`

Die Messung selbst.

```bash
python experiments/detection_rerank.py --n-queries 2000 --top-k 10
python experiments/detection_rerank.py --no-fetch      # nur aus dem Cache
```

Zieht Anfragen, bei denen im Top-k richtige **und** falsche Kandidaten
stehen — nur dort gibt es etwas zu unterscheiden. Holt die Detections, baut
IDF-gewichtete Histogramme und misst AUC sowie das tatsächliche Umsortieren.
Rund 20.000 API-Aufrufe, etwa zwei Minuten.

Die Detections landen in `cache/detections.jsonl` (gitignored). Ein zweiter
Lauf holt nur nach, was fehlt.

### `detection_probe.py`

Vorstufe: wie gut sind die Bilder überhaupt mit Detections abgedeckt? War
ursprünglich die letzte Zelle in `notebooks/03_image_download.ipynb` und ist
von dort herausgelöst worden — eine einmalige Erhebung, die mit dem Download
nichts zu tun hat.

```bash
python experiments/detection_probe.py          # gespeicherten Befund zeigen
python experiments/detection_probe.py --neu    # neu messen
```

Ergebnis in `detections_probe.json`: 500 Datenbankbilder, 94 % mit
Detections, Median 145 je Bild, 165 verschiedene Klassen. Die 94 % waren
leicht optimistisch — die größere Stichprobe oben kam auf 85 bis 90 %.

---

## Gemeinsame Deskriptorbreite (PCA-512)

**Frage:** Wieviel von MegaLocs Vorsprung ist Können, wieviel nur Breite?
Und hängt der Adapterschaden an der Parameterzahl (d×d) oder am Encoder?

### `pca_reduce.py`

Schreibt `{method}_pca512` und `{method}_pcaw512` (mit Whitening) als
eigene Encoder — Embeddings, Metadaten, Fingerabdruck — genau so, wie 04 es
täte. Danach laufen 05 bis 08 unverändert darüber. PCA nur auf `train`
angepasst, 50.000 Zeilen, randomisierte SVD.

```bash
python experiments/pca_reduce.py                    # alle konfigurierten Varianten
python run.py --method derived --adapter all        # dann die Pipeline
```

Stand 2026-09-12, alle 34 Zeilen. R@1 bei 25 m, Baselines ohne Adapter:

| Encoder | voll | pca512 | pcaw512 | pcaw volle Breite |
|---|---|---|---|---|
| clip (512) | 0.073 | 0.074 | **0.105** | — |
| anyloc (4096) | 0.204 | 0.175 | 0.263 | **0.321** |
| mixvpr (4096) | 0.426 | 0.408 | 0.424 | — |
| eigenplaces (2048) | 0.484 | 0.481 | **0.507** | 0.459 |
| megaloc (8448) | 0.568 | 0.545 | 0.541 | — |

Adapterschaden, also `linear` minus `none`:

| Encoder | voll | pca512 | pcaw512 | pcaw volle Breite |
|---|---|---|---|---|
| clip | +0.050 | +0.047 | +0.009 | — |
| anyloc | +0.130 | +0.130 | +0.045 | **−0.029** |
| mixvpr | −0.063 | −0.038 | −0.058 | — |
| eigenplaces | −0.040 | −0.043 | −0.069 | −0.037 |
| megaloc | **−0.126** | **−0.127** | **−0.125** | — |

**Drei Befunde.**

1. **Die Rangfolge hängt nicht an der Breite.** Bei 512 gewhitent:
   megaloc 0.541 > eigenplaces 0.507 > mixvpr 0.424 > anyloc 0.263 >
   clip 0.105 — dieselbe Reihenfolge wie bei voller Breite. MegaLocs
   Vorsprung ist Können, nicht Dimension; 8448 → 512 kostet 0.023.

2. **Whitening rettet AnyLoc und hilft CLIP.** AnyLoc +57 % auf voller
   Breite — VLAD-Deskriptoren sind stark anisotrop (Jégou & Chum 2012),
   und die Pipeline hatte sie ohne Whitening verglichen. CLIP +44 %. Bei
   den VPR-trainierten Encodern auf 512 neutral bis leicht positiv
   (`eigenplaces_pcaw512` schlägt seine eigene 2048er-Baseline), auf voller
   Breite negativ, weil die kleinsten Hauptrichtungen Rauschen verstärken.

3. **Der Adapterschaden hängt nicht an der Parameterzahl — und der
   Adaptergewinn war Whitening.** MegaLoc verliert bei 8448 und bei 512
   dasselbe; die Vermutung „Schaden wächst mit der Dimension" war Zufall.
   Und nach Whitening ist der Gewinn bei CLIP und AnyLoc fast weg: der
   Adapter hatte per Gradientenabstieg gelernt, was die geschlossene Formel
   besser kann. Kernsatz: **ein trainierter linearer Adapter fügt nichts
   hinzu, was ein festes Whitening nicht schon liefert.**

Konsequenz: Whitening gehört für AnyLoc in `04` (VLAD ohne Whitening ist
unüblich), nicht als allgemeiner Adapter — bei EigenPlaces auf voller Breite
schadet es.

---

## Datenbankdichte

**Frage:** Scheitert das System an Osnabrück oder an zu wenig
Referenzmaterial? Die Datenbank sind 15 % der Sequenzen; 36 % der Anfragen
haben darin kein Bild im Umkreis von 25 m.

### `database_density.py`

Nimmt `train` stufenweise zur Datenbank dazu (0 / 25 / 50 / 75 / 100 % der
train-Sequenzen) und trägt Recall gegen die Dichte auf. Nur für Baselines —
der Adapter wurde auf `train` trainiert, dort wäre es Leakage.

```bash
python experiments/database_density.py --method eigenplaces
```

Ergebnis in `results/database_density_{method}.json` und `.png`.

Whitening **und** Dichte zusammen (`--method eigenplaces_pcaw512` bzw.
`megaloc_pcaw512`, volle train-Referenz): EigenPlaces **0.715**, MegaLoc
**0.778**, R@5 0.845. Das sind die höchsten Zahlen im Projekt — ohne ein
einziges Modell zu ändern.

---

## Recall-Hebel: drei Wege, aus vorhandenen Trefferlisten mehr zu machen

Alle drei bewerten mit `src/evaluation.py`, also exakt wie 07, und erscheinen
in `compare.py` als eigene Zeilen. Gemessen 2026-09-12 auf `eigenplaces_pcaw512`
bzw. der Verkettung.

### `concat_embeddings.py` — Deskriptoren verketten

Zwei Encoder aneinanderhängen, neu normalisieren, als abgeleiteter Encoder
schreiben (`sources:` in der config). Richtet die Quellen über die `image_id`
aus — auf zwei Rechnern gerechnete Encoder halten dieselben Bilder in
verschiedener Reihenfolge.

| | Dim | R@1 | R@5 |
|---|---|---|---|
| eigenplaces_pcaw512 | 512 | 0.507 | 0.641 |
| megaloc_pcaw512 | 512 | 0.541 | 0.654 |
| megaloc (voll) | 8448 | 0.568 | 0.676 |
| **eigenplaces_megaloc_concat** | **1024** | **0.572** | **0.692** |

Die Verkettung schlägt MegaLoc auf voller Breite bei einem Achtel der
Dimensionen. **Das ist die Zeile „bestes System, Einzelbild".**

### `sequence_retrieval.py` — Nachbarframes aufsummieren

Trefferlisten der ±W Nachbarn einer Fahrt mit Dreiecksgewicht summieren.
**Hilft nicht:**

| Fenster | R@1 | R@5 | R@20 |
|---|---|---|---|
| einzeln | 0.507 | 0.641 | 0.727 |
| ±1 | 0.507 | 0.644 | 0.731 |
| ±3 | 0.498 | 0.645 | 0.736 |
| ±5 | 0.488 | 0.641 | 0.739 |

Benachbarte Frames sehen dieselbe Straße und machen denselben Fehler; ab ±5
überspannt das Fenster 33 m, mehr als die 25-m-Schwelle. Sequenzlokalisierung
setzt unabhängige Fehler voraus — die groben Verwechslungen hier sind
kohärent. Derselbe Befund wie bei Clustering und Snap in 08.

### `geometric_verification.py` — Top-k lokal nachprüfen

SuperPoint + LightGlue auf die Top-20, RANSAC gegen eine Fundamentalmatrix,
nach Inliern umsortieren. Der einzige Hebel, der die Fehlerart direkt
angreift: global ähnliche, lokal verschiedene Orte. Braucht die Bilder und
eine GPU — 53.414 × 20 Paare, auf der 3070 grob sechs Stunden, auf CPU
nicht sinnvoll. **Noch nicht gemessen.** Aufruf auf dem GPU-Rechner:

```bash
python experiments/geometric_verification.py --method eigenplaces_megaloc_concat --n-queries 2000
python experiments/geometric_verification.py --method eigenplaces_megaloc_concat --n-queries 0   # alle
```

Erwartung aus der Literatur: +0.05 bis +0.10 R@1. Abhängigkeit:
`pip install git+https://github.com/cvg/LightGlue.git` (steht in
`environment.yml`).
