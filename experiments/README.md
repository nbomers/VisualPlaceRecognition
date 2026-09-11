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

Stand 2026-09-11, R@1 bei 25 m:

| Encoder | voll | pca512 | Δ | erklärte Varianz |
|---|---|---|---|---|
| clip (512) | 0.073 | 0.074 | +0.001 | 100 % (Kontrolle) |
| eigenplaces (2048) | 0.484 | 0.481 | −0.003 | 86 % |
| mixvpr (4096) | 0.426 | 0.408 | −0.018 | 62 % |
| anyloc (4096) | 0.204 | offen | | |
| megaloc (8448) | 0.568 | offen | | |

EigenPlaces verliert auf einem Viertel der Breite praktisch nichts;
`eigenplaces_pca512` schlägt MixVPR mit vollen 4096 Dimensionen. Die
Rangfolge hängt nicht an der Breite. Whitening- und Adapter-Zeilen auf 512
stehen noch aus.

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
