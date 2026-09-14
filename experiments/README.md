# Nebenuntersuchungen

Einmalige Messungen, die **nicht** Teil der Pipeline sind. Sie beantworten je
eine Frage, laufen nicht bei `run.py` mit und schreiben nichts nach `results/`.

Hier liegen sie, damit die Zahlen auffindbar bleiben, ohne die Pipeline zu
belasten. Wer sie für eine Ausarbeitung oder Präsentation braucht, kopiert
sich heraus, was er zeigen will.

Jedes Skript beginnt mit `from _common import CFG, ROOT, RESULTS` und nutzt
die Bausteine aus `src/` — `src/retrieval.py` für Trefferlisten, lösbar
und Treffer je Anfrage, `src/evaluation.py` für jede Recall-Zahl.

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

## Volle Referenz — das zweite Protokoll

**Frage:** Was leistet das System, wenn es alles Referenzmaterial bekommt,
das es gibt — `train` und `database` zusammen, 279.453 Bilder statt 48.321?

### `full_reference.py`

Sucht blockweise (65.536 Referenzzeilen je FAISS-Index, Top-k zusammen-
geführt) und bewertet mit `standard_evaluations`, vier Ground Truths wie in
07. Nur Encoder ohne Adapter — `train` war das Trainingsmaterial des
Adapters. Zeilen heißen `<name>_fullref`, `compare.py --reference full`
zeigt sie, `bootstrap_ci.py --reference full` rechnet die Intervalle.

```bash
python experiments/full_reference.py                 # alle 18, rund zwei Stunden (MegaLoc 8448 d ist der Posten)
python compare.py --reference full --ci
python experiments/bootstrap_ci.py --reference full
```

Gemessen 2026-09-14, R@1 bei 25 m, 48.177 lösbare von 53.414 (90,2 %):

| Encoder | Dim | R@1 volle Referenz | 95 % | R@1 Benchmark | Zuwachs |
|---|---|---|---|---|---|
| megaloc | 8448 | **0.798** | [0.739, 0.848] | 0.568 | +0.230 |
| megaloc_pcaw512 | 512 | 0.778 | [0.716, 0.831] | 0.541 | +0.237 |
| eigenplaces_megaloc_concat | 1024 | 0.778 | [0.714, 0.832] | 0.572 | +0.206 |
| eigenplaces_pcaw512 | 512 | 0.715 | [0.646, 0.777] | 0.507 | +0.208 |
| eigenplaces | 2048 | 0.701 | [0.630, 0.765] | 0.484 | +0.217 |
| mixvpr | 4096 | 0.653 | [0.575, 0.724] | 0.426 | +0.227 |
| anyloc_pcaw4096 | 4096 | 0.540 | [0.456, 0.624] | 0.321 | +0.219 |
| anyloc | 4096 | 0.435 | [0.343, 0.533] | 0.204 | +0.231 |
| clip_pcaw512 | 512 | 0.290 | [0.202, 0.402] | 0.105 | +0.185 |
| clip | 512 | 0.232 | [0.144, 0.349] | 0.073 | +0.159 |

Alle 18 Zeilen: `python compare.py --reference full`.

**Drei Befunde.**

1. **Jeder Encoder gewinnt rund 0.2** — der Zuwachs ist über die Rangfolge
   fast konstant (+0.16 bis +0.24). Die Referenz hebt alle gleich; sie
   ändert nicht, wer besser ist.
2. **Die Rangfolge bleibt exakt**, alle Nachbarpaare schließen 0 aus:
   clip → anyloc +0.202, anyloc → mixvpr +0.219, mixvpr → eigenplaces
   +0.048 [+0.027, +0.074], eigenplaces → megaloc +0.096 [+0.070, +0.122].
3. **Die Reduktion kostet hier messbar.** MegaLoc 8448 → 512 gewhitent
   −0.019 [−0.024, −0.015], und die Verkettung liegt sicher *unter* MegaLoc
   (−0.019 [−0.027, −0.011]) — im Benchmark-Protokoll war beides innerhalb
   des Rauschens. Whitening hilft EigenPlaces auf 512 auch hier (+0.014
   [+0.006, +0.022]) und schadet auf voller Breite (−0.017).

Für die Ausarbeitung: die Kopfzeile ist 0.798 mit voller Referenz, und der
Benchmark mit 0.568 ist das Protokoll, auf dem Adapter und Varianten
verglichen werden. Beides steht im README.

---

## Konfidenzintervalle per Sequenz-Bootstrap

**Frage:** Welche der Unterschiede in `compare.py` sind belegt, welche
Rauschen? Die 53.414 Anfragen stammen aus 198 Fahrten; aufeinanderfolgende
Frames scheitern gemeinsam. Der binomiale Standardfehler (±0.005 bei R@1)
zählt sie als unabhängig.

### `bootstrap_ci.py`

Zieht 1.000-mal die 198 Query-Sequenzen mit Zurücklegen und rechnet Recall
über die gezogenen Sequenzen — dieselben Ziehungen für alle 36 Zeilen, damit
Differenzen gepaart ausgewertet werden. Distanzen einmal je Anfrage, der
Bootstrap ist Arithmetik auf Sequenz-Summen; 36 Zeilen in rund einer Minute.
Prüft jede Zeile gegen ihre 07-JSON und bricht bei Abweichung ab.

```bash
python experiments/bootstrap_ci.py                   # -> results/bootstrap_ci.json
python experiments/bootstrap_ci.py --reference full  # -> results/bootstrap_ci_fullref.json
python compare.py --ci                               # Intervall neben R@1
```

Gemessen 2026-09-14, R@1 bei 25 m, Seed 42.

**Die absoluten Zahlen sind unsicherer als gedacht.** Die 95-%-Halbbreite
liegt bei ±0.03 (CLIP) bis ±0.10 (MegaLoc) statt ±0.005 — ein Design-Effekt
von 100 bis 330. Grund: die Sequenzen sind extrem ungleich lang (Median 176
Frames, die größte 3.156; sieben Sequenzen stellen 19 % aller Anfragen),
und ob eine lange Fahrt gut oder schlecht läuft, entscheidet den Recall.
Eine einzelne Zahl wie „MegaLoc 0.568" ist deshalb nur auf eine
Nachkommastelle belastbar: [0.475, 0.666].

**Die Differenzen sind eng.** Dieselben Sequenzen sind für alle Encoder
schwer; in der gepaarten Differenz fällt das heraus. Halbbreiten von
±0.005 bis ±0.05.

| Vergleich (b − a) | Diff | 95 % | schließt 0 ein |
|---|---|---|---|
| eigenplaces → eigenplaces_pca512 | −0.003 | [−0.008, +0.002] | ja |
| eigenplaces → eigenplaces_pcaw512 | +0.022 | [+0.011, +0.039] | nein |
| eigenplaces → eigenplaces_pcaw2048 | −0.025 | [−0.042, −0.010] | nein |
| anyloc → anyloc_pcaw4096 | +0.117 | [+0.078, +0.162] | nein |
| megaloc → megaloc_pca512 | −0.023 | [−0.031, −0.017] | nein |
| megaloc → eigenplaces_megaloc_concat | +0.004 | [−0.005, +0.014] | **ja** |
| eigenplaces_pcaw512 → seq3 | −0.009 | [−0.017, −0.001] | nein |
| clip → clip_linear | +0.050 | [+0.038, +0.064] | nein |
| clip_pcaw512 → clip_pcaw512_linear | +0.009 | [−0.002, +0.021] | **ja** |
| anyloc → anyloc_linear | +0.130 | [+0.081, +0.187] | nein |
| anyloc_pcaw4096 → anyloc_pcaw4096_linear | −0.029 | [−0.074, +0.012] | **ja** |
| anyloc_pcaw512 → anyloc_pcaw512_linear | +0.045 | [+0.009, +0.088] | nein |
| eigenplaces → eigenplaces_linear | −0.040 | [−0.064, −0.017] | nein |
| mixvpr → mixvpr_linear | −0.063 | [−0.096, −0.032] | nein |
| megaloc → megaloc_linear | −0.126 | [−0.179, −0.073] | nein |
| megaloc_pca512 → megaloc_pca512_linear | −0.127 | [−0.160, −0.099] | nein |

Alle Paare stehen in `results/bootstrap_ci.json`, auch für R@5/10/20.

**Was die Intervalle überlebt — die sieben Befunde aus `docs/STAND.md`:**

1. *Encoder-Wechsel ist der größte Hebel* — **belegt.** CLIP → MegaLoc
   +0.496 [+0.405, +0.590]; jedes Nachbarpaar der Rangfolge
   (clip < anyloc < mixvpr < eigenplaces < megaloc) schließt 0 aus, auf
   voller Breite wie auf 512 gewhitent.
2. *Referenzdichte* — **nicht geprüft**; das ist ein anderes Experiment
   (`database_density.py`) mit anderer Datenbank. +0.22 liegt weit über
   jeder Halbbreite hier, aber ein eigener Bootstrap fehlt.
3. *Adapter fügt nichts hinzu, was Whitening nicht liefert* — **belegt,
   mit einer Einschränkung.** Ohne Whitening ist der Gewinn bei CLIP und
   AnyLoc klar; nach Whitening schließt er 0 ein (CLIP +0.009, AnyLoc auf
   4096 −0.029). Der Schaden bei den VPR-trainierten Encodern schließt 0
   in allen zwölf Paaren aus. Einschränkung: auf `anyloc_pcaw512` bleibt
   +0.045 [+0.009, +0.088] — 512 gewhitente Komponenten holen aus VLAD
   weniger heraus als 4096, dort hat der Adapter noch etwas zu tun.
4. *Rangfolge und Adapterschaden hängen nicht an der Breite* — **die
   Rangfolge ja** (Punkt 1). Der Adapterschaden bei MegaLoc ist auf 8448
   und 512 gleich groß (−0.126 vs −0.127) und die Intervalle decken sich;
   das ist verträglich mit „gleich", ein Beweis für Gleichheit ist ein
   Bootstrap nicht.
5. *AnyLoc war unfair behandelt* — **belegt.** Whitening auf 4096: +0.117
   [+0.078, +0.162]. Bei EigenPlaces auf voller Breite schadet es: −0.025
   [−0.042, −0.010]. Beide Vorzeichen sind sicher.
6. *Nichts schlägt Top-1* — **belegt, soweit hier messbar.** seq3 liegt
   sicher unter der Einzelbild-Zeile (−0.009 [−0.017, −0.001]). Die
   Verfahren aus 08 sind Lokalisierung, nicht Recall — nicht Gegenstand.
   **Aber: die Verkettung schlägt MegaLoc nicht.** +0.004 [−0.005, +0.014]
   schließt 0 ein. Richtig ist: `eigenplaces_megaloc_concat` ist bei einem
   Achtel der Breite *gleichauf* mit MegaLoc, nicht besser. README und
   Tabelle oben sind entsprechend zu lesen.
7. *Blickrichtung 14,5 %* — eine Zählung, kein Vergleich; nicht Gegenstand.

Ebenfalls bestätigt: 0.484 gegen 0.481 (eigenplaces vs pca512) ist
Rauschen, wie in `docs/STAND.md` vermutet; +0.023 durch `pcaw512` ist es nicht.

---

## Laufzeit und Speicher je Encoder

**Frage:** Trägt `eigenplaces_pcaw512` den Titel „effizientestes Modell"?
Drei Kosten, die getrennt anfallen: Encodieren (einmal je Bild, hängt am
Rückgrat), Suche (je Anfrage, hängt an der Breite), Index (Speicher).

### `timing.py`

```bash
python experiments/timing.py                            # Encodieren + Suche + Index
python experiments/timing.py --skip-encode              # nur Suche und Index
python experiments/timing.py --methods anyloc,megaloc   # auf dem GPU-Rechner
```

Encodieren: 200 feste Bilder aus dem Query-Split (Seed), inklusive Laden
und Dekodieren wie in 04, Aufwärmlauf ausgeschlossen. Suche: FAISS-Flat
über die 48.321 Datenbankzeilen, 1.000 Anfragen in Blöcken von 256, Median
aus fünf Runden. Die JSON (`results/timing.json`) ist mergefähig — je
Encoder ein Eintrag mit Hostname und Gerät, Einträge anderer Rechner
bleiben stehen.

Suche und Index, gemessen 2026-09-14 auf dem M1 Pro (FAISS CPU, 8 Threads):

| Encoder | Dim | ms/Anfrage | Index MB | .npy MB (alle 332k Zeilen) |
|---|---|---|---|---|
| alle `*_pca512` / `*_pcaw512` | 512 | 0.5–0.75 | 94 | 650 |
| eigenplaces_megaloc_concat | 1024 | 1.2 | 189 | 1.300 |
| eigenplaces, eigenplaces_pcaw2048 | 2048 | 1.9 | 378 | 2.601 |
| anyloc, anyloc_pcaw4096, mixvpr | 4096 | 3.2–3.4 | 755 | 5.201 |
| megaloc | 8448 | 6.4 | 1.557 | 10.727 |

Suchzeit und Index wachsen linear mit der Breite: 8448 → 512 ist Faktor 11
bei der Suche und Faktor 16,5 beim Speicher. Bei 53.414 Anfragen macht das
6 s gegen 34 s — auf dieser Datenbankgröße kein Argument. Bei einer
Datenbank in Millionengröße wäre es eines: der MegaLoc-Index läge bei
32 GB je Million Bilder, der 512er bei 2 GB.

Encodieren, gemessen 2026-09-14 auf dem M1 Pro (MPS, 200 Bilder, Batch 64):

| Encoder | Eingabe | Bilder/s | 332.868 Bilder |
|---|---|---|---|
| CLIP ViT-B/32 | 224 px | 145 | 38 min |
| MixVPR | 320 px | 73 | 75 min |
| EigenPlaces | 512 px | 33 | 2,8 h |
| MegaLoc | 322 px | 21 | 4,4 h |
| AnyLoc (RTX 3070, fp16, Batch 4) | 322 px | 15 | 6,3 h |

Der Durchsatz hängt am Rückgrat und der Eingabegröße, nicht an der PCA:
`eigenplaces_pcaw512` encodiert genau so schnell wie `eigenplaces`, die
Projektion ist ein Matrixprodukt.

**Antwort:** „effizientestes Modell" stimmt für Suche und Speicher, nicht
fürs Encodieren — dort ist EigenPlaces (33 Bilder/s bei 512 px) langsamer
als MixVPR (73) und CLIP (145), nur MegaLoc ist langsamer (21). Der
ehrliche Satz ist: *bester Recall je Byte Index*, mit 0.507 bei 94 MB gegen
0.568 bei 1.557 MB.

---

## Recall je Stadtteil

**Frage:** Wo in der Stadt scheitert das System — und liegt es an der
Referenzdichte?

### `recall_by_district.py`

Ordnet jede Anfrage per Point-in-Polygon einem OSM-Stadtteil zu — dieselbe
Gliederung wie die Abdeckungskarte in 01, die Abfrage steht in
`src/districts.py`. Je Stadtteil Anfragen, Sequenzen, Anteil lösbar bei
25 m, R@1 über die lösbaren, Datenbankbilder je km². Zwei Karten
nebeneinander: R@1 und Referenzdichte; Stadtteile unter 100 lösbaren
Anfragen grau.

```bash
python experiments/recall_by_district.py                          # megaloc
python experiments/recall_by_district.py --method eigenplaces_pcaw512
```

Gemessen 2026-09-14, 23 Stadtteile, alle 53.414 Anfragen zugeordnet.
Ergebnis in `results/recall_by_district_<name>.json` und `.png`.

| Stadtteil | Anfragen | Seq. | lösbar | R@1 megaloc | R@1 eigenpl._pcaw512 | DB/km² |
|---|---|---|---|---|---|---|
| Atter | 2.630 | 14 | 91 % | **0.83** | 0.77 | 244 |
| Schinkel | 311 | 4 | 48 % | 0.76 | 0.76 | 700 |
| Nahne | 4.956 | 16 | 88 % | 0.69 | 0.65 | 1.116 |
| Pye | 6.160 | 24 | 98 % | 0.67 | 0.60 | 287 |
| Innenstadt | 7.001 | 39 | 38 % | 0.67 | 0.56 | 2.623 |
| Darum-Gretesch-Lüstringen | 744 | 3 | 83 % | 0.63 | 0.59 | 95 |
| Schölerberg | 829 | 16 | 94 % | 0.62 | 0.57 | 565 |
| Fledder | 1.852 | 13 | 68 % | 0.61 | 0.61 | 599 |
| Voxtrup | 2.967 | 18 | 92 % | 0.59 | 0.53 | 228 |
| Wüste | 6.094 | 36 | 64 % | 0.57 | 0.49 | 1.915 |
| Dodesheide | 2.307 | 14 | 52 % | 0.48 | 0.32 | 412 |
| Kalkhügel | 2.853 | 14 | 63 % | 0.40 | 0.36 | 905 |
| Weststadt | 2.961 | 18 | 55 % | 0.38 | 0.33 | 970 |
| Westerberg | 744 | 14 | 26 % | 0.35 | 0.33 | 381 |
| Hafen | 2.014 | 20 | 54 % | 0.29 | 0.24 | 609 |
| Haste | 2.404 | 15 | 50 % | 0.23 | 0.21 | 206 |
| Hellern | 2.184 | 13 | 70 % | 0.21 | 0.18 | 147 |
| Sonnenhügel | 1.634 | 12 | 7 % | 0.18 | 0.14 | 403 |
| Sutthausen | 414 | 3 | 100 % | **0.07** | 0.06 | 230 |

Grau (unter 100 lösbare): Eversburg 13, Widukindland 73, Gartlage 1,
Schinkel-Ost 0 Anfragen.

**Das Muster — drei Befunde.**

1. **Die Spanne ist enorm: 0.07 bis 0.83 bei ein und demselben Encoder.**
   Der stadtweite Wert 0.568 ist ein Mittel über Stadtteile, die sich um
   den Faktor 12 unterscheiden.
2. **Die Dichte erklärt es nicht — nicht auf dieser Ebene.** Spearman R@1
   gegen Datenbankbilder/km²: ρ = +0.27 (p = 0.27) bei MegaLoc, +0.24 bei
   EigenPlaces. Die Innenstadt hat die dichteste Referenz (2.623/km²) und
   trotzdem nur 38 % lösbare Anfragen; Atter hat ein Zehntel der Dichte
   und den besten Recall. Bilder je km² messen die falsche Größe: was
   zählt, ist die Referenz *an der Straße der Anfrage*, nicht im Stadtteil.
   Der Dichte-Befund aus `database_density.py` (mehr Referenz → +0.22)
   bleibt; er ist auf Stadtteil-Ebene nur nicht sichtbar. Der direkte Test
   ist Recall gegen Nachbarzahl je Anfrage (Punkt 7 in `docs/AUFTRAG.md`).
3. **Die Karte zeigt die Daten, nicht den Encoder.** Die Rangfolge der
   Stadtteile ist bei MegaLoc und EigenPlaces praktisch dieselbe
   (Spearman 0.96). Was einen Stadtteil scheitern lässt, steckt in den
   Aufnahmen:
   - *Haste* (0.23): nur 29 % der lösbaren Anfragen haben überhaupt einen
     Nachbarn, der in dieselbe Richtung schaut; Median drei Jahre Abstand
     zum nächsten Referenzbild.
   - *Hellern* (0.21): Median 338 Tage Abstand — andere Jahreszeit.
   - *Sutthausen* (0.07): 35 Nachbarn im Median, sechs Tage Abstand,
     Blickrichtung passt bei 86 % — und trotzdem landet Top-1 bei 208 von
     414 Anfragen in Hellern, vier Kilometer entfernt. Eine einzige
     Sequenz mit 284 Frames, die ein Wohnviertel mit einem anderen
     verwechselt. Das ist die grobe Verwechslung aus Befund 6, auf der
     Karte sichtbar.

   Stadtteile mit drei Sequenzen (Sutthausen, Darum-Gretesch-Lüstringen)
   zeigen das Schicksal einer Fahrt, nicht des Ortes — dieselbe Lehre wie
   der Sequenz-Bootstrap.

---

## Verwechslungsatlas

**Frage:** Wohin schätzt das System, wenn es falsch liegt — knapp daneben
oder in einen anderen Stadtteil?

### `confusion_atlas.py`

Für jede lösbare Anfrage mit Top-1 jenseits von 25 m ein Pfeil von der
echten zur geschätzten Position auf dem Straßennetz (osmnx, aus dem
Cache), daneben dieselben Pfeile aggregiert nach Stadtteil-Paar. Dazu der
Straßentyp der nächsten Kante je Anfrage. Nur lösbare Anfragen — dort
hatte das System eine Referenz in Reichweite.

```bash
python experiments/confusion_atlas.py          # -> results/confusion_atlas_megaloc.{json,png}
```

Gemessen 2026-09-14, MegaLoc: **14.726 Fehlgriffe unter 34.112 lösbaren.**

**Zwei Sorten Fehler, fast nichts dazwischen.** Quartile des Fehlers
40 / 389 / 3.497 m: 45 % liegen unter 100 m (dieselbe Straße, knapp
jenseits der Schwelle), 44 % über 1 km, nur 11 % dazwischen. 53 % der
Fehlgriffe bleiben im eigenen Stadtteil. Die Verteilung ist bimodal; ein
Median (389 m hier, 1,5 km in `docs/STAND.md` für 08 über alle Anfragen)
beschreibt sie schlecht — die Masse liegt an beiden Enden.

Die häufigsten Paare zwischen Stadtteilen (echt → geschätzt):

| | Paar | n | Median |
|---|---|---|---|
| 1 | Voxtrup → Nahne | 240 | 540 m |
| 2 | Sutthausen → Hellern | 208 | 4.239 m |
| 3 | Hellern → Nahne | 147 | 6.894 m |
| 4 | Wüste → Weststadt | 141 | 63 m |
| 5 | Hellern → Kalkhügel | 139 | 3.590 m |
| 6 | Hafen → Fledder | 120 | 4.446 m |
| 7 | Haste → Nahne | 117 | 7.085 m |
| 8 | Pye → Atter | 113 | 4.748 m |
| 9 | Wüste → Hellern | 110 | 3.071 m |
| 10 | Kalkhügel → Hellern | 94 | 3.704 m |

Kein Paar trägt mehr als 1,6 % der Fehlgriffe: es gibt *keine* dominante
Verwechslung zweier Orte, sondern viele kleine. Nahne und Hellern tauchen
als Ziel am häufigsten auf — Wohn- und Gewerbegebiete am Stadtrand, die
sich gegenseitig ähneln.

**Autobahn ist nicht das Problem.** Das dicke Bündel in der linken Karte
folgt der A30, weil die Autobahn-Sequenzen lang sind — nicht, weil sie
schlechter laufen: R@1 auf der Autobahn 0.570, auf Hauptstraßen 0.576,
sonst 0.565. Autobahn-Fehlgriffe sind sogar *kurz* (Median 56 m gegen
776 m in Wohnstraßen), und unter den groben Fehlern über 1 km stellt die
Autobahn 23 % bei 32 % Anteil an den lösbaren Anfragen. Die groben
Verwechslungen sitzen in den Wohnstraßen (56 % der Fehler über 1 km).

---

## Aggregation in der Lokalisierung — fünf Wege, alle unterlegen

**Frage:** Macht aus der Trefferliste eine bessere Koordinate, wer die
Top-10 mittelt, clustert oder nur bei Einigkeit der Gruppe folgt?

### `localization_aggregation.py`

Schwerpunkt (roh und mit Softmax gespreizt), DBSCAN-Clustering, Snap
(bester echter Treffer der stärksten Gruppe) und Gated (Gruppe nur bei
≥ 70 % Einigkeit), gegen Top-1. Stand bis 2026-09-14 in 08; dort rechnet
jetzt nur noch Top-1.

```bash
python experiments/localization_aggregation.py --method megaloc
```

Gemessen (08, alle 53.414 Anfragen, Anteil unter 25 m / Median):
MegaLoc Top-1 **0.363 / 94 m**, Clustering 0.318 / 444 m, Schwerpunkt
gespreizt 0.267 / 575 m, Schwerpunkt roh 0.206 / 961 m; EigenPlaces Top-1
0.309 / 364 m, Clustering 0.265 / 656 m. Snap und Gated liegen zwischen
Clustering und Top-1, nie darüber. Grund: 44 % der Fehlgriffe sind grobe
Verwechslungen, bei denen die Nachbarn geschlossen am falschen Ort liegen
(Verwechslungsatlas) — Konsens bestätigt dann den Fehler.

---

## Ablehnungskurve — was „weiß ich nicht" bringt

**Frage:** Wie viel besser wird die Antwort, wenn das System bei niedriger
Konfidenz schweigen darf — und welche Konfidenz taugt dafür?

### `rejection_curve.py`

Drei Konfidenzmaße aus der Trefferliste: Ähnlichkeit des besten Treffers,
Marge zu Platz 2, Geschlossenheit der Top-10 (Anteil innerhalb 25 m um
Platz 1). Schwelle absenken, Präzision (Top-1 innerhalb 25 m) gegen
Abdeckung auftragen. Zwei Sichten: lösbare Anfragen und alle.

```bash
python experiments/rejection_curve.py --method megaloc
python experiments/rejection_curve.py --method eigenplaces_megaloc_concat
```

Gemessen 2026-09-14, MegaLoc:

| Sicht | Konfidenz | ohne Ablehnung | bei 80 % Abdeckung | bei 50 % | bei 20 % | AUC |
|---|---|---|---|---|---|---|
| lösbar | Ähnlichkeit | 0.568 | **0.689** | 0.793 | 0.887 | 0.791 |
| lösbar | Marge | 0.568 | 0.618 | 0.724 | 0.860 | 0.741 |
| lösbar | Geschlossenheit | 0.568 | 0.597 | 0.706 | 0.798 | 0.706 |
| alle | Ähnlichkeit | 0.363 | **0.451** | 0.673 | 0.828 | 0.659 |
| alle | Marge | 0.363 | 0.410 | 0.526 | 0.766 | 0.580 |

Die Verkettung liegt gleichauf (lösbar, Ähnlichkeit: 0.695 bei 80 %, AUC 0.790).

**Befund.** Die rohe Ähnlichkeit ist das beste Maß, die Marge das
schlechteste — anders als in der Literatur zu Klassifikatoren üblich. Wer
bei MegaLoc nur Anfragen mit cos ≥ 0.30 beantwortet, ist zu 82 % richtig
und beantwortet 37 % der lösbaren Anfragen; bei cos ≥ 0.20 noch 75 % bei
69 %. Im Betrieb (alle Anfragen, ohne Kenntnis der Referenz) sind es bei
80 % Abdeckung 45 % statt 36 % — die Konfidenz erkennt die unlösbaren
Anfragen nur zum Teil. `locate.py` meldet deshalb die Ähnlichkeit als
Konfidenz.

---

## Schwierigkeitsprofil — woran der Recall je Anfrage hängt

**Frage:** Der Dichte-Befund direkt: fällt R@1 bei wenigen Nachbarn ein?
Und was zählt sonst?

### `recall_by_difficulty.py`

Je Anfrage vier Eigenschaften aus den Metadaten — Nachbarn im Umkreis von
25 m, Tage zum nächsten Referenzbild, ob ein Nachbar in dieselbe Richtung
schaut, ob ein Nachbar vom selben Fotografen am selben Tag stammt — und
R@1 je Klasse über die lösbaren Anfragen.

```bash
python experiments/recall_by_difficulty.py --method megaloc
python experiments/recall_by_difficulty.py --method eigenplaces_pcaw512
```

Gemessen 2026-09-14, MegaLoc, R@1 gesamt 0.568:

| Merkmal | Klasse | n | R@1 |
|---|---|---|---|
| Nachbarn | 1–2 | 1,709 | 0.526 |
| | 3–5 | 3,508 | 0.481 |
| | 6–10 | 3,880 | 0.535 |
| | 11–20 | 6,832 | 0.448 |
| | 21–50 | 14,056 | 0.601 |
| | 51+ | 4,120 | **0.781** |
| Tage zum nächsten | 0–7 | 4,688 | 0.575 |
| | 8–30 | 3,675 | **0.819** |
| | 31–180 | 10,450 | 0.539 |
| | 181–365 | 5,769 | 0.612 |
| | 366+ | 9,366 | 0.473 |
| Blickrichtung passt | ja | 29,157 | 0.653 |
| | nein | 4,955 | **0.071** |
| Selber Fotograf, selber Tag | ja | 3,164 | 0.690 |
| | nein | 30,948 | 0.556 |

EigenPlaces (pcaw512) zeigt dieselben Muster auf niedrigerem Niveau
(`recall_by_difficulty_eigenplaces_pcaw512.json`).

**Befund — drei Faktoren, in dieser Reihenfolge.**

1. **Blickrichtung.** 4,955 lösbare Anfragen (14.5%) haben
   keinen Nachbarn, der in dieselbe Richtung schaut: R@1 0.071. Ohne sie
   läge MegaLoc bei 0.653. Das ist der Befund „Blickrichtung" aus 07,
   je Anfrage.
2. **Zeit.** 8–30 Tage Abstand: 0.819; über ein Jahr: 0.473. Ein
   Jahr kostet mehr als die Hälfte der Nachbarn.
3. **Dichte.** Ab 51 Nachbarn 0.781, darunter zwischen 0.45 und 0.60
   ohne klaren Verlauf. Der Dichte-Befund hält — aber als „viele Nachbarn
   helfen", nicht als „wenige schaden": bei 1–2 Nachbarn ist R@1 nicht
   schlechter als bei 11–20. Die Dichtekurve (+0.22 mit `train`) gewinnt
   nicht nur Nachbarn, sondern auch Blickrichtungen und Zeitpunkte.

---

## Stadtwahl — welche Stadt taugt als nächste?

**Frage:** Bilder je km² sagt wenig; Bamberg hat 25.000/km², weil dort
eine Kampagne die Hauptstraßen abgefahren hat. Was zählt, ist, ob *jede*
Straße ein Bild hat — sonst sind Anfragen aus Wohnstraßen unlösbar, wie
36 % in Osnabrück.

### `city_coverage.py`

Holt die Bildpunkte einer Stadt über dieselben Vector Tiles wie 01 und
misst die **Straßenabdeckung**: Anteil des OSM-Fahrnetzes (nach Länge) mit
einem Bild im Umkreis von 25 m, getrennt nach großen Straßen (bis
secondary) und Wohnstraßen (tertiary, residential, living_street,
unclassified). Dazu Sequenzen, Fotografen-Konzentration, Anteil seit 2022.

```bash
python experiments/city_coverage.py "Mainz, Germany" "Würzburg, Germany"
```

Gemessen 2026-09-14 (Mainz ausstehend — Overpass hatte die Verbindung
verweigert):

| Stadt | Bilder | /km² | Straßen km | gedeckt | große | Wohn | Seq. | Fotografen | größter | seit 2022 |
|---|---|---|---|---|---|---|---|---|---|---|
| Jena | 699.097 | 6.113 | 725 | **99%** | 100% | **99%** | 4.516 | 62 | 51% | 45% |
| Würzburg | 428.847 | 4.895 | 915 | **98%** | 100% | **98%** | 2.004 | 92 | 51% | 28% |
| Halle (Saale) | 919.790 | 6.787 | 1.305 | **92%** | 100% | **91%** | 3.931 | 72 | 30% | 87% |
| Heidelberg | 530.614 | 4.879 | 822 | **87%** | 98% | **83%** | 3.307 | 113 | 33% | 36% |
| Erlangen | 611.724 | 7.943 | 808 | **70%** | 100% | **63%** | 3.925 | 86 | 36% | 49% |
| Osnabrück | 336.168 | 2.808 | 1.327 | **44%** | 94% | **38%** | 1.334 | 57 | 47% | 84% |

**Befund.** Dichte und Abdeckung sind verschiedene Dinge: Erlangen hat die
höchste Dichte und nur 63 % der Wohnstraßen, Würzburg und Jena haben
praktisch jede Straße. Osnabrück mit 38 % Wohnstraßen erklärt seine 36 %
unlösbaren Anfragen direkt. Für eine zweite Stadt: **Würzburg** (98 %,
Osnabrücks Größe, aber nur 28 % der Bilder seit 2022 — großer Zeitabstand),
**Jena** (99 %, 45 % frisch), **Halle** (91 %, 87 % frisch, kein Fotograf
über 30 %, aber dreimal so viele Bilder wie Osnabrück).

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

Die Verkettung liegt bei einem Achtel der Dimensionen gleichauf mit MegaLoc
auf voller Breite — +0.004, das Bootstrap-Intervall schließt 0 ein (siehe
„Konfidenzintervalle"). **Das ist die Zeile „bestes System, Einzelbild"**,
zusammen mit `megaloc`.

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
