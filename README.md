# VPR Osnabrück — Visual Place Recognition auf Mapillary-Bildern

Projekt im Rahmen des Programmierpraktikums an der **Universität Osnabrück**.

Ein Foto rein, ein Ort raus: das System vergleicht das Anfragebild mit
48.321 Referenzbildern aus Osnabrück und gibt die Koordinate des ähnlichsten
zurück. Der Eigenanteil ist nicht das Modell — die Encoder kommen fertig
vortrainiert — sondern der **Benchmark**: ein sequenzbasierter Split ohne
Leakage, vier Ground-Truth-Definitionen, eine Zufallsbasis, Fingerabdrücke
gegen vertauschte Artefakte, Konfidenzintervalle über Fahrten statt über
Bilder, und 36 vergleichbare Zeilen über fünf Encoder und ihre Varianten.

**Eingabe:** ein Straßenfoto aus dem Stadtgebiet
**Ausgabe:** geschätzte Koordinate, eine Konfidenz, und die ähnlichsten
Referenzbilder — `python locate.py foto.jpg`

---

## Inhaltsverzeichnis

- [Schnellstart](#schnellstart)
- [Ergebnisse auf einen Blick](#ergebnisse-auf-einen-blick)
- [Über das Projekt](#über-das-projekt)
- [Ziel](#ziel)
- [Pipeline](#pipeline)
- [Konzeptioneller Aufbau](#konzeptioneller-aufbau)
- [Daten](#daten)
- [Projektstruktur](#projektstruktur)
- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Fremd-Repositories und Gewichte](#fremd-repositories-und-gewichte)
- [Konfiguration](#konfiguration)
- [Nutzung](#nutzung)
- [Reproduzierbarkeit](#reproduzierbarkeit)
- [Ergebnisse](#ergebnisse)
- [Mögliche Erweiterungen](#mögliche-erweiterungen)
- [Team](#team)
- [Credits](#credits)
- [Lizenz](#lizenz)

---

## Schnellstart

Von null bis zur Vergleichstabelle. Voraussetzung: conda, ein
Mapillary-Token, rund 60 GB Platz für Bilder und 25 GB je vollständigem
Encoder-Satz.

```bash
git clone https://github.com/nbomers/pytorch.git && cd pytorch
conda env create -f environment.yml && conda activate pytorch
nbstripout --install --attributes .gitattributes
cp .env.example .env            # MAPILLARY_TOKEN eintragen
python setup_external.py        # AnyLoc- und MixVPR-Repos, Gewichte, Vokabular
python run.py                   # 02 bis 08 mit dem Encoder aus config.yaml
python compare.py --ci          # die Tabelle, mit Konfidenzintervallen
```

`run.py` überspringt jede Stufe, deren Ergebnis schon vorliegt und zur
`config.yaml` passt. Die Metadaten und der Split liegen im Git, 01 läuft
deshalb nicht mit. Beim ersten Lauf dauert 03 (Bilddownload, 332.868
Bilder) Stunden, 04 (Encodieren) je nach Encoder und Rechner zwischen
zwanzig Minuten und einer Nacht. Alles danach sind Minuten.

Wer nur die Ergebnisse sehen will, braucht nichts davon: die JSONs unter
`results/` liegen im Git, und `python compare.py` liest sie direkt. `pytest
tests/` prüft in zwei Sekunden, dass Auswertung, Split und Ergebnisse
zueinander passen.

## Ergebnisse auf einen Blick

Zwei Protokolle, dieselben 53.414 Anfragen. **Volle Referenz**: alle
279.453 Bilder außerhalb der Anfragen als Datenbank — das, was ein System
mit dem ganzen Material leistet. **Benchmark**: nur 15 % der Sequenzen als
Datenbank — das strengere Protokoll, auf dem alle 36 Zeilen inklusive
Adapter und Varianten verglichen werden. R@1 bei 25 m über die lösbaren
Anfragen (90,2 % bzw. 63,9 %); in Klammern das 95-%-Intervall aus 1.000
Ziehungen der 198 Query-Fahrten. Raten trifft 0,05 %.

| | Encoder | Dim | R@1 volle Referenz | R@1 Benchmark |
|---|---|---|---|---|
| **Bester Encoder** | MegaLoc | 8448 | **0.798** [0.739, 0.848] | 0.568 [0.475, 0.666] |
| **Bei 1/16 der Breite** | MegaLoc, PCA-Whitening auf 512 | 512 | 0.778 [0.716, 0.831] | 0.541 |
| **Verkettung** | EigenPlaces + MegaLoc, gewhitent | 1024 | 0.778 [0.714, 0.832] | 0.572 |
| **Bester Recall je Byte Index** | EigenPlaces, PCA-Whitening auf 512 | 512 | 0.715 [0.646, 0.777] | 0.507 |
| **Schwächster** | CLIP ViT-B/32 (nicht für VPR trainiert) | 512 | 0.232 [0.144, 0.349] | 0.073 |

Die Intervalle sind breit, weil die Stichprobe aus 198 Fahrten besteht,
nicht aus 53.414 Bildern — sieben Fahrten stellen 19 % aller Anfragen.
Absolute Zahlen sind deshalb nur auf eine Nachkommastelle belastbar.
**Unterschiede zwischen zwei Zeilen** sind es dagegen auf drei: sie werden
gepaart über dieselben Fahrten gemessen, und dort fällt heraus, was für
alle Encoder gleich schwer ist.

Sieben Befunde, jeder gemessen, sechs davon mit Intervall belegt:

1. **Der Encoder ist der größte Hebel.** CLIP → MegaLoc ist Faktor 7,8,
   +0.496 [+0.405, +0.590]. Jedes Nachbarpaar der Rangfolge clip < anyloc <
   mixvpr < eigenplaces < megaloc schließt 0 aus, auf voller Breite wie auf
   512 gewhitent.
2. **Die Referenz ist der zweitgrößte.** 36 % der Anfragen haben in der
   15-%-Datenbank kein Bild im Umkreis von 25 m. Mit allen 279.453
   Referenzbildern steigt MegaLoc von 0.568 auf 0.798 und EigenPlaces von
   0.484 auf 0.701 — ohne dass sich am Modell ein Bit ändert. Je Anfrage
   zerlegt (Schwierigkeitsprofil): was zählt, ist ein Nachbar, der in
   dieselbe Richtung schaut (0.653 gegen 0.071 ohne), dann der Zeitabstand
   (8–30 Tage 0.819, über ein Jahr 0.473), erst dann die Zahl der Nachbarn
   (ab 51: 0.781). Auf Stadtteil-Ebene ist mit „Bilder je km²" nichts
   davon sichtbar (ρ = 0.27) — die Referenz muss an der Straße der Anfrage
   stehen, nicht im Stadtteil.
3. **Ein trainierter linearer Adapter fügt nichts hinzu, was ein festes
   Whitening nicht schon liefert.** Ohne Whitening hilft er CLIP (+0.050)
   und AnyLoc (+0.130); nach Whitening schließen beide Gewinne 0 ein
   (+0.009 [−0.002, +0.021], −0.029 [−0.074, +0.012]). Den VPR-trainierten
   Encodern schadet er in allen zwölf Paaren sicher (−0.037 bis −0.127).
   Einzige Einschränkung: auf `anyloc_pcaw512` bleibt +0.045 [+0.009,
   +0.088] — 512 gewhitente Komponenten holen aus VLAD weniger heraus als
   4096.
4. **Die Rangfolge hängt nicht an der Deskriptorbreite.** Auf 512
   Dimensionen bleibt sie identisch; MegaLoc verliert von 8448 auf 512
   0.023 [0.017, 0.031]. Der Adapterschaden bei MegaLoc ist auf 8448 und
   512 gleich groß (−0.126 / −0.127) — die Intervalle decken sich, ein
   Beweis für Gleichheit ist das nicht.
5. **Nichts schlägt die Position des besten Treffers.** Schwerpunkt,
   Clustering, zwei Hybride, Sequenz-Aggregation über Nachbarframes
   (−0.009 [−0.017, −0.001]) und semantisches Re-Ranking mit
   Mapillary-Detections — alle gemessen, alle schlechter oder gleich. Auch
   die Verkettung schlägt MegaLoc nicht: +0.004 [−0.005, +0.014].
6. **Fehler sind bimodal.** Unter den 14.726 Fehlgriffen von MegaLoc liegen
   45 % unter 100 m (dieselbe Straße, knapp jenseits der Schwelle) und
   44 % über 1 km (ein anderes Viertel); nur 11 % dazwischen. Ein Median
   beschreibt das schlecht. Die groben Verwechslungen sitzen in
   Wohnstraßen, nicht auf der Autobahn.
7. **Die Ähnlichkeit des besten Treffers ist eine brauchbare Konfidenz.**
   Darf MegaLoc die unsichersten 20 % der Anfragen ablehnen, steigt die
   Präzision von 0.568 auf 0.689; bei cos ≥ 0.30 sind 82 % der Antworten
   richtig. Marge und Geschlossenheit der Treffer taugen weniger.

## Über das Projekt

**Problemstellung.** Visual Place Recognition (VPR): einem Bild ohne
Metadaten den Ort zuordnen, an dem es entstand — indem man es mit einer
Datenbank verorteter Referenzbilder vergleicht. Die Forschung misst das auf
Standarddatensätzen wie Pittsburgh-30k oder MSLS. Die Frage hier: **wie gut
funktionieren die veröffentlichten Verfahren auf einer Stadt, die keiner
von ihnen je gesehen hat, mit Bildern, die niemand für sie kuratiert hat?**

**Motivation.** Drei Dinge wollten wir wissen: wie nah kommt man mit
Open-Source-Encodern an die Zahlen aus den Papern; welche Faktoren
begrenzen ein VPR-System in der Praxis — Modell, Daten, Deskriptor,
Nachbearbeitung; und wie baut man einen Vergleich so, dass die Zahlen etwas
bedeuten. Die dritte Frage wurde die wichtigste.

**Abgrenzung.** Kein Encoder wird trainiert. Alle fünf — CLIP, AnyLoc,
MixVPR, EigenPlaces, MegaLoc — kommen mit den Gewichten ihrer Autoren. Was
trainiert wird, ist ein linearer Adapter obendrauf, und der wird als
Vergleichszeile geführt, nicht als Beitrag. Osnabrück ist die einzige Stadt;
Generalisierung auf andere Städte war nicht Ziel. Das System ist ein
Benchmark und eine Demo, kein Produkt.

## Ziel

Am Ende sollte ein Vergleichsprotokoll stehen, das dieselbe Frage an fünf
Encoder und ihre Varianten stellt und Antworten liefert, die man
gegeneinander halten kann — und daraus eine Antwort auf „woran scheitert
Open-Source-VPR auf einer fremden Stadt".

**Erfolgskriterien**, so wie sie sich am Ende eingelöst haben:

| Kriterium | Stand |
|---|---|
| Reproduzierbarer Split ohne Leakage zwischen train / database / query | erfüllt — Sequenzen sind die Split-Einheit, Split-Listen und Metadaten im Git, ein Test rechnet den Split aus dem Seed nach |
| Jede Zeile der Tabelle unter identischen Bedingungen | erfüllt — Fingerabdrücke prüfen jedes Artefakt gegen die config, ein zweiter Rechenweg reproduziert jede 07-Zahl |
| Recall gegen mehrere Ground-Truth-Definitionen, nicht nur eine | erfüllt — Standard, Hard (anderer Fotograf oder > 180 Tage), Blickrichtung, plus Zufallsbasis |
| Unterschiede statistisch belegt | erfüllt — Sequenz-Bootstrap mit gepaarten Differenzen für 35 Vergleiche |
| Ergebnisse erklären, nicht nur berichten | erfüllt — Dichtekurve, Whitening-Vergleich, Fehlerstruktur, Stadtteilkarte, Verwechslungsatlas, sechs Negativergebnisse |
| Ein System, das man vorführen kann | erfüllt — `locate.py` und `demo/demo.ipynb`: ein eigenes Foto durch jeden Encoder inklusive PCA-, Whitening- und Verkettungsvarianten (nur AnyLoc nicht), mit Konfidenz und Karte |

## Pipeline

```mermaid
flowchart TD
    A[OSM-Stadtpolygon] --> B[01 · Mapillary Vector Tiles → Metadaten]
    B --> C[01 · Split nach Sequenzen<br/>train 70 % · database 15 % · query 15 %]
    C --> D[02 · Audit: Leakage, Zeit, Raum, Fotografen]
    C --> E[03 · Bilddownload, 1024 px]
    E --> F[04 · Embeddings<br/>CLIP · AnyLoc · MixVPR · EigenPlaces · MegaLoc]
    F --> G{vpr.adapter?}
    G -- linear --> H[05 · Adapter-Training auf train]
    H --> I
    G -- none --> I[06 · Retrieval, FAISS, Top-50]
    F -.-> X[experiments/<br/>PCA · Whitening · Verkettung]
    X -.-> I
    I --> J[07 · Recall@k bei 5/10/25/50/100 m, 4 Ground Truths]
    I --> K[08 · Lokalisierung, Fehler in m]
    J --> L[compare.py]
    K --> L
    I -.-> M[experiments/<br/>Bootstrap · Stadtteile · Atlas · Laufzeit]
    I -.-> N[demo/]
```

01 und 03 laufen einmal je Stadt; ihre Ergebnisse (Metadaten, Split,
Bilder) sind die Grundlage für alles Weitere. 04 bis 08 laufen je Encoder
und Variante.

## Konzeptioneller Aufbau

Vier Schichten, jede mit einer klaren Zuständigkeit:

**`config.yaml`** ist der einzige Schalter. Stadt, Split, Encoder, Radien,
Adapter-Training, Lokalisierung — alles steht dort, versioniert. Wer einen
Wert ändert, ändert den Fingerabdruck der betroffenen Artefakte, und die
Pipeline rechnet genau diese neu.

**`src/`** ist der Code, der von allem geteilt wird:

| Modul | Zuständigkeit |
|---|---|
| `config.py` | Projektwurzel finden, config lesen, `VPR_METHOD`/`VPR_ADAPTER` aus der Umgebung übernehmen, Embedding-Namen bilden |
| `paths.py` | alle Ablageorte für die konfigurierte Stadt — `data/<stadt>`, `results/<stadt>`, Bildordner (`VPR_IMAGE_ROOT`, `VPR_IMAGE_PATH`) |
| `run_guard.py` | Fingerabdrücke schreiben und prüfen, `validate_config` beim Laden, Code-Kennung für Ergebnis-JSONs |
| `split.py` | der Sequenz-Split: Listen übernehmen oder mit Seed würfeln |
| `pairs.py` | Bildpaare aus den Metadaten — Anchor/Positive für 05, Query/Datenbank für 02 |
| `evaluation.py` | die Recall-Auswertung mit vier Ground-Truth-Varianten — dieselbe für 07 und jedes Experiment |
| `retrieval.py` | Trefferlisten laden, „lösbar" und „Treffer unter Top-k" je Anfrage, Sequenz-Aggregation |
| `locate.py` | der Inferenz-Einstieg: Foto → Encoder → FAISS → Koordinate mit Konfidenz; nutzt Demo und `locate.py` |
| `districts.py` | OSM-Stadtteile, dieselbe Gliederung für 01 und die Experimente |
| `geo.py` | Haversine, Kompassdifferenz, UTM-Projektion |
| `device.py` | cuda / mps / cpu |
| `mapillary.py` | API-Token aus `.env`, Sessions mit Wiederholung, Vector-Tile-Dekodierung für 01 |
| `adapter_training.py` | fit/val-Split, Triplet-Dataset mit Hard Negatives, Trainingsschleife, val-Metrik — 05 ist damit ein kurzes Notebook |
| `models/` | ein Modul je Encoder, gemeinsame Basis (`base.py`), Adapter, `derived.py` für PCA-/Whitening-/Verkettungsvarianten, und `factory.py`, das aus einem Namen jeden Encoder baut |

**`notebooks/01–08`** sind die Pipeline. Jedes Notebook beginnt mit
denselben vier Zeilen, liest die config, prüft die Fingerabdrücke seiner
Eingaben und schreibt sein Ergebnis mit eigenem Fingerabdruck. `run.py`
führt sie der Reihe nach aus und überspringt, was passt.

**`experiments/`** stellt je Skript eine Frage und beantwortet sie mit
denselben Bausteinen — abgeleitete Encoder werden so geschrieben, wie 04 es
täte, und laufen danach durch die normale Pipeline; Trefferlisten werden mit
`src/evaluation.py` bewertet, damit die Zeile in `compare.py` vergleichbar
ist. Ergebnisse und Zahlen stehen in `experiments/README.md`.

**`tests/`** hält fest, was nicht mehr kaputtgehen darf: die Auswertung
gegen eine handgerechnete Erwartung, der Split gegen die versionierten
Listen, die Config gegen typische Tippfehler, und der Bootstrap gegen jede
07-Zahl. `pytest tests/` läuft ohne Torch und ohne Bilder — auch im CI.

Die Encoder selbst: **CLIP** (ViT-B/32, generischer Bild-Text-Encoder,
512 d), **AnyLoc** (DINOv2 ViT-G, Patch-Merkmale mit VLAD zu 49.152 d
aggregiert, per PCA auf 4096 d), **MixVPR** (ResNet-50 mit MLP-Mixer,
4096 d), **EigenPlaces** (ResNet-50, 2048 d) und **MegaLoc** (DINOv2-basiert,
8448 d). Die letzten drei sind überwacht auf Place Recognition trainiert,
die ersten zwei nicht — dieser Unterschied erklärt am Ende, wie sie auf
Adapter und Whitening reagieren.

## Daten

| | |
|---|---|
| **Quelle** | [Mapillary](https://www.mapillary.com) — Straßenbilder von Nutzern, per Vector Tiles (Zoom 14) über das OSM-Stadtpolygon von Osnabrück ermittelt, Thumbnails mit 1024 px Breite |
| **Lizenz** | CC BY-SA 4.0; Gesichter und Kennzeichen sind von Mapillary automatisch unkenntlich gemacht |
| **Umfang** | 332.868 Bilder aus 120 km² Stadtgebiet, alle innerhalb der administrativen Grenze; ein Bild ist beim Download gescheitert und dokumentiert |
| **Aufteilung** | nach **Sequenzen** (Fahrten), nie nach Einzelbildern: train 231.133 · database 48.321 · query 53.414 (70 / 15 / 15 %). Keine Sequenz in zwei Splits, keine Panoramen, keine fehlenden Werte |
| **Query-Struktur** | 198 Sequenzen, Median 176 Bilder, die längste 3.156; 0,17 s und 3,3 m zwischen Frames |
| **Fotografen** | 57; einer stellt 47,8 % aller Bilder, die drei größten zusammen 65,2 % |
| **Aufnahmejahre** | 2014 bis 2026; 2022 allein 29 %, 2016 ein zweiter Schwerpunkt |
| **Nähe zur Referenz** | 63,9 % der Anfragen haben ein Datenbankbild im Umkreis von 25 m (Median 22 Nachbarn, Median 107 Tage Abstand); 9,3 % davon eines vom selben Fotografen am selben Tag, 83,3 % eines von einem anderen Fotografen |
| **Ground Truth** | ein Datenbankbild zählt als richtig, wenn es höchstens 25 m entfernt liegt (Standard); Varianten: anderer Fotograf oder > 180 Tage Abstand („Hard"), Kompassabweichung ≤ 90° („Blickrichtung"; eine unbekannte Blickrichtung schließt nicht aus — Mapillary kodiert sie als −1, und roh verglichen läse sich das als 359°) |
| **Beschaffung** | `01` holt Metadaten und würfelt den Split, `03` lädt die Bilder nach `image_root/<stadt>` (Standard `~/Downloads/mapillary/osnabrueck`; je Rechner per `VPR_IMAGE_ROOT`) und legt daneben `test/` für eigene Fotos an. Metadaten und Split-Listen liegen im Git |
| **Speicher** | Bilder rund 50 GB, Embeddings 0,7 bis 11,3 GB je Encoder (alle Varianten zusammen 77 GB), Ergebnisse 1,1 GB |

**Warum 25 Meter.** Die Schwelle muss über dem GPS-Rauschen der Aufnahmen
liegen, sonst misst der Recall die Ortung der Kamera statt die Leistung des
Encoders. Mapillary-Bilder kommen überwiegend von Smartphones und Dashcams;
deren Einzelmessung streut im Stadtgebiet — Mehrwegeausbreitung an
Hausfassaden, enge Straßenschluchten — typisch 5 bis 15 m, einzelne Punkte
deutlich weiter. Bei 10 m zählte derselbe korrekt wiedererkannte Ort je nach
Rauschen mal als Treffer und mal nicht; bei 50 m fängt die Schwelle schon die
Nachbarstraße ein. 25 m liegt darüber und unter der typischen Blocklänge, und
es ist die Größenordnung, die MSLS und Pittsburgh-30k ebenfalls verwenden.
Weil das eine Setzung bleibt, berichtet `07` jede Zahl zusätzlich bei 5, 10,
50 und 100 m (`python compare.py --threshold 5`) — die Wahl ist damit
nachprüfbar, nicht nur begründet. Der Wert steht an genau einer Stelle:
`vpr.uncertain_radius_m` in der `config.yaml`.

Der Split ist bewusst sparsam auf der Datenbankseite: 15 % der Sequenzen
als Referenz heißt, dass 36 % der Anfragen kein Referenzbild im Umkreis von
25 m haben. Diese Anfragen zählen im Recall nicht mit („lösbar" = 34.112),
aber die Dichtekurve unter [Ergebnisse](#ergebnisse) zeigt, was mehr
Referenz bringen würde. Alle Zahlen dieser Tabelle stehen in
`results/dataset_audit.json` (aus `02`).

## Projektstruktur

```
.
├── config.yaml              # Alle Projektparameter, einziger Schalter
├── run.py                   # Pipeline am Stück, überspringt fertige Stufen
├── compare.py               # Vergleichstabelle und -abbildungen über alle Verfahren
├── locate.py                # ein Foto verorten: python locate.py foto.jpg
├── setup_external.py        # Fremd-Repos, Gewichte und Vokabular holen
├── environment.yml          # Conda-Umgebung; requirements.txt für pip
├── LICENSE
├── notebooks/               # Die Pipeline, in Reihenfolge
│   ├── 01_mapillary_coverage.ipynb   # Kacheln laden, Split festlegen, Stadtteile
│   ├── 02_dataset_audit.ipynb        # Konsistenz, Kennzahlen -> results/dataset_audit.json
│   ├── 03_image_download.ipynb       # Bilder holen und prüfen
│   ├── 04_embeddings.ipynb           # Encoder → Baseline-Embeddings
│   ├── 05_adapter.ipynb              # Adapter trainieren → adaptierte Embeddings
│   ├── 06_retrieval.ipynb            # Nächste Nachbarn je Query
│   ├── 07_evaluation.ipynb           # Recall-Tabellen, vier Ground-Truth-Varianten
│   └── 08_localization.ipynb         # Aus der Trefferliste eine Koordinate
├── demo/demo.ipynb          # Treffer, Karten, Encoder-Vergleich, eigenes Bild
├── experiments/             # Einmalige Messungen außerhalb der Pipeline, mit README
│   ├── pca_reduce.py, concat_embeddings.py     # abgeleitete Encoder (+ gespeicherte Projektion)
│   ├── full_reference.py                       # zweites Protokoll: database + train als Referenz
│   ├── bootstrap_ci.py                         # Konfidenzintervalle
│   ├── rejection_curve.py                      # Präzision gegen Abdeckung, Konfidenzmaße
│   ├── recall_by_difficulty.py                 # R@1 gegen Nachbarn, Zeit, Blickrichtung, Fotograf
│   ├── database_density.py                     # Recall gegen Referenzdichte
│   ├── recall_by_district.py, confusion_atlas.py   # Karten
│   ├── localization_aggregation.py             # fünf Aggregationsverfahren gegen Top-1
│   ├── sequence_retrieval.py, geometric_verification.py, detection_rerank.py
│   ├── detection_probe.py                      # taugen Mapillarys Detections ueberhaupt?
│   ├── city_coverage.py                        # Strassenabdeckung je Stadt -- welche Stadt als naechste
│   ├── timing.py                               # Laufzeit und Speicher
│   └── results/                                # JSONs und Abbildungen dazu
├── src/                     # geteilter Code, siehe Konzeptioneller Aufbau
├── tests/                   # pytest, ohne Torch
├── data/<stadt>/            # eine Stadt je Zweig, Slug aus config.yaml -> city
│   ├── raw/                 # Kachel-Rohdaten aus 01
│   ├── processed/           # metadata.parquet und Split-Listen (im Git)
│   └── embeddings/<name>/   # auch die abgeleiteten Varianten, mit <name>_pca.npz
├── results/<stadt>/
│   ├── dataset_audit.json   # aus 02                          (im Git)
│   ├── evaluation/          # Recall-JSONs aus 07             (im Git)
│   ├── localization/        # Lokalisierungs-JSONs aus 08     (im Git)
│   ├── retrieval/<name>/    # Trefferlisten je Verfahren
│   └── figures/             # coverage, dataset, evaluation, localization, demo
├── experiments/results/<stadt>/   # JSONs und Abbildungen der Experimente (im Git)
├── weights/
│   ├── adapter/<stadt>/     # trainierte Adapter je Encoder
│   └── mixvpr/              # heruntergeladener MixVPR-Checkpoint
├── cache/                   # OSMnx-Antworten und detections.jsonl, für alle Städte
└── external/                # geklonte Fremd-Repos, nicht im Git
```

Alle Ablageorte kommen aus `src/paths.py`: `city` in der `config.yaml`
bestimmt den Slug (`Osnabrück, Germany` → `osnabrueck`), und jede Stadt hat
ihren eigenen Zweig unter `data/`, `results/`, `experiments/results/` und
`weights/adapter/`. Eine zweite Stadt heißt: `city` umstellen, 01 und 03
laufen lassen, dann wie gewohnt — nichts überschreibt die erste. Bilder
liegen außerhalb des Repos unter `image_root/<stadt>`, eigene Fotos unter
`image_root/test`.

Im Git liegen neben dem Code die kleinen, versionswürdigen Ergebnisse:
Metadaten und Split-Listen, die JSONs unter `results/<stadt>/` und
`experiments/results/<stadt>/`, die Auswertungsabbildungen und der
Detection-Cache. Alles Große — Bilder, Embeddings, Trefferlisten, Gewichte,
Fremd-Repos — entsteht beim Durchlauf neu und wird über Fingerabdrücke
gegen die `config.yaml` geprüft.

## Voraussetzungen

| | |
|---|---|
| **Python** | 3.14 (`environment.yml`) |
| **Paketmanager** | conda (empfohlen) oder uv |
| **GPU** | nicht nötig, aber: 04 encodiert 332.868 Bilder. Auf einem M1 Pro sind CLIP, MixVPR und EigenPlaces in Stunden fertig; AnyLoc (ViT-G, 1,14 Mrd. Parameter) und MegaLoc gehören auf eine CUDA-GPU, dort mit `fp16` und kleinen Batches auf 8 GB VRAM |
| **Arbeitsspeicher** | 16 GB reichen für alles außer dem Adapter-Training auf MegaLoc (8448 d); dort braucht 05 rund 12 GB frei |
| **Speicherplatz** | 50 GB Bilder + 3 bis 25 GB je Encoder-Satz + 1 GB Ergebnisse |
| **Mapillary-Token** | kostenloser Developer-Account, siehe [Konfiguration](#konfiguration) |

## Installation

Empfohlen wird **conda** ([Miniconda](https://docs.conda.io/en/latest/miniconda.html)),
alternativ **uv**.

```bash
conda env create -f environment.yml
conda activate pytorch
nbstripout --install --attributes .gitattributes   # Zellenausgaben aus dem Git halten, einmal je Rechner
pytest tests/                                       # Installation pruefen, ohne Torch-Laufzeit
```

`environment.yml` holt alles mit kompilierten Abhängigkeiten (faiss, GDAL
hinter geopandas) über conda-forge und den Rest — torch und was darauf
aufbaut — über pip; die pip-Wheels wählen CUDA auf Linux und MPS auf dem
Mac von selbst. `conda install --file requirements.txt` funktioniert
**nicht**, mehrere Pakete gibt es nur über pip; `requirements.txt` ist für
die uv-Variante da:

```bash
uv venv --python 3.14 && source .venv/bin/activate && uv pip install -r requirements.txt
```

Die passende PyTorch-Variante (CUDA, MPS, CPU) gibt der Konfigurator aus:
https://pytorch.org/get-started/locally/

## Fremd-Repositories und Gewichte

AnyLoc und MixVPR werden aus ihren Original-Repos importiert und liegen
deshalb nicht im Git-Repo. Ein Skript holt sie auf die Commits, gegen die
hier entwickelt wurde:

```bash
python setup_external.py
```

EigenPlaces und MegaLoc brauchen das nicht — die kommen über `torch.hub`
und laden sich beim ersten Lauf selbst nach `~/.cache/torch/hub`.

**Die MixVPR-Gewichte holt das Skript selbst** (per `gdown` von Google Drive)
und prüft die Datei gegen eine hinterlegte SHA-256-Summe — Drive liefert bei
Überlastung gern eine HTML-Fehlerseite statt der Datei.

**Das AnyLoc-Vokabular holt es ebenfalls selbst** — AnyLocs eigene
OneDrive-Links sind tot, die HuggingFace-Space des Projekts hält dieselben
Dateien. Die Cluster-Zentren landen dort, wo die Konfiguration sie erwartet:

```
external/AnyLoc/cache/vocabulary/dinov2_vitg14/l31_value_c32/urban/c_centers.pt
```

Der Pfad wird aus `vpr.anyloc` gebaut — änderst du `vocabulary_domain`,
`num_clusters`, `desc_layer` oder `desc_facet`, prüft das Skript an der
passenden Stelle und meldet, wenn der Cache dafür nichts enthält.

Ohne dieses Vokabular fittet AnyLoc die Cluster-Zentren selbst auf den
Trainingsbildern. Das läuft, ist dann aber nicht mehr mit den Zahlen aus dem
Paper vergleichbar — für die Vergleichstabelle also vorher besorgen.

Beides ist nur nötig, wenn du das jeweilige Verfahren auch benutzt.

## Konfiguration

Alle Parameter stehen in `config.yaml` — Stadt, Split-Anteile, Encoder,
Radien, Adapter-Training. Die Datei ist versioniert; wer sie ändert, ändert
die Fingerabdrücke, und die betroffenen Stufen rechnen beim nächsten Lauf neu.
`validate_config` prüft sie beim Laden gegen Widersprüche (Anteile, Radien,
Tippfehler in `source`, `top_k`-Grenzen), bevor eine Stufe Stunden rechnet.

Für den Bilddownload braucht es einen Mapillary-Developer-Token (kostenloser
Account unter https://www.mapillary.com/developer). Zugangsdaten gehören in
eine lokale `.env`-Datei und **niemals ins Repository**.
Als Vorlage dient `.env.example`:

```bash
cp .env.example .env
```

Die Schlüssel, die man am ehesten anfasst:

| Schlüssel | Bedeutung |
|---|---|
| `vpr.method`, `vpr.adapter` | welcher Encoder, mit oder ohne linearen Adapter — `run.py --method/--adapter` überschreibt beides |
| `vpr.models` | Namen aller Encoder; abgeleitete Varianten haben darunter einen Block mit `source` (PCA) oder `sources` (Verkettung), die PCA-Blöcke teilen sich ihre Werte über YAML-Anker |
| `vpr.uncertain_radius_m` | die 25 m der Ground Truth — auch die Standardschwelle von `compare.py --threshold` |
| `vpr.max_heading_diff_deg` | die 90° der Blickrichtungs-Auswertung |
| `retrieval.top_k`, `k_values`, `thresholds` | wie viele Nachbarn 06 speichert, welche R@k und Schwellen 07 berichtet |
| `localization.top_k`, `eps_m`, … | Top-k für 08 und die Aggregationsverfahren in `experiments/localization_aggregation.py` |
| `city` | Stadt für Kacheln, Stadtgrenze, Stadtteile — und der Slug für alle Ablageorte |
| `max_missing_images_frac` | wie viele Bilder fehlen dürfen, bevor 03 und 04 abbrechen (Standard 0,01 = 1 %) |
| `verify_all_images` | `true` prüft in 03 den gesamten Bildbestand statt nur der neu geholten |
| `vpr.max_images` | Obergrenze für 04, nur zum Ausprobieren — `null` = alle |
| `image_root` | Wurzel der Bildordner (`<image_root>/<stadt>`, `<image_root>/test`); je Rechner per `VPR_IMAGE_ROOT` oder `VPR_IMAGE_PATH` überschreibbar |

## Nutzung

```bash
python run.py
```

Führt die Notebooks 01 bis 08 der Reihe nach aus. Eine Stufe wird
übersprungen, wenn ihr Ergebnis vorliegt **und** laut Fingerabdruck zur
`config.yaml` passt — nach einer Config-Änderung laufen also genau die
betroffenen Stufen neu. 07 und 08 tragen zusätzlich eine Kennung des
Auswertungscodes; ändert sich `src/evaluation.py`, werden sie neu
gerechnet. Ausnahmen: 01 bis 03 werden nur auf Existenz ihres Ergebnisses
geprüft (01 würde sonst den Split neu würfeln und alle Embeddings
entwerten; 02 und 03 hängen nicht am Verfahren), und 05 entfällt, solange
`vpr.adapter` auf `"none"` steht.

| Aufruf | Wirkung |
|---|---|
| `python run.py` | alles, was nötig ist |
| `python run.py --method mixvpr` | anderer Encoder, ohne `config.yaml` zu ändern |
| `python run.py --method clip --adapter linear` | Encoder und Variante zusammen |
| `python run.py --method all --adapter all` | jeden echten Encoder, je Baseline und Adapter |
| `python run.py --method derived --adapter all` | die abgeleiteten PCA-/Whitening-Varianten (siehe `experiments/`) |
| `python run.py --method clip,mixvpr` | nur diese beiden nacheinander |
| `python run.py --from 06` | ab dem Retrieval, erzwungen |
| `python run.py --force` | alles neu rechnen |

`--method` und `--adapter` erreichen die Notebooks über die
Umgebungsvariablen `VPR_METHOD` und `VPR_ADAPTER`; die `config.yaml` wird
nicht angefasst. Wer ein Notebook direkt in Jupyter öffnet, bekommt den
Wert aus der Datei.

`--method` und `--adapter` nehmen auch Kommalisten oder `all`. Dann rechnet
`run.py` eine Kombination nach der anderen: 01 bis 03 laufen dabei nur einmal,
weil sie nicht am Verfahren hängen, und eine Kombination, die abbricht, reißt
die übrigen nicht mit — am Ende steht, welche gescheitert sind.

### Ergebnisse vergleichen

Jeder Durchlauf von 07 legt seine Recall-Tabellen unter
`results/<stadt>/evaluation/<name>.json` ab. Daraus baut `compare.py` die
Vergleichstabelle über alle Verfahren:

```bash
python compare.py                 # R@k bei 25 m, die fünf Encoder mit und ohne Adapter
python compare.py --derived       # dazu alle PCA-, Whitening-, Verkettungs- und Sequenz-Zeilen
python compare.py --ci            # dazu das 95-%-Intervall neben R@1
python compare.py --threshold 5   # strengere Schwelle
python compare.py --split "Blickrichtung: Treffer nur bei <= 90 Grad Abweichung"
python compare.py --reference full   # zweites Protokoll: database + train als Referenz
python compare.py --plot          # drei Vergleichsabbildungen nach results/figures/evaluation/
python compare.py --plot --derived   # dieselben mit den PCA-/Whitening-Varianten
python compare.py --localization  # 08: Anteil unter 25 m und Median je Encoder
```

### Nebenuntersuchungen in `experiments/`

Jedes Skript beantwortet eine Frage, alle laufen auf vorhandenen Artefakten
und schreiben ihr Ergebnis dorthin, wo `compare.py` es findet, oder nach
`experiments/results/`. Details und Zahlen in `experiments/README.md`.

```bash
python experiments/pca_reduce.py            # PCA-/Whitening-Varianten als Encoder schreiben
python experiments/concat_embeddings.py     # zwei Encoder verketten
python run.py --method derived --adapter all   # dann die Pipeline darüber
python experiments/full_reference.py        # database + train als Referenz, alle Baselines
python experiments/bootstrap_ci.py          # Konfidenzintervalle für alle Zeilen, ~1 min
python experiments/rejection_curve.py       # Präzision gegen Abdeckung, welche Konfidenz taugt
python experiments/recall_by_difficulty.py  # R@1 gegen Nachbarn, Zeit, Blickrichtung, Fotograf
python experiments/timing.py                # Bilder/s, ms je Anfrage, Index-MB je Encoder
python experiments/database_density.py --method eigenplaces_pcaw512   # Recall gegen Referenzdichte
python experiments/recall_by_district.py    # Recall je Stadtteil, Karte
python experiments/confusion_atlas.py       # wohin die Fehlgriffe zeigen, Karte
python experiments/localization_aggregation.py   # Schwerpunkt, Clustering, Snap, Gated gegen Top-1
python experiments/sequence_retrieval.py --method eigenplaces_pcaw512  # Nachbarframes aufsummieren
python experiments/geometric_verification.py --method eigenplaces_megaloc_concat   # SuperPoint + LightGlue, GPU
python experiments/detection_rerank.py      # Mapillary-Detections als Re-Ranking-Signal
python experiments/detection_probe.py       # liefert Mapillary ueberhaupt brauchbare Detections?
python experiments/city_coverage.py "Würzburg, Germany"   # Strassenabdeckung einer Stadt, nur Metadaten
```

### Ein eigenes Foto verorten

```bash
python locate.py foto.jpg                                  # Encoder aus config.yaml
python locate.py foto.jpg --method eigenplaces_megaloc_concat --k 5
python locate.py ~/Downloads/mapillary/test                # alle Fotos im Testordner
python locate.py foto.jpg --json
```

Baut den Encoder über die Factory — auch die PCA-, Whitening- und
Verkettungsvarianten, deren Projektion neben den Embeddings liegt —, sucht
im FAISS-Index über die Datenbank und gibt Koordinate, Konfidenz (die
Ähnlichkeit des besten Treffers, siehe Ablehnungskurve) und die Top-k
zurück. Nur AnyLoc-Varianten gehen nicht: die AnyLoc-PCA aus 04 liegt
nicht neben den Embeddings. `demo/demo.ipynb` nutzt denselben Weg und zeigt
dazu Trefferreihen, Karten auf dem Straßennetz und alle Encoder am selben
Anfragebild.

### Tests

```bash
pytest tests/
```

Acht Dateien, drei Sekunden, kein Torch: die Recall-Auswertung gegen eine
handgerechnete Erwartung (Standard, Hard, Blickrichtung, Panorama), der Split gegen
die versionierten Listen, die Paarbildung, `validate_config` gegen die
echte `config.yaml` und gegen Tippfehler, die PCA-Projektion gegen sklearn,
`localizable` gegen die volle Distanzmatrix, und die versionierten
Ergebnis-JSONs gegen beide Bootstraps. Ein Test prüft außerdem, dass der in
jeder Ergebnis-JSON vermerkte Commit im Repository auffindbar ist — sonst ist
die Kennung wertlos. Dieselben Tests laufen bei jedem Push
(`.github/workflows/check.yml`).

## Reproduzierbarkeit

**Ein Seed für alles.** `vpr.split_seed: 42` in der `config.yaml` steuert
den Sequenz-Split in 01, die fit/val-Aufteilung und das Negative-Sampling
im Adapter-Training, die Stichproben der PCA-Anpassung, die Zufallsbasis in
07 und 08, den Bootstrap und die Stichproben aller Experimente.

**Metadaten und Split im Git.** `data/<stadt>/processed/metadata.parquet` (9 MB)
und die drei Split-Listen liegen versioniert; ein Test rechnet den Split
aus den Metadaten und dem Seed nach und vergleicht ihn mit den Listen. 01
läuft nur noch für eine neue Stadt oder einen neuen Split — Mapillary
ändert sich über die Zeit, ein frischer Lauf ergäbe einen anderen
Schnappschuss. Bildpaare für Audit und Adaptertraining werden aus den
Metadaten abgeleitet (`src/pairs.py`), nichts davon muss versioniert werden.

**Die Auswertung rechnet exakt, aber sparsam.** `src/evaluation.py` holt
Kandidaten innerhalb der größten Schwelle aus einem KDTree und rechnet die
Haversine-Distanz nur für diese Paare und die Trefferliste — dieselben
Zahlen wie mit der vollen Distanzmatrix (gegen alle gespeicherten JSONs
geprüft), aber Sekunden statt Minuten, auch bei 279.453 Referenzbildern.

**Fingerabdrücke statt Vertrauen.** Neben jedem Artefakt liegt eine
`.fingerprint.json` mit allem, was seinen Inhalt bestimmt — Encoder,
Modellkonfiguration, Split-Parameter, Hash der Metadaten. Jede Stufe prüft
ihre Eingaben dagegen und bricht ab, statt mit einer Datei aus einem anderen
Lauf falsche Zahlen zu rechnen. `run.py` nutzt dieselben Fingerabdrücke, um
zu entscheiden, was übersprungen werden darf. Die Ergebnis-JSONs aus 07 und
08 tragen zusätzlich den Fingerabdruck ihrer Trefferliste und eine Kennung
des Auswertungscodes (Git-Commit und Hash von `src/evaluation.py`).

**Zwei Rechner, ein Ergebnis.** Die Zahlen entstanden auf einem MacBook Pro
M1 Pro (16 GB, MPS) für CLIP, MixVPR und EigenPlaces samt Adaptern, und auf
einem Linux-Rechner mit RTX 3070 (8 GB) für AnyLoc und MegaLoc. AnyLoc läuft
dort mit fp16-Modell und Batch 4, sonst reicht der VRAM nicht. Alle
Ergebnis-JSONs und Abbildungen wandern über git; Embeddings und
Trefferlisten sind gitignored und wurden per `rsync` zusammengeführt.

Eine Folge davon, die man kennen sollte: **auf verschiedenen Rechnern
gerechnete Encoder halten dieselben Bilder in verschiedener
Zeilenreihenfolge**, und MegaLoc hat zwei `train`-Bilder weniger (Download
dort gescheitert). Für 07 und 08 ist das egal, jeder Encoder ist in sich
konsistent. Alles, was Encoder nebeneinanderlegt — die Verkettung, der
Encoder-Vergleich in der Demo, der Bootstrap — richtet deshalb über die
`image_id` aus, nie über die Zeilennummer.

**Was nicht bitidentisch reproduziert.** Die randomisierte SVD der
PCA-Varianten rundet auf anderem BLAS minimal anders; die Fingerabdrücke
prüfen die Konfiguration, nicht den Inhalt. Wer die exakten Zahlen aus
`results/` reproduzieren will, nimmt die Embeddings, mit denen sie
gerechnet wurden.

**Gewichte.** Trainierte Adapter unter `weights/adapter/`, der
MixVPR-Checkpoint unter `weights/mixvpr/`, EigenPlaces und MegaLoc über
`torch.hub` in `~/.cache/torch/hub`, das AnyLoc-Vokabular unter
`external/AnyLoc/cache/`. Alles außer den Adaptern holt `setup_external.py`.

## Ergebnisse

Alle Zahlen entstehen aus den genannten Befehlen; nichts hier ist von Hand
eingetragen.

### Recall, volle Referenz — `python compare.py --reference full --ci`

```
Alle Queries  |  Schwelle 25 m  |  48,177 loesbare Queries  |  Referenz: 279,453 Bilder (database + train)

Encoder       Variante     Dim        R@1          95-%-KI     R@5    R@10    R@20
----------------------------------------------------------------------------------
anyloc        none        4096      0.435  [0.343, 0.533]   0.512   0.549   0.588
clip          none         512      0.232  [0.144, 0.349]   0.288   0.318   0.353
eigenplaces   none        2048      0.701  [0.630, 0.765]   0.775   0.801   0.824
megaloc       none        8448      0.798  [0.739, 0.848]   0.858   0.876   0.888
mixvpr        none        4096      0.653  [0.575, 0.724]   0.723   0.751   0.777
----------------------------------------------------------------------------------
Zufall        (Raten)              0.0005                   0.0022  0.0043  0.0088

Intervall: Sequenz-Bootstrap, 2,5- und 97,5-Perzentil (experiments/bootstrap_ci.py). Fuer den Vergleich zweier Zeilen gilt die gepaarte Differenz dort, nicht die Ueberlappung.
```

Nur Encoder ohne Adapter: der Adapter wurde auf `train` trainiert, `train`
als Referenz wäre für ihn Leakage. Alle 18 Zeilen mit PCA-, Whitening- und
Verkettungsvarianten: `--derived`. Die Rangfolge ist dieselbe wie im
Benchmark-Protokoll, jede Zahl liegt rund 0.2 höher. MegaLoc auf 512
gewhitent (0.778) und die Verkettung (0.778) liegen gleichauf, MegaLoc auf
voller Breite 0.020 darüber — der Preis der Reduktion, bei einem
Sechzehntel des Index.

### Recall, Benchmark-Protokoll — `python compare.py --ci`

```
Alle Queries  |  Schwelle 25 m  |  34,112 loesbare Queries  |  Referenz: 48,321 Bilder (database)

Encoder       Variante     Dim        R@1          95-%-KI     R@5    R@10    R@20
----------------------------------------------------------------------------------
anyloc        none        4096      0.204  [0.153, 0.265]   0.294   0.340   0.384
anyloc        linear      4096      0.335  [0.277, 0.400]   0.504   0.570   0.631
clip          none         512      0.073  [0.044, 0.108]   0.106   0.129   0.158
clip          linear       512      0.123  [0.092, 0.161]   0.221   0.281   0.353
eigenplaces   none        2048      0.484  [0.405, 0.574]   0.608   0.650   0.695
eigenplaces   linear      2048      0.444  [0.372, 0.523]   0.593   0.648   0.696
megaloc       none        8448      0.568  [0.475, 0.666]   0.676   0.719   0.763
megaloc       linear      8448      0.442  [0.372, 0.517]   0.580   0.628   0.667
mixvpr        none        4096      0.426  [0.354, 0.509]   0.543   0.590   0.642
mixvpr        linear      4096      0.363  [0.302, 0.433]   0.508   0.568   0.624
----------------------------------------------------------------------------------
Zufall        (Raten)              0.0006                   0.0028  0.0047  0.0098

Intervall: Sequenz-Bootstrap, 2,5- und 97,5-Perzentil (experiments/bootstrap_ci.py). Fuer den Vergleich zweier Zeilen gilt die gepaarte Differenz dort, nicht die Ueberlappung.
```

Spalte „Variante": `none` = Encoder wie veröffentlicht, `linear` = mit
trainiertem linearem Adapter. `--derived` zeigt alle 36 Zeilen: Namen mit
`_pca512` sind per PCA auf 512 reduziert, `_pcaw512` zusätzlich gewhitent,
`_pcaw4096` / `_pcaw2048` gewhitent ohne Reduktion, `_concat` verkettet,
`seq3` = Trefferlisten über ±3 Nachbarframes aufsummiert.

Weitere Sichten: `--threshold 5` bis `100`, `--split "Hard: …"` und
`--split "Blickrichtung: …"` für die strengeren Ground Truths. Bei MegaLoc
etwa: 0.568 Standard, 0.543 Hard, 0.621 Blickrichtung — die letzte Zahl ist
höher, weil dort 4.945 Anfragen wegfallen, deren einzige nahen Nachbarn in
die andere Richtung schauen.

### Unsicherheit — `python experiments/bootstrap_ci.py`

Die 53.414 Anfragen stammen aus 198 Fahrten, und aufeinanderfolgende
Frames scheitern gemeinsam. Der Bootstrap zieht deshalb 1.000-mal die
Fahrten mit Zurücklegen, nicht die Bilder. Ergebnis: die 95-%-Halbbreite
einer einzelnen R@1-Zahl liegt bei ±0.03 (CLIP) bis ±0.10 (MegaLoc) — der
binomiale Standardfehler hätte ±0.005 behauptet. Gepaarte Differenzen
zwischen zwei Encodern sind dagegen eng (±0.005 bis ±0.05), weil dieselben
Fahrten für beide schwer sind:

| Vergleich | Differenz | 95 % | belegt |
|---|---|---|---|
| eigenplaces → eigenplaces_pca512 | −0.003 | [−0.008, +0.002] | nein — Rauschen |
| eigenplaces → eigenplaces_pcaw512 | +0.022 | [+0.011, +0.039] | ja |
| eigenplaces → eigenplaces_pcaw2048 | −0.025 | [−0.042, −0.010] | ja |
| anyloc → anyloc_pcaw4096 | +0.117 | [+0.078, +0.162] | ja |
| megaloc → eigenplaces_megaloc_concat | +0.004 | [−0.005, +0.014] | nein — gleichauf |
| megaloc → megaloc_linear | −0.126 | [−0.179, −0.073] | ja |
| clip_pcaw512 → clip_pcaw512_linear | +0.009 | [−0.002, +0.021] | nein |

Alle 35 Paare in `experiments/results/<stadt>/bootstrap_ci.json`.

### Lokalisierung — `python compare.py --localization`

```
Lokalisierung  |  Anteil unter 25 m  |  53,414 Anfragen, Top-10 je Anfrage
Median des Fehlers in Klammern.

Encoder                         Top-1
-------------------------------------
anyloc               0.130 ( 1,420 m)
anyloc_linear        0.214 ( 1,338 m)
clip                 0.046 ( 2,365 m)
clip_linear          0.078 ( 1,853 m)
eigenplaces          0.309 (   364 m)
eigenplaces_linear   0.283 (   758 m)
megaloc              0.363 (    94 m)
megaloc_linear       0.283 ( 1,496 m)
mixvpr               0.272 (   659 m)
mixvpr_linear        0.232 ( 1,335 m)
-------------------------------------
Zufall (DB-Bild)     0.000 ( 4,223 m)

Lesart: liegt Top-1 vorn, sind die Nachbartreffer zu oft falsch, als
dass Mitteln oder Clustern helfen koennte.
```

Bezogen auf alle 53.414 Anfragen, denn eine Koordinate wird immer
geschätzt. MegaLoc trifft mit dem besten Treffer im Median auf 94 m. Fünf
Aggregationsverfahren (Schwerpunkt, Clustering, Snap, Gated) sind gemessen
und unterliegen Top-1 bei jedem Encoder — MegaLoc: Clustering 0.318 bei
444 m, Schwerpunkt 0.267 bei 575 m; siehe
`experiments/localization_aggregation.py` und `experiments/README.md`.

### Referenzdichte — `experiments/database_density.py`

`train` stufenweise zur Datenbank dazugenommen, R@1 bei 25 m:

| train dazu | Referenzbilder | lösbar | EigenPlaces | EigenPlaces pcaw512 | MegaLoc pcaw512 |
|---|---|---|---|---|---|
| 0 % | 48.321 | 63,9 % | 0.484 | 0.507 | 0.541 |
| 25 % | 107.449 | 78,4 % | 0.545 | 0.568 | 0.615 |
| 50 % | 160.981 | 84,9 % | 0.595 | 0.615 | 0.670 |
| 75 % | 222.300 | 88,4 % | 0.672 | 0.688 | 0.754 |
| 100 % | 279.453 | 90,2 % | 0.701 | 0.715 | **0.778** |

Beides steigt gleichzeitig: mehr Anfragen werden überhaupt beantwortbar,
und der Recall unter den beantwortbaren steigt trotzdem — obwohl die
dazukommenden Anfragen die schwereren sind. Nur für Baselines sauber, weil
der Adapter auf `train` trainiert wurde. Für diese Kurve gibt es noch kein
Bootstrap-Intervall; der Anstieg ist mit +0.22 aber weit größer als jede
Halbbreite der Tabelle.

### Stadtteile — `experiments/recall_by_district.py`

R@1 von MegaLoc je OSM-Stadtteil reicht von 0.07 (Sutthausen) bis 0.83
(Atter) — Faktor 12 beim selben Encoder, und EigenPlaces ordnet die
Stadtteile praktisch gleich (Spearman 0.96): die Karte zeigt die Daten,
nicht den Encoder. Mit der Referenzdichte in Bildern je km² hängt das
nicht zusammen (ρ = 0.27, p = 0.27): die Innenstadt hat die dichteste
Referenz und nur 38 % lösbare Anfragen, Atter ein Zehntel der Dichte und
den besten Recall. Was einen Stadtteil scheitern lässt, steckt in den
Aufnahmen — Blickrichtung (Haste: nur 29 % der lösbaren Anfragen haben einen
Nachbarn, der in dieselbe Richtung schaut), Jahre Abstand (Hellern), oder
eine einzelne Fahrt, die ein Wohnviertel mit einem anderen verwechselt
(Sutthausen: 208 von 414 Anfragen landen in Hellern, 4 km entfernt).
Karten unter `experiments/results/<stadt>/recall_by_district_*.png`.

### Ablehnung — `experiments/rejection_curve.py`

Darf das System schweigen, wenn es sich nicht sicher ist? Drei
Konfidenzmaße aus der Trefferliste, MegaLoc, Präzision = Top-1 innerhalb
25 m:

| Sicht | Konfidenz | ohne Ablehnung | bei 80 % Abdeckung | bei 50 % | AUC |
|---|---|---|---|---|---|
| lösbare Anfragen | Ähnlichkeit des besten Treffers | 0.568 | **0.689** | 0.793 | 0.791 |
| lösbare Anfragen | Marge zu Platz 2 | 0.568 | 0.618 | 0.724 | 0.741 |
| lösbare Anfragen | Geschlossenheit der Top-10 | 0.568 | 0.597 | 0.706 | 0.706 |
| alle Anfragen | Ähnlichkeit des besten Treffers | 0.363 | 0.451 | 0.673 | 0.659 |

Die rohe Ähnlichkeit ist das beste Maß; `locate.py` meldet sie als
Konfidenz. Bei cos ≥ 0.30 sind 82 % der Antworten richtig (37 % der
lösbaren Anfragen), bei cos ≥ 0.20 noch 75 % (69 %).

### Schwierigkeit je Anfrage — `experiments/recall_by_difficulty.py`

R@1 von MegaLoc nach Eigenschaften der Anfrage, aus den Metadaten:

| Merkmal | | R@1 |
|---|---|---|
| Nachbar mit passender Blickrichtung | ja (29.157) / nein (4.955) | 0.653 / **0.071** |
| Tage zum nächsten Referenzbild | 8–30 / 31–180 / über 365 | **0.819** / 0.539 / 0.473 |
| Nachbarn im Umkreis von 25 m | 1–2 / 11–20 / 51+ | 0.526 / 0.448 / **0.781** |
| Nachbar vom selben Fotografen am selben Tag | ja / nein | 0.690 / 0.556 |

Blickrichtung zuerst, Zeit danach, Dichte zuletzt — und die Dichte wirkt
als „viele Nachbarn helfen", nicht als „wenige schaden".

### Struktur der Fehler — `experiments/confusion_atlas.py`

Unter den 14.726 Fehlgriffen von MegaLoc (Top-1 weiter als 25 m, nur
lösbare Anfragen) liegen 45 % unter 100 m — dieselbe Straße, knapp jenseits
der Schwelle — und 44 % über 1 km; nur 11 % dazwischen. 53 % bleiben im
eigenen Stadtteil. Kein Stadtteil-Paar trägt mehr als 1,6 % der Fehlgriffe;
es gibt keine dominante Verwechslung zweier Orte, sondern viele kleine.
Autobahn-Anfragen scheitern nicht öfter als andere (R@1 0.570 gegen 0.565),
ihre Fehlgriffe sind sogar kurz (Median 56 m); die groben Verwechslungen
sitzen in Wohnstraßen.

Über alle 53.414 Anfragen (08, auch die unlösbaren) liegt der Median eines
falschen Top-1 bei 1,7 km (MegaLoc) bzw. 1,5 km (EigenPlaces). Beides
zusammen erklärt, warum Aggregation scheitert: bei einer groben
Verwechslung liegen die Nachbarn geschlossen am falschen Ort, und bei einem
knappen Fehlgriff ist der beste Treffer schon die beste Antwort.

### Laufzeit und Speicher — `experiments/timing.py`

Suche im FAISS-Flat-Index über 48.321 Datenbankbilder (M1 Pro, CPU):

| Dim | ms je Anfrage | Index |
|---|---|---|
| 512 (alle PCA-Varianten) | 0.5–0.75 | 94 MB |
| 1024 (Verkettung) | 1.2 | 189 MB |
| 2048 (EigenPlaces) | 1.9 | 378 MB |
| 4096 (AnyLoc, MixVPR) | 3.2–3.4 | 755 MB |
| 8448 (MegaLoc) | 6.4 | 1.557 MB |

Linear in der Breite: 8448 → 512 ist Faktor 11 bei der Suche und 16,5 beim
Speicher. Bei 48.321 Bildern kein Argument (6 s gegen 34 s für alle
Anfragen), bei einer Million wäre es eines.

Encodieren (200 Bilder, M1 Pro auf MPS, inklusive Laden und Dekodieren):
CLIP **145** Bilder/s (224 px), MixVPR **73** (320 px), EigenPlaces
**33** (512 px), MegaLoc **21** (322 px); AnyLoc **15** auf der RTX 3070
(fp16, Batch 4 — auf dem Mac nicht praktikabel).
Der Durchsatz hängt am Rückgrat und der Eingabegröße, nicht an der PCA —
`eigenplaces_pcaw512` encodiert genauso schnell wie `eigenplaces`. Für
332.868 Bilder heißt das auf dem Mac: CLIP 40 Minuten, MixVPR 75 Minuten,
EigenPlaces 2,8 Stunden, MegaLoc 4,4 Stunden; AnyLoc 6,3 Stunden auf der GPU.

### Abbildungen

`results/figures/evaluation/` — `vergleich_r1_25m.png` (R@1 je Zeile mit
Bootstrap-Intervall, liegende Balken nach Wert sortiert, Farbe = Encoder),
`vergleich_recall_k_25m.png` und `vergleich_schwellen.png` (Kurven, Farbe =
Encoder, gestrichelt = Adapter). Mit `--derived` heißen sie `_derived` und
werden zu kleinen Vielfachen: ein Feld je Encoder, Farbe und Markerform =
Deskriptorvariante, Linienstil = Adapter bzw. Sequenz. 37 Zeilen in eine
Legende zu zwingen war vorher der Punkt, an dem die Abbildung unlesbar wurde. `results/figures/localization/` —
je Encoder die Fehlerverteilung. `results/figures/demo/` — Trefferreihen,
Karten auf dem Straßennetz, Encoder-Vergleich. `experiments/results/` —
Dichtekurven, Stadtteilkarten, Verwechslungsatlas.

## Mögliche Erweiterungen

**Bewusst vereinfacht.** Ein Split-Seed, eine Stadt, eine Auflösung je
Encoder. AnyLoc wurde zunächst ohne Whitening verglichen — das ist für
VLAD unüblich und kostete 0,12 R@1; die gewhitente Variante steht in der
Tabelle, in `04` eingebaut ist sie nicht, und für ein neues Bild ist
AnyLoc nicht vorführbar, weil seine PCA aus 04 nicht neben den Embeddings
liegt.

**Eine zweite Stadt.** Seit `src/paths.py` hat jede Stadt ihren eigenen
Zweig in `data/`, `results/` und `experiments/results/`: `city` umstellen,
01 und 03 laufen lassen, dann die Pipeline wie gewohnt. Ein Encoder auf
einer Stadt von Osnabrücks Größe kostet auf dem M1 Pro eine Nacht (04) plus
Minuten (06–08). Auf Mapillary dichter erschlossen als Osnabrück (2.808
Bilder/km²) sind etwa Erlangen (7.943), Mainz (6.593), Jena (6.113),
Würzburg (4.895) und Heidelberg (4.879);
Bamberg (25.248/km², 1,38 Mio. Bilder auf 55 km²) ist ein Sonderfall.

**Nächster Schritt mit mehr Zeit.** Die geometrische Verifikation
(`experiments/geometric_verification.py`) ist gebaut und geprüft, aber
nicht gemessen — sie braucht die Bilder und eine GPU, rund sechs Stunden
für alle Anfragen. Sie ist der einzige Hebel, der die grobe Verwechslung
direkt angreift: global ähnliche, lokal verschiedene Orte. Erwartung aus
der Literatur +0.05 bis +0.10 R@1.

Danach bleibt als einziger offener Punkt ein **zweiter Split-Seed**: alle
Zahlen stehen auf `split_seed: 42`, und wie viel davon am Seed hängt, ist
nicht gemessen. Recall gegen Nachbarzahl und Zeitabstand je Anfrage
(`recall_by_difficulty.py`) und die Ablehnungskurve (`rejection_curve.py`)
sind inzwischen gerechnet und stehen oben unter [Ergebnisse](#ergebnisse).

**An den Fremd-Repositories.** MixVPR hat keine Lizenzdatei. AnyLocs
Download-Links für das Vokabular sind tot; `setup_external.py` holt es aus
der HuggingFace-Space, und die Datei heißt dort `c_centers.pt`, während das
README des Projekts `c_center.pt` nennt. AnyLocs `VLAD.generate()` legt
Zwischenergebnisse auf der CPU an und bricht mit CUDA-Tensoren ab.

## Team

| GitHub | Name |
|---|---|
| [@nbomers](https://github.com/nbomers) | Noah Bomers |
| [@D4ne2kk](https://github.com/D4ne2kk) | Niels Dähne |
| [@eknight04](https://github.com/eknight04) | Erasmus Ritter | 

Entwickelt wird in Feature-Branches, `main` bleibt lauffähig; vor dem Push
`git pull --rebase origin main`, damit die Historie linear bleibt.
Notebook-Ausgaben entfernt `nbstripout` beim Commit von selbst, `ruff` und
`pytest` laufen bei jedem Push.

## Credits

**Genutzte Repositories und Datensätze**

| | Lizenz | Verwendung |
|---|---|---|
| [Mapillary](https://www.mapillary.com) — Bilder und Metadaten | [CC BY-SA 4.0](https://www.mapillary.com/terms) | Datensatz: 332.868 Straßenbilder aus Osnabrück samt GPS, Kompass, Sequenz |
| [AnyLoc](https://github.com/AnyLoc/AnyLoc) | BSD-3-Clause | DINOv2-ViT-G-Merkmale, VLAD, Domänen-Vokabular |
| [EigenPlaces](https://github.com/gmberton/EigenPlaces) | MIT | vortrainierter ResNet-50-Encoder |
| [MegaLoc](https://github.com/gmberton/MegaLoc) | MIT | vortrainierter Encoder |
| [MixVPR](https://github.com/amaralibey/MixVPR) | keine Lizenzdatei im Repository | vortrainierter ResNet-50-Encoder und Gewichte |
| [CLIP](https://github.com/openai/CLIP) über 🤗 Transformers | MIT | ViT-B/32 als Baseline ohne VPR-Training |
| [LightGlue](https://github.com/cvg/LightGlue) | Apache-2.0 | SuperPoint + LightGlue für die geometrische Verifikation |
| [FAISS](https://github.com/facebookresearch/faiss) | MIT | die Suche in 06 |
| [OSMnx](https://github.com/gboeing/osmnx) / OpenStreetMap | MIT / [ODbL](https://www.openstreetmap.org/copyright) | Stadtgrenze, Straßennetz, Stadtteile |

Die Mapillary-Bilder werden von Mapillary vor der Veröffentlichung
automatisch unkenntlich gemacht — Gesichter und Kennzeichen sind verpixelt.
Sie liegen nicht im Repository, sondern werden über `03_image_download`
mit eigenem API-Token geladen. Die CC-BY-SA-Lizenz verlangt Namensnennung
und Weitergabe unter gleichen Bedingungen; wer abgeleitete Datensätze
veröffentlicht, muss das beachten.

**Werkzeuge**

- [claude.ai](https://claude.ai) — Unterstützung bei Recherche und Entwicklung

**Literatur**

- CLIP — Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021
- DINOv2 (Basis von AnyLoc und MegaLoc) — Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024
- AnyLoc — Keetha et al., *AnyLoc: Towards Universal Visual Place Recognition*, IEEE RA-L 2023
- MixVPR — Ali-bey, Chaib-draa, Giguère, *MixVPR: Feature Mixing for Visual Place Recognition*, WACV 2023
- EigenPlaces — Berton, Trivigno, Caputo, Masone, *EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition*, ICCV 2023
- MegaLoc — Berton, Masone, *MegaLoc: One Retrieval to Place Them All*, arXiv 2502.17237, 2025
- VLAD — Jégou, Douze, Schmid, Pérez, *Aggregating local descriptors into a compact image representation*, CVPR 2010
- PCA-Whitening für VLAD — Jégou, Chum, *Negative evidences and co-occurences in image retrieval: The benefit of PCA and whitening*, ECCV 2012
- MSLS — Warburg et al., *Mapillary Street-Level Sequences: A Dataset for Lifelong Place Recognition*, CVPR 2020
- SuperPoint — DeTone, Malisiewicz, Rabinovich, *SuperPoint: Self-Supervised Interest Point Detection and Description*, CVPR Workshops 2018
- LightGlue — Lindenberger, Sarlin, Pollefeys, *LightGlue: Local Feature Matching at Light Speed*, ICCV 2023
- FAISS — Johnson, Douze, Jégou, *Billion-scale similarity search with GPUs*, IEEE Transactions on Big Data 2019

## Lizenz

Der Code in diesem Repository steht unter der [MIT-Lizenz](LICENSE).

Nicht darunter fallen die Mapillary-Daten (CC BY-SA 4.0), die geklonten
Fremd-Repositories unter `external/` und die Modellgewichte unter `weights/`
— deren Lizenzen stehen oben unter Credits.
