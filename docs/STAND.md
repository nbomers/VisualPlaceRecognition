# Stand des Projekts — Übergabe

Für jemanden, der nur dieses Repository sieht. Stand 2026-09-15. Zweck:
ohne den Verlauf der letzten Wochen im Kontext weiterarbeiten zu können.
Alles hier ist gemessen, nichts geschätzt; jede Zahl lässt sich mit dem
genannten Befehl neu erzeugen. `docs/AUFTRAG.md` ist die Aufgabenliste des
Autors (er kreuzt an), `README.md` die Nutzerdokumentation,
`experiments/README.md` hat alle Zahlen der Nebenuntersuchungen.

## Was das Projekt ist

Visual Place Recognition auf 332.868 Mapillary-Bildern aus Osnabrück:
Anfragebild rein, Ort raus, per Vergleich mit Referenzbildern.
Programmierpraktikum, Universität Osnabrück (Noah Bomers, Niels Dähne,
Erasmus Ritter). Der Eigenanteil ist der **Benchmark** — Sequenz-Split
70/15/15 (train/database/query), vier Ground-Truth-Varianten (Alle, Hard,
Blickrichtung, Nicht-Panorama), R@k bei 5/10/25/50/100 m, Fingerabdrücke,
Konfidenzintervalle, zwei Referenzprotokolle, 36 vergleichbare Zeilen —
nicht die Modelle. Kein Encoder wird trainiert; fünf veröffentlichte
Encoder (CLIP, AnyLoc, MixVPR, EigenPlaces, MegaLoc) plus 13 abgeleitete
Varianten (PCA, PCA-Whitening, Verkettung) werden verglichen.

## Zustand

- `ruff check .` sauber, `pytest tests/` 23 Tests in 4 s, CI
  (`.github/workflows/check.yml`) läuft bei jedem Push.
- Git-Historie ist umgebaut (2026-09-15): ein Autorname, keine
  Claude-Zeilen, Notebook-Ausgaben aus allen alten Versionen entfernt,
  `.git` 29 MB. `.mailmap` liegt bei, `docs/history/` enthält das
  benutzte Skript (muss nicht committet werden).
- Alle Pfade haben eine **Stadt-Dimension** (`src/paths.py`): `city` in
  `config.yaml` → Slug (`Osnabrück, Germany` → `osnabrueck`,
  `Halle (Saale), Germany` → `halle-saale`) → `data/<stadt>/`,
  `results/<stadt>/`, `experiments/results/<stadt>/`,
  `weights/adapter/<stadt>/`, Bilder unter `image_root/<stadt>`
  (`~/Downloads/mapillary/<stadt>`, je Rechner per `VPR_IMAGE_ROOT` oder
  `VPR_IMAGE_PATH`), eigene Fotos unter `image_root/test`. Ein frischer
  Klon kann mit einer anderen Stadt laufen, ohne Osnabrück anzufassen.
- Demo vollständig: `python locate.py foto.jpg` (auch Ordner), Notebook
  `demo/demo.ipynb`; alle Encoder außer AnyLoc-Varianten (deren PCA aus 04
  liegt nicht neben den Embeddings — bewusst ausgeklammert).
- Auf dem Mac liegen alle Embeddings und Trefferlisten für Osnabrück
  (78 GB unter `data/osnabrueck/embeddings/`). Der Linux-Klon (RTX 3070)
  war zuletzt auf der alten Ordnerstruktur ohne Daten; er muss auf
  `origin/main` gesetzt werden und Embeddings/Trefferlisten/Adapter/`.env`
  in die neue Struktur kopiert bekommen (siehe „Umgebung").
- Nicht committet auf dem Mac (Stand 2026-09-15): `experiments/README.md`
  (Stadtwahl-Abschnitt mit Mainz, Gütersloh, Kachel-Survey),
  `experiments/city_coverage.py` (`--tiles-only`, `--pause`),
  `experiments/results/city_coverage.json` (36 Städte),
  `experiments/results/osnabrueck/localization_aggregation_megaloc.{json,png}`,
  drei neu gezeichnete `vergleich_*_derived.png`, `docs/` (untracked).

## Wo was liegt

| | |
|---|---|
| `config.yaml` | einziger Schalter: `city`, `image_root`, `vpr.method`/`adapter`, Split, Radien, Encoder-Blöcke (YAML-Anker `&pca512`/`&pcaw512` für die abgeleiteten). Alles darin fließt in die Fingerabdrücke (`run_guard._PFLICHT`) |
| `run.py` | Pipeline 01–08, überspringt Fertiges per Fingerabdruck; `--method all` = 5 Encoder, `--method derived` = 13 abgeleitete, `--adapter all` = none,linear; `--from 06`, `--force` |
| `compare.py` | Tabelle (`--split`, `--threshold`, `--ci`, `--reference database\|full`, `--derived` für alle Zeilen), `--plot`, `--localization` |
| `locate.py` | ein Foto oder Ordner verorten; dahinter `src/locate.py` (`Locator`: lat/lon, Konfidenz = Top-1-Ähnlichkeit, Marge, Streuung, Treffer) |
| `src/` | `paths` (Stadt-Dimension), `config`, `geo` (Haversine, UTM), `mapillary` (Kacheln, Download), `split`, `pairs`, `districts` (OSM-Stadtteile, place-Rückfall), `evaluation` (07-Rechnung, KDTree-Kandidaten, exakt), `retrieval` (Trefferlisten laden, auch fullref), `adapter_training` (05), `locate`, `device`, `run_guard` (Fingerabdrücke, `validate_config`, `code_version`), `models/factory.py` + `models/derived.py` |
| `notebooks/01–08` | 01 Kacheln + Split + Stadtteile (nur je Stadt), 02 Audit (`results/<stadt>/dataset_audit.json`), 03 Download (legt `image_root/<stadt>` und `test/README.txt` an), 04 Embeddings, 05 Adapter (132 Zeilen auf `src/adapter_training`), 06 Retrieval (memmap, FAISS), 07 Evaluation, 08 Lokalisierung (nur Top-1) |
| `tests/` | 23 Tests: Auswertung handgerechnet (auch Panoramen), Split gegen versionierte Listen, Paare, Config, Ergebnis-JSONs gegen Bootstrap, abgeleitete Projektion, Adaptertraining. Tests für fehlende Artefakte werden übersprungen — eine neue Stadt bricht sie nicht |
| `experiments/` | 16 Skripte, je eine Frage; `_common.py` liefert `CFG, PATHS, RESULTS, ROOT`. `bootstrap_ci.py` (`--reference full`), `full_reference.py`, `rejection_curve.py`, `recall_by_difficulty.py`, `recall_by_district.py`, `confusion_atlas.py`, `timing.py`, `localization_aggregation.py`, `database_density.py`, `pca_reduce.py` (`--projection-only`), `concat_embeddings.py`, `sequence_retrieval.py`, `geometric_verification.py` (gebaut, nie mit Bildern gelaufen), `detection_rerank.py`, `detection_probe.py`, `city_coverage.py` |
| `data/<stadt>/processed/` | `metadata.parquet` (im Git, 9 MB) und drei `*_sequences.txt` — Zeilenreihenfolge ist Teil jedes Fingerabdrucks; Rest von `data/` gitignored |
| `results/<stadt>/evaluation/*.json` | 55 JSONs im Git (18 Encoder × Benchmark / `_linear` / `_fullref`, plus `seq3`); `localization/` 36; `figures/` im Git |
| `experiments/results/<stadt>/` | JSONs und Abbildungen der Experimente, im Git; `experiments/results/city_coverage.json` stadtübergreifend |
| `weights/adapter/<stadt>/` | trainierte Adapter, gitignored; MixVPR-Checkpoint `weights/mixvpr/`, AnyLoc-Vokabular `external/AnyLoc/cache/` (`setup_external.py`) |
| `cache/` | osmnx-Antworten (Stadtgrenze, Stadtteile, Straßennetz), Detections — ein Cache für alle Städte |
| `.env` | `MAPILLARY_TOKEN`, gitignored, nie committen |

## Ergebnisse — R@1 bei 25 m, Osnabrück

Zwei Protokolle. **Benchmark:** Referenz = database (48.321 Bilder),
34.112 lösbare von 53.414 Anfragen. **Volle Referenz:** database + train
(279.453 Bilder), 48.177 lösbar — `variant: "fullref"`,
`compare.py --reference full --ci`.

| Encoder | Benchmark | volle Referenz |
|---|---|---|
| clip | 0.073 | 0.232 |
| anyloc / anyloc_pcaw4096 | 0.204 / 0.321 | 0.435 / 0.540 |
| mixvpr | 0.426 | 0.653 |
| eigenplaces / eigenplaces_pcaw512 | 0.484 / 0.507 | 0.701 / 0.715 |
| megaloc / megaloc_pcaw512 | **0.568** / 0.541 | **0.798** / 0.778 |
| eigenplaces_megaloc_concat (1024) | 0.572 | 0.778 |

**Unsicherheit** (`bootstrap_ci.py`, 1.000 Ziehungen über 198
Query-Sequenzen): absolute R@1 tragen ±0.03 (CLIP) bis ±0.10 (MegaLoc
[0.475, 0.666]); gepaarte Differenzen sind eng. Verkettung schlägt MegaLoc
nicht (+0.004, [−0.005, +0.014]); mit voller Referenz liegt sie sogar
darunter (−0.019). Adaptergewinn = Whitening (nach PCAW schließen die
Gewinne 0 ein, der Schaden bei VPR-trainierten Encodern bleibt).

**Weitere Befunde** (alle in README/experiments-README mit Befehl):
Blickrichtung ist der größte Einzelfaktor (kein Nachbar mit passender
Richtung → R@1 0.071 statt 0.653); Zeitabstand 8–30 Tage 0.819, > 365 Tage
0.473; 51+ Nachbarn 0.781; selber Fotograf am selben Tag 0.690 statt 0.556.
Ablehnung: Top-1-Ähnlichkeit ist das beste Konfidenzmaß (Präzision 0.689
bei 80 % Abdeckung, AUC 0.791). Fehler bimodal (45 % < 100 m, 44 % > 1 km).
Stadtteile streuen Faktor 12, Rangfolge encoderunabhängig (Spearman 0.96).
Nichts schlägt Top-1 (sechs Negativergebnisse). Encodieren auf dem M1 Pro:
CLIP 145, MixVPR 73, EigenPlaces 33, MegaLoc 21 Bilder/s; AnyLoc 15 auf
der RTX 3070 (fp16, Batch 4). Suche 0.5–6.4 ms je Anfrage (FAISS flat).

## Nächstes Vorhaben: zweite Stadt

Frage: funktioniert das Ergebnis in einer Stadt, die Mapillary dicht
erschlossen hat? Osnabrück hat nur 44 % der Straßen (38 % der
Wohnstraßen) mit einem Bild im Umkreis von 25 m — daher 36 % unlösbare
Anfragen. `experiments/city_coverage.py` misst das für Kandidaten
(Kacheln wie 01 + OSM-Straßennetz); 36 Städte in
`experiments/results/city_coverage.json`, Tabelle und Befund in
`experiments/README.md` → Stadtwahl.

**Entscheidung offen zwischen drei Städten** (Heilbronn und alle anderen
sind verworfen):

| | Bilder | ×Osnabrück | Straßen / Wohn | Seq. | Fotografen | größter | seit 2022 |
|---|---|---|---|---|---|---|---|
| Würzburg | 428.847 | 1,3 | 98 % / 98 % | 2.004 | 92 | 51 % | 28 % |
| Jena | 699.097 | 2,1 | 99 % / 99 % | 4.516 | 62 | 51 % | 45 % |
| Halle (Saale) | 919.790 | 2,7 | 92 % / 91 % | 3.931 | 72 | 30 % | 87 % |

Abwägung: Würzburg ist am billigsten und hat die meisten Fotografen, aber
72 % der Bilder sind älter als 2022 (großer Zeitabstand, ältere Kameras).
Jena hat doppelt so viele Sequenzen → engere Bootstrap-Intervalle (die
Osnabrück-Intervalle sind ±0.05–0.10 breit, weil nur 198 Query-Sequenzen
die Stichprobe sind). Halle hat die sauberste Fotografenverteilung und
frische Bilder, kostet aber fast dreimal Osnabrück. Fotografenanteil
~50 % entspricht Osnabrück (47 %) und ist für den direkten Vergleich
erwünscht.

**Gütersloh** (99 % Straßen, 10.636 Sequenzen, 75 % frisch) ist als
*drittes* Experiment vorgemerkt: 80 % der Bilder und 94 % der Sequenzen
stammen von einem Mapper (seit 2014, 259 Aufnahmetage). Query und
Datenbank wären dasselbe Gerät, ein Vergleich mit Osnabrück nicht von der
Kamera zu trennen — aber genau das lässt sich dort messen
(Query-Sequenzen des Top-Fotografen gegen die übrigen 684). Mainz (74 %)
hat dasselbe Problem. Detroit/Berlin: verworfen.

**Ablauf für eine neue Stadt** (der Autor führt alles aus, was Bilder oder
Overpass braucht — siehe „Umgebung"):

```bash
# config.yaml: city: "Würzburg, Germany"   (bzw. Jena / Halle (Saale))
python run.py --method megaloc          # 01 Kacheln+Split+Stadtteile, 02, 03 Download, 04, 06, 07, 08
python run.py --method eigenplaces
python experiments/pca_reduce.py --methods eigenplaces_pcaw512,megaloc_pcaw512
python experiments/concat_embeddings.py
python run.py --method eigenplaces_pcaw512,megaloc_pcaw512,eigenplaces_megaloc_concat
python experiments/full_reference.py
python experiments/bootstrap_ci.py && python experiments/bootstrap_ci.py --reference full
python compare.py --ci && python compare.py --reference full --ci
python experiments/recall_by_difficulty.py && python experiments/rejection_curve.py
python experiments/recall_by_district.py && python experiments/confusion_atlas.py
```

Empfehlung: nur MegaLoc und EigenPlaces (die beiden besten) samt
pcaw512 und Verkettung, kein Adapter (die Adapter-These ist mit
Osnabrück abgeschlossen); CLIP/MixVPR/AnyLoc nur, wenn Zeit übrig ist.
Was zu vergleichen ist: Anteil lösbarer Anfragen (erwartet deutlich über
64 %), R@1 beider Protokolle mit Intervallen, das Schwierigkeitsprofil
(Blickrichtung/Zeit/Fotograf), die Ablehnungskurve. Die
Osnabrück-Ergebnisse bleiben unberührt unter `results/osnabrueck/`;
`compare.py` und die Experimente lesen immer die konfigurierte Stadt.

Kosten (Mac-Durchsatz, 04 je Encoder): Würzburg MegaLoc 5,7 h, EigenPlaces
3,6 h; Jena 9,3 h / 5,9 h; Halle 12 h / 7,7 h. Auf der 3070 schneller
(nicht gemessen außer AnyLoc). Speicher: Bilder ~180 KB je Bild (Würzburg
~77 GB, Jena ~125 GB, Halle ~165 GB), MegaLoc-Embeddings 33,8 KB je Bild
(Würzburg 14 GB), EigenPlaces 8,2 KB. 03 dauert Stunden, proportional zur
Bildzahl. 01 braucht Overpass (osmnx: Stadtgrenze, Stadtteile) und den
Mapillary-Token; für Halle müssen die `admin_levels` in `config.yaml`
eventuell angepasst werden — `src/districts.py` fällt sonst auf
`place`-Tags zurück.

Bekannt und in der Interpretation zu nennen: 01 zieht einen frischen
Mapillary-Schnappschuss (Bildzahl weicht von `city_coverage.json` leicht
ab); der Split-Seed 42 bleibt; alle Fingerabdrücke enthalten `city`.

## Umgebung

- **Mac** (M1 Pro, 16 GB, MPS): Python der Projektumgebung
  `/opt/homebrew/anaconda3/envs/pytorch/bin/python` (3.14; das `python`
  im PATH ist miniforge ohne numpy), ruff unter
  `/opt/homebrew/anaconda3/pkgs/ruff-0.12.0-py313h59dbcda_0/bin/ruff`,
  pytest per pip in der Umgebung. Die Login-Shell ist fish, das Bash-Tool
  läuft zsh — `echo ====` scheitert, `set -x` setzt nichts.
- **`~/Downloads` ist für den Agenten-Prozess gesperrt** (macOS-TCC,
  auch ohne Sandbox). Alles mit Bildzugriff — 03, 04, `timing.py` ohne
  `--skip-encode`, `geometric_verification.py`, `locate.py` — startet der
  Autor im eigenen Terminal (`conda activate pytorch`).
- **Overpass** (osmnx: Stadtgrenze, Stadtteile, Straßennetz) ist aus der
  Agenten-Sandbox seit 2026-09-14 „Connection refused"; Mapillary-Kacheln
  gehen. Cache unter `cache/` deckt Osnabrück ab; 01 und `city_coverage.py`
  für neue Städte laufen beim Autor (Mac oder Linux).
- **Linux** (RTX 3070, 8 GB, cuda): AnyLoc nur fp16 mit Batch 4; MegaLoc
  und AnyLoc für Osnabrück wurden dort gerechnet. Der Klon dort muss auf
  `origin/main` (`git fetch && git reset --hard origin/main`), danach
  `.env` anlegen und die Osnabrück-Artefakte in die neue Struktur kopieren:
  Embeddings → `data/osnabrueck/embeddings/`, Trefferlisten →
  `results/osnabrueck/retrieval/`, Adapter → `weights/adapter/osnabrueck/`,
  Bilder → `~/Downloads/mapillary/osnabrueck` (oder `VPR_IMAGE_PATH`).
- Auf verschiedenen Rechnern gerechnete Encoder halten dieselben Bilder in
  verschiedener Zeilenreihenfolge; alles, was Encoder nebeneinanderlegt,
  richtet über `image_id` aus.

## Offen

- **README:** der Absatz „Nächster Schritt mit mehr Zeit" unter „Mögliche
  Erweiterungen" nennt Ablehnungskurve und Recall-gegen-Nachbarzahl noch
  als offen — beide sind umgesetzt; der Absatz „Eine zweite Stadt" sollte
  auf `experiments/README.md` → Stadtwahl verweisen statt Dichten zu
  listen. Der Satz zu claude.ai-Credits ist Entscheidung des Autors.
- `AUFTRAG.md` Punkte 6 (Recall über alle Anfragen als zweite Spalte —
  `rejection_curve.py` hat die Zahl: 0.363 für MegaLoc), 9–12 (Merkmal-
  Cache für die geometrische Verifikation, diagonaler Adapter, zweite
  Adapter-Hyperparameter, Shrinkage-Test), Rest von 15. Punkte 7, 8, 14
  sind faktisch erledigt (`recall_by_difficulty.py`, `tests/` + CI,
  `rejection_curve.py`), aber nicht angekreuzt.
- Geometrische Verifikation (`geometric_verification.py`) ist gebaut, nie
  mit Bildern gemessen — braucht GPU und Bilder, ~15 min für 2.000
  Anfragen auf der 3070.
- `anyloc_pcaw4096`-Projektion ist nicht reproduzierbar (min. Kosinus
  0.992 zur gespeicherten Variante), die `.npz` wurde gelöscht; AnyLoc
  bleibt in der Demo ausgeklammert.

## Bekannte Eigenheiten — kein Fehler, aber wissenswert

- „Hard" ist ein ODER: anderer Fotograf *oder* > 180 Tage. Milder als MSLS.
- `compass_angle` hat −1,0 („unbekannt") und Werte über 360; `% 360`
  faltet korrekt.
- Lösbar-Zahlen unterscheiden sich um 7 Anfragen zwischen UTM (34.105)
  und Haversine (34.112) — Grenzfälle bei exakt 25 m; 07/08 rechnen
  Haversine.
- MegaLoc hat zwei `train`-Bilder weniger (Download gescheitert); für 07
  und 08 egal.
- Die randomisierte SVD der PCA-Varianten rundet auf anderem BLAS minimal
  anders; Fingerabdrücke prüfen die Konfiguration, nicht den Inhalt.
- `--method all` nimmt nur die 5 Basis-Encoder; abgeleitete brauchen
  `derived` oder den Namen.
- Notebooks brauchen `E402` als ruff-Ausnahme (`sys.path` vor `from src`).
- `git shortlog -sn` ohne `HEAD` liest stdin und hängt im Hintergrund.

## Arbeitsweise, die gilt

- Der Autor committet und pusht selbst. **Nie committen, nie pushen.**
- Alles, was die Git-Historie oder gelöschte Dateien betrifft, vorher
  zur Bestätigung vorlegen.
- Kommentare Deutsch, knapp, ohne „vorher war es anders"; Funktionsnamen
  Englisch. `ruff check .` sauber vor jedem Bericht; Notebooks müssen
  parsen.
- Auswertungen laufen über `src/evaluation.py`; Encoder werden über
  `image_id` ausgerichtet. Zahlen immer mit dem erzeugenden Befehl.
- Erst messen, dann bauen; Negativergebnisse werden in
  `experiments/README.md` dokumentiert, nicht gelöscht.
- Notebook-Ausgaben werden erst ganz am Ende aller Arbeiten entfernt
  (`nbstripout` ist installiert).

## Schlussprüfung — was man laufen lassen kann

```bash
ruff check .                                   # sauber
pytest tests/                                  # 23 Tests, 4 s
python compare.py --ci && python compare.py --reference full --ci
python compare.py --localization
python compare.py --plot && python compare.py --plot --derived
python run.py --method clip --adapter none --from 07    # 4 min, prüft 07+08 mit src/evaluation
python experiments/bootstrap_ci.py             # ~1 min, prüft alle Zeilen gegen 07
python experiments/bootstrap_ci.py --reference full    # ~12 min
python experiments/timing.py --skip-encode     # ~1.5 min
python experiments/recall_by_district.py       # ~1 min, osmnx aus dem Cache
python experiments/confusion_atlas.py          # ~1 min
python experiments/rejection_curve.py && python experiments/recall_by_difficulty.py
```

## Literatur — für README-Credits und Ausarbeitung

Aus dem Gedächtnis; Jahr und Venue vor dem Zitieren prüfen.

- CLIP — Radford et al., *Learning Transferable Visual Models From Natural
  Language Supervision*, ICML 2021.
- DINOv2 (Basis von AnyLoc und MegaLoc) — Oquab et al., *DINOv2: Learning
  Robust Visual Features without Supervision*, TMLR 2024.
- AnyLoc — Keetha et al., *AnyLoc: Towards Universal Visual Place
  Recognition*, IEEE Robotics and Automation Letters 2023.
- MixVPR — Ali-bey, Chaib-draa, Giguère, *MixVPR: Feature Mixing for Visual
  Place Recognition*, WACV 2023.
- EigenPlaces — Berton, Trivigno, Caputo, Masone, *EigenPlaces: Training
  Viewpoint Robust Models for Visual Place Recognition*, ICCV 2023.
- MegaLoc — Berton, Masone, *MegaLoc: One Retrieval to Place Them All*,
  arXiv 2502.17237, 2025.
- VLAD — Jégou, Douze, Schmid, Pérez, *Aggregating local descriptors into
  a compact image representation*, CVPR 2010.
- PCA-Whitening für VLAD — Jégou, Chum, *Negative evidences and
  co-occurences in image retrieval: The benefit of PCA and whitening*,
  ECCV 2012.
- MSLS — Warburg et al., *Mapillary Street-Level Sequences: A Dataset for
  Lifelong Place Recognition*, CVPR 2020.
- SuperPoint — DeTone, Malisiewicz, Rabinovich, *SuperPoint:
  Self-Supervised Interest Point Detection and Description*, CVPR
  Workshops 2018.
- LightGlue — Lindenberger, Sarlin, Pollefeys, *LightGlue: Local Feature
  Matching at Light Speed*, ICCV 2023.
- FAISS — Johnson, Douze, Jégou, *Billion-scale similarity search with
  GPUs*, IEEE Transactions on Big Data 2019.
