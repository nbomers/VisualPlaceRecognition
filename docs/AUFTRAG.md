# Auftrag: Absicherung und Abschluss

Du arbeitest an einem VPR-Benchmark (Visual Place Recognition, Osnabrück).
**Lies zuerst `STAND.md` vollständig** — dort stehen Aufbau, Ergebnisse,
Befunde, Eigenheiten und die Regeln. Diese Datei hier sagt, was zu tun ist.

## Regeln

- Du committest nie und pushst nie. Der Autor macht das.
- Jede Änderung einzeln: was du vorhast, in zwei Sätzen — dann umsetzen —
  dann berichten, was sich geändert hat und mit welchem Befehl es geprüft
  wurde. Bei Entscheidungen, die die Ergebnisse verändern (Split, Ground
  Truth, Definitionen), erst fragen.
- Kommentare auf Deutsch, knapp, ohne „vorher war es anders".
  Funktionsnamen Englisch. Stil wie im Bestand: eine Anweisung je Zeile.
- Vor jedem Bericht: `ruff check .` muss sauber sein. Notebooks nach dem
  Ändern parsen (`ast.parse` je Code-Zelle). Nichts, was Fingerabdrücke
  entwertet, ohne es zu sagen.
- Alles Neue, das Trefferlisten bewertet, nutzt `src/evaluation.py`.
  Alles Neue, das Encoder nebeneinanderlegt, richtet über `image_id` aus.
- Erst messen, dann bauen. Ein Negativergebnis ist ein Ergebnis — es wird
  in `experiments/README.md` dokumentiert, nicht gelöscht.
- Zahlen im Bericht immer mit dem Befehl, der sie erzeugt.

## Freigabe

Der Autor kreuzt an. Nicht angekreuzte Punkte werden nicht bearbeitet.

**Stufe 1 — vor der Abgabe**
- [x] 1 Konfidenzintervalle per Sequenz-Bootstrap
- [x] 2 Laufzeit und Speicher je Encoder
- [x] 3 Metadaten versionieren, 01 deterministisch machen
- [x] 4 Recall je Stadtteil entscheide ob nur für den vollsten oder alle?
- [ ] 5 README zusammenführen

**Stufe 2 — die Geschichte abrunden**
- [ ] 6 Recall über alle Anfragen als zweite Spalte
- [ ] 7 Recall gegen Nachbarzahl und Zeitabstand je Anfrage
- [ ] 8 Test und CI
- [ ] 9 Merkmal-Cache für die geometrische Verifikation
- [ ] 10 Diagonaler Adapter
- [ ] 11 Adapter mit zweiter Hyperparameter-Einstellung
- [ ] 12 Shrinkage-Test für Whitening auf voller Breite

**Stufe 3 — nice to have**
- [x] 13 Verwechslungsatlas
- [ ] 14 Ablehnungskurve
- [ ] 15 Kleinkram (siehe unten)

Reihenfolge = Nummer. Nach jedem Punkt kurz berichten und weitermachen;
nicht am Ende alles auf einmal.

---

## Stufe 1

### 1 · Konfidenzintervalle per Sequenz-Bootstrap

**Warum.** Die 53.414 Anfragen stammen aus 198 Sequenzen; aufeinander-
folgende Frames scheitern gemeinsam. Der naive Standardfehler (±0.005)
unterschätzt die Unsicherheit deutlich. Ohne Intervalle sind Unterschiede
wie 0.484 vs 0.481 oder +0.004 der Verkettung gegenüber MegaLoc nicht
belegbar.

**Wo.** Neues Skript `experiments/bootstrap_ci.py`. Ergebnis nach
`experiments/results/bootstrap_ci.json`. Zusätzlich ein Schalter
`compare.py --ci`, der die Intervalle als Spalte neben R@1 zeigt, wenn die
JSON vorliegt.

**Wie.**
- Je Recall-JSON in `results/evaluation/` die zugehörige `.npz` und
  Metadaten laden (Pfade wie in `experiments/sequence_retrieval.py`).
- Einmal je Encoder und je k ∈ {1, 5, 10, 20} pro Anfrage berechnen: ist
  sie bei 25 m lösbar, und ist ein Treffer unter den ersten k? Das sind
  zwei Bool-Vektoren je k — mit `src.geo.haversine_distance` blockweise
  wie in `src/evaluation.py`, nicht per Anfrage.
- Bootstrap auf **Sequenz**-Ebene: 1.000-mal die 198 Query-Sequenzen mit
  Zurücklegen ziehen, Recall = Σ Treffer / Σ lösbar über die gezogenen
  Sequenzen. Das ist mit den vorberechneten Bool-Vektoren und einem
  `np.bincount` je Sequenz reine Arithmetik — keine Distanzen im Bootstrap.
- Seed aus `vpr.split_seed`. 95-%-Intervall = 2,5- und 97,5-Perzentil.
- Zusätzlich für die Paare, die im Text verglichen werden, den
  **gepaarten** Bootstrap der Differenz (dieselben gezogenen Sequenzen für
  beide Encoder): eigenplaces vs eigenplaces_pca512, eigenplaces vs
  eigenplaces_pcaw512, megaloc vs eigenplaces_megaloc_concat, jeder Encoder
  gegen seine `_linear`-Variante, eigenplaces vs mixvpr.

**Abnahme.** Tabelle mit R@1 und Intervall für alle 36 Zeilen; Liste der
gepaarten Differenzen mit Intervall und der Aussage „schließt 0 ein: ja /
nein". Ein Absatz in `experiments/README.md`, welche der sieben Befunde in
`STAND.md` die Intervalle überleben und welche nicht. Laufzeit unter zehn
Minuten für alle Encoder.

### 2 · Laufzeit und Speicher je Encoder

**Warum.** Das README nennt `eigenplaces_pcaw512` das effizienteste
Modell, ohne Zeitangabe.

**Wo.** Neues Skript `experiments/timing.py`, Ergebnis
`experiments/results/timing.json`; `compare.py` bekommt keine neue Spalte,
die Tabelle kommt ins README.

**Wie.**
- Encodierdurchsatz: für jeden Basis-Encoder, der auf diesem Rechner
  baubar ist (`src.models.factory.build_embedder`), 200 feste Bilder
  (Seed) encodieren, Bilder/s messen, Aufwärmlauf ausschließen. AnyLoc
  braucht Vokabular und PCA — wenn nicht vorhanden, überspringen und
  sagen. Gerät (`mps`/`cuda`/`cpu`) mit ins Ergebnis.
- Suchzeit: FAISS-Flat-Index über die Datenbank-Embeddings jedes
  Encoders (memmap laden, nur database-Zeilen), 1.000 Anfragen in Blöcken
  von 256, ms je Anfrage. Für alle 18 Encoder mit `.npy`, auch abgeleitete.
- Indexgröße: `n_database × dim × 4` Byte, und tatsächliche Größe der
  `.npy`.
- Die JSON ist **mergefähig**: je Encoder ein Eintrag mit Hostname und
  Gerät, bestehende Einträge anderer Rechner bleiben stehen. AnyLoc und
  MegaLoc werden auf dem GPU-Rechner gemessen; der Befehl dafür steht im
  Docstring.

**Abnahme.** Tabelle Encoder · Dim · Bilder/s (Gerät) · ms/Anfrage ·
Index MB. Ein Satz, ob `eigenplaces_pcaw512` die Effizienz-Behauptung
trägt (Erwartung: Suchzeit ∝ Dim, Encodierzeit hängt am Rückgrat, nicht an
der PCA).

### 3 · Metadaten versionieren, 01 deterministisch machen

**Warum.** `data/processed/metadata.parquet` ist gitignored; jeder Rechner
hat seinen eigenen Mapillary-Schnappschuss (Linux: zwei Bilder weniger)
und eine andere Zeilenreihenfolge (Threadpool-Ankunft in 01).

**Wie.**
- `.gitignore`: Ausnahme für `data/processed/metadata.parquet` (9 MB),
  nach dem Muster der bestehenden Ausnahmen für `*_sequences.txt`.
- In `notebooks/01_mapillary_coverage.ipynb` vor `metadata.to_parquet(...)`
  ein `sort_values(["sequence_id", "captured_at", "image_id"])` mit
  `reset_index(drop=True)` und einem Kommentar, dass das nur frische Läufe
  betrifft.
- **Nicht** 01 ausführen. **Nicht** die vorhandene Parquet sortieren — das
  würde alle Fingerabdrücke entwerten. Die Datei wird so versioniert, wie
  sie ist.
- README (Reproduzierbarkeit): ein Satz, dass die Metadaten im Git liegen
  und 01 nur für eine neue Stadt oder einen neuen Split läuft.

**Abnahme.** `git check-ignore data/processed/metadata.parquet` meldet
nichts; `git status` zeigt die Datei als neu; alle 35 Fingerabdrücke
passen weiterhin (Prüfskript in `STAND.md`).

### 4 · Recall je Stadtteil

**Warum.** Die anschaulichste Abbildung, die das Projekt haben kann: wo in
der Stadt scheitert es. 01 berechnet die Stadtteile bereits aus OSM.

**Wo.** Neues Skript `experiments/recall_by_district.py`, Ergebnis
`experiments/results/recall_by_district_<method>.json` und `.png`.

**Wie.**
- Stadtteile wie in 01 holen (osmnx, `ox.settings.cache_folder = cache/`,
  Tags und Filter aus `config.yaml → districts`; den Code aus 01 nicht
  duplizieren, sondern — falls 01 die Polygone irgendwo ablegt — laden;
  sonst dieselbe osmnx-Abfrage mit denselben Parametern).
- Jede Anfrage per Point-in-Polygon einem Stadtteil zuordnen (geopandas
  `sjoin`). Je Stadtteil: Anzahl Anfragen, Anteil lösbar, R@1 bei 25 m
  über die lösbaren — mit den Bool-Vektoren aus Punkt 1, falls vorhanden,
  sonst wie dort berechnet.
- Choroplethe: Stadtteile eingefärbt nach R@1, Beschriftung mit Name und
  n; Stadtteile mit unter 100 lösbaren Anfragen grau. Standard-Encoder
  `megaloc`, `--method` wählbar.
- Zweite Abbildung daneben: dieselbe Karte mit der Referenzdichte
  (Datenbankbilder je km²). Wenn beide Karten dasselbe Muster zeigen, ist
  der Dichte-Befund auf Stadtteil-Ebene belegt.

**Abnahme.** Tabelle Stadtteil · n · lösbar · R@1, sortiert nach R@1;
zwei PNGs; ein Absatz in `experiments/README.md` mit dem Muster.

### 5 · README zusammenführen

Nur wenn angekreuzt — das README ist die Stimme des Autors.

**Wie.** `README_VORSCHLAG.md` wird zur Basis. Aus dem bestehenden
`README.md` übernehmen, was dort neuer oder persönlicher ist (Abschnitt
„Über das Projekt", falls der Autor ihn geändert hat). Den Absatz „Bester
Adapter-Einfluss" auf die Lesart aus `STAND.md` Befund 3 bringen. Die
Literatur aus `STAND.md` unter Credits einsetzen; „MLSL" → „MSLS". Das
Pipeline-Diagramm als Mermaid-Block (`flowchart TD`) unter „Pipeline".
Ergebnisse aus Punkt 1, 2 und 4 einarbeiten, sobald sie vorliegen.
Danach `README_VORSCHLAG.md` löschen — zwei READMEs sind schlimmer als
eines.

**Abnahme.** Ein README; jeder Befehl darin existiert (Prüfschleife in
`STAND.md`); keine TODOs außer Team-Namen und -Rollen.

---

## Stufe 2

### 6 · Recall über alle Anfragen als zweite Spalte

**Warum.** „Lösbar" wandert mit der Schwelle, deshalb fällt R@1 von 50 auf
100 m. Eine Spalte, die unlösbare Anfragen als Fehlschlag zählt, ist
monoton in der Schwelle und erzählt die Dichte-Geschichte von selbst.

**Wie.** In `src/evaluation.py` je Schwelle zusätzlich
`recall_alle = hits / n_queries` in die JSON schreiben (Schlüssel
`recall_alle`), ohne bestehende Schlüssel zu ändern. `compare.py` bekommt
`--all-queries`, das diese Spalte statt `recall` zeigt. Danach 07 für alle
36 Kombinationen (`run.py --method all --adapter all --from 07` und
`--method derived …`), rund zwei Stunden — vorher fragen.

**Abnahme.** `compare.py --all-queries` zeigt bei jedem Encoder monoton
steigende Werte über `--threshold 5 … 100`.

### 7 · Recall gegen Nachbarzahl und Zeitabstand je Anfrage

**Wo.** `experiments/recall_by_difficulty.py`, Ergebnis unter
`experiments/results/`.

**Wie.** Je Anfrage: Anzahl Datenbankbilder im Umkreis von 25 m (KDTree
wie in `database_density.py`) und Zeitabstand zum nächsten davon in Tagen
(`captured_at` ist Millisekunden-Epoch). R@1 in Klassen: 1–2, 3–5, 6–10,
11–20, 21–50, 51+ Nachbarn; Zeitabstand 0–7 Tage, 8–30, 31–180, 181–365,
366+. Je Klasse n und R@1, als Balken. Für `megaloc` und
`eigenplaces_pcaw512`.

**Abnahme.** Zwei Abbildungen, Tabelle, ein Absatz: fällt R@1 bei wenigen
Nachbarn ein, und bei großem Zeitabstand?

### 8 · Test und CI

**Wie.**
- `tests/test_evaluation.py`: fünf synthetische Anfragen, zehn
  Datenbankbilder mit bekannten Koordinaten und Kompass, handgerechnete
  Erwartung für R@1/R@5 bei zwei Schwellen, für Standard und Blickrichtung.
  Dazu ein Test, dass `validate_config` die echte `config.yaml` annimmt und
  einen Tippfehler in `source` ablehnt. `pytest` in `environment.yml` und
  `requirements.txt`.
- `.github/workflows/check.yml`: Python 3.14, `pip install ruff pytest
  numpy pandas pyyaml tqdm`, dann `ruff check .` und `pytest tests/`. Keine
  Torch-Installation im CI — die Tests dürfen kein Torch brauchen.

**Abnahme.** `pytest tests/` grün lokal; Workflow-Datei parst (yaml).

### 9 · Merkmal-Cache für die geometrische Verifikation

**Warum.** `experiments/geometric_verification.py` extrahiert die
Datenbank-Merkmale je Anfrage neu — beim vollen Lauf 1,07 Mio.
Extraktionen für 48.321 Bilder.

**Wie.** Vorlauf `--build-cache`: SuperPoint-Merkmale aller
Datenbankbilder einmal berechnen, je Bild Keypoints (float16) und
Deskriptoren (float16) in eine `.npz` unter `cache/superpoint_<max_side>/`
(gitignored). Im Hauptlauf aus dem Cache laden, wenn vorhanden. Anfrage-
Merkmale weiter frisch.

**Abnahme.** Ein Lauf mit `--n-queries 200 --top-k 10` auf dem GPU-Rechner
ist mit Cache mindestens dreimal schneller als ohne; die Inlier-Zahlen sind
identisch.

### 10 · Diagonaler Adapter

**Warum.** Befund 3 in `STAND.md` — „der lineare Adapter lernt nur
Whitening" — hat einen direkten Test: ein Adapter mit nur einer Skalierung
je Dimension (d Parameter statt d²). Erreicht er dasselbe wie der volle,
ist die These bewiesen.

**Wie.** `src/models/adapter.py`: `DiagonalAdapter` (Parameter-Vektor
Skalierung, initialisiert mit 1, plus Bias 0, dann L2-Normalisierung wie
`LinearAdapter`). Auswahl über `config.yaml → vpr.adapter_training.kind:
linear | diagonal`, Standard `linear`. Der Adapter-Name in den Dateien
bleibt `linear` für `kind: linear`; für `diagonal` heißt er `diagonal` —
das erfordert Änderungen an `validate_config` (erlaubte Adapter), `run.py`
(`_stages`, Adapter-Artefakt), 05 (`ADAPTER_KIND`) und `_ERZEUGER` in
`run_guard.py`. Alle vier Stellen sind in `STAND.md` und `run.py` benannt;
**vorher die Liste der Stellen zeigen**, dann ändern. Der Kind fließt in
`adapter_fingerprint`.

Laufen für `eigenplaces_pcaw512` und `clip_pcaw512` (schnell) und für
`anyloc` roh (der Fall mit +0.130).

**Abnahme.** Drei Zeilen `diagonal` in `compare.py`. Aussage: liegt
`anyloc diagonal` nahe bei `anyloc linear` (0.335) oder bei `anyloc`
(0.204)?

### 11 · Adapter mit zweiter Hyperparameter-Einstellung

**Warum.** „Der Adapter schadet" ruht auf einer einzigen Konfiguration.

**Wie.** Nur die Werte in `config.yaml → vpr.adapter_training` ändern
(lr 0.0001, epochs 10, sonst gleich); die Fingerabdrücke entwerten dann
die bestehenden `_linear`-Artefakte korrekt. Damit die alten Zeilen
erhalten bleiben: die `_linear`-Ergebnisse von `eigenplaces_pcaw512` und
`megaloc_pcaw512` vorher unter `experiments/results/adapter_lr1e-3/`
sichern (JSONs kopieren). Dann 05–08 für beide laufen lassen
(`run.py --method eigenplaces_pcaw512,megaloc_pcaw512 --adapter linear`),
Werte zurücksetzen, Ergebnis-JSONs unter `experiments/results/adapter_lr1e-4/`
sichern, alte zurückkopieren. GPU-Rechner für MegaLoc. **Vorher fragen**
— das dauert Stunden und berührt versionierte Ergebnisse.

**Abnahme.** Tabelle beide Konfigurationen nebeneinander, R@1 mit
Bootstrap-Intervall aus Punkt 1. Aussage: bleibt das Vorzeichen des
Adaptereffekts?

### 12 · Shrinkage-Test

**Warum.** Die Erklärung „Whitening auf voller Breite verstärkt Rauschen
der kleinsten Hauptrichtungen" ist plausibel, nicht gemessen.

**Wie.** `experiments/pca_reduce.py` bekommt `whiten_eps` (Standard 0):
statt durch √λ durch √(λ + ε·λ_max) teilen. Eine Variante
`eigenplaces_pcaw2048_eps` in der config mit ε = 0.01, erzeugen, 06/07
laufen lassen.

**Abnahme.** Liegt `eigenplaces_pcaw2048_eps` deutlich über 0.459 (Richtung
0.484 oder darüber), ist die Erklärung belegt; sonst streichen und im
README als offen markieren.

---

## Stufe 3

### 13 · Verwechslungsatlas
Aus `results/retrieval/megaloc/…npz`: für falsche Top-1 (> 25 m) Pfeile
von echter zu geschätzter Position auf dem Straßennetz (osmnx, wie die
Demo), aggregiert nach Stadtteil-Paar (Punkt 4). Die zehn häufigsten Paare
als Tabelle. `experiments/confusion_atlas.py`.

### 14 · Ablehnungskurve
Für `megaloc`: Ähnlichkeit des besten Treffers als Konfidenz. Kurve
Präzision (Anteil richtig unter den beantworteten) gegen Abdeckung (Anteil
beantwortet) bei fallender Schwelle. Zusätzlich die Geschlossenheit aus 08
als zweite Konfidenz. `experiments/rejection_curve.py`. Aussage: bei 80 %
Abdeckung, wie hoch ist die Präzision?

### 15 · Kleinkram
- `run.py._is_valid`: die geschluckte Ausnahme mit einer Zeile drucken.
- `notebooks/08`: Inhaltsprobe wie in 07 (256 Similarities nachrechnen).
- `experiments/sequence_retrieval.py`: `--windows 0` abfangen.
- Demo, Encoder-Vergleich: fehlende `image_id` als Text in die Zeile
  schreiben statt leer lassen.
- `img_download_path`: Standard `data/images`, Override per
  `VPR_IMAGE_PATH` in `src/config.py`.
- `config.yaml`: die 13 abgeleiteten Blöcke über YAML-Anker entdoppeln
  (`&pca512 {pca_dim: 512, fit_images: 50000, whiten: false}` und `<<:`).
- `experiments/_common.py` für ROOT-Bootstrap und `load_config`.
- 05: `val_recall_at_1` durch `src.evaluation.evaluate_retrieval` ersetzen.

---

## Abschluss

Wenn alle freigegebenen Punkte durch sind:

1. Die Prüfschleife aus `STAND.md` (Linter, Notebooks, Fingerabdrücke,
   Einstiegspunkte, README-Befehle) einmal komplett laufen lassen und das
   Ergebnis berichten.
2. `STAND.md` aktualisieren: neue Skripte in „Wo was liegt", neue Zahlen
   bei den Befunden, erledigte Punkte aus „Offen" streichen, Datum setzen.
3. Ein Absatz für den Autor: welche Befunde haben die Intervalle
   überlebt, welche nicht, und was das für die Ausarbeitung heißt.
