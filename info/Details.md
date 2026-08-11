# Osnabrück VPR – Detaillierter Ablauf

## 1. OSM / Stadtpolygon

Das Stadtpolygon definiert, welche Mapillary-Daten zum Datensatz gehören.

```text
OSM → Stadtpolygon → Mapillary-Abfrage
```

Für räumliche Berechnungen ein geeignetes CRS verwenden.

## 2. Mapillary Harvest

Wichtige Metadaten:

```text
image_id
sequence_id
captured_at
lat
lon
compass_angle
is_pano
creator_id
```

`sequence_id` ist später die zentrale Einheit für den Split.

## 3. Sequence-aware Split

Nicht einzelne Bilder zufällig splitten:

```text
Bild 1 → Database
Bild 2 → Query
```

sondern komplette Sequenzen:

```text
Sequence A → TRAIN
Sequence B → DATABASE
Sequence C → QUERY
```

Prüfung:

```text
TRAIN ∩ DATABASE = ∅
TRAIN ∩ QUERY = ∅
DATABASE ∩ QUERY = ∅
```

## 4. Aktueller Datensatz

```text
332.544 Bilder
1.322 Sequenzen
Median: 182 Bilder/Sequenz
Maximum: 3.156 Bilder
Zeitraum: 2014-08 bis 2026-06
Stadtteile: 23

TRAIN: 925 Sequenzen
DATABASE: 198 Sequenzen
QUERY: 199 Sequenzen
```

## 5. Dataset Audit

`02_dataset_audit.ipynb` verändert die Daten zunächst nicht.

### Grundstatistik
- Bilder
- Sequenzen
- Bilder/Sequenz
- Minimum / Maximum / Median / Mittelwert

### Missing Values
Für:
`image_id`, `sequence_id`, `captured_at`, `lat`, `lon`, `compass_angle`, `is_pano`, `creator_id`.

### Sequence Audit
- Verteilung der Sequenzgrößen
- Histogramm
- Top-Sequenzen
- sehr kurze/lange Sequenzen prüfen

### Split Audit
- Sequenzen je Split
- Bilder je Split
- Schnittmengen der Splits müssen leer sein

### Zeitliche Analyse
- Bilder pro Jahr/Monat
- getrennt nach Train, Database, Query
- möglichen Temporal Shift untersuchen

### Räumliche Analyse
Aus `lat/lon` eine GeoDataFrame erzeugen.

Darstellen:
- gesamte Coverage
- Database vs. Query
- Abdeckung nach Stadtteil

Zentrale Frage:

> Gibt es in Query-Gebieten ausreichend Database-Abdeckung?

## 6. Räumliche Ground Truth

Für spätere Evaluation:

```text
≤ 10 m      Positive
10–25 m     Uncertain
> 25 m      Negative
```

Der Bereich 10–25 m ist kein sicherer Negativfall.

## 7. Embeddings

Erst nach bestandenem Audit.

```text
Bild → vortrainierter Backbone → Embedding
```

Zunächst Baseline ohne Fine-Tuning.

Mögliche Backbones werden später verglichen.

## 8. Retrieval

```text
Query-Bild
 ↓
Query-Embedding
 ↓
Similarity Search gegen DATABASE
 ↓
Top-1 / Top-5 / Top-10
```

Für effiziente Suche kann FAISS eingesetzt werden.

## 9. GPS-Aggregation

Mindestens drei Varianten:

### Top-1
GPS des besten Treffers.

### Score-weighted centroid
Ähnlichere Treffer erhalten mehr Gewicht.

### Spatial clustering
Räumlich zusammenliegende Top-k-Treffer bilden einen dominanten Cluster.

## 10. Evaluation

Für jedes Query:

```text
Prediction → Ground Truth → geografische Distanz
```

Metriken:
- Recall@1
- Recall@5
- Recall@10
- bei 5 m, 10 m, 25 m, 50 m, 100 m
- Median Localization Error

## 11. OOD-Rejection

Ein Query außerhalb Osnabrücks darf nicht zwangsläufig einer Osnabrück-Position zugeordnet werden.

```text
Query
 ↓
Retrieval
 ↓
Confidence
 ↓
Location oder UNKNOWN
```

Die Schwelle wird experimentell untersucht.

## 12. Sequence-VPR

Erst nach der Image-Baseline:

```text
Image → Sequence
Sequence → Image
Sequence → Sequence
```

Sequenzinformation kann später für robustere VPR-Auswertung genutzt werden.

## 13. GitHub vs. eigener Code

### Selbst schreiben
- Coverage
- Dataset Split
- Audit
- Ground Truth
- GPS-Aggregation
- Evaluation
- OOD

### Bestehende Komponenten nutzen
- PyTorch
- GeoPandas
- Pandas
- FAISS
- vortrainierte Backbones
- geeignete VPR-Repositories

Bei übernommenem GitHub-Code immer die Lizenz prüfen.

## 14. Projektstruktur

```text
project/
├── data/
│   ├── metadata.parquet
│   ├── train_sequences.txt
│   ├── database_sequences.txt
│   └── query_sequences.txt
├── notebooks/
│   ├── 01_coverage.ipynb
│   ├── 02_dataset_audit.ipynb
│   ├── 03_embeddings.ipynb
│   ├── 04_retrieval.ipynb
│   └── 05_evaluation.ipynb
├── embeddings/
├── results/
└── figures/
```

## Arbeitsprinzip

```text
Daten korrekt
   ↓
Daten verstanden
   ↓
Baseline reproduzierbar
   ↓
Retrieval
   ↓
Evaluation
   ↓
Optimierung
```

Nicht zum nächsten Schritt springen, bevor der vorherige belastbar ist.
