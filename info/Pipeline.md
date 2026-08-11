# Osnabrück VPR – Pipeline

## Ziel

Reproduzierbare Visual-Place-Recognition-Pipeline für Osnabrück auf Basis von Mapillary.

```text
OSM / Stadtpolygon
        ↓
Mapillary Harvest / Coverage
        ↓
Metadata
        ↓
Data Cleaning
        ↓
Sequence-aware Split
        ↓
Dataset Audit
        ↓
Embeddings
        ↓
Image / Sequence Retrieval
        ↓
GPS Aggregation
        ↓
Localization
        ↓
Evaluation
        ↓
OOD Rejection
```

## Notebooks

### 01_coverage.ipynb
- Stadtpolygon
- Mapillary Harvest
- Metadaten
- Coverage
- Sequenzen
- reproduzierbare Splits

Artefakte:
- `metadata.parquet`
- `train_sequences.txt`
- `database_sequences.txt`
- `query_sequences.txt`

### 02_dataset_audit.ipynb
Prüft, ob der Datensatz für VPR geeignet ist:
- Grundstatistik
- Missing Values
- Sequenzgrößen
- Split-Leakage
- Zeit
- räumliche Verteilung
- Stadtteile
- GPS-Qualität
- Database-/Query-Abdeckung

### 03_embeddings.ipynb
```text
Bild → Backbone → Embedding D
```
Zunächst Baseline ohne Fine-Tuning.

### 04_retrieval.ipynb
```text
Query → Embedding → Similarity Search → Top-k
```
Top-1, Top-5, Top-10. Für die Suche kann FAISS verwendet werden.

### 05_evaluation.ipynb
- GPS Aggregation
- Recall@1/5/10
- 5/10/25/50/100 m
- Median Localization Error
- OOD-Rejection

## GPS-Definition

- `≤ 10 m` → Positive
- `10–25 m` → Uncertain
- `> 25 m` → Negative

## Erweiterungen

Erst nach einer stabilen Baseline:
- Backbone-Vergleich
- Fine-Tuning
- Sequence Retrieval
- Re-Ranking
- bessere GPS-Aggregation
- OOD-Methoden

