# Teil 1 — Pipeline-Übersicht

**Projekt:** Visuelle Geolokalisierung in Osnabrück
**Ansatz:** Visual Place Recognition (Image Retrieval), nicht Geo-Regression
**Stand Datenlage:** ~7.569 Mapillary-Bilder innerhalb der Stadtgrenze, ~63 Bilder/km²

---

## Korrekturen am ursprünglichen Plan

Vier Punkte, die die Pipeline substanziell verändern:

1. **Drei Pflichtschritte fehlten:** Train/Val/Test-Split (Schritt 2), Aggregation k→Koordinate (Schritt 8), Evaluation (Schritt 9). Ohne diese existiert kein bewertbares System, nur eine Demo.
2. **FAISS ist bei 7.569 Vektoren nicht aus Effizienzgründen nötig.** Brute-Force-Suche dauert hier unter 1 ms. FAISS wird wegen sauberer API und Skalierbarkeit genutzt, nicht wegen Performance.
3. **EXIF-Metadaten-Entfernung ist im Modellpfad ein Scheinproblem.** CLIP liest keine EXIF-Daten, nur Pixel. Das echte Leakage-Risiko liegt in der Code-Struktur.
4. **SfM zur Richtungsschätzung ist ein Kategoriefehler.** SfM braucht mehrere überlappende Bilder und löst nicht das beschriebene Problem.

---

## Hauptschritte

| # | Name | Input | Output | Zweck |
|---|------|-------|--------|-------|
| 0 | Gebietsabgrenzung | OSM-Polygon (osmnx) | Bbox / Stadtpolygon | Definiert das Zielgebiet — **erledigt** |
| 1 | Metadaten-Harvest | Mapillary API + Polygon | JSON: `id`, GPS, `sequence_id`, `compass_angle`, `quality_score` | Datensatz-Skelett ohne Pixel |
| 2 | **Split** | Metadaten | train/val/test-Listen nach `sequence_id` | Verhindert Data Leakage — **methodisch kritisch** |
| 3 | Bild-Download | Bild-IDs | JPEGs auf Disk (~1,1 GB) | Pixel beschaffen |
| 4 | Preprocessing | JPEGs | Tensoren 224×224, normalisiert | CLIP-kompatibles Eingabeformat |
| 5 | Embedding | Tensoren | Matrix `(N, 512)` float32, L2-normalisiert | Bilder numerisch vergleichbar machen |
| 6 | Indexierung | Embedding-Matrix | FAISS-Index + `idx→GPS`-Mapping | Durchsuchbare Struktur |
| 7 | Retrieval | Query-Embedding | Top-k IDs + Similarity-Scores | Kandidaten finden |
| 8 | **Aggregation** | Top-k GPS + Scores | **eine** Koordinatenschätzung | k Treffer → 1 Antwort |
| 9 | **Evaluation** | Schätzungen + Ground Truth | Recall@d, Median-Fehler | Systemqualität messbar machen |
| 10 | Re-Ranking *(opt.)* | Top-k + semantische Features | umsortierte Top-k | Präzisionsgewinn |
| 11 | Fine-Tuning *(opt.)* | Train-Embeddings | trainierter Projection Head | Domänenanpassung |
| 12 | Heading-Schätzung *(opt.)* | Bild | Blickrichtung in Grad | Zusatzinformation |
| 13 | OOD-Rejection *(opt.)* | Query-Embedding | akzeptieren / ablehnen | Falschantworten vermeiden |

---

## Abgrenzung MVP vs. Forschungsrichtung

**MVP — Pflicht (Schritte 0–9)**
Ohne Ausnahme alle nötig. Ein System, das Schritt 2, 8 oder 9 auslässt, liefert entweder keine verwertbare Ausgabe oder keine belastbaren Zahlen.

**Optional (Schritte 10–13)**
Nach Aufwand-Nutzen-Verhältnis priorisiert:

| Priorität | Schritt | Aufwand | Erwarteter Nutzen |
|-----------|---------|---------|-------------------|
| 1 | Backbone-Vergleich (Teil von 5) | sehr gering | potenziell sehr hoch |
| 2 | Fine-Tuning (11) | mittel | plausibel, unsicher |
| 3 | OOD-Rejection (13) | gering | hoch für Systemehrlichkeit |
| 4 | Aggregationsvarianten (8) | sehr gering | mittel |
| 5 | Heading-Regression (12a) | gering | peripher |
| 6 | Semantisches Re-Ranking (10) | hoch | unsicher, Verschlechterung möglich |
| 7 | SfM / hloc (12b) | sehr hoch | nur als Future Work |

---

## Grundsätzliche Reihenfolge-Empfehlung

1. **Baseline zuerst vollständig bauen** (0–9), auch wenn sie schwach ist
2. **Fehlerbilder anschauen** (Bild-Grid der schlechtesten Queries)
3. **Erst dann entscheiden**, welcher Verbesserungsschritt sich lohnt

Alles andere wäre Optimieren ins Blaue. Die absolute Höhe der zu erwartenden Verbesserungen lässt sich vorab nicht seriös beziffern — sie hängt an Sequenzstruktur, visueller Diversität Osnabrücks und Train/Test-Überlappung, alles erst nach der Baseline bekannt.
