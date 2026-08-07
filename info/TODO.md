# Teil 3 — Eigenanteil, Libraries und Repos

Bewertung pro Komponente: **selbst schreiben** / **Library** / **Repo klonen** — plus konkrete Repo-Empfehlungen.

---

## Wichtiger Fund vorweg: VPR-Spezialmodelle existieren bereits

Die Annahme "CLIP + FAISS" ist ein solider Einstieg, aber es gibt eine ganze Modellfamilie, die **exakt für diese Aufgabe** gebaut wurde und CLIP darin vermutlich deutlich schlägt.

Es existiert ein Wrapper-Repo, das mehrere dieser Modelle über einen einzigen Parameter austauschbar macht: <cite index="8-1">NetVLAD, AP-GeM, SFRS, CosPlace, Conv-AP, MixVPR, EigenPlaces, AnyLoc, SALAD, CricaVPR, CliqueMining, SuperVLAD und MegaLoc</cite>.

Warum das relevant ist:
- <cite index="9-1">EigenPlaces und MixVPR schneiden in Benchmarks durchgehend am besten ab — EigenPlaces erreicht Recall@1 von 92,5 % auf Pitts30k, MixVPR 94,6 % auf Pitts250k</cite>
- <cite index="12-1">EigenPlaces trainiert gezielt mit mehreren Blickwinkeln desselben Ortes, um robuste Deskriptoren zu erzeugen</cite> — das adressiert direkt das Problem "Query-Foto aus anderer Richtung als die Referenzaufnahmen"
- <cite index="12-1">SALAD nutzt ein fine-getuntes DINOv2 als Grundlage</cite>

**Konsequenz für die Projektplanung:** CLIP als Baseline behalten (gut erklärbar, bekannt), aber mindestens ein VPR-Spezialmodell als Vergleich einbauen. Der Modellvergleich CLIP vs. DINOv2 vs. EigenPlaces/MixVPR ist mit dem Wrapper-Repo ein Nachmittag Arbeit und liefert vermutlich die größte Einzelverbesserung der gesamten Pipeline.

---

## Bewertungsmatrix

| # | Komponente | Kategorie | Konkret | Begründung |
|---|-----------|-----------|---------|------------|
| 0 | Gebietsabgrenzung | **Library** | `osmnx` | Standardlösung, erledigt |
| 1 | Metadaten-Harvest | **selbst** ✍️ | `requests` + API v4 | Mapillary-Eigenheiten (Bbox-Limit, 500-als-Rate-Limit) sind projektspezifisch — bereits gelöst |
| 2 | Train/Val/Test-Split | **selbst** ✍️ | ~30 Zeilen | Kernstück der Methodik, hängt an `sequence_id`-Struktur der eigenen Daten |
| 3 | Bild-Download | **selbst** ✍️ | `requests` + Threads | 40 Zeilen, Fremdcode bringt mehr Reibung als Ersparnis |
| 4 | Preprocessing | **Library** | `open_clip.preprocess` | Kommt mit dem Modell, nicht selbst bauen |
| 5 | Embedding | **Library + Repo** | `open_clip` / VPR-Wrapper | Modelle nie selbst implementieren |
| 6 | FAISS-Index | **Library** | `faiss-cpu` | 5 Zeilen, kein Eigenanteil möglich |
| 7 | Retrieval | **Library** | `index.search()` | trivial |
| 8 | Aggregation | **selbst** ✍️ | `sklearn.DBSCAN` + eigene Logik | Design-Entscheidung, kein Standardverfahren |
| 9 | Evaluation | **selbst** ✍️ | Haversine + Recall@d | Muss verstanden werden, ~50 Zeilen |
| 10 | Re-Ranking | **Repo/Library** | SegFormer o. Mapillary API | eigene Detektoren zu bauen wäre Verschwendung |
| 11 | Fine-Tuning | **selbst + Library** ✍️ | `pytorch-metric-learning` | **Kern-Eigenanteil** — Loss/Sampling selbst definieren |
| 12a | Heading-Regression | **selbst** ✍️ | 20 Zeilen MLP | trivial, kein Repo nötig |
| 12b | SfM/6-DoF | **Repo** | `hloc` | niemals selbst — Future Work |
| 13 | OOD-Rejection | **selbst** ✍️ | Threshold + ROC | ~30 Zeilen |

**Zusammenfassung:** Etwa **7 von 14 Komponenten** sind sinnvoller Eigenanteil, konzentriert auf die Stellen, die von den eigenen Daten und der eigenen Fragestellung abhängen.

---

## Entscheidungsregel

> **Selbst schreiben,** wenn die Komponente ohne die konkreten Daten/Fragestellung nicht existieren würde.
> **Library,** wenn es ein gelöstes, domänenunabhängiges Standardproblem ist.
> **Repo klonen,** wenn es ein trainiertes Modell oder eine komplexe Forschungspipeline ist, die nachzubauen Wochen kostet und nichts lehrt.

---

## Konkrete Repo-Empfehlungen

### 🥇 Priorität 1 — sofort nutzen

**`gmberton/VPR-methods-evaluation`**
`https://github.com/gmberton/VPR-methods-evaluation`
<cite index="8-1">Wrapper für 10+ VPR-Modelle; der Code jedes Modells stammt aus dem jeweiligen Original-Repo</cite>. Modellwechsel über einen Parameter.

- **Nutzen:** Modellvergleich in Stunden statt Wochen
- **Eigenanteil bleibt:** Anwendung auf eigenen Osnabrück-Datensatz, eigene Evaluation, eigene Interpretation
- **Empfehlung:** unbedingt einbauen

**`mlfoundations/open_clip`**
`https://github.com/mlfoundations/open_clip`
Über `pip install open_clip_torch` — kein Klonen nötig. Für die CLIP-Baseline.

---

### 🥈 Priorität 2 — als Vergleichsmodelle

**`gmberton/EigenPlaces`**
`https://github.com/gmberton/EigenPlaces`
<cite index="3-1">Trainingsprotokoll für Robustheit gegen Blickwinkeländerungen; laut Autoren bei 60 % weniger GPU-Speicher im Training und 50 % kompakteren Deskriptoren</cite>. Der geringere Speicherbedarf ist bei 8 GB VRAM relevant.

**`amaralibey/MixVPR`**
`https://github.com/amaralibey/MixVPR`
<cite index="5-1">All-MLP-Aggregationsmethode, deutlich effizienter in Latenz und Parameterzahl als bestehende Methoden; alle Modelle wurden auf dem GSV-Cities-Datensatz trainiert</cite>.

⚠️ **Achtung bei MixVPR:** <cite index="5-1">Bilder müssen auf 320×320 skaliert werden</cite> — andere Auflösung als CLIP (224×224). Preprocessing muss pro Modell angepasst werden, sonst sind die Ergebnisse ungültig.

⚠️ **Achtung beim Training:** <cite index="11-1">MixVPR benötigt im Training über 18 GB Speicher bei Batch Size 480</cite> — Training scheidet mit 8 GB aus, **Inference** mit vortrainierten Gewichten ist aber problemlos.

**`gmberton/auto_VPR`**
`https://github.com/gmberton/auto_VPR`
<cite index="3-1">Codebase für Experimente mit NetVLAD, SFRS, CosPlace, Conv-AP, MixVPR und EigenPlaces, lädt die Gewichte automatisch aus den offiziellen Repos</cite>.

---

### 🥉 Priorität 3 — Referenz & Orientierung

**`gmberton/awesome-Visual-Place-Recognition`**
`https://github.com/gmberton/awesome-Visual-Place-Recognition`
Kuratierte Paper-Liste mit GitHub-Links. Guter Startpunkt für die Literaturrecherche der Abschlussarbeit.

**`amaralibey/gsv-cities`**
`https://github.com/amaralibey/gsv-cities`
Trainingsdatensatz, auf dem viele VPR-Modelle trainiert wurden. Relevant zum Verständnis, worauf die vortrainierten Gewichte basieren.

---

### Mapillary-Download (nur falls eigene Skripte ersetzt werden sollen)

Die eigenen Skripte funktionieren bereits — diese Repos sind primär als Referenz interessant:

| Repo | Eignung |
|------|---------|
| `gchoumos/mapillary` | <cite index="14-1">Python-Skripte für API v4: Sequenzen abrufen, Bild-IDs und Metadaten laden, Bilder in konfigurierbarer Qualität herunterladen, Detections abrufen und visualisieren; Konfiguration über `settings.py`</cite> — **bester Kandidat**, deckt auch Detections ab |
| `tobesucht/mapillary_downloader` | <cite index="18-1">Docker-basiert, lädt Vektordaten und Street Imagery für eine Region; Bbox über Umgebungsvariablen</cite> — sauber, aber Docker-Overhead |
| `Stefal/mapillary_download` | <cite index="15-1">Lädt Bilder einer Sequenz, geotagged und orientiert</cite> — nur sequenzbasiert, nicht bbox-basiert |
| cbeddow Gists | Offizielle Mapillary-Beispiele für Bbox-Download — gute Referenz zum Abgleich der eigenen Implementierung |

**Empfehlung:** Bei den eigenen Skripten bleiben. Sie sind bereits auf die Osnabrück-Eigenheiten (Kachelung, Retry, Checkpointing) angepasst. Bei `gchoumos/mapillary` lohnt ein Blick auf den **Detections-Teil**, falls Schritt 10 verfolgt wird.

---

### Future Work — nicht für dieses Projekt

**`cvg/Hierarchical-Localization` (hloc)**
`https://github.com/cvg/Hierarchical-Localization`
<cite index="22-1">Unterstützt SuperPoint, DISK, D2-Net, SIFT, R2D2 als lokale Feature-Extraktoren; SuperGlue, LightGlue und Nearest-Neighbor-Matching; dichtes Matching mit LoFTR; NetVLAD, AP-GeM, OpenIBL und MegaLoc für Retrieval</cite>. <cite index="28-1">Plugin-Architektur, bei der Algorithmen ohne Änderung des Pipeline-Codes ausgetauscht werden können</cite>.

Technisch exzellent, aber: braucht dichte überlappende Abdeckung für die SfM-Rekonstruktion. Bei ~63 Bildern/km² in Osnabrück vermutlich zu dünn. **Als Ausblick erwähnen, nicht implementieren.**

---

## Empfohlene Arbeitsreihenfolge

| Phase | Inhalt | Kategorie |
|-------|--------|-----------|
| 1 | Metadaten + Split + Download | **selbst** (fast fertig) |
| 2 | CLIP-Embeddings + FAISS + Retrieval | Library |
| 3 | Aggregation + Evaluation + Baselines | **selbst** |
| 4 | **Erste Zahlen** — Recall@d, Fehlerbilder ansehen | — |
| 5 | Modellvergleich via VPR-methods-evaluation | Repo |
| 6 | Fine-Tuning auf bestem Backbone | **selbst** (Kern) |
| 7 | OOD-Rejection | **selbst** |
| 8 | Optional: Re-Ranking / Heading | gemischt |

**Kritisch:** Phase 4 vor Phase 5–8. Ohne echte Fehlerbilder ist jede Optimierungsentscheidung geraten.

---

## Was in der Präsentation als Eigenanteil zählt

Nicht: "Ich habe CLIP implementiert" (hat niemand, das ist ein `pip install`).

Sondern:
- **Datensatzkonstruktion** mit korrektem, leakage-freiem Split für einen Datentyp, bei dem naives Splitten systematisch zu optimistische Ergebnisse liefert
- **Methodischer Modellvergleich** auf einem selbst erstellten, bisher nicht existierenden Osnabrück-Benchmark
- **Domänenspezifisches Fine-Tuning** mit selbst definiertem räumlichem Positiv/Negativ-Kriterium
- **Evaluationsframework** inklusive der Erkenntnis, dass Metergenauigkeit durch GPS-Labelrauschen (5–15 m) nach oben begrenzt ist
- **Ehrliche Fehleranalyse**: wo und warum das System versagt

Das ist mehr Eigenanteil als "ein Modell from scratch bauen" — und deutlich näher an tatsächlicher Forschungspraxis.
