# Teil 2 — Detailausarbeitung

Jeder Schritt einzeln: Tools, Vorwissen, Metriken, Visualisierung, Status, Qualitätseinfluss.

**Legende Status:**
`NOTWENDIG` = MVP-Pflicht · `VERBESSERUNG` = optional · `ABKÜRZUNG` = weglassbar mit Trade-off

---

## Schritt 1 — Metadaten-Harvest

**Status: NOTWENDIG**

### Tools
- `requests` + Mapillary Graph API v4 (direkt, transparent)
- Alternative: `mapillary-python-sdk` (offiziell, nimmt Pagination ab, aber weniger Kontrolle)

### Was du wissen musst
- Felder gezielt anfordern: `fields=id,geometry,compass_angle,sequence,captured_at,quality_score` — ohne `fields` gibt die API nur Minimalantworten
- **`geometry.coordinates` ist `[lon, lat]`** (GeoJSON-Konvention), umgekehrt zur üblichen `(lat, lon)`-Schreibweise. Häufigste Fehlerquelle in Geo-Pipelines überhaupt
- `sequence` identifiziert die Aufnahmefahrt — wichtigstes Feld für Schritt 2
- `captured_at` ist Unix-**Millisekunden**, nicht Sekunden
- Bbox-Limit: strikt unter 0.01° pro Achse
- Rate-Limiting äußert sich als HTTP 500, nicht 429

### Metriken
- Anzahl Bilder, Anzahl unique Sequenzen
- Verteilung Bilder/Sequenz
- Zeitliche Spanne (Jahre) — entscheidet, ob Temporal-Split möglich ist

### Visualisierung
**Karte mit GPS-Punkten, eingefärbt nach `sequence_id`.** Zeigt sofort, ob die Daten aus 20 langen Fahrten oder 500 kurzen Schnipseln bestehen — das ändert die Split-Strategie fundamental.

### Qualitätseinfluss
Indirekt, aber fundamental. Fehlt `sequence_id`, ist die gesamte spätere Evaluation wertlos. Kein Qualitätsregler, sondern Voraussetzung.

---

## Schritt 2 — Train/Val/Test-Split

**Status: NOTWENDIG — der Schritt, den die meisten falsch machen**

### Das Problem
Mapillary-Bilder entstehen als Fahrt. Bild 47 und Bild 48 derselben Sequenz liegen 3 m auseinander und zeigen praktisch dasselbe. Bei zufälligem Split landet Bild 47 im Index, Bild 48 im Testset. Das Retrieval findet Bild 47, meldet 3 m Fehler — gemessen wurde aber nur, ob ein fast identisches Bild wiedererkannt wird.

### Drei Split-Strategien

| Strategie | Vorgehen | Was sie misst | Empfehlung |
|-----------|----------|---------------|------------|
| **Sequence-Split** | ganze Sequenzen zu train **oder** test | Generalisierung über Fahrten | **Standard** |
| **Temporal-Split** | ältere Fahrten = Index, neuere = Query | gleicher Ort, andere Zeit/Wetter/Jahreszeit | zusätzliches, härteres Szenario |
| Geographic-Split | Viertel A im Index, Viertel B als Query | — | **nicht nutzen**: Query-Ort ist gar nicht im Index, unlösbar |

### Wichtige Klarstellung
Wenn zwei *verschiedene* Sequenzen dieselbe Straße befahren haben (auf Hauptstraßen sehr wahrscheinlich), ist das **kein** Leakage, sondern genau der Testfall. Sequence-Split ist also nicht zu streng.

### Code

```python
import numpy as np
from collections import defaultdict

seqs = defaultdict(list)
for img in metadata:
    seqs[img["sequence"]].append(img)

seq_ids = sorted(seqs.keys())          # sortieren = reproduzierbar
rng = np.random.default_rng(42)
rng.shuffle(seq_ids)

n = len(seq_ids)
train_seqs = seq_ids[: int(0.8 * n)]
test_seqs  = seq_ids[int(0.8 * n):]
```

### Metriken
**Anteil Test-Queries, für die überhaupt ein Index-Bild innerhalb 50 m existiert.** Liegt der unter ~70 %, ist die Aufgabe teilweise unlösbar und alle späteren Fehlerzahlen sind irreführend.

### Visualisierung
Zwei-Farben-Karte: Index-Punkte vs. Query-Punkte. Räumliche Lücken springen sofort ins Auge.

### Qualitätseinfluss
Kein Einfluss auf die *tatsächliche* Systemqualität — vollständiger Einfluss darauf, ob die *gemessene* Qualität etwas bedeutet. Falscher Split kann Recall@25m um geschätzt 30–50 Prozentpunkte künstlich aufblähen (Größenordnung abhängig von Sequenzdichte).

---

## Schritt 3 — Bild-Download

**Status: NOTWENDIG**

### Tools
- `requests` + `concurrent.futures.ThreadPoolExecutor`
- Alternative: `aiohttp` (höherer Durchsatz, mehr Komplexität — bei 7.500 Bildern unnötig)

### Was du wissen musst
- Bild-URLs über `fields=thumb_1024_url` — **zeitlich begrenzt gültig**, nicht wochenlang zwischenspeicherbar
- Auflösungsvarianten: `thumb_256_url`, `thumb_1024_url`, `thumb_2048_url`, `thumb_original_url`
- **Empfehlung: `thumb_1024_url`** — CLIP braucht nur 224px, aber spätere Schritte (Local Features, Re-Ranking) brauchen mehr. Einmal laden, mehrfach nutzen
- Speicherbedarf: 7.500 × ~150 KB ≈ 1,1 GB — unproblematisch
- **Idempotenz einbauen**: vor Download prüfen, ob Datei existiert

### Metriken
Erfolgsquote Download, Anzahl korrupte/leere Dateien.

### Visualisierung
**Bild-Grid mit 25 Zufallsbildern.** Klingt banal, ist aber der einzige Weg, systematische Probleme zu erkennen: verschwommene Bilder, Innenaufnahmen aus Fahrzeugen, Nachtaufnahmen, Regentropfen auf der Linse.

### Qualitätseinfluss
Gering, solange die Bilder ankommen. **Aber:** Sind 20 % unbrauchbar, merkst du das nur durch visuelle Inspektion, nie durch Metriken.

---

## Schritt 4 — Preprocessing

### 4a) Downsizing

**Status: NOTWENDIG (aber ohne Wahlfreiheit)**

CLIP ViT-B/32 hat eine **feste** Eingabeauflösung von 224×224. Keine Empfehlung, sondern Architekturvorgabe: 32×32-Patches ergeben bei 224px genau 7×7 = 49 Patches. Andere Auflösungen brechen die Positional Embeddings.

```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
# preprocess = Resize(224) → CenterCrop(224) → ToTensor → Normalize
```

**Offener Punkt mit Ablation-Potenzial:** `CenterCrop` wirft bei 4:3-Bildern die Ränder weg — bei Straßenszenen oft informative Fassaden. Alternative: `Resize((224,224))` ohne Crop, verzerrt, behält aber alles. Was besser funktioniert, ist empirisch offen.

- **Metrik:** keine eigene
- **Visualisierung:** Vorher/Nachher-Paare von 5 Bildern
- **Qualitätseinfluss:** Crop vs. Resize vermutlich wenige Prozentpunkte Recall, Richtung unklar. Nicht vorab optimieren, als Ablation am Ende testen.

### 4b) Patch-Einteilung

**Status: ABKÜRZUNG — weglassen**

Hier werden zwei Dinge verwechselt:
- **CLIP-interne Patches** (32×32 im ViT) — passieren automatisch, kein Eingriffspunkt
- **Manuelles Bild-Tiling** (Bild in Quadranten schneiden, separat embedden) — bewusste Design-Entscheidung mit völlig anderer Wirkung

Tiling hilft bei **lokalen** Übereinstimmungen (ein Gebäude an unterschiedlicher Bildposition). Das ist der Weg von Local-Feature-Matching (SuperPoint/LightGlue), nicht von Global-Descriptor-Retrieval.

Für Global Retrieval ist gerade der **ganzheitliche** Eindruck der Punkt — Tiling zerstört ihn und verfünffacht den Index. Falls später lokale Features gewünscht sind, mit dem richtigen Werkzeug arbeiten, nicht mit CLIP-Tiling.

### 4c) Metadaten entfernen

**Status: ABKÜRZUNG im Trainingspfad · NOTWENDIG bei eigenen Testfotos**

**Klarstellung: CLIP liest keine EXIF-Daten.** Es sieht ausschließlich Pixelwerte des dekodierten Bildes. Ein GPS-Tag im EXIF-Header kann das Modell physisch nicht beeinflussen. Es gibt kein Data-Leakage-Risiko durch EXIF im Modellpfad.

Das *echte* Risiko liegt in der **Code-Struktur**:

```python
# Query-Funktion bekommt NUR den Bildpfad, nie die Koordinate
def localize(image_path, index, id2gps) -> tuple[float, float]:
    ...

# Ground Truth taucht erst im Evaluations-Code auf
pred = localize(test_img["path"], index, id2gps)
error = haversine(pred, test_img["gps"])   # nur hier
```

**Wann EXIF-Stripping doch nötig wird:** Bei eigenen Handyfotos, die veröffentlicht werden — das ist Datenschutz, nicht ML-Methodik.

---

## Schritt 5 — Embedding

**Status: NOTWENDIG**

### Tools
- `open_clip` (empfohlen — mehr Pretrained-Checkpoints, LAION-Modelle)
- Alternativen: `transformers` (`CLIPModel`), OpenAIs originales `clip`-Repo (veraltet)

### Modellwahl — wichtiger als erwartet

| Modell | Dim | VRAM (Inference) | Eignung für VPR |
|--------|-----|------------------|-----------------|
| CLIP ViT-B/32 | 512 | ~1 GB | Baseline, schnell |
| CLIP ViT-L/14 | 768 | ~3 GB | stärker, 4–5× langsamer |
| **DINOv2 ViT-B/14** | 768 | ~2 GB | **oft besser für reines Place Recognition** |
| **VPR-Spezialmodelle** | variabel | variabel | **siehe Teil 3 — purpose-built** |

**Wichtiger Einwand:** CLIP ist auf *Bild-Text-Alignment* trainiert. Es lernt "was ist auf dem Bild" (semantisch), nicht "welcher konkrete Ort ist das" (instanzspezifisch). Zwei Wohnstraßen mit Backsteinfassaden sind für CLIP nahezu identisch — für die Aufgabe sind sie maximal verschieden.

DINOv2 ist selbstüberwacht auf visueller Ähnlichkeit trainiert und schneidet in VPR-Benchmarks häufig besser ab. **Empfehlung: beide embedden, beide evaluieren** — kostet einen halben Tag, ist ein methodisch sauberer Modellvergleich. 8 GB VRAM reichen für beide im Inference-Modus locker.

### Code

```python
import torch, numpy as np

model.eval().to("cuda")
embs = []
with torch.no_grad():                              # kein Gradientenspeicher
    with torch.autocast("cuda", dtype=torch.float16):
        for batch in dataloader:                   # batch_size 64–128 problemlos
            e = model.encode_image(batch.to("cuda"))
            e = e / e.norm(dim=-1, keepdim=True)   # L2-Normalisierung
            embs.append(e.float().cpu().numpy())

embs = np.concatenate(embs).astype("float32")
np.save("embeddings.npy", embs)
```

### Drei Details, die oft schiefgehen
1. **L2-Normalisierung ist Pflicht**, wenn später Cosine Similarity genutzt wird
2. `torch.no_grad()` — ohne läuft der VRAM voll
3. **Reihenfolge speichern**: Der Index kennt nur Positionen `0…N-1`. Das `idx → mapillary_id → GPS`-Mapping muss exakt der Embedding-Reihenfolge entsprechen und **zusammen** mit den Embeddings gespeichert werden, nie separat rekonstruiert

### Metriken
- Embedding-Norm-Verteilung (nach Normalisierung exakt 1.0 — guter Sanity Check)
- Paarweise Cosine-Similarity-Verteilung

### Visualisierung
**UMAP oder t-SNE der Embeddings, eingefärbt nach GPS-Position** (z. B. nach Längengrad).
Spiegelt sich räumliche Nachbarschaft in Cluster-Struktur wider, funktioniert der Ansatz grundsätzlich. Ist es reines Rauschen, wird kein Fine-Tuning das retten.

**Das ist der billigste Frühindikator überhaupt — diese Visualisierung vor allem anderen erstellen.**

### Qualitätseinfluss
**Sehr hoch.** Die Backbone-Wahl dominiert vermutlich alles andere in der Pipeline. CLIP vs. DINOv2 vs. VPR-Spezialmodell kann größere Unterschiede machen als das gesamte spätere Fine-Tuning.

---

## Schritt 6 — Indexierung

**Status: NOTWENDIG — aber die übliche Begründung stimmt nicht**

### Korrektur
Naives Nearest-Neighbor ist bei dieser Datengröße **nicht** zu langsam: 7.569 Vektoren × 512 Dim = eine Matrixmultiplikation `(1,512) × (512,7569)` ≈ 3,9 Mio. Multiplikationen → **unter 1 ms auf CPU**. Bei 1.500 Test-Queries: unter 2 Sekunden gesamt.

FAISS wird genutzt wegen sauberer API, SIMD-Optimierung, GPU-Fähigkeit und Skalierbarkeit — nicht wegen Performance-Not.

### Was speichert der Index?
Nur die **Vektoren** plus implizit deren Position `0…N-1`. FAISS weiß nichts über GPS, Bild-IDs oder Dateipfade.

```python
import faiss, numpy as np

embs = np.load("embeddings.npy")            # (N, 512), L2-normalisiert
index = faiss.IndexFlatIP(embs.shape[1])    # Inner Product
index.add(embs)
faiss.write_index(index, "osnabrueck.faiss")

# Separat, aber IMMER zusammen halten:
id2gps = np.array([[img["lat"], img["lon"]] for img in metadata])  # (N, 2)
np.save("id2gps.npy", id2gps)
```

Bei L2-normalisierten Vektoren ist **Inner Product identisch mit Cosine Similarity** — daher `IndexFlatIP`, nicht `IndexFlatL2`. Ohne Normalisierung misst IP etwas anderes (skaliert mit Vektorlänge) und Ergebnisse werden subtil falsch.

### Indextyp-Wahl

| Typ | Exakt? | Sinnvoll ab | Für dieses Projekt |
|-----|--------|-------------|---------------------|
| **`IndexFlatIP`** | ja | immer | **← nutzen** |
| `IndexIVFFlat` | approximativ | ~100k Vektoren | unnötig |
| `IndexHNSWFlat` | approximativ | ~100k–1M | unnötig |
| `IndexIVFPQ` | stark komprimiert | ~1M+ | schädlich |

Bei 7.569 Vektoren bringt jeder approximative Index **keinen messbaren Geschwindigkeitsvorteil**, führt aber Recall-Verluste ein. Genauigkeit gegen nichts eintauschen. IVF/PQ zu nutzen "weil man das so macht" wäre klassische vorzeitige Optimierung — bei Skalierung auf Millionen Bilder ändert sich genau eine Zeile.

### Metriken
Index-Größe auf Disk, Query-Latenz (einmal messen → belastbares Argument für die Präsentation).

### Visualisierung
Balkendiagramm Query-Zeit Flat vs. IVF. Zeigt, dass der Unterschied bei dieser Größe irrelevant ist — guter Slide-Inhalt, weil er eine bewusste Designentscheidung belegt.

### Qualitätseinfluss
Bei Flat: **null** (exakt). Das ist genau der Punkt.

---

## Schritt 7 — Retrieval

**Status: NOTWENDIG**

```python
model.eval()
with torch.no_grad():
    q = model.encode_image(preprocess(img).unsqueeze(0).cuda())
    q = (q / q.norm(dim=-1, keepdim=True)).float().cpu().numpy()

scores, indices = index.search(q, k=10)   # scores: Cosine Similarity ∈ [-1, 1]
candidate_gps = id2gps[indices[0]]        # (10, 2)
```

### Was du wissen musst
- `index.search` gibt **absteigend sortierte** Scores zurück — höher = ähnlicher (bei IP)
- `k` ist ein echter Parameter, kein Detail (wirkt in Schritt 8)
- Der absolute Score-Wert ist schwer interpretierbar. Aussagekräftiger: **Abstand Top-1 zu Top-2** — eng beieinander = Modell unsicher

### Metriken
Score-Verteilung der Top-1-Treffer; Score-Gap Top-1 vs. Top-2.

### Visualisierung
**Bild-Grid: links Query, rechts Top-5 mit Score und Distanz-in-Metern darunter.**
Wichtigste Debugging-Visualisierung der gesamten Pipeline — Fehlerfälle werden nur so verständlich.

**Wichtig: für die schlechtesten 20 Queries erstellen, nicht für zufällige.**

### Qualitätseinfluss
Der Schritt selbst ist deterministisch; `k` beeinflusst Schritt 8.

---

## Schritt 8 — Aggregation k → Koordinate

**Status: NOTWENDIG — fehlte im ursprünglichen Plan**

10 Kandidaten mit 10 Koordinaten, aber die Ausgabe muss **eine** Koordinate sein.

| Strategie | Vorgehen | Trade-off |
|-----------|----------|-----------|
| Top-1 | GPS des besten Treffers | einfach, anfällig für einzelnen Ausreißer |
| Mittelwert Top-k | arithmetisches Mittel | schlecht bei gestreuten Kandidaten — Mittel zweier Stadtteile liegt dazwischen, also nirgends |
| **Score-gewichtetes Mittel** | Σ wᵢ·pᵢ mit wᵢ ∝ scoreᵢ | robuster, guter Default |
| **Cluster + größter Cluster** | DBSCAN über Kandidaten-GPS, Zentrum des dichtesten Clusters | am robustesten gegen Ausreißer |

**Empfehlung:** Top-1 als triviale Baseline, dann Cluster-basiert. Der Vergleich beider ist eine saubere kleine Ablation.

Warum Clustering überlegen sein dürfte: Liegen 7 von 10 Kandidaten in derselben Straße und 3 in einem anderen Viertel (visuell ähnliche Reihenhäuser), zieht der Mittelwert die Schätzung ins Nichts. Clustering erkennt die Mehrheitsmeinung.

```python
from sklearn.cluster import DBSCAN
import numpy as np

# eps in Grad: 0.0005 ≈ 50 m
db = DBSCAN(eps=0.0005, min_samples=2).fit(candidate_gps)
labels = db.labels_
if (labels >= 0).any():
    best = np.bincount(labels[labels >= 0]).argmax()
    pred = candidate_gps[labels == best].mean(axis=0)
else:
    pred = candidate_gps[0]      # Fallback: Top-1
```

### Metriken
Vergleich der Endmetriken (Schritt 9) zwischen den Aggregationsstrategien.

### Visualisierung
Karte für einzelne Queries: 10 Kandidatenpunkte + aggregierte Schätzung + wahre Position. Zeigt unmittelbar, ob die Aggregation der Datenlage hilft oder gegen sie arbeitet.

### Qualitätseinfluss
**Mittel bis hoch, und meist unterschätzt.** Grobe Erwartung: Cluster-Aggregation ändert den **Median**-Fehler kaum, verbessert aber den **Mittelwert** deutlich, weil sie katastrophale Ausreißer abfängt. Wie stark, ist ohne die konkreten Daten nicht seriös vorhersagbar.

---

## Schritt 9 — Evaluation

**Status: NOTWENDIG — fehlte im ursprünglichen Plan**

### Metriken, nach Wichtigkeit

1. **Recall@d** — Anteil Queries mit Fehler < d Meter, für d ∈ {10, 25, 50, 100, 250}. Standardmetrik der VPR-Literatur, das Hauptergebnis
2. **Median-Fehler in Metern** — robuster als Mittelwert
3. **Mittlerer Fehler** — nur im Vergleich zum Median interessant: große Differenz = Ausreißerproblem
4. **Straßenabschnitt-Accuracy** — die eigentlich ehrliche Metrik: Vorhersage und Wahrheit auf OSM-Segmente snappen, Übereinstimmung prüfen

```python
import numpy as np

def haversine(p1, p2):
    R = 6371000.0
    lat1, lon1 = np.radians(p1[..., 0]), np.radians(p1[..., 1])
    lat2, lon2 = np.radians(p2[..., 0]), np.radians(p2[..., 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

errors = haversine(preds, truths)
for d in [10, 25, 50, 100, 250]:
    print(f"Recall@{d}m: {(errors < d).mean():.1%}")
print(f"Median: {np.median(errors):.1f} m | Mean: {errors.mean():.1f} m")
```

### Kritische Einordnung
GPS-Labels haben 5–15 m Rauschen. **Recall@10m ist damit nicht sinnvoll interpretierbar** — dort wird überwiegend Labelrauschen gemessen. Ruhig berichten, aber über Recall@25m aufwärts argumentieren.

### Baselines — unverzichtbar zur Einordnung
- **Zufall:** zufälliges Index-Bild als Vorhersage → wo liegt "null Können"?
- **Konstant:** immer Stadtzentrum vorhersagen → bei kompaktem Gebiet überraschend stark, entlarvt scheinbar gute Ergebnisse

Schlägt das System die Konstant-Baseline nicht deutlich, wurde nichts gelernt. Diese Baseline zu berichten ist ein Zeichen methodischer Sorgfalt.

### Visualisierungen
1. **Kumulative Fehlerverteilung** (x: Distanz log-skaliert, y: Anteil Queries darunter) — informativste Einzelgrafik, enthält alle Recall@d gleichzeitig
2. **Fehler-Heatmap über die Stadt** — zeigt, *wo* das System versagt. Erwartung: Innenstadt gut, monotone Wohngebiete schlecht
3. **Scatter Top-1-Score (x) vs. Fehler (y)** — falls korreliert, ist das ein geschenktes Konfidenzmaß für Schritt 13

### Qualitätseinfluss
Kein Einfluss auf die Qualität, vollständiger Einfluss darauf, ob sie bekannt ist.

---

## Schritt 10 — Objekt-Detection + OSM-Abgleich

**Status: VERBESSERUNG — Aufwand hoch, Nutzen unsicher**

### Kann CLIP das selbst liefern?
**Nein.** CLIP hat keinen Detection-Kopf — keine Bounding Boxes, keine Objektzählung, keine Positionen. Nur ein globaler Vektor pro Bild.

Was CLIP kann: Zero-Shot-**Klassifikation** durch Text-Vergleich.
```python
prompts = ["a street with traffic lights",
           "a residential street without traffic lights"]
# → Ähnlichkeit Bild-Embedding vs. Text-Embeddings
```
Gibt eine grobe Ja/Nein-Tendenz, aber keine verlässliche Zählung.

### Optionen

| Ansatz | Aufwand | Bemerkung |
|--------|---------|-----------|
| **Mapillary-Detections API** | sehr gering | bereits berechnet, keine eigene Inferenz |
| YOLOv8 / RT-DETR (COCO) | mittel | erkennt `traffic light`, `stop sign` — Klassen begrenzt |
| Mapillary Vistas-Model | hoch | 66 straßenspezifische Klassen, eigenes Setup |
| GroundingDINO | mittel-hoch | Open-Vocabulary, flexibel, langsam |

### Das übersehene Konsistenzproblem
Für **Index**-Bilder liefert Mapillary Detections gratis. Für das **Query**-Bild (fremdes Foto) gibt es keine — es muss also doch ein eigenes Detektionsmodell laufen. Dann stammen die Detections beider Seiten aus verschiedenen Quellen mit verschiedenen Klassendefinitionen. Das wird gern übersehen und untergräbt den Vergleich.

### Realistischere Variante
Statt Objektdetection ein **semantisches Segmentierungsmodell** (z. B. SegFormer auf Cityscapes) auf **beiden** Seiten laufen lassen und die Klassenverteilung (% Himmel, % Vegetation, % Gebäude, % Straße) als Zusatz-Feature nutzen. Konsistent, schnell, diskriminativer als Objektzählungen.

### Metriken
Recall@25m mit vs. ohne Re-Ranking (Ablation).

### Visualisierung
Top-5 vor und nach Re-Ranking nebeneinander, mit Fehler in Metern — zeigt direkt, ob Umsortierung hilft oder schadet.

### Aufwand-Nutzen
**Eher ungünstig.** Implementierungsaufwand mehrere Tage, erwarteter Gewinn wenige Prozentpunkte Recall — mit realer Chance auf **Verschlechterung**, wenn der semantische Filter falsche Kandidaten bevorzugt. Wissenschaftlich als Ablation interessant ("hilft symbolisches Wissen dem dense retrieval?"), als Qualitätshebel unattraktiv. Bei knapper Zeit: streichen.

---

## Schritt 11 — Fine-Tuning mit frozen Backbone

**Status: VERBESSERUNG — bestes Aufwand-Nutzen-Verhältnis der optionalen Schritte**

### Warum hilft das, wenn der Backbone unverändert bleibt?
Der Backbone erzeugt einen Vektorraum, der für *semantische* Ähnlichkeit optimiert wurde. Zwei Osnabrücker Wohnstraßen mit Backsteinfassaden liegen darin sehr nah — semantisch sind sie dasselbe ("Wohnstraße"). Für die Aufgabe sind es **völlig verschiedene Orte**, die maximal weit auseinanderliegen sollten.

Der trainierbare Kopf lernt eine **Umformung dieses Raums**: eine Funktion f: ℝ⁵¹² → ℝᵈ, die Richtungen streckt, die Orte unterscheiden, und Richtungen staucht, die nur generische Szenentypen kodieren.

Der Backbone extrahiert weiterhin dieselben Informationen — der Kopf lernt nur, **welche Anteile davon für die Aufgabe wichtig sind**. Das ist überraschend wirksam, weil die relevante Information im CLIP-Vektor meist bereits enthalten, nur ungünstig gewichtet ist.

### Loss-Funktion — der eigentliche Kern
Kontrastives Lernen mit räumlicher Definition von positiv/negativ:
- **Positiv:** zwei Bilder < 25 m auseinander
- **Negativ:** zwei Bilder > 200 m auseinander
- **Grauzone 25–200 m:** ignorieren (sonst widersprüchliche Signale)

Praktisch: `pytorch-metric-learning` bietet fertige Losses (`NTXentLoss`, `TripletMarginLoss`, `MultiSimilarityLoss`) und nimmt das fehleranfällige Mining ab.

### Hardware (RTX 3070, 8 GB)
Embeddings einmal vorberechnen → Training läuft auf `(N, 512)`-Matrizen. Ein 2–3-Layer-MLP auf 7.500 Vektoren trainiert in **Minuten**, nicht Stunden. Bei diesem Schritt verschwindet die Hardwarebeschränkung praktisch.

### Kritischer Fallstrick
Das Fine-Tuning darf **ausschließlich** auf Train-Sequenzen laufen. Sonst wird direkt auf das Testset optimiert und alle Zahlen sind wertlos.

### Metriken
Recall@25m vor/nach Fine-Tuning, auf demselben Testset.

### Visualisierung
**UMAP der Embeddings vorher/nachher, eingefärbt nach GPS.** Idealerweise sichtbar, wie sich räumlich getrennte Orte im Embedding-Raum separieren — sehr überzeugende Grafik für die Abschlusspräsentation.

### Aufwand-Nutzen
**Günstig.** Klar abgegrenzter Eigenanteil, moderater Aufwand (wenige Tage), plausibler Gewinn. Größenordnung schwer vorhersagbar — bei wenigen tausend Trainingsbildern besteht **reales Overfitting-Risiko**, und es ist nicht ausgeschlossen, dass der Kopf gar nichts verbessert. Das offen zu berichten wäre ein legitimes Ergebnis, kein Scheitern.

---

## Schritt 12 — Richtungsschätzung

**Status: teils VERBESSERUNG, teils falsch konzipiert**

### Der Kategoriefehler
Structure-from-Motion rekonstruiert 3D-Struktur aus **mehreren überlappenden Bildern**. Es ist kein Verfahren, das aus *einem* Bild eine Richtung abliest. Gemeint ist eines von zwei völlig verschiedenen Dingen:

### (a) Heading-Regression aus dem Bild
Frage: "In welche Himmelsrichtung zeigt die Kamera?" Dafür braucht es kein SfM — `compass_angle` liegt als **fertiges Label** für tausende Bilder vor. Simples überwachtes Lernen:

```python
# Zirkuläre Kodierung — 359° und 1° müssen benachbart sein
target = np.stack([np.sin(np.radians(angle)),
                   np.cos(np.radians(angle))], axis=-1)
# Kopf auf CLIP-Features, MSE-Loss auf (sin, cos)
# Rückrechnung: angle = np.degrees(np.arctan2(sin_pred, cos_pred)) % 360
```

- **Aufwand:** sehr gering (derselbe vorberechnete Embedding-Cache wie Schritt 11)
- **Nutzen für Lokalisierung:** fraglich — verbessert das Retrieval nicht direkt, eher interessantes Nebenergebnis

### (b) Visual Localization mit 6-DoF-Pose
Das vermutlich eigentlich Gemeinte: 3D-Modell der Stadt aus SfM bauen (COLMAP / OpenSfM), dann für ein Query-Bild 2D-3D-Korrespondenzen finden und per PnP exakte Kameraposition **und** -orientierung berechnen. Toolchain: `hloc` mit SuperPoint + LightGlue.

Das ist der **genaueste** existierende Ansatz — Zentimeter- bis Meterbereich statt zehn Metern. Aber:

- Benötigt **dichte, überlappende** Bildabdeckung. Bei ~63 Bildern/km² für weite Teile Osnabrücks vermutlich zu dünn → fragmentierte Teilmodelle statt eines zusammenhängenden Stadtmodells
- Rechenaufwand Rekonstruktion: Stunden bis Tage
- Implementierungskomplexität deutlich höher als alles andere in der Pipeline

### Wichtig
Die eigentliche Sorge — "Bilder aus anderer Richtung als die Referenzaufnahmen" — löst SfM **nicht automatisch**. Das ist ein Datenabdeckungsproblem: Ist nie jemand in Gegenrichtung gefahren, existiert diese Ansicht schlicht nicht. Realistischere Gegenmaßnahmen: Augmentierung (horizontales Spiegeln), rotationsrobustere Descriptoren, Local-Feature-Matching mit Toleranz für partielle Überlappung, oder ein viewpoint-robustes VPR-Modell (siehe Teil 3).

### Aufwand-Nutzen
- **(a) Heading-Regression:** günstig, aber geringer Nutzen für die Kernaufgabe. Nettes Zusatzergebnis
- **(b) SfM / hloc:** ungünstig für das Zeitbudget. Hoher Aufwand, unsichere Datengrundlage. Als "Future Work" erwähnen, nicht implementieren

---

## Schritt 13 — OOD-Rejection

**Status: VERBESSERUNG — unterschätzt, sehr günstiges Verhältnis**

Der einfachste sinnvolle Ansatz kostet fast nichts:

**Score-Threshold:** Liegt der Top-1-Cosine-Score unter einem Schwellwert → "kann ich nicht zuordnen" statt einer Koordinate. Schwellwert empirisch aus der Score-Verteilung des Testsets vs. einer Negativmenge (beliebige Nicht-Osnabrück-Bilder) bestimmen.

### Metriken
- **ROC-Kurve / AUROC** für "in-distribution vs. out-of-distribution"
- **Recall@25m unter den akzeptierten Queries** — sollte deutlich höher liegen als über alle

### Visualisierung
Zwei überlagerte Histogramme der Top-1-Scores (Osnabrück-Queries vs. Fremdbilder). Starke Überlappung → simple Score-Schwelle funktioniert nicht, echter Klassifikator nötig.

### Aufwand-Nutzen
**Sehr günstig.** Ein Nachmittag Arbeit, macht das System deutlich ehrlicher, liefert eine gut präsentierbare Grafik.
