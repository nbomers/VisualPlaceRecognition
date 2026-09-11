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
