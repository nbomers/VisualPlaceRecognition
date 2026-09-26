# Was nicht unter der MIT-Lizenz steht

Die [MIT-Lizenz](LICENSE) gilt für den **Code** in diesem Repository.
Nicht darunter fallen:

## Mapillary-Daten

`data/<stadt>/processed/metadata.parquet` **liegt im Repository** (rund
66 MB über sechs Städte) und enthält Mapillary-Metadaten: `image_id`,
`sequence_id`, `captured_at`, `lat`, `lon`, `compass_angle`, `is_pano`,
`creator_id`. Ebenso `cache/detections.jsonl` (Bild-ID und Zähler je
semantischer Klasse).

Diese Daten stehen unter [CC BY-SA 4.0](https://www.mapillary.com/terms).
Namensnennung: **Bilder und Metadaten von [Mapillary](https://www.mapillary.com),
CC BY-SA 4.0.** Wer daraus abgeleitete Datensätze veröffentlicht, muss sie
unter denselben Bedingungen weitergeben.

Der **Bildbestand** liegt nicht im Repository — `notebooks/03_image_download`
lädt ihn mit einem eigenen API-Token nach `image_root/<stadt>`.

Eine Ausnahme sind die Abbildungen unter `results/<stadt>/figures/demo/`:
sie zeigen einzelne Anfrage- und Datenbankbilder nebeneinander, um Treffer
und Fehlgriffe sichtbar zu machen, und enthalten damit Mapillary-Bildinhalt.
Als abgeleitetes Werk stehen sie unter denselben Bedingungen — CC BY-SA 4.0,
**Bilder von Mapillary, CC BY-SA 4.0**. Dasselbe gilt für dieselben
Abbildungen, wenn sie in Vorträgen oder Berichten auftauchen.

CC BY-SA verlangt außerdem den **Urheber jedes einzelnen Bildes**, nicht
nur die Plattform. Nach Abschnitt 3(a)(2) der Lizenz genügt dafür ein Link
auf eine Seite, die ihn nennt — die Bildseite bei Mapillary.
[`QUELLEN.md`](results/osnabrueck/figures/demo/QUELLEN.md) im selben Ordner
verlinkt jedes Bild jeder Abbildung; `src/quellen.py` schreibt die Datei,
sobald die Demo oder `experiments/beispielbilder.py` eine Abbildung
speichert. Wer eine Abbildung weiterverwendet, nimmt die Links mit.

`creator_id` ist eine pseudonyme Mapillary-Konto-ID. Zusammen mit `lat`,
`lon` und `captured_at` ergibt sie eine Aufnahmespur je Konto. Sie steht
nicht zum Schmuck in den Metadaten: die „Hard"-Ground-Truth in
`src/evaluation.py` braucht genau diesen Vergleich („anderer Fotograf oder
mehr als 180 Tage Abstand"), und ohne sie ließe sich der Dubletteneffekt
im Städtevergleich nicht messen. Gesichter und Kennzeichen macht Mapillary
vor der Veröffentlichung automatisch unkenntlich; das betrifft die Bilder,
nicht die Metadaten.

## OpenStreetMap

Stadtgrenze, Straßennetz und Stadtteile kommen über OSMnx/Overpass aus
OpenStreetMap. Diese Daten stehen unter der
[ODbL](https://www.openstreetmap.org/copyright). Namensnennung:
**© OpenStreetMap-Mitwirkende**. Betroffen sind alle Karten unter
`results/<stadt>/figures/` und `experiments/results/<stadt>/*.png` sowie
die Stadtteilspalten in den zugehörigen JSONs.

## Fremd-Repositories und Modellgewichte

Nichts davon liegt im Repository. `setup_external.py`, `torch.hub`,
Hugging Face und `pip` holen es beim ersten Lauf auf den eigenen Rechner;
es gilt jeweils die Lizenz des Projekts, nicht die MIT-Lizenz hier.
Geprüft am 2026-09-23 gegen die Lizenzdateien der Projekte, bei den
Hugging-Face-Gewichten gegen die Modellkarte.

| Komponente | Woher | Lizenz | Urheber |
|---|---|---|---|
| **MegaLoc** — Code | `torch.hub`, [gmberton/MegaLoc](https://github.com/gmberton/MegaLoc) | MIT | © 2024 Gabriele Berton, Carlo Masone |
| **MegaLoc** — Gewichte | Hugging Face, [gberton/MegaLoc](https://huggingface.co/gberton/MegaLoc) (`model.safetensors`) | MIT (laut Modellkarte) | dieselben |
| **EigenPlaces** — Code und Gewichte | `torch.hub`, [gmberton/EigenPlaces](https://github.com/gmberton/EigenPlaces), lädt Teile von [gmberton/CosPlace](https://github.com/gmberton/CosPlace) | MIT | © 2023 Berton, Trivigno, Masone, Caputo; CosPlace © 2022 Berton, Masone, Caputo |
| **AnyLoc** — Code und Vokabular | `external/AnyLoc` ([AnyLoc/AnyLoc](https://github.com/AnyLoc/AnyLoc)), Vokabular aus der Hugging-Face-Space des Projekts | BSD-3-Clause | © 2023 AnyLoc |
| **DINOv2 ViT-G/14** — Gewichte für AnyLoc | `torch.hub`, [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Apache-2.0 | Meta Platforms |
| **MixVPR** — Code und Gewichte | `external/MixVPR` ([amaralibey/MixVPR](https://github.com/amaralibey/MixVPR)), Gewichte von Google Drive | **keine Lizenzdatei** — alle Rechte bei den Autoren | Ali-bey, Chaib-draa, Giguère |
| **CLIP ViT-B/32** — Gewichte | Hugging Face, `openai/clip-vit-base-patch32`, über 🤗 Transformers (Apache-2.0) | MIT ([openai/CLIP](https://github.com/openai/CLIP)) | © 2021 OpenAI |
| **LightGlue** — Code und Gewichte | `pip`, [cvg/LightGlue](https://github.com/cvg/LightGlue) | Apache-2.0 | Lindenberger, Sarlin, Pollefeys |
| **SuperPoint** — Gewichte und Inferenzcode | mit LightGlue installiert | **[Magic Leap, nur nichtkommerzielle Forschung](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE)** | Magic Leap, Inc. |

Zwei Einträge schränken ein, was man mit dem Projekt tun darf:

- **SuperPoint** darf nur für eigene, nichtkommerzielle Forschung benutzt
  und nicht weitergegeben werden. Betroffen ist ausschließlich
  `experiments/geometric_verification.py`; die Pipeline 01–08, `locate.py`
  und die Demo kommen ohne SuperPoint aus. Wer die geometrische
  Verifikation kommerziell nutzen will, braucht einen anderen Detektor —
  LightGlue gibt es auch für DISK (Apache-2.0) und ALIKED (BSD-3-Clause).
- **MixVPR** hat keine Lizenzdatei. Ohne Lizenz ist nichts ausdrücklich
  erlaubt; das Projekt klont Code und Gewichte nur auf den eigenen Rechner,
  um die veröffentlichten Zahlen nachzumessen, und gibt nichts davon weiter.

Keine der Komponenten verlangt eine Zitation. Die Autoren bitten darum; die
Einträge stehen im README unter „Literatur".
