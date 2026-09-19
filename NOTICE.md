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

Die **Bilder selbst** liegen nicht im Repository — `notebooks/03_image_download`
lädt sie mit einem eigenen API-Token nach `image_root/<stadt>`.

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

`external/` und `weights/` liegen nicht im Repository und werden von
`setup_external.py` bzw. `torch.hub` geholt. Ihre Lizenzen stehen in den
jeweiligen Projekten; eine Übersicht im README unter „Credits".
