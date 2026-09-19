"""
Alle Pfade an einer Stelle -- mit der Stadt als Dimension.

Jede Stadt bekommt ihren eigenen Zweig unter data/, results/,
experiments/results/ und weights/adapter/, benannt nach dem Slug aus
config.yaml -> city ("Osnabrück, Germany" -> osnabrueck). Damit laeuft eine
zweite Stadt im selben Klon, ohne die erste zu ueberschreiben, und die
versionierten Ergebnisse (Metadaten, Split-Listen, JSONs) liegen je Stadt
im Git. Bilder liegen ausserhalb des Repos unter image_root/<slug>, eigene
Fotos unter image_root/test.

    paths = Paths(cfg, ROOT)
    paths.processed / "metadata.parquet"
    paths.embedding_file("clip_linear", "clip")
    paths.retrieval_file("megaloc", "megaloc")
    paths.evaluation / "megaloc.json"

Umgebungsvariablen: VPR_IMAGE_ROOT setzt image_root, VPR_IMAGE_PATH den
Bildordner der Stadt direkt -- je Rechner, ohne die Datei zu aendern.
"""

import os
import re
import unicodedata
from pathlib import Path

# Was die NFKD-Zerlegung unten NICHT aufloest, weil es keine Grundform mit
# Akzent ist, sondern ein eigener Buchstabe: deutsche Umlaute wuerden sonst
# zu "o"/"u" statt "oe"/"ue", und aeltere Staebe wie ø, æ, ł, đ fielen beim
# ascii-ignore ersatzlos heraus -- "Ærøskøbing" wurde so zu "rskbing", und
# zwei verschiedene Staedte koennten auf denselben Slug fallen.
_SONDERZEICHEN = str.maketrans({
    "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
    "Ä": "ae", "Ö": "oe", "Ü": "ue",
    "ø": "oe", "Ø": "oe", "æ": "ae", "Æ": "ae", "å": "aa", "Å": "aa",
    "ł": "l", "Ł": "l", "đ": "d", "Đ": "d", "ð": "d", "Ð": "d",
    "þ": "th", "Þ": "th", "ı": "i", "İ": "i", "œ": "oe", "Œ": "oe",
})


def city_slug(city):
    """'Osnabrück, Germany' -> 'osnabrueck', 'Halle (Saale), Germany' -> 'halle-saale'.

    Nicht-deutsche Sonderzeichen gehen ueber _SONDERZEICHEN, alles Uebrige
    ueber NFKD + ascii-ignore (é -> e). Ein Name, von dem nichts uebrig
    bleibt, ist ein Fehler und keine leere Ablage.
    """
    name = str(city).split(",")[0].strip().translate(_SONDERZEICHEN)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not name:
        raise ValueError(f"Kein brauchbarer Stadtname in {city!r}")
    return name


class Paths:
    def __init__(self, cfg, root):
        self.root = Path(root)
        self.city = city_slug(cfg["city"])

        data = self.root / "data" / self.city
        self.raw = data / "raw"
        self.processed = data / "processed"
        self.embeddings = data / "embeddings"

        results = self.root / "results" / self.city
        self.results = results
        self.retrieval = results / "retrieval"
        self.evaluation = results / "evaluation"
        self.localization = results / "localization"
        self.figures = results / "figures"
        self.dataset_audit = results / "dataset_audit.json"

        self.experiments = self.root / "experiments" / "results" / self.city
        self.adapters = self.root / "weights" / "adapter" / self.city
        # osmnx-Antworten und Detections sind ueber image_id bzw. Abfrage
        # eindeutig -- ein Cache fuer alle Staedte.
        self.cache = self.root / "cache"

        image_root = Path(os.environ.get("VPR_IMAGE_ROOT", cfg["image_root"])).expanduser()
        self.images = Path(os.environ.get("VPR_IMAGE_PATH", image_root / self.city)).expanduser()
        self.own_images = image_root / "test"

    # -- Artefakte je Encoder -----------------------------------------------

    def embedding_dir(self, method):
        return self.embeddings / method

    def embedding_file(self, name, method):
        return self.embeddings / method / f"{name}_embeddings.npy"

    def metadata_file(self, name, method):
        return self.embeddings / method / f"{name}_metadata.parquet"

    def retrieval_file(self, name, method):
        return self.retrieval / method / f"{name}_retrieval.npz"

    def adapter_file(self, method, kind="linear"):
        return self.adapters / f"{method}_{kind}.pt"

    def image_file(self, image_id):
        return self.images / f"{image_id}.jpg"
