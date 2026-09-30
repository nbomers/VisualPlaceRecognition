"""
Ein Foto rein, ein Ort raus -- der Inferenz-Einstieg, den Demo und
Kommandozeile teilen.

    locator = Locator(cfg, ROOT, "eigenplaces_megaloc_concat")
    antwort = locator.locate("foto.jpg", k=5)
    antwort["lat"], antwort["lon"], antwort["konfidenz"], antwort["treffer"]

Baut den Encoder ueber die Factory (auch abgeleitete: PCA, Whitening,
Verkettung), legt den Adapter darueber, wenn einer konfiguriert ist, und
sucht exakt ueber die Datenbank-Embeddings -- der Weg von 04 bis 06, nur
fuer ein Bild. Die Datenbank kommt per memmap: nur die database-Zeilen
landen im Speicher.

Gesucht wird mit numpy, nicht mit FAISS. Der Locator braucht Torch fuer den
Encoder, und Torch und FAISS bringen auf macOS je ihre eigene
OpenMP-Bibliothek mit: beide im selben Prozess beenden ihn ohne Traceback
(siehe tests/blockwise_check.py) -- in Jupyter als "kernel died". FAISS-Flat
ist ohnehin nur das vollstaendige Skalarprodukt; fuer eine Handvoll Fotos
gegen die Datenbank einer Stadt (Osnabrueck: 48.321 Bilder) sind das
Millisekunden, mit derselben Rangfolge (bis auf die Reihenfolge exakt
gleicher Aehnlichkeiten). Die
Pipeline (06) behaelt FAISS -- dort laeuft kein Torch im selben Prozess.

Konfidenz: die Aehnlichkeit des besten Treffers. experiments/
rejection_curve.py hat sie gegen Marge und Geschlossenheit gemessen -- sie
trennt am besten (AUC 0.79 auf loesbaren Anfragen). Bei MegaLoc: cos >= 0.30
heisst 82 % richtig bei 37 % der loesbaren Anfragen, >= 0.20 noch 75 % bei
69 % (experiments/README.md, Ablehnungskurve).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import embedding_name
from .device import pick_device
from .geo import haversine_distance
from .paths import Paths
from .run_guard import embedding_fingerprint, require_fingerprint


class Locator:
    def __init__(self, cfg, project_root, method=None, adapter=None, device=None, verbose=True):
        self.cfg = {**cfg, "vpr": dict(cfg["vpr"])}
        if method is not None:
            self.cfg["vpr"]["method"] = method
        if adapter is not None:
            self.cfg["vpr"]["adapter"] = adapter
        self.method = self.cfg["vpr"]["method"]
        self.adapter = self.cfg["vpr"].get("adapter", "none")
        self.name = embedding_name(self.cfg)
        self.root = Path(project_root)
        self.paths = Paths(self.cfg, self.root)
        self.device = pick_device(device)
        self.verbose = verbose
        self._embedder = None
        self._adapter = None
        self._vektoren = None
        self._load_database()

    # -- Datenbank ---------------------------------------------------------

    def _load_database(self):
        meta_datei = self.paths.metadata_file(self.name, self.method)
        if not meta_datei.exists():
            raise FileNotFoundError(
                f"Keine Embeddings fuer {self.name!r} unter {meta_datei.parent}.\n"
                "Entweder wurde der Encoder hier nie gerechnet, oder er liegt auf "
                "einem anderen Rechner (Embeddings sind gitignored).\n"
                "  python run.py --bestand        zeigt, was hier vollstaendig ist\n"
                f"  python run.py --method {self.method}   rechnet ihn hier"
            )
        meta = pd.read_parquet(meta_datei)
        npy = self.paths.embedding_file(self.name, self.method)
        require_fingerprint(npy, embedding_fingerprint(self.cfg, self.method, self.adapter, meta),
                            "Embeddings")
        zeilen = np.flatnonzero((meta["split"] == "database").to_numpy())
        self.database = meta.iloc[zeilen].reset_index(drop=True)
        emb = np.load(npy, mmap_mode="r")
        self._vektoren = np.ascontiguousarray(emb[zeilen], dtype=np.float32)
        self.dim = int(self._vektoren.shape[1])
        if self.verbose:
            print(f"Datenbank: {len(self.database):,} Bilder, {self.dim} Dimensionen ({self.name})")

    # -- Encoder -----------------------------------------------------------

    def _ensure_embedder(self):
        if self._embedder is not None:
            return
        from .models.factory import build_embedder

        self._embedder = build_embedder(self.method, self.cfg, self.device, self.root)
        if self._embedder.embedding_dim != self.dim:
            raise RuntimeError(
                f"Encoder liefert {self._embedder.embedding_dim} Dimensionen, "
                f"die Datenbank hat {self.dim}."
            )
        if self.adapter == "linear":
            import torch

            from .models.adapter import LinearAdapter

            pfad = self.paths.adapter_file(self.method)
            self._adapter = LinearAdapter(embedding_dim=self.dim).to(self.device).eval()
            self._adapter.load_state_dict(torch.load(pfad, map_location=self.device))
            if self.verbose:
                print(f"Adapter geladen: {pfad.name}")

    def embed(self, image_paths):
        """L2-normalisierte Vektoren fuer beliebige Bilder -- wie 04/05 sie schreiben."""
        self._ensure_embedder()
        vecs = self._embedder.embed_images([Path(p).expanduser() for p in image_paths],
                                           batch_size=min(32, len(image_paths)))
        if self._adapter is not None:
            import torch

            with torch.inference_mode():
                vecs = self._adapter(torch.from_numpy(vecs).float().to(self.device)).cpu().numpy()
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return np.ascontiguousarray(vecs, dtype=np.float32)

    # -- Suche -------------------------------------------------------------

    def search(self, vecs, k=10):
        """Exakte Suche nach Skalarprodukt, wie faiss.IndexFlatIP: (indices, aehnlichkeiten),
        je Zeile absteigend sortiert."""
        sims = np.ascontiguousarray(vecs, dtype=np.float32) @ self._vektoren.T
        k = min(int(k), sims.shape[1])
        # Erst die k groessten finden (linear), dann nur diese sortieren.
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        ordnung = np.argsort(-np.take_along_axis(sims, idx, axis=1), axis=1, kind="stable")
        idx = np.take_along_axis(idx, ordnung, axis=1)
        return idx, np.take_along_axis(sims, idx, axis=1)

    def locate(self, image_path, k=10):
        """Koordinate des besten Treffers, seine Aehnlichkeit als Konfidenz, die Top-k dazu."""
        idx, sims = self.search(self.embed([image_path]), k)
        idx, sims = idx[0], sims[0]
        treffer = self.database.iloc[idx]
        beste = treffer.iloc[0]
        streuung = haversine_distance(beste["lat"], beste["lon"],
                                      treffer["lat"].to_numpy(), treffer["lon"].to_numpy())
        return {
            "lat": float(beste["lat"]),
            "lon": float(beste["lon"]),
            "konfidenz": float(sims[0]),
            "marge": float(sims[0] - sims[1]) if len(sims) > 1 else float("nan"),
            "streuung_m": float(streuung.max()),
            "treffer": [
                {"image_id": int(z["image_id"]), "lat": float(z["lat"]), "lon": float(z["lon"]),
                 "aehnlichkeit": float(s), "abstand_zu_top1_m": float(d)}
                for (_, z), s, d in zip(treffer.iterrows(), sims, streuung)
            ],
        }

    def close(self):
        if self._embedder is not None:
            self._embedder.close()
