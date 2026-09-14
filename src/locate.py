"""
Ein Foto rein, ein Ort raus -- der Inferenz-Einstieg, den Demo und
Kommandozeile teilen.

    locator = Locator(cfg, ROOT, "eigenplaces_megaloc_concat")
    antwort = locator.locate("foto.jpg", k=5)
    antwort["lat"], antwort["lon"], antwort["konfidenz"], antwort["treffer"]

Baut den Encoder ueber die Factory (auch abgeleitete: PCA, Whitening,
Verkettung), legt den Adapter darueber, wenn einer konfiguriert ist, und
sucht im FAISS-Index ueber die Datenbank-Embeddings -- exakt der Weg von
04 bis 06, nur fuer ein Bild. Die Datenbank kommt per memmap: nur die
database-Zeilen landen im Speicher.

Konfidenz: die Aehnlichkeit des besten Treffers. experiments/
rejection_curve.py hat sie gegen Marge und Geschlossenheit gemessen -- sie
trennt am besten (AUC 0.79 auf loesbaren Anfragen). Bei MegaLoc: cos >= 0.30
heisst 82 % richtig (37 % der Anfragen), >= 0.20 noch 75 %, unter 0.17
faellt es unter zwei Drittel.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import embedding_name
from .device import pick_device
from .geo import haversine_distance
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
        self.device = pick_device(device)
        self.verbose = verbose
        self._embedder = None
        self._adapter = None
        self._index = None
        self._load_database()

    # -- Datenbank ---------------------------------------------------------

    def _load_database(self):
        import faiss

        emb_dir = self.root / "data" / "embeddings" / self.method
        meta = pd.read_parquet(emb_dir / f"{self.name}_metadata.parquet")
        npy = emb_dir / f"{self.name}_embeddings.npy"
        require_fingerprint(npy, embedding_fingerprint(self.cfg, self.method, self.adapter, meta),
                            "Embeddings")
        zeilen = np.flatnonzero((meta["split"] == "database").to_numpy())
        self.database = meta.iloc[zeilen].reset_index(drop=True)
        emb = np.load(npy, mmap_mode="r")
        vektoren = np.ascontiguousarray(emb[zeilen], dtype=np.float32)
        self.dim = int(vektoren.shape[1])
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(vektoren)
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

            pfad = self.root / "weights" / "adapter" / f"{self.method}_linear.pt"
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
        sims, idx = self._index.search(np.ascontiguousarray(vecs, dtype=np.float32), k)
        return idx, sims

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
