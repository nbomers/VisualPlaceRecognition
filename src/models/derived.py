"""
Abgeleitete Encoder fuer ein neues Bild: Quelle -> PCA(-Whitening) -> L2,
oder die Verkettung mehrerer Quellen.

experiments/pca_reduce.py und concat_embeddings.py schreiben die Embeddings
der Datenbank; dieses Modul bringt ein einzelnes Foto (Demo, locate.py) in
denselben Raum. Die Projektion liegt als {name}_pca.npz neben den
Embeddings, die Verkettung braucht nur ihre Quellen. Rechnet exakt wie
pca_reduce.py: (x - mean) @ components.T, bei Whitening geteilt durch
sqrt(explained_variance), dann normalisiert.

AnyLoc-Varianten sind nicht vorfuehrbar, solange die AnyLoc-PCA aus 04
nicht neben den Embeddings liegt -- build_embedder sagt das dann.
"""

from pathlib import Path

import numpy as np


def load_projection(path):
    d = np.load(path)
    return {
        "mean": d["mean"].astype(np.float32),
        "components": d["components"].astype(np.float32),
        "scale": (np.sqrt(d["explained_variance"]).astype(np.float32)
                  if bool(d["whiten"]) else None),
    }


def project(vecs, proj):
    """Wie sklearn.decomposition.PCA.transform, danach L2-normalisiert."""
    out = (np.asarray(vecs, dtype=np.float32) - proj["mean"]) @ proj["components"].T
    if proj["scale"] is not None:
        out = out / proj["scale"]
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norm > 0, norm, 1.0)


class DerivedEmbedder:
    """Dieselbe Schnittstelle wie BaseEmbedder: embed_images, embedding_dim, close."""

    def __init__(self, name, cfg, device, project_root):
        from .factory import build_embedder

        block = cfg["vpr"][name]
        emb_dir = Path(project_root) / "data" / "embeddings" / name
        self.name = name
        if "sources" in block:
            self.parts = [build_embedder(q, cfg, device, project_root) for q in block["sources"]]
            self.projection = None
        else:
            pfad = emb_dir / f"{name}_pca.npz"
            if not pfad.exists():
                raise FileNotFoundError(
                    f"{pfad.relative_to(project_root)} fehlt -- "
                    "python experiments/pca_reduce.py --projection-only"
                )
            self.parts = [build_embedder(block["source"], cfg, device, project_root)]
            self.projection = load_projection(pfad)
        self.image_size = self.parts[0].image_size
        self.embedding_dim = (int(self.projection["components"].shape[0]) if self.projection
                              else sum(p.embedding_dim for p in self.parts))
        print(f"{name} bereit: {self.embedding_dim} Dimensionen aus "
              + " + ".join(getattr(p, "name", type(p).__name__) for p in self.parts))

    def embed_images(self, image_paths, batch_size=32, **_):
        teile = [p.embed_images(image_paths, batch_size=batch_size) for p in self.parts]
        if self.projection is not None:
            return project(teile[0], self.projection)
        out = np.concatenate(teile, axis=1).astype(np.float32)
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norm > 0, norm, 1.0)

    def close(self):
        for p in self.parts:
            p.close()
