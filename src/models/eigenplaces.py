"""
eigenplaces.py -- EigenPlaces als Embedder mit derselben Schnittstelle wie
CLIPEmbedder.

Modell aus https://github.com/gmberton/EigenPlaces (MIT), geladen ueber
torch.hub -- kein Repo-Klon noetig.

Berton, Trivigno, Caputo, Masone: "EigenPlaces: Training Viewpoint Robust
Models for Visual Place Recognition", ICCV 2023.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm.auto import tqdm

# backbone -> zulaessige fc_output_dim
AVAILABLE = {
    "ResNet18": (256, 512),
    "ResNet50": (128, 256, 512, 2048),
    "ResNet101": (128, 256, 512, 2048),
    "VGG16": (512,),
}


class EigenPlacesEmbedder:
    def __init__(
        self,
        model_id="ResNet50",
        device="cuda",
        revision=None,  # nur fuer Interface-Kompatibilitaet
        fc_output_dim=2048,
        image_size=512,
        num_workers=8,
        use_amp=True,
    ):
        if model_id not in AVAILABLE:
            raise ValueError(
                f"Unbekanntes Backbone {model_id!r}. Moeglich: {sorted(AVAILABLE)}"
            )
        if fc_output_dim not in AVAILABLE[model_id]:
            raise ValueError(
                f"{model_id} gibt es nur mit fc_output_dim aus "
                f"{AVAILABLE[model_id]}, nicht {fc_output_dim}."
            )

        self.device = torch.device(device)
        self.device_type = self.device.type
        self.use_amp = use_amp and self.device_type in ("cuda", "mps")
        self.image_size = image_size

        self.executor = (
            ThreadPoolExecutor(max_workers=num_workers) if num_workers > 0 else None
        )

        print(f"EigenPlaces laden: {model_id}, {fc_output_dim} Dimensionen")
        # trust_repo: torch.hub fuehrt Code aus dem Repo aus und fragt sonst
        # interaktiv nach -- das haengt in einem Notebook.
        self.model = (
            torch.hub.load(
                "gmberton/eigenplaces",
                "get_trained_model",
                backbone=model_id,
                fc_output_dim=fc_output_dim,
                trust_repo=True,
            )
            .to(self.device)
            .eval()
        )

        # GeM pooled ueber die ganze Feature-Map, deshalb ist keine feste
        # Eingabegroesse noetig -- fuer Batching brauchen wir aber eine.
        self.transform = T.Compose(
            [
                T.Resize(
                    (image_size, image_size), interpolation=T.InterpolationMode.BILINEAR
                ),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Dimension messen statt der Config glauben.
        with torch.no_grad():
            probe = torch.zeros(1, 3, image_size, image_size, device=self.device)
            self.embedding_dim = int(self.model(probe).shape[-1])
        if self.embedding_dim != fc_output_dim:
            raise RuntimeError(
                f"Modell liefert {self.embedding_dim} statt {fc_output_dim} Dimensionen."
            )
        print(
            f"EigenPlaces bereit: {self.embedding_dim} Dimensionen, "
            f"{image_size}x{image_size}"
        )

    # ------------------------------------------------------------------

    def _load_one(self, path):
        try:
            img = Image.open(path)
            img.draft("RGB", (self.image_size, self.image_size))
            return self.transform(img.convert("RGB"))
        except Exception as e:
            raise RuntimeError(f"Bild nicht lesbar: {path}") from e

    def _load_batch(self, paths):
        if self.executor is not None:
            return list(self.executor.map(self._load_one, paths))
        return [self._load_one(p) for p in paths]

    def embed_images(
        self, image_paths, batch_size=32, checkpoint_path=None, checkpoint_every=500
    ):
        """
        L2-normalisierte float32-Embeddings, Shape (len(image_paths),
        embedding_dim), in der Reihenfolge der Eingabe.
        """
        n = len(image_paths)
        if n == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        out = np.zeros((n, self.embedding_dim), dtype=np.float32)

        done = 0
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                saved = np.load(checkpoint_path)
                if (
                    saved.ndim == 2
                    and saved.shape[1] == self.embedding_dim
                    and saved.shape[0] <= n
                ):
                    out[: saved.shape[0]] = saved
                    done = saved.shape[0]
                    print(f"Checkpoint: {done:,} von {n:,} Bildern uebernommen.")

        starts = list(range(done, n, batch_size))
        for i, start in enumerate(tqdm(starts, desc="EigenPlaces embeddings")):
            paths = image_paths[start : start + batch_size]
            batch = torch.stack(self._load_batch(paths)).to(self.device)

            with torch.inference_mode():
                if self.use_amp:
                    with torch.autocast(
                        device_type=self.device_type, dtype=torch.float16
                    ):
                        vecs = self.model(batch)
                else:
                    vecs = self.model(batch)
                # Das Modell endet auf L2Norm -- hier nur zur Sicherheit,
                # damit alle Embedder dieselbe Zusage einhalten.
                vecs = torch.nn.functional.normalize(vecs.float(), dim=-1)

            out[start : start + len(paths)] = vecs.cpu().numpy()

            if checkpoint_path is not None and (i + 1) % checkpoint_every == 0:
                self._save_checkpoint(checkpoint_path, out[: start + len(paths)])

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, out)
        return out

    @staticmethod
    def _save_checkpoint(path, array):
        tmp = path.with_name(path.name + ".tmp.npy")
        np.save(tmp, array)
        tmp.replace(path)

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
