"""
Gemeinsames Geruest fuer alle Embedder: Laden, Batching, Checkpointing.
Eine Unterklasse setzt self.model und self.transform und ruft _measure_dim().
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm.auto import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def imagenet_transform(image_size):
    return T.Compose(
        [
            T.Resize(
                (image_size, image_size), interpolation=T.InterpolationMode.BILINEAR
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class BaseEmbedder:
    def __init__(self, device, image_size, num_workers=8, use_amp=True):
        self.device = torch.device(device)
        self.device_type = self.device.type
        self.use_amp = use_amp and self.device_type in ("cuda", "mps")
        self.image_size = image_size
        self.executor = (
            ThreadPoolExecutor(max_workers=num_workers) if num_workers > 0 else None
        )
        self.model = None  # Unterklasse setzt
        self.transform = None  # Unterklasse setzt
        self.embedding_dim = None

    def _measure_dim(self):
        """Dimension messen statt aus der Config ableiten."""
        with torch.no_grad():
            probe = torch.zeros(
                1, 3, self.image_size, self.image_size, device=self.device
            )
            self.embedding_dim = int(self.model(probe).shape[-1])
        return self.embedding_dim

    # -- Laden ---------------------------------------------------------

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

    def _collate(self, items):
        """
        Liste aus _load_one -> Batch-Tensor.

        Default: die Elemente sind bereits Tensoren. CLIP ueberschreibt das,
        weil sein Processor auf PIL-Bildern arbeitet und selbst batcht.
        """
        return torch.stack(items)

    # -- Inferenz ------------------------------------------------------

    def _forward(self, batch):
        """Unterklassen mit mehrstufigem Modell koennen das ueberschreiben."""
        return self.model(batch)

    def embed_images(
        self, image_paths, batch_size=32, checkpoint_path=None, checkpoint_every=500
    ):
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

        desc = f"{type(self).__name__} embeddings"
        for i, start in enumerate(tqdm(list(range(done, n, batch_size)), desc=desc)):
            paths = image_paths[start : start + batch_size]
            batch = self._collate(self._load_batch(paths)).to(
                self.device, non_blocking=(self.device_type == "cuda")
            )

            with torch.inference_mode():
                if self.use_amp:
                    with torch.autocast(
                        device_type=self.device_type, dtype=torch.float16
                    ):
                        vecs = self._forward(batch)
                else:
                    vecs = self._forward(batch)
                vecs = torch.nn.functional.normalize(vecs.float(), dim=-1)

            out[start : start + len(paths)] = vecs.cpu().numpy()

            if checkpoint_path is not None and (i + 1) % checkpoint_every == 0:
                self._save_checkpoint(checkpoint_path, out[: start + len(paths)])

        # Bewusst noch nicht loeschen: zwischen hier und dem np.save im
        # Notebook liegen ein paar Zellen, und der Checkpoint ist der einzige
        # Wiedereinstieg, falls der Kernel dazwischen stirbt.
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
