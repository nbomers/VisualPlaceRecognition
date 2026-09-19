"""
Gemeinsames Geruest fuer alle Embedder: Laden, Batching, Checkpointing.
Eine Unterklasse setzt self.model und self.transform und ruft _measure_dim().
"""

from __future__ import annotations

import json
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
        """
        Vektoren fuer alle Bilder. Mit checkpoint_path liegt das Ergebnis als
        memmap auf der Platte, ohne bleibt es im Arbeitsspeicher.

        Der Unterschied ist bei grossen Staedten kein Feinschliff: 699.120
        Bilder mal MegaLocs 8.448 Dimensionen sind 23,6 GB. Als np.zeros stand
        das komplett im RAM, und die Pruefungen in 04 legten noch einmal so
        viel obendrauf -- np.linalg.norm(x, axis=1) allokiert ein Temporaer in
        Arraygroesse. Bei Jena starb der Kernel genau dort, nach anderthalb
        Stunden fertigem Encodieren.

        Ohne Checkpoint bleibt es beim Array im Speicher: locate.py, die
        abgeleiteten Encoder und timing.py encodieren eine Handvoll Bilder,
        da waere eine Datei nur im Weg.
        """
        n = len(image_paths)
        if n == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        if checkpoint_path is None:
            out, done = np.zeros((n, self.embedding_dim), dtype=np.float32), 0
        else:
            checkpoint_path = Path(checkpoint_path)
            out, done = self._open_checkpoint(checkpoint_path, n)

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
                self._save_checkpoint(checkpoint_path, out, start + len(paths), n)

        # Der Checkpoint bleibt liegen, bis 04 ihn zum Ergebnis macht -- er ist
        # der einzige Wiedereinstieg, wenn der Kernel dazwischen stirbt.
        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, out, n, n)

        return out

    # -- Checkpoint --------------------------------------------------------
    #
    # Die Datei hat von Anfang an die volle Form (n, dim); wie weit sie
    # gefuellt ist, steht daneben in <datei>.fortschritt.json. Frueher war die
    # Datei selbst kuerzer und ihre Zeilenzahl der Fortschritt -- das ging nur,
    # solange alles im Speicher stand und am Stueck geschrieben wurde.

    @staticmethod
    def _fortschritt_datei(path):
        return path.with_name(path.name + ".fortschritt.json")

    @classmethod
    def _schreibe_fortschritt(cls, path, done, n):
        datei = cls._fortschritt_datei(path)
        tmp = datei.with_name(datei.name + ".tmp")
        tmp.write_text(json.dumps({"done": int(done), "n": int(n)}), encoding="utf-8")
        tmp.replace(datei)          # atomar: ein Absturz beim Schreiben laesst
                                    # keine halbe JSON zurueck

    def _open_checkpoint(self, path, n):
        """memmap fuer das Ergebnis, und ab welcher Zeile es weitergeht."""
        path.parent.mkdir(parents=True, exist_ok=True)
        form = (n, self.embedding_dim)

        if path.exists():
            vorhanden = np.load(path, mmap_mode="r")
            gefunden = vorhanden.shape
            passt_neu = gefunden == form
            passt_alt = (vorhanden.ndim == 2 and gefunden[1] == self.embedding_dim
                         and gefunden[0] <= n)
            del vorhanden

            if passt_neu:
                out = np.lib.format.open_memmap(path, mode="r+")
                done = 0
                datei = self._fortschritt_datei(path)
                if datei.exists():
                    try:
                        done = int(json.loads(datei.read_text(encoding="utf-8"))["done"])
                    except (ValueError, KeyError, OSError):
                        done = 0    # unlesbar -> lieber neu encodieren als
                                    # auf halben Daten aufsetzen
                done = max(0, min(done, n))
                if done:
                    print(f"Checkpoint: {done:,} von {n:,} Bildern uebernommen.")
                return out, done

            if passt_alt:
                # Checkpoint aus der Zeit vor der memmap: kuerzeres Array, das
                # blockweise umzieht -- nie beides zugleich im Speicher.
                done = int(gefunden[0])
                print(f"Checkpoint (altes Format): {done:,} von {n:,} Bildern werden uebernommen.")
                alt = np.load(path, mmap_mode="r")
                neu_datei = path.with_name(path.name + ".neu.npy")
                neu = np.lib.format.open_memmap(neu_datei, mode="w+",
                                                dtype=np.float32, shape=form)
                for s in range(0, done, 8192):
                    # bis done begrenzen: neu ist laenger als alt, ein
                    # ungebremstes Slice haette dort verschiedene Formen
                    ende = min(s + 8192, done)
                    neu[s:ende] = alt[s:ende]
                neu.flush()
                del alt, neu
                neu_datei.replace(path)
                self._schreibe_fortschritt(path, done, n)
                return np.lib.format.open_memmap(path, mode="r+"), done

            print(f"Checkpoint {path.name} passt nicht (Form {gefunden}, erwartet {form}) "
                  "-- wird neu angelegt.")
            path.unlink()
            self._fortschritt_datei(path).unlink(missing_ok=True)

        out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=form)
        self._schreibe_fortschritt(path, 0, n)
        return out, 0

    @classmethod
    def _save_checkpoint(cls, path, array, done, n):
        array.flush()
        cls._schreibe_fortschritt(path, done, n)

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
