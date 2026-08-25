from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor


class CLIPEmbedder:
    """
    CLIP-Bildembeddings fuer eine Liste von Dateipfaden.

    Laeuft auf CUDA, MPS (Apple Silicon) und CPU. Die Reihenfolge der
    Rueckgabe entspricht exakt der Reihenfolge von image_paths -- darauf
    verlassen sich alle nachgelagerten Notebooks.
    """

    def __init__(
        self,
        model_id,
        device,
        revision=None,
        num_workers=8,
        use_amp=True,
        use_fast_processor=True,
    ):
        # torch.device statt String: dann funktionieren beide Aufrufformen,
        # und device.type ist die einzige Quelle fuer alle Fallunterscheidungen.
        self.device = torch.device(device)
        self.device_type = self.device.type

        # ThreadPool fuer paralleles Laden UND Vorskalieren der Bilder.
        # Bei 0 wird sequentiell geladen.
        self.executor = (
            ThreadPoolExecutor(max_workers=num_workers) if num_workers > 0 else None
        )

        # Mixed Precision auf CUDA und MPS. Auf CPU bringt fp16 nichts.
        # Falls auf MPS NaNs auftauchen (aeltere PyTorch-Versionen haben da
        # Ecken): use_amp=False uebergeben, kostet nur Zeit, keine Qualitaet.
        self.use_amp = use_amp and self.device_type in ("cuda", "mps")
        self.amp_dtype = torch.float16

        self.model = (
            CLIPModel.from_pretrained(model_id, revision=revision)
            .to(self.device)
            .eval()
        )

        # use_fast=True nutzt den torchvision-basierten Prozessor statt PIL.
        # Deutlich schneller; faellt auf den langsamen zurueck, falls die
        # transformers-Version ihn nicht kennt.
        try:
            self.processor = CLIPProcessor.from_pretrained(
                model_id, revision=revision, use_fast=use_fast_processor
            )
        except (TypeError, ValueError):
            self.processor = CLIPProcessor.from_pretrained(model_id, revision=revision)

        self.embedding_dim = self.model.config.projection_dim

        # Zielkantenlaenge des Modells, fuer das Vorskalieren im Worker.
        size = getattr(self.processor.image_processor, "size", None) or {}
        self.target_size = int(size.get("shortest_edge") or size.get("height") or 224)

    # ------------------------------------------------------------------
    # Bild laden
    # ------------------------------------------------------------------

    def _load_one(self, path):
        """
        Laedt ein Bild und dekodiert es direkt kleiner.

        draft() nutzt aus, dass der JPEG-Decoder nativ in 1/2, 1/4 oder 1/8
        aufloesen kann. Aus einem 1024er-Thumb wird so 512 oder 256, bevor
        ueberhaupt Pixel entstehen -- der teure Resize im Prozessor arbeitet
        danach auf einem Bruchteil der Daten.
        """
        try:
            img = Image.open(path)
            img.draft("RGB", (self.target_size, self.target_size))
            return img.convert("RGB")
        except Exception as e:
            # Pfad mitgeben: sonst weiss man nach 300.000 Bildern nicht,
            # welche Datei den Lauf abgebrochen hat.
            raise RuntimeError(f"Bild nicht lesbar: {path}") from e

    def _load_batch(self, batch_paths):
        if self.executor is not None:
            return list(self.executor.map(self._load_one, batch_paths))
        return [self._load_one(p) for p in batch_paths]

    # ------------------------------------------------------------------
    # Ein Batch durch CLIP
    # ------------------------------------------------------------------

    def _embed_batch(self, images):
        pixel_values = self.processor(images=images, return_tensors="pt")[
            "pixel_values"
        ]
        pixel_values = pixel_values.to(
            self.device, non_blocking=(self.device_type == "cuda")
        )

        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast(device_type=self.device_type, dtype=self.amp_dtype):
                    output = self.model.get_image_features(pixel_values=pixel_values)
            else:
                output = self.model.get_image_features(pixel_values=pixel_values)

            # Je nach transformers-Version ein Tensor oder ein ModelOutput.
            if torch.is_tensor(output):
                vectors = output
            elif hasattr(output, "image_embeds"):
                vectors = output.image_embeds
            elif hasattr(output, "pooler_output"):
                vectors = output.pooler_output
            else:
                raise TypeError(
                    f"Unerwarteter Rueckgabetyp von get_image_features(): {type(output)}"
                )

            vectors = torch.nn.functional.normalize(vectors, dim=-1)

        return vectors.float().cpu().numpy()

    # ------------------------------------------------------------------
    # Oeffentliche API
    # ------------------------------------------------------------------

    def embed_images(
        self,
        image_paths,
        batch_size=32,
        checkpoint_path=None,
        checkpoint_every=500,
    ):
        """
        Erstellt L2-normalisierte float32-Embeddings, Shape
        (len(image_paths), embedding_dim), in der Reihenfolge der Eingabe.

        checkpoint_path : optional
            Pfad einer .npy-Datei. Alle checkpoint_every Batches wird der
            bisherige Stand geschrieben. Existiert die Datei beim Start,
            wird dort fortgesetzt.
        checkpoint_every : int
            Abstand in Batches zwischen zwei Zwischenstaenden.
        """
        n_images = len(image_paths)

        if n_images == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        # Vorallokiert statt Liste + concatenate: spart bei 332k Bildern
        # rund 700 MB Spitzenspeicher.
        out = np.zeros((n_images, self.embedding_dim), dtype=np.float32)

        done = 0
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                saved = np.load(checkpoint_path)
                if (
                    saved.ndim == 2
                    and saved.shape[1] == self.embedding_dim
                    and saved.shape[0] <= n_images
                ):
                    out[: saved.shape[0]] = saved
                    done = saved.shape[0]
                    print(
                        f"Checkpoint gefunden: {done:,} von {n_images:,} Bildern uebernommen."
                    )
                else:
                    print(
                        f"Checkpoint passt nicht zu diesem Lauf ({saved.shape}) -- wird ignoriert."
                    )

        starts = list(range(done, n_images, batch_size))

        for i, start in enumerate(tqdm(starts, desc="CLIP embeddings")):
            batch_paths = image_paths[start : start + batch_size]
            images = self._load_batch(batch_paths)
            out[start : start + len(batch_paths)] = self._embed_batch(images)

            if checkpoint_path is not None and (i + 1) % checkpoint_every == 0:
                self._save_checkpoint(checkpoint_path, out[: start + len(batch_paths)])

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, out)

        return out

    @staticmethod
    def _save_checkpoint(path, array):
        # Erst daneben schreiben, dann umbenennen. Ein Abbruch mitten im
        # Schreiben laesst sonst eine halbe Datei zurueck, die beim naechsten
        # Start als gueltiger Checkpoint gelesen wuerde.
        # Endung .npy behalten -- np.save haengt sonst nochmal eine an.
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
