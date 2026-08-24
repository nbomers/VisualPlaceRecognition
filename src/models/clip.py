import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
from transformers import CLIPModel, CLIPProcessor


class CLIPEmbedder:
    def __init__(
        self,
        model_id,
        device,
        revision=None,
        num_workers=8,
        use_amp=True,
    ):
        self.device = device
        self.num_workers = num_workers

        # ThreadPool für paralleles Laden der Bilder.
        # Bei 0 wird sequentiell geladen.
        self.executor = (
            ThreadPoolExecutor(max_workers=num_workers)
            if num_workers > 0
            else None
        )

        # Mixed Precision nur auf CUDA verwenden.
        self.use_amp = use_amp and device == "cuda" and str(device).startswith("cuda")

        # CLIP-Modell laden.
        self.model = (
            CLIPModel.from_pretrained(
                model_id,
                revision=revision,
            )
            .to(device)
            .eval()
        )

        # Processor laden.
        self.processor = CLIPProcessor.from_pretrained(
            model_id,
            revision=revision,
        )

        # Dimension des finalen CLIP-Embeddings.
        self.embedding_dim = self.model.config.projection_dim

    def _load_one(self, path):
        """
        Lädt ein einzelnes Bild und konvertiert es zu RGB.
        Das eigentliche Preprocessing übernimmt später der Processor
        für den kompletten Batch.
        """
        with Image.open(path) as img:
            return img.convert("RGB")

    def embed_images(self, image_paths, batch_size=32):
        """
        Erstellt CLIP-Embeddings für eine Liste von Bildern.

        Parameters
        ----------
        image_paths : list
            Liste mit Bildpfaden.

        batch_size : int
            Anzahl Bilder, die gleichzeitig durch CLIP laufen.

        Returns
        -------
        np.ndarray
            Array mit Shape:

                (Anzahl Bilder, embedding_dim)

            Die Embeddings sind L2-normalisiert und float32.
        """

        if not image_paths:
            return np.empty(
                (0, self.embedding_dim),
                dtype=np.float32,
            )

        batches = [
            image_paths[i : i + batch_size]
            for i in range(0, len(image_paths), batch_size)
        ]

        embeddings = []

        for batch_paths in tqdm(
            batches,
            desc="CLIP embeddings",
        ):
            # ---------------------------------------------------------
            # 1. Bilder laden
            # ---------------------------------------------------------

            if self.executor is not None:
                images = list(
                    self.executor.map(
                        self._load_one,
                        batch_paths,
                    )
                )
            else:
                images = [
                    self._load_one(path)
                    for path in batch_paths
                ]

            # ---------------------------------------------------------
            # 2. KOMPLETTEN Batch preprocessen
            # ---------------------------------------------------------

            pixel_values = self.processor(
                images=images,
                return_tensors="pt",
            )["pixel_values"]

            # ---------------------------------------------------------
            # 3. Batch auf GPU / Device übertragen
            # ---------------------------------------------------------

            pixel_values = pixel_values.to(
                self.device,
                non_blocking=True,
            )

            # ---------------------------------------------------------
            # 4. CLIP Inference
            # ---------------------------------------------------------

            with torch.inference_mode():

                if self.use_amp:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    ):
                        output = self.model.get_image_features(
                            pixel_values=pixel_values,
                        )
                else:
                    output = self.model.get_image_features(
                        pixel_values=pixel_values,
                    )

                # -----------------------------------------------------
                # 5. Output behandeln
                # -----------------------------------------------------

                if torch.is_tensor(output):
                    batch_embeddings = output

                elif hasattr(output, "image_embeds"):
                    batch_embeddings = output.image_embeds

                elif hasattr(output, "pooler_output"):
                    batch_embeddings = output.pooler_output

                else:
                    raise TypeError(
                        "Unerwarteter Rückgabetyp von "
                        f"get_image_features(): {type(output)}"
                    )

                # -----------------------------------------------------
                # 6. L2-Normalisierung
                # -----------------------------------------------------

                batch_embeddings = torch.nn.functional.normalize(
                    batch_embeddings,
                    dim=-1,
                )

            # ---------------------------------------------------------
            # 7. Zurück auf CPU und float32
            # ---------------------------------------------------------

            embeddings.append(
                batch_embeddings
                .float()
                .cpu()
                .numpy()
            )

        # -------------------------------------------------------------
        # 8. Alle Batches zusammenfügen
        # -------------------------------------------------------------

        return np.concatenate(
            embeddings,
            axis=0,
        )

    def close(self):
        """
        Beendet den ThreadPool sauber.
        """
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def __del__(self):
        """
        Sicherheitsnetz für den ThreadPool.
        """
        try:
            self.close()
        except Exception:
            pass


'''


import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor


class CLIPEmbedder:
    def __init__(self, model_id, device, revision=None, num_workers=8, use_amp=True):
        self.device = device
        # ThreadPool statt multiprocessing.DataLoader: spawnt keine neuen
        # Prozesse (funktioniert daher in Jupyter), PIL gibt beim Decodieren
        # trotzdem den GIL frei -> echter Parallelitäts-Gewinn beim I/O.
        self.num_workers = num_workers
        self.executor = (
            ThreadPoolExecutor(max_workers=num_workers) if num_workers > 0 else None
        )
        # Mixed Precision (fp16) lohnt sich primär auf CUDA (Tensor Cores)
        self.use_amp = use_amp and device == "cuda"

        self.model = (
            CLIPModel.from_pretrained(model_id, revision=revision).to(device).eval()
        )

        self.processor = CLIPProcessor.from_pretrained(model_id, revision=revision)
        self.embedding_dim = self.model.config.projection_dim

    def _load_one(self, path):
        with Image.open(path) as img:
            img = img.convert("RGB")
        # [0] da der Processor pro Bild einen Batch der Größe 1 zurückgibt
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]

    def embed_images(self, image_paths, batch_size=32):
        batches = [
            image_paths[i : i + batch_size]
            for i in range(0, len(image_paths), batch_size)
        ]

        emb = []

        for batch_paths in tqdm(batches, desc="CLIP embeddings"):
            if self.executor is not None:
                pixel_list = list(self.executor.map(self._load_one, batch_paths))
            else:
                pixel_list = [self._load_one(p) for p in batch_paths]

            pixel_values = torch.stack(pixel_list).to(self.device, non_blocking=True)

            with torch.inference_mode():
                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = self.model.get_image_features(
                            pixel_values=pixel_values
                        )
                else:
                    output = self.model.get_image_features(pixel_values=pixel_values)

                # Je nach transformers-Version liefert get_image_features()
                # entweder direkt einen Tensor oder ein ModelOutput-Objekt
                # (z.B. mit .pooler_output / .image_embeds). Beides abfangen.
                if torch.is_tensor(output):
                    batch_embeddings = output
                elif hasattr(output, "image_embeds"):
                    batch_embeddings = output.image_embeds
                elif hasattr(output, "pooler_output"):
                    batch_embeddings = output.pooler_output
                else:
                    raise TypeError(
                        f"Unerwarteter Rückgabetyp von get_image_features(): {type(output)}"
                    )

                batch_embeddings = torch.nn.functional.normalize(
                    batch_embeddings, dim=-1
                )

            # zurück zu float32 vor dem Export nach numpy
            emb.append(batch_embeddings.float().cpu().numpy())

        return np.concatenate(emb, axis=0)


'''