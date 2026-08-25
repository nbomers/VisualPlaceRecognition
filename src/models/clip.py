"""
clip.py -- CLIP-Bildembeddings.

Laden, Batching und Checkpointing kommen aus BaseEmbedder. Hier steht nur,
was an CLIP anders ist: der Processor arbeitet auf PIL-Bildern und batcht
selbst, und die Bildmerkmale kommen ueber get_image_features().
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .base import BaseEmbedder


class CLIPEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_id,
        device,
        revision=None,
        num_workers=8,
        use_amp=True,
        use_fast_processor=True,
    ):
        # Die echte Bildgroesse steht erst nach dem Laden des Processors fest.
        super().__init__(
            device, image_size=224, num_workers=num_workers, use_amp=use_amp
        )

        self.model = (
            CLIPModel.from_pretrained(model_id, revision=revision)
            .to(self.device)
            .eval()
        )

        # use_fast=True nutzt den torchvision-basierten Processor statt PIL.
        # Faellt auf den langsamen zurueck, falls die transformers-Version
        # ihn nicht kennt.
        try:
            self.processor = CLIPProcessor.from_pretrained(
                model_id, revision=revision, use_fast=use_fast_processor
            )
        except (TypeError, ValueError):
            self.processor = CLIPProcessor.from_pretrained(model_id, revision=revision)

        size = getattr(self.processor.image_processor, "size", None) or {}
        self.image_size = int(size.get("shortest_edge") or size.get("height") or 224)

        # CLIP kennt seine Ausgabedimension aus der Modellconfig. Ein
        # Probe-Forward wie in _measure_dim() ginge hier nicht, weil
        # self.model(tensor) nicht der Bildpfad ist.
        self.embedding_dim = int(self.model.config.projection_dim)

        print(
            f"CLIP bereit: {self.embedding_dim} Dimensionen, "
            f"{self.image_size}x{self.image_size}"
        )

    # ------------------------------------------------------------------
    # CLIP-spezifisch
    # ------------------------------------------------------------------

    def _load_one(self, path):
        """
        Gibt ein PIL-Bild zurueck, keinen Tensor -- das Skalieren macht der
        Processor fuer den ganzen Batch auf einmal.

        draft() dekodiert das JPEG direkt kleiner: aus einem 1024er-Thumb
        wird 512 oder 256, bevor ueberhaupt Pixel entstehen.
        """
        try:
            img = Image.open(path)
            img.draft("RGB", (self.image_size, self.image_size))
            return img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Bild nicht lesbar: {path}") from e

    def _collate(self, images):
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    def _forward(self, batch):
        output = self.model.get_image_features(pixel_values=batch)

        # Je nach transformers-Version ein Tensor oder ein ModelOutput.
        if torch.is_tensor(output):
            return output
        if hasattr(output, "image_embeds"):
            return output.image_embeds
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        raise TypeError(
            f"Unerwarteter Rueckgabetyp von get_image_features(): {type(output)}"
        )
