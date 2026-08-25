"""
eigenplaces.py -- EigenPlaces ueber torch.hub.

https://github.com/gmberton/EigenPlaces (MIT)
Berton, Trivigno, Caputo, Masone: "EigenPlaces: Training Viewpoint Robust
Models for Visual Place Recognition", ICCV 2023.

Laden, Batching und Checkpointing kommen aus BaseEmbedder.
"""

from __future__ import annotations

import torch

from .base import BaseEmbedder, imagenet_transform

# backbone -> zulaessige fc_output_dim
AVAILABLE = {
    "ResNet18": (256, 512),
    "ResNet50": (128, 256, 512, 2048),
    "ResNet101": (128, 256, 512, 2048),
    "VGG16": (512,),
}


class EigenPlacesEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_id="ResNet50",
        device="cuda",
        revision=None,
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

        super().__init__(device, image_size, num_workers, use_amp)

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
        self.transform = imagenet_transform(image_size)

        self._measure_dim()
        if self.embedding_dim != fc_output_dim:
            raise RuntimeError(
                f"Modell liefert {self.embedding_dim} statt {fc_output_dim} Dimensionen."
            )
        print(
            f"EigenPlaces bereit: {self.embedding_dim} Dimensionen, "
            f"{image_size}x{image_size}"
        )
