"""
megaloc.py -- MegaLoc ueber torch.hub.

https://github.com/gmberton/MegaLoc (MIT)
Berton, Masone: "MegaLoc: One Retrieval to Place Them All", 2025.
"""

import torch

from .base import BaseEmbedder, imagenet_transform


class MegaLocEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_id=None,
        device="cuda",
        revision=None,
        image_size=322,
        num_workers=8,
        use_amp=True,
    ):
        super().__init__(device, image_size, num_workers, use_amp)

        # MegaLoc hat keine Konfiguration -- ein Modell, feste Dimension.
        self.model = (
            torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True)
            .to(self.device)
            .eval()
        )

        self.transform = imagenet_transform(image_size)
        self._measure_dim()
        print(
            f"MegaLoc bereit: {self.embedding_dim} Dimensionen, "
            f"{image_size}x{image_size}"
        )
