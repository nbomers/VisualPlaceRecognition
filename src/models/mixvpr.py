"""
mixvpr.py -- MixVPR als Embedder mit derselben Schnittstelle wie CLIPEmbedder.

Basiert auf https://github.com/amaralibey/MixVPR
(Ali-bey, Chaib-draa, Giguere: "MixVPR: Feature Mixing for Visual Place
Recognition", WACV 2023).

Backbone und Aggregator werden aus dem geklonten Repo importiert, nicht
nachgebaut. Dieses Modul ist nur der Adapter auf unsere Pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from .base import BaseEmbedder, imagenet_transform


def _import_mixvpr(repo_path):
    """
    Importiert den helper aus dem geklonten MixVPR-Repo.

    Bewusst NICHT auf Modulebene: sonst scheitert schon
    `import src.models.mixvpr`, auch wenn gerade CLIP benutzt wird.
    """
    repo = Path(repo_path).expanduser().resolve()
    if not (repo / "models" / "helper.py").exists():
        raise FileNotFoundError(
            f"MixVPR-Repo nicht gefunden unter {repo}. "
            "git clone https://github.com/amaralibey/MixVPR.git und "
            "vpr.mixvpr.repo_path in config.yaml setzen."
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from models import helper  # noqa: E402

    return helper


class _MixVPRNet(nn.Module):
    """
    Attributnamen backbone/aggregator entsprechen VPRModel aus dem Repo --
    nur dann passt der veroeffentlichte state_dict.
    """

    def __init__(self, backbone, aggregator):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator

    def forward(self, x):
        return self.aggregator(self.backbone(x))


class MixEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_id="resnet50",
        device="cuda",
        revision=None,  # nur fuer Interface-Kompatibilitaet
        repo_path="external/MixVPR",
        weights=None,
        image_size=320,
        agg_config=None,
        num_workers=8,
        use_amp=True,
    ):
        if weights is None:
            raise ValueError("vpr.mixvpr.weights muss auf die .ckpt-Datei zeigen.")

        super().__init__(device, image_size, num_workers, use_amp)

        helper = _import_mixvpr(repo_path)

        agg_config = dict(
            agg_config
            or {
                "in_channels": 1024,
                "in_h": 20,
                "in_w": 20,
                "out_channels": 1024,
                "mix_depth": 4,
                "mlp_ratio": 1,
                "out_rows": 4,
            }
        )

        # in_h/in_w haengen an der Bildgroesse: ResNet50 ohne layer4 hat
        # Stride 16. Stillschweigend falsche Werte geben spaeter kryptische
        # Shape-Fehler tief im Aggregator.
        expected = image_size // 16
        if agg_config["in_h"] != expected or agg_config["in_w"] != expected:
            raise ValueError(
                f"Bei image_size={image_size} muessen in_h und in_w {expected} sein, "
                f"sind aber {agg_config['in_h']}x{agg_config['in_w']}."
            )

        # pretrained=False: die ImageNet-Gewichte werden ohnehin ueberschrieben.
        backbone = helper.get_backbone(
            backbone_arch=model_id,
            pretrained=False,
            layers_to_freeze=0,
            layers_to_crop=[4],
        )
        aggregator = helper.get_aggregator(agg_arch="MixVPR", agg_config=agg_config)

        self.model = _MixVPRNet(backbone, aggregator)
        self._load_weights(Path(weights).expanduser())
        self.model = self.model.to(self.device).eval()

        self.transform = imagenet_transform(image_size)

        self._measure_dim()
        print(
            f"MixVPR bereit: {self.embedding_dim} Dimensionen, {image_size}x{image_size}"
        )

    def _load_weights(self, path):
        if not path.exists():
            raise FileNotFoundError(f"MixVPR-Gewichte nicht gefunden: {path}")

        obj = torch.load(path, map_location="cpu")
        # Je nach Speicherweg ein roher state_dict oder ein Lightning-Checkpoint.
        state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
        state = {
            k.replace("model.", "", 1) if k.startswith("model.") else k: v
            for k, v in state.items()
        }

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(
                f"{len(missing)} Gewichte fehlen, z.B. {missing[:3]}. "
                "Passen agg_config und Gewichtsdatei zusammen? "
                "channels und rows stehen im Dateinamen."
            )
        if unexpected:
            print(
                f"Hinweis: {len(unexpected)} ungenutzte Eintraege im Checkpoint "
                f"(z.B. {unexpected[:3]}) -- meist Trainings-Ballast, unkritisch."
            )
