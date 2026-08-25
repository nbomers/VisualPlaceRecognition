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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from tqdm.auto import tqdm


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


class MixEmbedder:
    def __init__(
        self,
        model_id="resnet50",
        device="cuda",
        revision=None,  # nur fuer Interface-Kompatibilitaet
        repo_path="~/third_party/MixVPR",
        weights=None,
        image_size=320,
        agg_config=None,
        num_workers=8,
        use_amp=True,
    ):
        if weights is None:
            raise ValueError("vpr.mixvpr.weights muss auf die .ckpt-Datei zeigen.")

        helper = _import_mixvpr(repo_path)

        self.device = torch.device(device)
        self.device_type = self.device.type
        self.use_amp = use_amp and self.device_type in ("cuda", "mps")
        self.image_size = image_size

        self.executor = (
            ThreadPoolExecutor(max_workers=num_workers) if num_workers > 0 else None
        )

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

        self.transform = T.Compose(
            [
                T.Resize(
                    (image_size, image_size), interpolation=T.InterpolationMode.BILINEAR
                ),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Dimension messen statt aus der Config ableiten.
        with torch.no_grad():
            probe = torch.zeros(1, 3, image_size, image_size, device=self.device)
            self.embedding_dim = int(self.model(probe).shape[-1])
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
        for i, start in enumerate(tqdm(starts, desc="MixVPR embeddings")):
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
