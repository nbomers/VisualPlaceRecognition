"""
anyloc.py -- AnyLoc (DINOv2 + VLAD) als Embedder.

Basiert auf https://github.com/AnyLoc/AnyLoc (BSD-3-Clause).
Modell und VLAD kommen aus dem geklonten Repo, Laden/Batching/Checkpointing
aus BaseEmbedder.

Standardweg ist das offizielle Domaenen-Vokabular: die Cluster-Zentren
stammen aus einem anderen Datensatz und sehen unsere Query-Bilder nie.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .base import BaseEmbedder, imagenet_transform


def _import_anyloc(repo_path):
    """
    Importiert VLAD und DinoV2ExtractFeatures aus dem geklonten AnyLoc-Repo.

    Bewusst NICHT auf Modulebene: sonst scheitert schon
    `import src.models.anyloc`, auch wenn gerade CLIP benutzt wird.
    """
    repo = Path(repo_path).expanduser().resolve()
    if not (repo / "utilities.py").exists():
        raise FileNotFoundError(
            f"AnyLoc-Repo nicht gefunden unter {repo}. "
            "git clone https://github.com/AnyLoc/AnyLoc.git und "
            "vpr.anyloc.repo_path in config.yaml setzen."
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from utilities import VLAD, DinoV2ExtractFeatures  # noqa: E402

    return VLAD, DinoV2ExtractFeatures


class AnyLocEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_id="dinov2_vitg14",
        device="cuda",
        revision=None,
        repo_path="external/AnyLoc",
        vocabulary_domain="urban",
        desc_layer=31,
        desc_facet="value",
        num_clusters=32,
        image_size=322,
        pca_dim=None,
        fit_fallback_paths=None,
        fit_sample_images=5000,
        fit_sample_patches=500_000,
        num_workers=8,
        use_amp=True,
    ):
        if image_size % 14 != 0:
            raise ValueError(
                f"image_size muss ein Vielfaches von 14 sein, ist {image_size}."
            )

        super().__init__(device, image_size, num_workers, use_amp)

        VLAD, DinoV2ExtractFeatures = _import_anyloc(repo_path)
        self._VLAD = VLAD

        self.num_clusters = num_clusters
        self.pca_dim = pca_dim
        self._pca = None  # (mean, components) nach fit_pca

        self.transform = imagenet_transform(image_size)

        print(f"DINOv2 laden: {model_id}, Layer {desc_layer}, Facet {desc_facet}")
        self.dino = DinoV2ExtractFeatures(
            model_id, desc_layer, desc_facet, device=str(self.device)
        )
        # BaseEmbedder erwartet ein self.model; hier ist es der Extraktor.
        # _forward() ist ueberschrieben, _measure_dim() wird nicht benutzt.
        self.model = self.dino

        with torch.no_grad():
            probe = torch.zeros(1, 3, image_size, image_size, device=self.device)
            self.desc_dim = int(self.dino(probe).shape[-1])
        print(f"Patch-Deskriptor: {self.desc_dim} Dimensionen")

        self.vlad = VLAD(num_clusters=num_clusters, desc_dim=self.desc_dim)
        self._load_or_fit_vocabulary(
            repo_path,
            model_id,
            desc_layer,
            desc_facet,
            vocabulary_domain,
            fit_fallback_paths,
            fit_sample_images,
            fit_sample_patches,
        )

        self.vlad_dim = num_clusters * self.desc_dim
        self.embedding_dim = pca_dim or self.vlad_dim
        print(
            f"VLAD-Deskriptor: {self.vlad_dim} Dimensionen"
            + (f" -> PCA auf {pca_dim}" if pca_dim else "")
        )

    # ------------------------------------------------------------------
    # Vokabular
    # ------------------------------------------------------------------

    def _load_or_fit_vocabulary(
        self,
        repo_path,
        model_id,
        layer,
        facet,
        domain,
        fallback_paths,
        n_imgs,
        n_patches,
    ):
        ordner = (
            Path(repo_path).expanduser()
            / "cache"
            / "vocabulary"
            / model_id
            / f"l{layer}_{facet}_c{self.num_clusters}"
            / domain
        )
        # VLAD.can_use_cache_vlad() erkennt ausschliesslich diesen Namen. Das
        # demo-README von AnyLoc nennt c_center.pt, der Code liest c_centers.pt.
        cache = ordner / "c_centers.pt"
        einzahl = ordner / "c_center.pt"

        if not cache.exists() and einzahl.exists():
            raise FileNotFoundError(
                f"Vokabular liegt als {einzahl.name} vor, VLAD liest aber nur "
                f"{cache.name}. Umbenennen:\n  mv {einzahl} {cache}"
            )

        if cache.exists():
            centers = torch.load(cache, map_location="cpu")
            if centers.shape != (self.num_clusters, self.desc_dim):
                raise ValueError(
                    f"Vokabular {cache} hat Form {tuple(centers.shape)}, "
                    f"erwartet ({self.num_clusters}, {self.desc_dim})."
                )

            # Ueber cache_dir laden statt c_centers von Hand zu setzen: fit()
            # traegt dabei auch kmeans.centroids nach, die generate_multi braucht.
            self.vlad = self._VLAD(
                num_clusters=self.num_clusters,
                desc_dim=self.desc_dim,
                cache_dir=str(ordner),
            )
            self.vlad.fit(None)
            self.vocabulary_source = f"offiziell:{domain}"
            print(f"Vokabular geladen: {cache}")
            return

        if not fallback_paths:
            raise FileNotFoundError(
                f"Kein Vokabular unter {ordner}/c_centers.pt\n"
                "python setup_external.py holt es, alternativ cache.zip aus den "
                "AnyLoc-Public-Data entpacken oder fit_fallback_paths mit "
                "TRAIN-Bildern uebergeben (niemals query!)."
            )

        print(
            f"Kein offizielles Vokabular -- fitte auf "
            f"{min(n_imgs, len(fallback_paths)):,} Trainingsbildern. "
            "Ergebnisse sind dann nicht mehr mit dem Paper vergleichbar."
        )
        rng = np.random.default_rng(42)
        sample = [
            fallback_paths[i]
            for i in rng.choice(
                len(fallback_paths), min(n_imgs, len(fallback_paths)), replace=False
            )
        ]

        pool, collected = [], 0
        per_image = max(1, n_patches // len(sample))
        for start in tqdm(range(0, len(sample), 8), desc="Vokabular"):
            descs = self._descriptors(sample[start : start + 8])
            flat = descs.reshape(-1, self.desc_dim)
            take = min(per_image * descs.shape[0], flat.shape[0])
            idx = torch.from_numpy(rng.choice(flat.shape[0], take, replace=False))
            pool.append(flat[idx].cpu())
            collected += take
            if collected >= n_patches:
                break

        self.vlad.fit(torch.cat(pool, dim=0))
        self.vocabulary_source = f"selbst gefittet auf {collected:,} Patches"

    # ------------------------------------------------------------------
    # Extraktion
    # ------------------------------------------------------------------

    def _descriptors(self, batch_paths):
        """Patch-Deskriptoren fuer einen Batch: [B, num_patches, desc_dim]."""
        batch = self._collate(self._load_batch(batch_paths)).to(self.device)
        with torch.no_grad():
            # Ohne autocast liefe die PCA-Phase in fp32, waehrend der
            # eigentliche Lauf ueber base.embed_images fp16 nutzt. Das waere
            # nicht nur langsamer, die PCA wuerde auch auf anderen Zahlen
            # fitten als denen, auf die sie spaeter angewendet wird.
            if self.use_amp:
                with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                    descs = self.dino(batch)
            else:
                descs = self.dino(batch)
        return descs.float()

    def _forward(self, batch):
        """
        Deskriptoren pro Batch sofort zu VLAD verdichten und verwerfen --
        der Speicherbedarf bleibt konstant statt linear in der Bildzahl.
        VLAD selbst laeuft bewusst in float32, auch unter autocast.
        """
        # VLAD.generate() legt seine Zwischenergebnisse auf der CPU an und
        # ruft labels.numpy() -- mit CUDA-Tensoren bricht es ab. AnyLocs
        # eigene Demo schiebt die Deskriptoren aus demselben Grund herunter.
        descs = self.dino(batch).float().cpu()
        vecs = self.vlad.generate_multi(descs)
        if isinstance(vecs, list):
            vecs = torch.stack(vecs)
        return self._project(vecs)

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------

    def fit_pca(self, image_paths, n_images=10_000, batch_size=4, seed=42):
        """
        PCA auf einem Subsample fitten. 49.152 Dimensionen sind fuer 100k
        Bilder nicht speicherbar; 4.096 sind es.

        WICHTIG: nur mit train- oder database-Bildern aufrufen, nie mit query.
        """
        if self.pca_dim is None:
            return
        rng = np.random.default_rng(seed)
        n = min(n_images, len(image_paths))
        sample = [
            image_paths[i] for i in rng.choice(len(image_paths), n, replace=False)
        ]

        chunks = []
        for start in tqdm(range(0, n, batch_size), desc="PCA-Sample"):
            descs = self._descriptors(sample[start : start + batch_size])
            vecs = self.vlad.generate_multi(descs.float().cpu())
            if isinstance(vecs, list):
                vecs = torch.stack(vecs)
            chunks.append(vecs.cpu())
        X = torch.cat(chunks, dim=0).float()

        mean = X.mean(dim=0, keepdim=True)
        _, _, V = torch.pca_lowrank(X - mean, q=min(self.pca_dim, min(X.shape) - 1))
        self._pca = (mean, V[:, : self.pca_dim])
        print(
            f"PCA gefittet: {self.vlad_dim} -> {self._pca[1].shape[1]} Dimensionen "
            f"auf {n:,} Bildern"
        )

    def _project(self, vecs):
        vecs = vecs.float().cpu()
        if self._pca is not None:
            mean, components = self._pca
            vecs = (vecs - mean) @ components
        return vecs

    # ------------------------------------------------------------------

    def embed_images(self, image_paths, batch_size=4, **kwargs):
        if self.pca_dim is not None and self._pca is None:
            raise RuntimeError(
                "pca_dim gesetzt, aber fit_pca() wurde nicht aufgerufen."
            )
        return super().embed_images(image_paths, batch_size=batch_size, **kwargs)
