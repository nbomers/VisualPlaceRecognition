"""
anyloc.py -- AnyLoc (DINOv2 + VLAD) mit derselben Schnittstelle wie CLIPEmbedder.

Basiert auf https://github.com/AnyLoc/AnyLoc (BSD-3-Clause).
Die Modell- und VLAD-Implementierung wird aus dem geklonten Repo importiert,
nicht nachgebaut. Dieses Modul ist nur der Adapter auf unsere Pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm.auto import tqdm


def _import_anyloc(repo_path):
    """
    Importiert VLAD und DinoV2ExtractFeatures aus dem geklonten AnyLoc-Repo.

    Bewusst NICHT auf Modulebene: sonst scheitert schon `import
    src.models.anyloc`, auch wenn gerade CLIP benutzt wird.
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


class AnyLocEmbedder:
    """
    Erzeugt AnyLoc-VLAD-Deskriptoren.

    Standardweg ist das offizielle Domaenen-Vokabular: die Cluster-Zentren
    stammen aus einem anderen Datensatz und sehen unsere Query-Bilder nie.
    Nur wenn kein Vokabular gefunden wird und fit_fallback_paths gesetzt ist,
    wird auf einem Subsample selbst gefittet -- dann aber ausschliesslich auf
    Bildern, die der Aufrufer als unbedenklich uebergibt (train).
    """

    def __init__(
        self,
        model_id="dinov2_vitg14",
        device="cuda",
        revision=None,  # nur fuer Interface-Kompatibilitaet
        repo_path="~/third_party/AnyLoc",
        vocabulary_domain="urban",
        desc_layer=31,
        desc_facet="value",
        num_clusters=32,
        image_size=322,  # Vielfaches von 14
        pca_dim=None,
        fit_fallback_paths=None,
        fit_sample_images=5000,
        fit_sample_patches=500_000,
    ):
        if image_size % 14 != 0:
            raise ValueError(
                f"image_size muss ein Vielfaches von 14 sein, ist {image_size}."
            )

        VLAD, DinoV2ExtractFeatures = _import_anyloc(repo_path)

        self.device = torch.device(device)
        self.image_size = image_size
        self.num_clusters = num_clusters
        self.pca_dim = pca_dim
        self._pca = None  # (mean, components) nach fit_pca

        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        print(f"DINOv2 laden: {model_id}, Layer {desc_layer}, Facet {desc_facet}")
        self.dino = DinoV2ExtractFeatures(
            model_id, desc_layer, desc_facet, device=str(self.device)
        )

        # Deskriptordimension einmal empirisch bestimmen, statt sie zu raten.
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
        cache = (
            Path(repo_path).expanduser()
            / "cache"
            / "vocabulary"
            / model_id
            / f"l{layer}_{facet}_c{self.num_clusters}"
            / domain
            / "c_center.pt"
        )

        if cache.exists():
            centers = torch.load(cache, map_location="cpu")
            if centers.shape != (self.num_clusters, self.desc_dim):
                raise ValueError(
                    f"Vokabular {cache} hat Form {tuple(centers.shape)}, "
                    f"erwartet ({self.num_clusters}, {self.desc_dim})."
                )
            self.vlad.c_centers = centers
            self.vlad.fit(None)  # nur laden, nicht clustern
            self.vocabulary_source = f"offiziell:{domain}"
            print(f"Vokabular geladen: {cache}")
            return

        if not fallback_paths:
            raise FileNotFoundError(
                f"Kein Vokabular unter {cache}.\n"
                "Entweder cache.zip aus der AnyLoc-Public-Data entpacken, oder "
                "fit_fallback_paths mit TRAIN-Bildern uebergeben (niemals query!)."
            )

        print(
            f"Kein offizielles Vokabular -- fitte auf {min(n_imgs, len(fallback_paths)):,} "
            "Trainingsbildern. Ergebnisse sind dann nicht mehr mit dem Paper vergleichbar."
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
            descs = self._descriptors(sample[start : start + 8])  # [B, P, D]
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
        tensors = []
        for path in batch_paths:
            try:
                with Image.open(path) as img:
                    tensors.append(self.transform(img.convert("RGB")))
            except Exception as e:
                raise RuntimeError(f"Bild nicht lesbar: {path}") from e

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            return self.dino(batch)

    def embed_images(
        self, image_paths, batch_size=4, checkpoint_path=None, checkpoint_every=200
    ):
        """
        L2-normalisierte float32-Embeddings, Shape (len(image_paths), embedding_dim),
        in der Reihenfolge der Eingabe.

        Deskriptoren werden pro Batch verworfen -- der Speicherbedarf ist
        konstant, nicht linear in der Bildzahl.
        """
        n = len(image_paths)
        if n == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        if self.pca_dim is not None and self._pca is None:
            raise RuntimeError(
                "pca_dim gesetzt, aber fit_pca() wurde nicht aufgerufen."
            )

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
        for i, start in enumerate(tqdm(starts, desc="AnyLoc embeddings")):
            paths = image_paths[start : start + batch_size]
            descs = self._descriptors(paths)
            vecs = self.vlad.generate_multi(descs)  # [B, K*D]
            vecs = self._project(vecs)
            out[start : start + len(paths)] = vecs.cpu().numpy().astype(np.float32)

            if checkpoint_path is not None and (i + 1) % checkpoint_every == 0:
                self._save_checkpoint(checkpoint_path, out[: start + len(paths)])

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, out)
        return out

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
            chunks.append(self.vlad.generate_multi(descs).cpu())
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
        return torch.nn.functional.normalize(vecs, dim=-1)

    @staticmethod
    def _save_checkpoint(path, array):
        tmp = path.with_name(path.name + ".tmp.npy")
        np.save(tmp, array)
        tmp.replace(path)
