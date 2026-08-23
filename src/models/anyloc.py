
"""
anyloc.py

Simple AnyLoc pipeline based on:
    DINOv2 local patch descriptors + VLAD aggregation.

Pipeline:
    Images
      ↓
    DINOv2 patch descriptors
      ↓
    VLAD cluster vocabulary
      ↓
    VLAD global descriptors
      ↓
    .pt file

The resulting embeddings can later be used with FAISS for retrieval.

Expected input:
    A folder containing images.

Example:
    python anyloc.py ~/Downloads/images

Or:
    python anyloc.py ~/Downloads/images \
        --num-clusters 8 \
        --model-type dinov2_vits14 \
        --desc-layer 11 \
        --desc-facet key
"""

from __future__ import annotations

import argparse
from pathlib import Path

import einops as ein
import torch
import yaml
from PIL import Image
from torchvision import transforms as T


def find_project_root():
    for dir in (Path.cwd(), *Path.cwd().parents):
        if (dir / "config.yaml").exists():
            return dir
    raise FileNotFoundError("Projektroot nicht gefunden")


def find_upwards(name):
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / name).exists():
            return d / name
    return None


CFG_FILE = find_upwards("config.yaml")
assert CFG_FILE, "config.yaml nicht gefunden (liegt im Projektwurzelverzeichnis)."
CFG = yaml.safe_load(CFG_FILE.read_text())

IMAGE_DIR = CFG["img_download_path"]

# ---------------------------------------------------------------------------
# AnyLoc imports
# ---------------------------------------------------------------------------
from utilities import VLAD, DinoV2ExtractFeatures

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Select CUDA, MPS or CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------



def load_image(
    image_path: Path,
    image_size: int = 320,
) -> torch.Tensor:
    """
    Load and preprocess one image.

    Returns:
        Tensor with shape [C, H, W].
    """

    image = Image.open(image_path).convert("RGB")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        T.Resize((image_size, image_size)),
    ])

    return transform(image)


# ---------------------------------------------------------------------------
# DINOv2
# ---------------------------------------------------------------------------

def load_dino(
    model_type: str,
    desc_layer: int,
    desc_facet: str,
    device: torch.device,
) -> DinoV2ExtractFeatures:
    """
    Load the DINOv2 feature extractor.
    """

    print("Loading DINOv2...")
    print(f"  Model:  {model_type}")
    print(f"  Layer:  {desc_layer}")
    print(f"  Facet:  {desc_facet}")
    print(f"  Device: {device}")

    dino = DinoV2ExtractFeatures(
        model_type,
        desc_layer,
        desc_facet,
        device=device,
    )

    print("DINOv2 loaded.")

    return dino


@torch.no_grad()
def extract_dino_descriptors(
    image_paths: list[Path],
    dino: DinoV2ExtractFeatures,
    device: torch.device,
    image_size: int = 320,
) -> torch.Tensor:
    """
    Extract DINOv2 patch descriptors for all images.

    Returns:
        Tensor:
            [N, num_patches, descriptor_dim]
    """

    descriptors = []

    print(f"Extracting DINOv2 descriptors from {len(image_paths)} images...")

    for i, image_path in enumerate(image_paths):

        image = load_image(
            image_path,
            image_size=image_size,
        )

        _, h, w = image.shape

        # DINOv2 patch size is 14x14.
        h_new = (h // 14) * 14
        w_new = (w // 14) * 14

        image = T.CenterCrop(
            (h_new, w_new)
        )(image)

        image = image.unsqueeze(0).to(device)

        descriptor = dino(image)

        # Expected:
        # [1, num_patches, descriptor_dim]
        descriptors.append(
            descriptor.cpu()
        )

        if (i + 1) % 100 == 0 or i == len(image_paths) - 1:
            print(
                f"  {i + 1}/{len(image_paths)}"
            )

    descriptors = torch.cat(
        descriptors,
        dim=0,
    )

    print(
        f"DINO descriptors: {tuple(descriptors.shape)}"
    )

    return descriptors


# ---------------------------------------------------------------------------
# VLAD
# ---------------------------------------------------------------------------

def build_vlad(
    descriptors: torch.Tensor,
    num_clusters: int = 8,
    assignment: str = "hard",
    soft_temp: float = 1.0,
) -> VLAD:
    """
    Build the VLAD vocabulary from DINO descriptors.

    The cluster centers are learned from all patch descriptors.
    """

    print()
    print("Building VLAD vocabulary...")
    print(f"  Clusters:   {num_clusters}")
    print(f"  Assignment: {assignment}")

    vlad = VLAD(
        num_clusters=num_clusters,
        desc_dim=None,
        vlad_mode=assignment,
        soft_temp=soft_temp,
    )

    # [N, patches, D]
    # ->
    # [N * patches, D]
    all_descriptors = ein.rearrange(
        descriptors,
        "n k d -> (n k) d",
    )

    print(
        f"Clustering {all_descriptors.shape[0]} "
        f"local descriptors..."
    )

    vlad.fit(all_descriptors)

    print(
        f"VLAD centers: {tuple(vlad.c_centers.shape)}"
    )

    return vlad


@torch.no_grad()
def compute_vlad_embeddings(
    descriptors: torch.Tensor,
    vlad: VLAD,
) -> torch.Tensor:
    """
    Convert DINOv2 patch descriptors into VLAD descriptors.

    Returns:
        [N, num_clusters * descriptor_dim]
    """

    print()
    print("Generating VLAD embeddings...")

    embeddings = vlad.generate_multi(
        descriptors
    )

    print(
        f"VLAD embeddings: {tuple(embeddings.shape)}"
    )

    return embeddings


# ---------------------------------------------------------------------------
# Saving / loading
# ---------------------------------------------------------------------------

def save_embeddings(
    output_path: Path,
    image_paths: list[Path],
    embeddings: torch.Tensor,
    vlad: VLAD,
    config: dict,
) -> None:
    """
    Save embeddings, image paths and VLAD vocabulary.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "embeddings": embeddings,
        "image_paths": [
            str(path)
            for path in image_paths
        ],
        "cluster_centers": vlad.c_centers,
        "config": config,
    }

    torch.save(
        data,
        output_path,
    )

    print()
    print("Saved AnyLoc embeddings to:")
    print(f"  {output_path}")


def load_embeddings(
    path: Path,
):
    """
    Load a previously generated AnyLoc file.
    """

    return torch.load(
        path,
        map_location="cpu",
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_anyloc(
    image_dir: Path,
    output_path: Path,
    model_type: str = "dinov2_vits14",
    desc_layer: int = 11,
    desc_facet: str = "key",
    num_clusters: int = 8,
    image_size: int = 320,
    assignment: str = "hard",
    soft_temp: float = 1.0,
) -> None:
    """
    Complete AnyLoc pipeline.
    """

    # -------------------------------------------------------
    # Find images
    # -------------------------------------------------------

    print("=" * 70)
    print("AnyLoc")
    print("=" * 70)

    print("Image directory:")
    print(f"  {image_dir}")

    image_paths = IMAGE_DIR

    if not image_paths:
        raise RuntimeError(
            f"No images found in {image_dir}"
        )

    print(
        f"Found {len(image_paths)} images."
    )

    # -------------------------------------------------------
    # Device
    # -------------------------------------------------------

    device = get_device()

    # -------------------------------------------------------
    # DINOv2
    # -------------------------------------------------------

    dino = load_dino(
        model_type=model_type,
        desc_layer=desc_layer,
        desc_facet=desc_facet,
        device=device,
    )

    # -------------------------------------------------------
    # Extract local descriptors
    # -------------------------------------------------------

    descriptors = extract_dino_descriptors(
        image_paths=image_paths,
        dino=dino,
        device=device,
        image_size=image_size,
    )

    # -------------------------------------------------------
    # VLAD vocabulary
    # -------------------------------------------------------

    vlad = build_vlad(
        descriptors=descriptors,
        num_clusters=num_clusters,
        assignment=assignment,
        soft_temp=soft_temp,
    )

    # -------------------------------------------------------
    # Generate global embeddings
    # -------------------------------------------------------

    embeddings = compute_vlad_embeddings(
        descriptors=descriptors,
        vlad=vlad,
    )

    # -------------------------------------------------------
    # Save
    # -------------------------------------------------------

    config = {
        "model_type": model_type,
        "desc_layer": desc_layer,
        "desc_facet": desc_facet,
        "num_clusters": num_clusters,
        "image_size": image_size,
        "assignment": assignment,
        "soft_temp": soft_temp,
        "num_images": len(image_paths),
        "descriptor_shape": list(descriptors.shape),
        "embedding_shape": list(embeddings.shape),
    }

    save_embeddings(
        output_path=output_path,
        image_paths=image_paths,
        embeddings=embeddings,
        vlad=vlad,
        config=config,
    )

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------

    print()
    print("=" * 70)
    print("Done")
    print("=" * 70)

    print(f"Images:       {len(image_paths)}")
    print(f"DINO shape:   {tuple(descriptors.shape)}")
    print(f"VLAD shape:   {tuple(embeddings.shape)}")
    print(f"Clusters:     {num_clusters}")
    print(f"Output:       {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate AnyLoc/DINOv2+VLAD embeddings."
    )

    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing images.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("anyloc_embeddings.pt"),
        help="Output .pt file.",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="dinov2_vits14",
        choices=[
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ],
    )

    parser.add_argument(
        "--desc-layer",
        type=int,
        default=11,
        help="DINOv2 layer used for descriptors.",
    )

    parser.add_argument(
        "--desc-facet",
        type=str,
        default="key",
        choices=[
            "query",
            "key",
            "value",
            "token",
        ],
    )

    parser.add_argument(
        "--num-clusters",
        type=int,
        default=8,
        help="Number of VLAD clusters.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=320,
        help="Image resize before DINOv2.",
    )

    parser.add_argument(
        "--assignment",
        type=str,
        default="hard",
        choices=[
            "hard",
            "soft",
        ],
        help="VLAD descriptor assignment.",
    )

    parser.add_argument(
        "--soft-temp",
        type=float,
        default=1.0,
        help="Soft assignment temperature.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    run_anyloc(
        image_dir=args.image_dir,
        output_path=args.output,
        model_type=args.model_type,
        desc_layer=args.desc_layer,
        desc_facet=args.desc_facet,
        num_clusters=args.num_clusters,
        image_size=args.image_size,
        assignment=args.assignment,
        soft_temp=args.soft_temp,
    )

