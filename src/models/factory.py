"""
Ein Encoder je Name, gebaut aus der config.yaml.

Ein neuer Encoder heisst: eine Klasse in src/models/, eine Bauvorschrift
hier, eine Zeile in _BAUER. Abgeleitete Encoder (PCA, Whitening,
Verkettung) brauchen nichts davon -- sie entstehen aus ihrem source-Block
in config.yaml ueber derived.py.

Die Importe liegen in den Bauvorschriften, nicht am Dateianfang: jeder
Encoder zieht sein eigenes Fremd-Repo nach, und wer CLIP rechnet, soll nicht
an einem fehlenden MixVPR-Klon scheitern.
"""

from pathlib import Path


def _from_root(project_root, path):
    """Relative Pfade aus der config.yaml gegen den Projektroot aufloesen --
    resolve() allein haengt am Arbeitsverzeichnis, in Jupyter ist das
    notebooks/."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else project_root / path


def _clip(cfg, model_id, device, root):
    from .clip import CLIPEmbedder

    return CLIPEmbedder(
        model_id=model_id,
        device=device,
        revision=cfg["vpr"].get("clip", {}).get("revision"),
        num_workers=8,
        use_amp=True,
    )


def _anyloc(cfg, model_id, device, root):
    from ..paths import Paths
    from .anyloc import AnyLocEmbedder

    a = cfg["vpr"]["anyloc"]
    embedder = AnyLocEmbedder(
        model_id=model_id,
        device=device,
        repo_path=_from_root(root, a["repo_path"]),
        vocabulary_domain=a["vocabulary_domain"],
        desc_layer=a["desc_layer"],
        desc_facet=a["desc_facet"],
        num_clusters=a["num_clusters"],
        image_size=a["image_size"],
        pca_dim=a.get("pca_dim"),
        seed=int(cfg["vpr"]["split_seed"]),
    )
    # AnyLoc reduziert seine 49.152 VLAD-Dimensionen mit einer PCA, die 04 auf
    # den train-Bildern anpasst und neben die Embeddings legt. Ohne sie liefert
    # der Encoder den rohen Vektor -- andere Breite als die Datenbank, und der
    # Fehler faellt erst beim Suchen auf. Deshalb hier laden oder abbrechen.
    if a.get("pca_dim"):
        pfad = Paths(cfg, root).embedding_dir("anyloc") / "anyloc_pca.npz"
        if not pfad.exists():
            raise FileNotFoundError(
                f"AnyLocs PCA fehlt: {pfad}\n"
                "Sie entsteht in 04 und liegt neben den Embeddings.\n"
                "  python run.py --method anyloc --from 04"
            )
        embedder.load_pca(pfad)
    return embedder


def _eigenplaces(cfg, model_id, device, root):
    from .eigenplaces import EigenPlacesEmbedder

    e = cfg["vpr"]["eigenplaces"]
    return EigenPlacesEmbedder(
        model_id=model_id,
        device=device,
        fc_output_dim=e["fc_output_dim"],
        image_size=e["image_size"],
    )


def _mixvpr(cfg, model_id, device, root):
    from .mixvpr import MixEmbedder

    m = cfg["vpr"]["mixvpr"]
    return MixEmbedder(
        model_id=model_id,
        device=device,
        repo_path=_from_root(root, m["repo_path"]),
        weights=_from_root(root, m["weights"]),
        image_size=m["image_size"],
        agg_config=m["agg_config"],
    )


def _megaloc(cfg, model_id, device, root):
    from .megaloc import MegaLocEmbedder

    # Braucht keinen eigenen Block in der config. Wer einen anlegt -- etwa
    # image_size fuer hoehere Aufloesung -- aendert damit den Fingerabdruck.
    # Das ist richtig so: die Embeddings aendern sich dann auch.
    m = cfg["vpr"].get("megaloc") or {}
    return MegaLocEmbedder(device=device, image_size=m.get("image_size", 322))


_BAUER = {
    "clip": _clip,
    "anyloc": _anyloc,
    "eigenplaces": _eigenplaces,
    "mixvpr": _mixvpr,
    "megaloc": _megaloc,
}


def build_embedder(method, cfg, device, project_root):
    """Basis-Encoder aus _BAUER; abgeleitete (source/sources) ueber derived.py."""
    block = cfg["vpr"].get(method)
    if method not in _BAUER and isinstance(block, dict) and ("source" in block or "sources" in block):
        from .derived import DerivedEmbedder

        return DerivedEmbedder(method, cfg, device, project_root)
    if method not in _BAUER:
        raise ValueError(f"Unbekannter Encoder {method!r}. Bekannt: {sorted(_BAUER)}.")
    model_id = cfg["vpr"]["models"][method]
    embedder = _BAUER[method](cfg, model_id, device, project_root)
    embedder.name = method
    return embedder
