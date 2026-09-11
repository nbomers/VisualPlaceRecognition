"""
Bringt alle Encoder auf dieselbe Deskriptorbreite.

Zwei Fragen stehen dahinter:

  1. Fairer Vergleich. MegaLoc hat 8.448 Dimensionen, CLIP 512 -- wieviel des
     Vorsprungs ist Koennen und wieviel nur Breite?
  2. Adapter-Test. Der lineare Adapter ist eine d-x-d-Matrix, seine
     Parameterzahl waechst also quadratisch: 4,2 Mio bei 2048, 71,4 Mio bei
     8448. Gemessen waechst der Schaden monoton mit der Breite. Auf gleicher
     Breite hat jeder Encoder dieselben 262.144 Parameter. Bleibt die
     Rangfolge des Schadens, liegt es am Encoder; wird sie flach, war es die
     Parameterzahl.

Das Skript rechnet weder Retrieval noch Recall. Es schreibt die reduzierten
Embeddings so, wie 04 sie schreiben wuerde -- mit Metadaten und einem
Fingerabdruck aus run_guard. Danach ist `{method}_pca512` fuer die Pipeline
ein gewoehnlicher Encoder:

    python experiments/pca_reduce.py
    python run.py --method eigenplaces_pca512 --adapter all

Die PCA wird ausschliesslich auf dem train-Split angepasst. database und
query fliessen nie in die Anpassung ein, sonst waere es Leakage.

    python experiments/pca_reduce.py                       # alle konfigurierten
    python experiments/pca_reduce.py --methods eigenplaces_pca512
    python experiments/pca_reduce.py --force               # auch neu schreiben
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Liegt in experiments/, die Pipeline eine Ebene darueber.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.run_guard import embedding_fingerprint, write_fingerprint  # noqa: E402

CFG = load_config(ROOT)
EMBEDDING_ROOT = ROOT / "data" / "embeddings"

# Blockweise transformieren: MegaLoc sind 332.867 x 8448 float32, also
# 11,2 GB. Vollstaendig laden waere auf 16 GB nicht drin.
BLOCK = 8192


def _args():
    ap = argparse.ArgumentParser(
        description="Reduziert Encoder-Embeddings auf gemeinsame Breite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--methods", metavar="LISTE",
                    help="Kommaliste. Standard: alle Eintraege aus vpr.models, "
                         "die einen source-Eintrag haben.")
    ap.add_argument("--force", action="store_true",
                    help="Auch schreiben, wenn das Ergebnis schon passt")
    return ap.parse_args()


def pca_methoden():
    """Alle Modelle, deren Konfigurationsblock ein source-Feld hat."""
    vpr = CFG["vpr"]
    return [name for name in vpr["models"]
            if isinstance(vpr.get(name), dict) and "source" in vpr[name]]


def reduce_one(name, force):
    block = CFG["vpr"][name]
    quelle = block["source"]
    dim = int(block["pca_dim"])
    fit_images = int(block.get("fit_images", 50000))
    whiten = bool(block.get("whiten", False))

    quelle_dir = EMBEDDING_ROOT / quelle
    quelle_npy = quelle_dir / f"{quelle}_embeddings.npy"
    quelle_meta = quelle_dir / f"{quelle}_metadata.parquet"
    for p in (quelle_npy, quelle_meta):
        if not p.exists():
            print(f"  uebersprungen: {p.relative_to(ROOT)} fehlt")
            return False

    ziel_dir = EMBEDDING_ROOT / name
    ziel_npy = ziel_dir / f"{name}_embeddings.npy"
    ziel_meta = ziel_dir / f"{name}_metadata.parquet"

    metadata = pd.read_parquet(quelle_meta)
    fingerprint = embedding_fingerprint(CFG, name, "none", metadata)

    if ziel_npy.exists() and not force:
        vorhanden = ziel_npy.with_name(ziel_npy.name + ".fingerprint.json")
        if vorhanden.exists():
            import json
            gespeichert = json.loads(vorhanden.read_text()).get("fingerprint")
            if gespeichert == fingerprint:
                print(f"  liegt vor und passt: {ziel_npy.name}")
                return True
        print(f"  vorhanden, passt aber nicht -- wird neu geschrieben")

    # mmap: die Quelle wird nur der Reihe nach blockweise gelesen.
    quelle_emb = np.load(quelle_npy, mmap_mode="r")
    n, d_quelle = quelle_emb.shape
    if dim > d_quelle:
        print(f"  uebersprungen: pca_dim={dim} > Quelldimension {d_quelle}")
        return False

    # ------------------------------------------------------------------
    # Anpassen -- ausschliesslich auf train.
    # ------------------------------------------------------------------
    train_rows = np.flatnonzero((metadata["split"] == "train").to_numpy())
    if len(train_rows) == 0:
        print("  uebersprungen: keine train-Zeilen in den Metadaten")
        return False

    rng = np.random.default_rng(int(CFG["vpr"]["split_seed"]))
    if len(train_rows) > fit_images:
        gezogen = np.sort(rng.choice(train_rows, fit_images, replace=False))
    else:
        gezogen = train_rows

    print(f"  Quelle {quelle}  {n:,} x {d_quelle}  ->  {dim}")
    print(f"  PCA angepasst auf {len(gezogen):,} train-Zeilen"
          f"{' , mit Whitening' if whiten else ''}")

    fit_daten = np.ascontiguousarray(quelle_emb[gezogen], dtype=np.float32)
    pca = PCA(n_components=dim, svd_solver="randomized", whiten=whiten,
              random_state=int(CFG["vpr"]["split_seed"]))
    pca.fit(fit_daten)
    erklaert = float(pca.explained_variance_ratio_.sum())
    del fit_daten
    print(f"  erklaerte Varianz: {erklaert:.1%}")

    # ------------------------------------------------------------------
    # Anwenden -- blockweise auf die Platte.
    # ------------------------------------------------------------------
    ziel_dir.mkdir(parents=True, exist_ok=True)
    ziel = np.lib.format.open_memmap(
        ziel_npy, mode="w+", dtype=np.float32, shape=(n, dim)
    )
    for start in range(0, n, BLOCK):
        block = np.ascontiguousarray(
            quelle_emb[start : start + BLOCK], dtype=np.float32
        )
        reduziert = pca.transform(block).astype(np.float32)
        # 06 sucht ueber das innere Produkt und setzt L2-normalisierte
        # Vektoren voraus -- wie bei allen anderen Encodern auch.
        norm = np.linalg.norm(reduziert, axis=1, keepdims=True)
        reduziert /= np.where(norm > 0, norm, 1.0)
        assert np.isfinite(reduziert).all()
        ziel[start : start + BLOCK] = reduziert
    ziel.flush()

    metadata.to_parquet(ziel_meta, index=False)
    write_fingerprint(
        ziel_npy, fingerprint,
        derived_from=quelle_npy.name,
        pca_dim=dim,
        whiten=whiten,
        fit_images=int(len(gezogen)),
        explained_variance_ratio=erklaert,
    )
    print(f"  geschrieben: {ziel_npy.relative_to(ROOT)}")
    return True


def main():
    args = _args()
    methoden = ([m.strip() for m in args.methods.split(",") if m.strip()]
                if args.methods else pca_methoden())
    if not methoden:
        raise SystemExit(
            "Keine PCA-Varianten konfiguriert. In config.yaml unter vpr.models "
            "einen Namen eintragen und einen Block mit source/pca_dim anlegen."
        )

    fertig = []
    for name in methoden:
        print(f"\n{name}")
        print("-" * len(name))
        if name not in CFG["vpr"].get("models", {}):
            print(f"  uebersprungen: kein Eintrag unter vpr.models")
            continue
        if reduce_one(name, args.force):
            fertig.append(name)

    print()
    print("=" * 62)
    if fertig:
        print("Bereit fuer die Pipeline:")
        print(f"  python run.py --method {','.join(fertig)} --adapter all")
    else:
        print("Nichts geschrieben.")
    print("=" * 62)


if __name__ == "__main__":
    main()
