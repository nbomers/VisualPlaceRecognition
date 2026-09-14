"""
Zwei oder mehr Encoder zu einem Deskriptor verketten.

Ensembles schlagen in der Bildsuche fast immer ihr bestes Mitglied um ein
bis drei Punkte: was der eine Encoder verwechselt, verwechselt der andere
oft nicht. Die Vektoren sind L2-normalisiert, also traegt jeder Encoder
gleich viel zum inneren Produkt bei; nach dem Verketten wird noch einmal
normalisiert, damit 06 wie gewohnt ueber das innere Produkt suchen kann.

Das Skript schreibt den verketteten Encoder so, wie 04 es taete -- mit
Metadaten und Fingerabdruck aus run_guard. Danach ist er fuer die Pipeline
ein gewoehnlicher Encoder:

    python experiments/concat_embeddings.py
    python run.py --method eigenplaces_megaloc_concat --adapter none

Konfiguriert wird er in config.yaml unter vpr.models plus einem Block mit
`sources`, analog zu den PCA-Varianten:

    eigenplaces_megaloc_concat:
      sources: ["eigenplaces_pcaw512", "megaloc_pcaw512"]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Liegt in experiments/, die Pipeline eine Ebene darueber.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.run_guard import embedding_fingerprint, metadata_digest, write_fingerprint  # noqa: E402

CFG = load_config(ROOT)
EMBEDDING_ROOT = ROOT / "data" / "embeddings"
BLOCK = 8192


def _args():
    ap = argparse.ArgumentParser(
        description="Verkettet Encoder-Embeddings zu einem Ensemble-Deskriptor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--methods", metavar="LISTE",
                    help="Kommaliste. Standard: alle Eintraege aus vpr.models "
                         "mit sources-Eintrag.")
    ap.add_argument("--force", action="store_true",
                    help="Auch schreiben, wenn das Ergebnis schon passt")
    return ap.parse_args()


def concat_methoden():
    vpr = CFG["vpr"]
    return [name for name in vpr["models"]
            if isinstance(vpr.get(name), dict) and "sources" in vpr[name]]


def concat_one(name, force):
    quellen = list(CFG["vpr"][name]["sources"])
    if len(quellen) < 2:
        print("  uebersprungen: weniger als zwei Quellen")
        return False

    quell_npy, quell_meta = [], []
    for q in quellen:
        npy = EMBEDDING_ROOT / q / f"{q}_embeddings.npy"
        meta = EMBEDDING_ROOT / q / f"{q}_metadata.parquet"
        if not npy.exists() or not meta.exists():
            print(f"  uebersprungen: {npy.relative_to(ROOT)} fehlt")
            return False
        quell_npy.append(npy)
        quell_meta.append(meta)

    # Die Quellen koennen dieselben Bilder in verschiedener Reihenfolge
    # halten -- auf zwei Rechnern gerechnete Encoder tun das. Deshalb wird
    # ueber die image_id ausgerichtet, nie ueber die Zeilennummer. Bilder,
    # die nicht in allen Quellen liegen, fallen weg; database und query
    # muessen dabei vollstaendig erhalten bleiben.
    metadaten = [pd.read_parquet(m) for m in quell_meta]
    gemeinsam = set(metadaten[0].image_id)
    for m in metadaten[1:]:
        gemeinsam &= set(m.image_id)
    metadata = metadaten[0][metadaten[0].image_id.isin(gemeinsam)].reset_index(drop=True)
    for q, m in zip(quellen, metadaten):
        weg = m[~m.image_id.isin(gemeinsam)]
        if len(weg):
            print(f"  {q}: {len(weg):,} Bilder nicht in allen Quellen, fallen weg "
                  f"(Splits: {weg.split.value_counts().to_dict()})")
            if (weg.split != "train").any():
                print("  uebersprungen: database oder query waeren unvollstaendig")
                return False
    zeilen = []
    for m in metadaten:
        pos = pd.Series(np.arange(len(m)), index=m.image_id.to_numpy())
        zeilen.append(pos.loc[metadata.image_id.to_numpy()].to_numpy())
    if any(metadata_digest(m) != metadata_digest(metadaten[0]) for m in metadaten[1:]):
        print("  Quellen in verschiedener Reihenfolge -- ueber image_id ausgerichtet")

    ziel_dir = EMBEDDING_ROOT / name
    ziel_npy = ziel_dir / f"{name}_embeddings.npy"
    fingerprint = embedding_fingerprint(CFG, name, "none", metadata)

    if ziel_npy.exists() and not force:
        sidecar = ziel_npy.with_name(ziel_npy.name + ".fingerprint.json")
        if sidecar.exists() and json.loads(sidecar.read_text()).get("fingerprint") == fingerprint:
            print(f"  liegt vor und passt: {ziel_npy.name}")
            return True
        print("  vorhanden, passt aber nicht -- wird neu geschrieben")

    arrays = [np.load(p, mmap_mode="r") for p in quell_npy]
    n = len(metadata)
    for q, a, m in zip(quellen, arrays, metadaten):
        if len(a) != len(m):
            print(f"  uebersprungen: {q} hat {len(a):,} Zeilen, Metadaten {len(m):,}")
            return False
    dims = [a.shape[1] for a in arrays]
    dim = sum(dims)
    print(f"  {' + '.join(f'{q} ({d})' for q, d in zip(quellen, dims))}  ->  {dim}")

    ziel_dir.mkdir(parents=True, exist_ok=True)
    ziel = np.lib.format.open_memmap(ziel_npy, mode="w+", dtype=np.float32, shape=(n, dim))
    for start in range(0, n, BLOCK):
        block = np.concatenate(
            [np.asarray(a[z[start : start + BLOCK]], dtype=np.float32)
             for a, z in zip(arrays, zeilen)],
            axis=1,
        )
        # Jede Quelle ist L2-normalisiert; verkettet hat der Vektor Norm
        # sqrt(k). Neu normalisieren, damit inneres Produkt = Cosinus bleibt.
        norm = np.linalg.norm(block, axis=1, keepdims=True)
        block /= np.where(norm > 0, norm, 1.0)
        assert np.isfinite(block).all()
        ziel[start : start + BLOCK] = block
    ziel.flush()

    metadata.to_parquet(ziel_dir / f"{name}_metadata.parquet", index=False)
    write_fingerprint(ziel_npy, fingerprint, derived_from=[p.name for p in quell_npy], dims=dims)
    print(f"  geschrieben: {ziel_npy.relative_to(ROOT)}")
    return True


def main():
    args = _args()
    methoden = ([m.strip() for m in args.methods.split(",") if m.strip()]
                if args.methods else concat_methoden())
    if not methoden:
        raise SystemExit(
            "Keine Verkettung konfiguriert. In config.yaml unter vpr.models einen "
            "Namen eintragen und einen Block mit sources anlegen."
        )
    fertig = []
    for name in methoden:
        print(f"\n{name}")
        print("-" * len(name))
        if name not in CFG["vpr"].get("models", {}):
            print("  uebersprungen: kein Eintrag unter vpr.models")
            continue
        if concat_one(name, args.force):
            fertig.append(name)
    print()
    print("=" * 62)
    if fertig:
        print("Bereit fuer die Pipeline:")
        print(f"  python run.py --method {','.join(fertig)} --adapter none")
    else:
        print("Nichts geschrieben.")
    print("=" * 62)


if __name__ == "__main__":
    main()
