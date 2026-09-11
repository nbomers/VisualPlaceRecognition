"""
Scheitert das System an Osnabrueck oder an zu wenig Referenzmaterial?

Die Datenbank sind 15 % der Sequenzen. 36 % der Anfragen haben darin kein
einziges Bild im Umkreis von 25 m -- die zaehlen im Recall nicht mit, zeigen
aber, wie duenn die Referenz ist. Die 70 % train liegen ungenutzt daneben.

Hier wird train stufenweise zur Datenbank dazugenommen und Recall gegen die
Dichte aufgetragen. Fuer die Baseline-Encoder ist das sauber: sie haben
diese Bilder nie gesehen. Fuer die Adapter-Varianten NICHT -- der Adapter
wurde auf train trainiert. Das Skript nimmt deshalb nur Baselines an.

Die Treppe laeuft ueber Sequenzen, nicht ueber Einzelbilder, so wie der
Split selbst. Ergebnis: eine Tabelle und eine Kurve unter experiments/results/.

    python experiments/database_density.py                     # eigenplaces
    python experiments/database_density.py --method megaloc
    python experiments/database_density.py --fractions 0,0.5,1
"""

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

# Liegt in experiments/, die Pipeline eine Ebene darueber.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.geo import haversine_distance  # noqa: E402

CFG = load_config(ROOT)
OUT_DIR = Path(__file__).resolve().parent / "results"


def _args():
    ap = argparse.ArgumentParser(
        description="Recall gegen Datenbankdichte.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--method", default="eigenplaces",
                    help="Baseline-Encoder (Standard: eigenplaces)")
    ap.add_argument("--fractions", default="0,0.25,0.5,0.75,1.0",
                    help="Anteil der train-Sequenzen, der dazukommt")
    ap.add_argument("--radius", type=float,
                    default=float(CFG["vpr"]["uncertain_radius_m"]))
    ap.add_argument("--top-k", type=int, default=20)
    return ap.parse_args()


def metric_xy(lat, lon, lat0):
    """Lokale Naeherung in Metern -- fuer einen 25-m-Radius voellig ausreichend."""
    return np.c_[np.radians(lon) * 6371000.0 * np.cos(np.radians(lat0)),
                 np.radians(lat) * 6371000.0]


def main():
    args = _args()
    method = args.method
    if "_linear" in method:
        raise SystemExit("Nur Baselines: der Adapter wurde auf train trainiert.")

    emb_dir = ROOT / "data" / "embeddings" / method
    emb_path = emb_dir / f"{method}_embeddings.npy"
    meta_path = emb_dir / f"{method}_metadata.parquet"
    for p in (emb_path, meta_path):
        if not p.exists():
            raise SystemExit(f"Fehlt: {p.relative_to(ROOT)}")

    meta = pd.read_parquet(meta_path)
    emb = np.load(emb_path, mmap_mode="r")
    assert len(emb) == len(meta)

    split = meta["split"].to_numpy()
    q_rows = np.flatnonzero(split == "query")
    db_rows = np.flatnonzero(split == "database")
    tr_rows = np.flatnonzero(split == "train")

    # train-Sequenzen in fester, zufaelliger Reihenfolge -- jede Stufe ist
    # eine Obermenge der vorigen, sonst springt die Kurve.
    rng = np.random.default_rng(int(CFG["vpr"]["split_seed"]))
    tr_seqs = meta["sequence_id"].to_numpy()[tr_rows]
    seq_ids = np.unique(tr_seqs)
    rng.shuffle(seq_ids)

    lat, lon = meta["lat"].to_numpy(), meta["lon"].to_numpy()
    lat0 = float(lat[q_rows].mean())
    q_xy = metric_xy(lat[q_rows], lon[q_rows], lat0)
    q_emb = np.ascontiguousarray(emb[q_rows], dtype=np.float32)

    fractions = [float(f) for f in args.fractions.split(",")]
    ks = [k for k in (1, 5, 10, 20) if k <= args.top_k]
    zeilen = []

    print(f"Verfahren {method}  |  Anfragen {len(q_rows):,}  |  "
          f"database {len(db_rows):,}  |  train {len(tr_rows):,} in {len(seq_ids):,} Sequenzen")
    print()
    print(f"{'train-Anteil':>12} {'Referenz':>10} {'loesbar':>9} {'Anteil':>7} "
          + " ".join(f"{'R@'+str(k):>7}" for k in ks) + f" {'Nachb.':>7}")
    print("-" * (12 + 11 + 10 + 8 + 8 * len(ks) + 8))

    for f in fractions:
        n_seq = int(round(f * len(seq_ids)))
        dazu = tr_rows[np.isin(tr_seqs, seq_ids[:n_seq])] if n_seq else np.array([], dtype=int)
        ref_rows = np.concatenate([db_rows, dazu])

        ref_xy = metric_xy(lat[ref_rows], lon[ref_rows], lat0)
        baum = cKDTree(ref_xy)
        nachbarn = baum.query_ball_point(q_xy, args.radius)
        loesbar = np.array([len(n) > 0 for n in nachbarn])
        n_nachbarn = np.array([len(n) for n in nachbarn])

        index = faiss.IndexFlatIP(emb.shape[1])
        for start in range(0, len(ref_rows), 65536):
            index.add(np.ascontiguousarray(emb[ref_rows[start:start + 65536]], dtype=np.float32))

        idx = np.empty((len(q_rows), args.top_k), dtype=np.int64)
        for start in tqdm(range(0, len(q_rows), 2048),
                          desc=f"train {f:.0%}", leave=False):
            _, idx[start:start + 2048] = index.search(q_emb[start:start + 2048], args.top_k)

        treffer_rows = ref_rows[idx]
        d = haversine_distance(lat[q_rows][:, None], lon[q_rows][:, None],
                               lat[treffer_rows], lon[treffer_rows])
        recall = {}
        for k in ks:
            hit = (d[:, :k] <= args.radius).any(axis=1) & loesbar
            recall[k] = float(hit.sum() / loesbar.sum()) if loesbar.any() else float("nan")

        zeile = {
            "train_fraction": f,
            "n_reference": int(len(ref_rows)),
            "n_loesbar": int(loesbar.sum()),
            "anteil_loesbar": float(loesbar.mean()),
            "median_nachbarn": float(np.median(n_nachbarn[loesbar])) if loesbar.any() else 0.0,
            "recall": {str(k): v for k, v in recall.items()},
        }
        zeilen.append(zeile)
        print(f"{f:>11.0%} {len(ref_rows):>10,} {loesbar.sum():>9,} {loesbar.mean():>6.1%} "
              + " ".join(f"{recall[k]:>7.3f}" for k in ks)
              + f" {zeile['median_nachbarn']:>7.0f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"database_density_{method}.json"
    out.write_text(json.dumps({
        "datum": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "method": method,
        "radius_m": args.radius,
        "n_queries": int(len(q_rows)),
        "stufen": zeilen,
    }, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["figure.dpi"] = 150
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = [z["n_reference"] for z in zeilen]
    for k in ks:
        ax.plot(x, [z["recall"][str(k)] for z in zeilen], marker="o",
                markersize=4, label=f"R@{k}")
    ax2 = ax.twinx()
    ax2.plot(x, [z["anteil_loesbar"] for z in zeilen], "k--", linewidth=1,
             marker="s", markersize=3, label="Anteil loesbar")
    ax2.set_ylabel("Anteil loesbarer Anfragen")
    ax2.set_ylim(0, 1)
    ax.set_xlabel("Referenzbilder (database + train-Anteil)")
    ax.set_ylabel(f"Recall bei {args.radius:g} m")
    ax.set_title(f"{method}: Recall gegen Datenbankdichte")
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"database_density_{method}.png", bbox_inches="tight")
    print(f"\ngeschrieben: {out.relative_to(ROOT)} und .png")


if __name__ == "__main__":
    main()
