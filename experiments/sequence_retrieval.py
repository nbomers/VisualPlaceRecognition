"""
Sequenzbasiertes Retrieval: die Trefferlisten benachbarter Frames stuetzen
sich gegenseitig.

Die Anfragen sind keine Einzelbilder, sondern Fahrten -- 0,17 s und 3,3 m
zwischen aufeinanderfolgenden Frames. Fuer jede Anfrage werden die
Trefferlisten ihrer +-W Nachbarn in derselben Sequenz aufsummiert, mit
Dreiecksgewicht (der eigene Frame zaehlt voll, der W-te Nachbar fast nichts).
Ein Datenbankbild, das bei mehreren Nachbarn auftaucht, steigt; eines, das
nur bei einem auftaucht, faellt.

Das ist eine ANDERE Aufgabe als Einzelbild-Retrieval -- Sequenzlokalisierung
-- und wird deshalb als eigene Zeile berichtet, nicht als bessere Version
derselben. Bewertet wird trotzdem exakt wie in 07: gegen die echte Position
des mittleren Frames, mit denselben vier Ground-Truth-Varianten.

Reines Nachbearbeiten der .npz aus 06; kein Modell, keine GPU.

    python experiments/sequence_retrieval.py --method eigenplaces_pcaw512
    python experiments/sequence_retrieval.py --method megaloc --window 5
    python experiments/sequence_retrieval.py --method megaloc --windows 1,2,3,5,10

Fenstergroesse gegen Schwelle: bei 3,3 m je Frame reicht +-3 knapp 10 m
weit, +-5 gut 16 m -- beides unter den 25 m der Ground Truth. +-10 (33 m)
liegt darueber; dort misst man nicht mehr denselben Ort.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Liegt in experiments/, die Pipeline eine Ebene darueber.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.evaluation import standard_evaluations, write_evaluation  # noqa: E402
from src.run_guard import embedding_fingerprint, require_fingerprint  # noqa: E402

CFG = load_config(ROOT)


def _args():
    ap = argparse.ArgumentParser(
        description="Trefferlisten ueber benachbarte Frames einer Sequenz aufsummieren.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--method", default="eigenplaces_pcaw512")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--windows", default="3",
                    help="Kommaliste von Fenstergroessen W (+-W Frames), Standard 3")
    ap.add_argument("--quiet", action="store_true", help="Keine Zwischentabellen")
    return ap.parse_args()


def sequence_windows(query_metadata):
    """Je Query-Zeile: die Zeilen derselben Sequenz in zeitlicher Reihenfolge
    und die eigene Position darin."""
    order = np.lexsort((query_metadata["captured_at"].to_numpy(),
                        query_metadata["sequence_id"].to_numpy()))
    seq = query_metadata["sequence_id"].to_numpy()[order]
    grenzen = np.flatnonzero(np.r_[True, seq[1:] != seq[:-1], True])
    fenster = [None] * len(query_metadata)
    for a, b in zip(grenzen[:-1], grenzen[1:]):
        zeilen = order[a:b]
        for pos, zeile in enumerate(zeilen):
            fenster[zeile] = (zeilen, pos)
    return fenster


def aggregate(indices, similarities, fenster, w, top_k):
    """Neue Trefferliste je Query aus den gewichteten Listen der Nachbarn."""
    n, k_max = indices.shape
    neu_idx = np.empty((n, top_k), dtype=indices.dtype)
    neu_sim = np.empty((n, top_k), dtype=np.float32)
    gewichte = 1.0 - np.abs(np.arange(-w, w + 1)) / (w + 1)   # Dreieck, Mitte = 1

    for qi in tqdm(range(n), desc=f"Fenster +-{w}", leave=False):
        zeilen, pos = fenster[qi]
        lo, hi = max(0, pos - w), min(len(zeilen), pos + w + 1)
        nachbarn = zeilen[lo:hi]
        gw = gewichte[(lo - pos + w):(hi - pos + w)]

        kandidaten = indices[nachbarn].ravel()
        scores = (similarities[nachbarn] * gw[:, None]).ravel()
        uniq, inv = np.unique(kandidaten, return_inverse=True)
        summe = np.zeros(len(uniq), dtype=np.float64)
        np.add.at(summe, inv, scores)

        if len(uniq) > top_k:
            beste = np.argpartition(-summe, top_k - 1)[:top_k]
        else:
            beste = np.arange(len(uniq))
        beste = beste[np.argsort(-summe[beste])]
        m = len(beste)
        neu_idx[qi, :m] = uniq[beste]
        neu_sim[qi, :m] = summe[beste]
        if m < top_k:                      # kann bei sehr kurzen Sequenzen passieren
            neu_idx[qi, m:] = uniq[beste[-1]]
            neu_sim[qi, m:] = -np.inf
    return neu_idx, neu_sim


def main():
    args = _args()
    method, adapter = args.method, args.adapter
    name = method if adapter in ("none", "None") else f"{method}_{adapter}"
    CFG["vpr"]["method"], CFG["vpr"]["adapter"] = method, adapter

    emb_dir = ROOT / "data" / "embeddings" / method
    meta = pd.read_parquet(emb_dir / f"{name}_metadata.parquet")
    npz = ROOT / "results" / "retrieval" / method / f"{name}_retrieval.npz"
    require_fingerprint(npz, embedding_fingerprint(CFG, method, adapter, meta), "Retrieval-Ergebnis")
    r = np.load(npz)
    indices, similarities = r["indices"], r["similarities"]
    top_k = indices.shape[1]

    database = meta[meta.split == "database"].reset_index(drop=True)
    query = meta[meta.split == "query"].reset_index(drop=True)
    dim = int(np.load(emb_dir / f"{name}_embeddings.npy", mmap_mode="r").shape[1])
    fenster = sequence_windows(query)

    n_seq = query["sequence_id"].nunique()
    print(f"{name}: {len(query):,} Anfragen in {n_seq:,} Sequenzen, Top-{top_k} je Anfrage")

    basis = standard_evaluations(indices, query, database, CFG, verbose=False)
    b25 = basis["Alle Queries"]["schwellen"]["25"]["recall"]
    print(f"{'Fenster':>8}   {'R@1':>6} {'R@5':>6} {'R@10':>6}   {'dR@1':>6}")
    print(f"{'einzeln':>8}   {b25['1']:>6.3f} {b25['5']:>6.3f} {b25['10']:>6.3f}")

    for w in [int(x) for x in args.windows.split(",") if x.strip()]:
        neu_idx, neu_sim = aggregate(indices, similarities, fenster, w, top_k)
        befunde = standard_evaluations(neu_idx, query, database, CFG, verbose=not args.quiet)
        r25 = befunde["Alle Queries"]["schwellen"]["25"]["recall"]
        print(f"{'+-' + str(w):>8}   {r25['1']:>6.3f} {r25['5']:>6.3f} {r25['10']:>6.3f}   "
              f"{r25['1'] - b25['1']:>+6.3f}")

        # Als eigene Variante in results/evaluation: compare.py zeigt sie in
        # der Spalte "Variante" neben none und linear.
        CFG["vpr"]["adapter"] = f"{adapter}+seq{w}" if adapter not in ("none", "None") else f"seq{w}"
        pfad = write_evaluation(
            ROOT / "results" / "evaluation" / f"{name}_seq{w}.json",
            CFG, f"{name}_seq{w}", dim, len(database), befunde,
            sequence_window=w, sequence_weighting="dreieck",
        )
        CFG["vpr"]["adapter"] = adapter
        print(f"           -> {pfad.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
