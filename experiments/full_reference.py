"""
Volle Referenz: database UND train als Datenbank, alle Baselines.

Der Benchmark-Split haelt nur 15 % der Sequenzen als Referenz; 36 % der
Anfragen haben dort kein Bild im Umkreis von 25 m. Die 70 % train liegen
daneben und werden nur vom Adapter gebraucht -- der verloren hat. Hier
werden sie zur Datenbank dazugenommen: dieselben Anfragen, dieselben vier
Ground Truths, 279.453 statt 48.321 Referenzbilder. Das ist das zweite
Protokoll in compare.py (--reference full) -- naeher an dem, was ein
System mit aller verfuegbaren Referenz leistet.

Nur fuer Encoder ohne Adapter: der Adapter wurde auf train trainiert,
train als Referenz waere fuer ihn Leakage. Sequenz- und andere Varianten
gehoeren zum Benchmark-Protokoll, nicht hierher.

Die Suche laeuft blockweise ueber die Referenz (65.536 Zeilen je FAISS-
Index) und fuehrt die Top-k zusammen -- bei 8448 Dimensionen waere ein
Index ueber alles 9,4 GB.

    python experiments/full_reference.py                      # alle Encoder mit .npy
    python experiments/full_reference.py --methods megaloc,eigenplaces_pcaw512

Ergebnis: results/retrieval/<method>/<name>_fullref_retrieval.npz und
results/evaluation/<name>_fullref.json (Variante "fullref").
"""

import argparse
import time

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from _common import CFG, ROOT
from src.evaluation import standard_evaluations, write_evaluation
from src.run_guard import embedding_fingerprint, require_fingerprint, write_fingerprint

EMB_DIR = ROOT / "data" / "embeddings"
REFERENZ = ("database", "train")
BLOCK_REF = 65_536
BLOCK_Q = 2_048


def _args():
    ap = argparse.ArgumentParser(
        description="Retrieval und Recall mit database + train als Referenz.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--methods", help="Kommaliste; Standard: alle Encoder mit Embeddings")
    ap.add_argument("--force", action="store_true", help="Auch rechnen, wenn das Ergebnis passt")
    return ap.parse_args()


def encoders_with_npy():
    return [m for m in CFG["vpr"]["models"] if (EMB_DIR / m / f"{m}_embeddings.npy").exists()]


def search_blockwise(emb, ref_rows, q_rows, top_k):
    """Top-k ueber alle Referenzzeilen, Index fuer Index -- Ergebnis wie ein Index ueber alles."""
    q = np.ascontiguousarray(emb[q_rows], dtype=np.float32)
    beste_sim = np.full((len(q_rows), top_k), -np.inf, dtype=np.float32)
    beste_idx = np.zeros((len(q_rows), top_k), dtype=np.int64)
    for start in tqdm(range(0, len(ref_rows), BLOCK_REF), desc="Referenzbloecke", leave=False):
        zeilen = ref_rows[start:start + BLOCK_REF]
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(np.ascontiguousarray(emb[zeilen], dtype=np.float32))
        for qs in range(0, len(q), BLOCK_Q):
            sim, idx = index.search(q[qs:qs + BLOCK_Q], min(top_k, len(zeilen)))
            # Bisherige und neue Kandidaten zusammen, die besten top_k behalten.
            sim_alle = np.concatenate([beste_sim[qs:qs + BLOCK_Q], sim], axis=1)
            idx_alle = np.concatenate([beste_idx[qs:qs + BLOCK_Q], start + idx], axis=1)
            wahl = np.argsort(-sim_alle, axis=1, kind="stable")[:, :top_k]
            beste_sim[qs:qs + BLOCK_Q] = np.take_along_axis(sim_alle, wahl, axis=1)
            beste_idx[qs:qs + BLOCK_Q] = np.take_along_axis(idx_alle, wahl, axis=1)
    return beste_idx, beste_sim


def run_one(method, force):
    name = f"{method}_fullref"
    meta = pd.read_parquet(EMB_DIR / method / f"{method}_metadata.parquet")
    npy = EMB_DIR / method / f"{method}_embeddings.npy"
    fingerprint = embedding_fingerprint(CFG, method, "none", meta)
    require_fingerprint(npy, fingerprint, "Embeddings")
    fp_voll = {**fingerprint, "reference_splits": list(REFERENZ)}

    npz = ROOT / "results" / "retrieval" / method / f"{name}_retrieval.npz"
    eval_json = ROOT / "results" / "evaluation" / f"{name}.json"
    npz_passt = False
    if npz.exists() and not force:
        try:
            require_fingerprint(npz, fp_voll, "")
            npz_passt = True
        except Exception:
            pass
    if npz_passt and eval_json.exists():
        print(f"  liegt vor und passt: {name}")
        return

    split = meta["split"].to_numpy()
    q_rows = np.flatnonzero(split == "query")
    ref_rows = np.flatnonzero(np.isin(split, REFERENZ))
    emb = np.load(npy, mmap_mode="r")
    top_k = int(CFG["retrieval"]["top_k"])
    if npz_passt:
        r = np.load(npz)
        idx, sim = r["indices"], r["similarities"]
        print(f"  Trefferliste liegt vor: {npz.name}")
    else:
        t0 = time.time()
        idx, sim = search_blockwise(emb, ref_rows, q_rows, top_k)
        print(f"  Suche: {len(q_rows):,} Anfragen gegen {len(ref_rows):,} Referenzbilder, "
              f"{emb.shape[1]} d, {time.time() - t0:.0f} s")
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, indices=idx, similarities=sim)
        write_fingerprint(npz, fp_voll, retrieval_method="faiss", top_k=top_k)

    query = meta.iloc[q_rows].reset_index(drop=True)
    referenz = meta.iloc[ref_rows].reset_index(drop=True)
    cfg = {**CFG, "vpr": {**CFG["vpr"], "method": method, "adapter": "none"}}
    befunde = standard_evaluations(idx, query, referenz, cfg, verbose=False)
    r25 = befunde["Alle Queries"]["schwellen"]["25"]
    print(f"  R@1 {r25['recall']['1']:.3f}  R@5 {r25['recall']['5']:.3f}  "
          f"loesbar {r25['loesbar']:,} von {len(query):,}")
    write_evaluation(eval_json, cfg, name, emb.shape[1], len(referenz), befunde,
                     variant="fullref", fingerprint=fp_voll, root=ROOT,
                     reference_splits=list(REFERENZ))


def main():
    args = _args()
    methoden = ([m.strip() for m in args.methods.split(",") if m.strip()]
                if args.methods else encoders_with_npy())
    for m in methoden:
        print(f"\n{m}")
        run_one(m, args.force)
    print("\npython compare.py --reference full")


if __name__ == "__main__":
    main()
