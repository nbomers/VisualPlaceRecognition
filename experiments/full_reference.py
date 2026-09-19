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

Die Suche laeuft blockweise ueber die Referenz (src/retrieval.py ->
blockwise_search, 65.536 Zeilen je FAISS-Index) und fuehrt die Top-k
zusammen. Ein Index ueber alles braucht Referenzbilder x Dimension x 4 Byte
und waechst mit der Stadt: bei 8448 Dimensionen sind das in Osnabrueck
9,4 GB, in Jena 19,8 GB. Geblockt bleibt es bei rund 2 GB je Index.

    python experiments/full_reference.py                      # alle Encoder mit .npy
    python experiments/full_reference.py --methods megaloc,eigenplaces_pcaw512

Ergebnis: results/<stadt>/retrieval/<method>/<name>_fullref_retrieval.npz und
results/<stadt>/evaluation/<name>_fullref.json (Variante "fullref").
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from _common import CFG, PATHS, ROOT
from src.evaluation import standard_evaluations, write_evaluation
from src.retrieval import blockwise_search
from src.run_guard import (code_version, embedding_fingerprint, require_fingerprint,
                           write_fingerprint)

EMB_DIR = PATHS.embeddings
REFERENZ = ("database", "train")


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


def _auswertung_veraltet(eval_json):
    """Wurde die JSON mit einem anderen Auswertungscode gerechnet als dem hier?"""
    try:
        gespeichert = json.loads(eval_json.read_text(encoding="utf-8")).get("code_version") or {}
    except (ValueError, OSError):
        return True
    if not gespeichert.get("evaluation"):
        return True        # alte Datei ohne Kennung -- lieber neu rechnen
    return gespeichert["evaluation"] != code_version(ROOT)["evaluation"]


def run_one(method, force):
    name = f"{method}_fullref"
    meta = pd.read_parquet(EMB_DIR / method / f"{method}_metadata.parquet")
    npy = EMB_DIR / method / f"{method}_embeddings.npy"
    fingerprint = embedding_fingerprint(CFG, method, "none", meta)
    require_fingerprint(npy, fingerprint, "Embeddings")
    fp_voll = {**fingerprint, "reference_splits": list(REFERENZ)}

    npz = PATHS.retrieval_file(name, method)
    eval_json = PATHS.evaluation / f"{name}.json"
    npz_passt = False
    if npz.exists() and not force:
        try:
            require_fingerprint(npz, fp_voll, "")
            npz_passt = True
        except Exception:
            pass
    # Auch die Kennung des Auswertungscodes pruefen, nicht nur die Existenz.
    # run.py macht das in _result_current laengst so; hier fehlte es, und die
    # fullref-Zeilen blieben nach einer Aenderung an src/evaluation.py mit
    # veralteter Kennung liegen -- ohne dass irgendwo etwas rot wurde.
    if npz_passt and eval_json.exists() and not _auswertung_veraltet(eval_json):
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
        idx, sim = blockwise_search(emb, ref_rows, q_rows, top_k)
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
