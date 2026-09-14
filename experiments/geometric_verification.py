"""
Geometrische Verifikation: die Top-k nach lokalen Merkmalen umsortieren.

Der Deskriptor vergleicht Bilder GLOBAL -- zwei Vorstadtstrassen sechs
Kilometer auseinander sehen sich dann aehnlich. 88 % der Fehlgriffe sind
solche groben Verwechslungen. Lokale Merkmale sehen das anders: Fenster,
Schilder und Dachkanten zweier verschiedener Orte passen geometrisch nicht
zusammen. Fuer jede Anfrage werden die Top-k-Kandidaten mit SuperPoint +
LightGlue gematcht, die Matches per RANSAC gegen eine Fundamentalmatrix
geprueft, und die Kandidaten nach Inlier-Zahl neu sortiert. Wer unterhalb
von --min-inliers bleibt, behaelt seine alte Reihenfolge hinter den
verifizierten.

Bewertet wird exakt wie in 07. Das ist der teuerste Hebel: je Anfrage k
Bildpaare durch ein Matching-Netz. Auf der GPU rund 50 Paare je Sekunde,
auf CPU ein Bruchteil davon -- deshalb standardmaessig eine Stichprobe.
Ein Lauf ueber alle Anfragen gehoert auf den Rechner mit der GPU.

    python experiments/geometric_verification.py --method eigenplaces_megaloc_concat
    python experiments/geometric_verification.py --method megaloc --n-queries 0   # alle
    python experiments/geometric_verification.py --method megaloc --top-k 10 --max-side 512

Ergebnis: bei einer Stichprobe nach experiments/results/, bei allen Anfragen
als eigene Zeile nach results/evaluation/ (Variante "gv<k>").
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from _common import CFG, PATHS, RESULTS, ROOT
from src.device import pick_device
from src.evaluation import standard_evaluations, write_evaluation
from src.retrieval import load_retrieval
from src.run_guard import embedding_fingerprint

OUT_DIR = RESULTS


def _args():
    ap = argparse.ArgumentParser(
        description="Top-k mit SuperPoint + LightGlue geometrisch verifizieren.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--method", default="eigenplaces_megaloc_concat")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--top-k", type=int, default=20,
                    help="Wieviele Kandidaten je Anfrage verifiziert werden")
    ap.add_argument("--n-queries", type=int, default=2000,
                    help="Stichprobe; 0 = alle Anfragen (GPU!)")
    ap.add_argument("--max-side", type=int, default=640,
                    help="Bilder auf diese laengste Kante verkleinern")
    ap.add_argument("--max-keypoints", type=int, default=1024)
    ap.add_argument("--min-inliers", type=int, default=15,
                    help="Darunter gilt ein Paar als nicht verifiziert")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (Standard: automatisch)")
    return ap.parse_args()


def load_gray(path, max_side):
    """Graustufen-Tensor [1,1,H,W] in [0,1], laengste Kante auf max_side.
    None, wenn das Bild fehlt oder nicht lesbar ist."""
    from PIL import Image
    try:
        im = Image.open(path)
    except OSError:
        return None
    with im:
        im.draft("L", (max_side, max_side))
        im = im.convert("L")
        w, h = im.size
        s = max_side / max(w, h)
        if s < 1:
            im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, None]


def inlier_count(kp0, kp1, min_matches=8):
    """RANSAC gegen eine Fundamentalmatrix; Rueckgabe: Anzahl Inlier."""
    import cv2
    if len(kp0) < min_matches:
        return 0
    _, mask = cv2.findFundamentalMat(kp0, kp1, cv2.FM_RANSAC, 3.0, 0.999)
    return int(mask.sum()) if mask is not None else 0


def main():
    args = _args()
    method, adapter = args.method, args.adapter
    name = method if adapter in ("none", "None") else f"{method}_{adapter}"
    CFG["vpr"]["method"], CFG["vpr"]["adapter"] = method, adapter
    image_path = PATHS.images

    emb_dir = PATHS.embedding_dir(method)
    query, database, indices, _ = load_retrieval(ROOT, CFG, method, adapter)
    k = min(args.top_k, indices.shape[1])
    dim = int(np.load(emb_dir / f"{name}_embeddings.npy", mmap_mode="r").shape[1])
    fingerprint = embedding_fingerprint(
        CFG, method, adapter, pd.read_parquet(emb_dir / f"{name}_metadata.parquet"))

    db_ids = database["image_id"].to_numpy()
    q_ids = query["image_id"].to_numpy()

    # Stichprobe oder alle
    rng = np.random.default_rng(int(CFG["vpr"]["split_seed"]))
    if args.n_queries and args.n_queries < len(query):
        auswahl = np.sort(rng.choice(len(query), args.n_queries, replace=False))
    else:
        auswahl = np.arange(len(query))
    voll = len(auswahl) == len(query)

    try:
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import rbd
    except ImportError as e:
        raise SystemExit(
            "LightGlue fehlt:  pip install git+https://github.com/cvg/LightGlue.git"
        ) from e

    device = torch.device(pick_device(args.device))
    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    print(f"{name}: {len(auswahl):,} von {len(query):,} Anfragen, Top-{k}, "
          f"{len(auswahl) * k:,} Bildpaare auf {device}")

    neu_idx = indices.copy()
    inlier_stat = np.zeros((len(auswahl), k), dtype=np.int32)
    fehlend = 0
    t0 = time.time()

    with torch.inference_mode():
        for pos, qi in enumerate(tqdm(auswahl, desc="verifizieren")):
            bild = load_gray(image_path / f"{q_ids[qi]}.jpg", args.max_side)
            if bild is None:
                fehlend += 1
                continue
            f0 = extractor.extract(bild.to(device))

            inliers = np.zeros(k, dtype=np.int32)
            for j in range(k):
                bild = load_gray(image_path / f"{db_ids[indices[qi, j]]}.jpg", args.max_side)
                if bild is None:
                    continue
                f1 = extractor.extract(bild.to(device))
                m = rbd(matcher({"image0": f0, "image1": f1}))
                paare = m["matches"]
                if len(paare) == 0:
                    continue
                kp0 = rbd(f0)["keypoints"][paare[:, 0]].cpu().numpy()
                kp1 = rbd(f1)["keypoints"][paare[:, 1]].cpu().numpy()
                inliers[j] = inlier_count(kp0, kp1)

            inlier_stat[pos] = inliers
            # Verifizierte nach Inliern absteigend, der Rest in alter Reihenfolge
            # dahinter -- stabil, damit Gleichstaende die Deskriptor-Ordnung behalten.
            ok = inliers >= args.min_inliers
            reihenfolge = np.r_[
                np.argsort(-inliers[ok], kind="stable") if ok.any() else np.array([], int),
            ]
            reihenfolge = np.r_[np.flatnonzero(ok)[reihenfolge], np.flatnonzero(~ok)]
            neu_idx[qi, :k] = indices[qi, reihenfolge]

    dauer = time.time() - t0
    print(f"{dauer / 60:.1f} min, {len(auswahl) * k / max(dauer, 1):.1f} Paare/s"
          + (f", {fehlend} Anfragebilder fehlten" if fehlend else ""))

    # Vorher/nachher auf derselben Auswahl
    filt = np.zeros(len(query), dtype=bool)
    filt[auswahl] = True
    from src.evaluation import evaluate_retrieval
    vorher = evaluate_retrieval(indices, query, database, CFG, label="vorher",
                                query_filter=filt, verbose=False)
    nachher = evaluate_retrieval(neu_idx, query, database, CFG, label="nachher",
                                 query_filter=filt, verbose=False)
    print()
    print(f"{'':<10}" + "".join(f"{'R@' + str(kk):>8}" for kk in CFG["retrieval"]["k_values"]))
    for label, b in (("vorher", vorher), ("nachher", nachher)):
        rec = b["schwellen"]["25"]["recall"]
        print(f"{label:<10}" + "".join(f"{rec[str(kk)]:>8.3f}" for kk in CFG["retrieval"]["k_values"]))
    d1 = nachher["schwellen"]["25"]["recall"]["1"] - vorher["schwellen"]["25"]["recall"]["1"]
    print(f"{'dR@1':<10}{d1:>+8.3f}   (bei 25 m, {vorher['schwellen']['25']['loesbar']:,} loesbare)")
    print(f"verifizierte Paare (>= {args.min_inliers} Inlier): "
          f"{(inlier_stat >= args.min_inliers).mean():.1%}, Median Inlier {np.median(inlier_stat):.0f}")

    if voll:
        befunde = standard_evaluations(neu_idx, query, database, CFG, verbose=False)
        variante = f"{adapter}+gv{k}" if adapter not in ("none", "None") else f"gv{k}"
        pfad = write_evaluation(PATHS.evaluation / f"{name}_gv{k}.json",
                                CFG, f"{name}_gv{k}", dim, len(database), befunde,
                                variant=variante, fingerprint=fingerprint, root=ROOT,
                                verification_top_k=k, min_inliers=args.min_inliers)
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pfad = OUT_DIR / f"geometric_verification_{name}.json"
        pfad.write_text(json.dumps({
            "datum": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "embedding_name": name, "top_k": k, "min_inliers": args.min_inliers,
            "n_queries": int(len(auswahl)), "stichprobe": True,
            "vorher": vorher, "nachher": nachher,
            "anteil_verifiziert": float((inlier_stat >= args.min_inliers).mean()),
        }, indent=2))
    print(f"-> {pfad.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
