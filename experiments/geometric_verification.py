"""
Geometrische Verifikation: die Top-k nach lokalen Merkmalen umsortieren.

Der Deskriptor vergleicht Bilder GLOBAL -- zwei Vorstadtstrassen sechs
Kilometer auseinander sehen sich dann aehnlich. 44 % der Fehlgriffe von
MegaLoc liegen ueber einen Kilometer daneben (confusion_atlas.py). Lokale
Merkmale sehen das anders: Fenster,
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

    python experiments/geometric_verification.py                   # Stichprobe, 2.000 Anfragen
    python experiments/geometric_verification.py --n-queries 0     # alle, eigene Zeile
    python experiments/geometric_verification.py --top-k 10 --max-side 512

Verifiziert werden nur Anfragen, die bei der groessten Schwelle aus
retrieval.thresholds (100 m) ueberhaupt ein Datenbankbild haben. Die
uebrigen zaehlen in keinem Recall mit; sie umzusortieren kostete je nach
Stadt 8 bis 33 % der Laufzeit und aenderte keine einzige Zahl.

Gespeichert werden die Inlier je Anfrage und Kandidat, nicht die neue
Reihenfolge (src.verification.verification_file, neben der Trefferliste aus 06).
Daraus rechnet bootstrap_ci.py die gv-Zeile nach, ohne neu zu matchen, und
ein abgebrochener Lauf setzt beim naechsten Start dort fort, wo er stand --
bei gleichen Parametern. Ein Lauf ueber alle Anfragen einer grossen Stadt
dauert einen halben Tag.

Ergebnis: bei einer Stichprobe nach experiments/results/, bei allen Anfragen
als eigene Zeile nach results/<stadt>/evaluation/ (Variante "gv<k>").
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from _common import CFG, PATHS, RESULTS, ROOT
from src.device import pick_device
from src.evaluation import standard_evaluations, write_evaluation
from src.retrieval import descriptor_dim, load_retrieval, localizable
from src.run_guard import embedding_fingerprint, require_fingerprint, write_fingerprint
from src.verification import rerank_by_inliers, verification_file, verification_fingerprint

OUT_DIR = RESULTS
# So oft wird die Inlier-Matrix zwischengespeichert. Ein Absturz kostet
# hoechstens diese Zeit, das Schreiben (wenige MB) faellt nicht ins Gewicht.
SICHERN_ALLE_S = 120


def _args():
    ap = argparse.ArgumentParser(
        description="Top-k mit SuperPoint + LightGlue geometrisch verifizieren.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--method", default="megaloc",
                    help="Standard: megaloc -- der Encoder, den es in allen Staedten gibt")
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
    ap.add_argument("--ransac-px", type=float, default=3.0,
                    help="Zulaessiger Abstand zur Epipolarlinie in Pixeln")
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


def inlier_count(kp0, kp1, ransac_px=3.0, min_matches=8):
    """
    RANSAC gegen eine Fundamentalmatrix; Rueckgabe: Anzahl Inlier.

    ransac_px ist der Abstand in Pixeln, den ein Match von der Epipolarlinie
    haben darf. Er steht hier als Argument und nicht als Konstante im Aufruf,
    weil er das Ergebnis verschiebt: enger heisst weniger, aber zuverlaessigere
    Inlier. Der Wert landet ueber --ransac-px im Ergebnis-JSON, damit zwei
    Laeufe vergleichbar bleiben.
    """
    import cv2
    if len(kp0) < min_matches:
        return 0
    _, mask = cv2.findFundamentalMat(kp0, kp1, cv2.FM_RANSAC, ransac_px, 0.999)
    return int(mask.sum()) if mask is not None else 0


def _kandidaten(query, database, auswahl_n, rng):
    """
    Die Anfragen, die verifiziert werden: alle, die bei der groessten Schwelle
    ein Datenbankbild haben -- oder eine Stichprobe daraus.
    """
    groesste = max(float(t) for t in CFG["retrieval"]["thresholds"])
    zaehlt = np.flatnonzero(localizable(query, database, groesste))
    if auswahl_n and auswahl_n < len(zaehlt):
        return np.sort(rng.choice(zaehlt, auswahl_n, replace=False)), groesste, False
    return zaehlt, groesste, True


def _fortsetzen(datei, gv_fp, auswahl, k):
    """
    Inlier, Zahl der fertigen Anfragen und bisherige Rechenzeit aus einem
    frueheren Lauf -- oder ein leerer Anfang, wenn es keinen gibt oder er
    nicht passt.
    """
    leer = np.zeros((len(auswahl), k), dtype=np.int32), 0, 0.0
    if not datei.exists():
        return leer
    try:
        require_fingerprint(datei, gv_fp, "Inlier eines frueheren Laufs")
        with np.load(datei) as alt:
            if not np.array_equal(alt["rows"], auswahl):
                raise RuntimeError("andere Anfragen")
            inliers, fertig = alt["inliers"].astype(np.int32), int(alt["fertig"])
            sekunden = float(alt["sekunden"])
    except (RuntimeError, FileNotFoundError, KeyError, ValueError) as e:
        erste = str(e).splitlines()[0]
        print(f"Frueherer Lauf passt nicht ({erste}) -- beginne von vorn.")
        return leer
    if fertig:
        print(f"Setze fort: {fertig:,} von {len(auswahl):,} Anfragen sind schon verifiziert.")
    return inliers, fertig, sekunden


def _sichern(datei, gv_fp, inliers, auswahl, fertig, sekunden):
    """
    Atomar ersetzen -- ein Absturz beim Schreiben laesst den alten Stand
    stehen. Der Fingerabdruck kommt NACH den Daten: stirbt der Prozess
    dazwischen, passt er nicht und der naechste Lauf beginnt von vorn. In der
    anderen Reihenfolge stuenden alte Inlier unter neuem Fingerabdruck.
    """
    tmp = datei.with_name(datei.stem + ".tmp.npz")
    np.savez(tmp, inliers=inliers, rows=auswahl, fertig=np.int64(fertig),
             sekunden=np.float64(sekunden))
    os.replace(tmp, datei)
    write_fingerprint(datei, gv_fp)


def main():
    args = _args()
    method, adapter = args.method, args.adapter
    name = method if adapter in ("none", "None") else f"{method}_{adapter}"
    CFG["vpr"]["method"], CFG["vpr"]["adapter"] = method, adapter
    image_path = PATHS.images

    emb_dir = PATHS.embedding_dir(method)
    query, database, indices, _ = load_retrieval(ROOT, CFG, method, adapter)
    k = min(args.top_k, indices.shape[1])
    dim = descriptor_dim(ROOT, CFG, method, adapter)
    fingerprint = embedding_fingerprint(
        CFG, method, adapter, pd.read_parquet(emb_dir / f"{name}_metadata.parquet"))

    db_ids = database["image_id"].to_numpy()
    q_ids = query["image_id"].to_numpy()

    rng = np.random.default_rng(int(CFG["vpr"]["split_seed"]))
    auswahl, groesste, voll = _kandidaten(query, database, args.n_queries, rng)

    params = {"top_k": k, "max_side": args.max_side,
              "max_keypoints": args.max_keypoints, "ransac_px": args.ransac_px}
    # Die Stichprobe bekommt eine eigene Datei -- sonst ueberschriebe sie den
    # Stand eines laufenden Gesamtlaufs.
    datei = verification_file(ROOT, CFG, method, adapter, k)
    if not voll:
        datei = datei.with_name(datei.stem + f"_stichprobe{len(auswahl)}.npz")
    datei.parent.mkdir(parents=True, exist_ok=True)
    gv_fp = verification_fingerprint(fingerprint, params)
    inlier_stat, fertig, vorher_s = _fortsetzen(datei, gv_fp, auswahl, k)

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
    print(f"{name}: {len(auswahl):,} von {len(query):,} Anfragen "
          f"(loesbar bei {groesste:g} m{'' if voll else ', Stichprobe'}), Top-{k}, "
          f"{len(auswahl) * k:,} Bildpaare auf {device}")

    fehlend = 0
    t0 = time.time()
    gesichert = t0

    with torch.inference_mode():
        for pos in tqdm(range(fertig, len(auswahl)), desc="verifizieren",
                        initial=fertig, total=len(auswahl)):
            qi = auswahl[pos]
            bild = load_gray(image_path / f"{q_ids[qi]}.jpg", args.max_side)
            if bild is None:
                # Bleibt bei null Inliern -- die Anfrage behaelt ihre Reihenfolge.
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
                inliers[j] = inlier_count(kp0, kp1, args.ransac_px)
            inlier_stat[pos] = inliers

            if time.time() - gesichert > SICHERN_ALLE_S:
                _sichern(datei, gv_fp, inlier_stat, auswahl, pos + 1,
                         vorher_s + time.time() - t0)
                gesichert = time.time()

    dauer = time.time() - t0
    _sichern(datei, gv_fp, inlier_stat, auswahl, len(auswahl), vorher_s + dauer)
    neu = len(auswahl) - fertig
    print(f"{dauer / 60:.1f} min fuer {neu:,} Anfragen, {neu * k / max(dauer, 1):.1f} Paare/s"
          + (f", {fehlend} Anfragebilder fehlten" if fehlend else ""))

    neu_idx = rerank_by_inliers(indices, inlier_stat, auswahl, args.min_inliers)

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
    anteil = float((inlier_stat >= args.min_inliers).mean())
    median = float(np.median(inlier_stat))
    print(f"{'dR@1':<10}{d1:>+8.3f}   (bei 25 m, {vorher['schwellen']['25']['loesbar']:,} loesbare)")
    print(f"verifizierte Paare (>= {args.min_inliers} Inlier): {anteil:.1%}, Median Inlier {median:.0f}")

    # Alles, was die Zahlen bestimmt oder einordnet, steht in der JSON --
    # city_comparison.py liest die Tabelle daraus, nicht aus der Konsole.
    protokoll = {"verification_top_k": k, "min_inliers": args.min_inliers,
                 "ransac_px": args.ransac_px, "max_keypoints": args.max_keypoints,
                 "max_side": args.max_side, "verifiziert_bis_m": groesste,
                 "n_verifiziert": int(len(auswahl)), "anteil_verifiziert": anteil,
                 "median_inlier": median, "dauer_s": round(vorher_s + dauer),
                 "fehlende_anfragebilder": fehlend}
    if voll:
        befunde = standard_evaluations(neu_idx, query, database, CFG, verbose=False)
        variante = f"{adapter}+gv{k}" if adapter not in ("none", "None") else f"gv{k}"
        pfad = write_evaluation(PATHS.evaluation / f"{name}_gv{k}.json",
                                CFG, f"{name}_gv{k}", dim, len(database), befunde,
                                variant=variante, fingerprint=fingerprint, root=ROOT,
                                **protokoll)
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pfad = OUT_DIR / f"geometric_verification_{name}.json"
        pfad.write_text(json.dumps({
            "datum": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "embedding_name": name, **protokoll,
            "n_queries": int(len(auswahl)), "stichprobe": True,
            "vorher": vorher, "nachher": nachher,
        }, indent=2), encoding="utf-8")
    print(f"-> {pfad.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
