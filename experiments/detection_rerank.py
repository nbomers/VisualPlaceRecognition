"""
Traegt Mapillarys Detection-Histogramm Information bei, die im Deskriptor
noch nicht steckt?

Die Frage entscheidet, ob semantisches Re-Ranking lohnt. Gemessen wird an
einer Stichprobe, nicht am ganzen Datensatz: fuer jede gezogene Anfrage die
Top-k-Kandidaten aus dem vorhandenen Retrieval, dazu die Detections von
Anfrage und Kandidaten.

Zwei Zahlen kommen heraus:

  AUC   Liegt die Histogramm-Aehnlichkeit beim richtigen Kandidaten hoeher
        als bei den falschen derselben Anfrage? 0.5 = kein Signal.
  R@1   Was Umsortieren tatsaechlich braechte, fuer mehrere Gewichte.

Die Detections landen in cache/detections.jsonl. Ein zweiter Lauf liest von
dort und kommt ohne API-Aufrufe aus.

    python experiments/detection_rerank.py                 # eigenplaces, 2000
    python experiments/detection_rerank.py --method mixvpr
    python experiments/detection_rerank.py --n-queries 5000 --top-k 20
    python experiments/detection_rerank.py --no-fetch      # nur aus dem Cache
"""

import argparse
import collections
import json
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Liegt in experiments/, die Pipeline eine Ebene darueber.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.geo import haversine_distance  # noqa: E402
from src.mapillary import get_session, load_token  # noqa: E402

CFG = load_config(ROOT)
CACHE_PATH = ROOT / "cache" / "detections.jsonl"

# Klassen, die an der Tageszeit haengen und nicht am Ort. Werden in der
# zweiten Variante ausgeblendet, um ihren Anteil am Ergebnis zu zeigen.
TRANSIENT_PREFIXES = ("object--vehicle", "human--", "object--bicycle")



def _args():
    ap = argparse.ArgumentParser(
        description="Misst, ob Detection-Histogramme Re-Ranking tragen.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--method", default="eigenplaces",
                    help="Welches Retrieval-Ergebnis (Standard: eigenplaces)")
    ap.add_argument("--adapter", default="none",
                    help="none oder linear (Standard: none)")
    ap.add_argument("--n-queries", type=int, default=2000,
                    help="Groesse der Stichprobe (Standard: 2000)")
    ap.add_argument("--top-k", type=int, default=10,
                    help="Wieviele Kandidaten je Anfrage (Standard: 10)")
    ap.add_argument("--radius", type=float,
                    default=float(CFG["vpr"]["uncertain_radius_m"]),
                    help="Ab welchem Abstand ein Kandidat falsch ist")
    ap.add_argument("--workers", type=int, default=32,
                    help="Parallele API-Abrufe (Standard: 32)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="Nichts holen, nur mit dem rechnen, was im Cache liegt")
    return ap.parse_args()


# ----------------------------------------------------------------------
# Mapillary
# ----------------------------------------------------------------------

def fetch_counts(image_id, token, workers):
    """Klassenzaehlung eines Bildes. None = Abruf fehlgeschlagen."""
    try:
        r = get_session(workers).get(
            f"https://graph.mapillary.com/{image_id}/detections",
            params={"access_token": token, "fields": "value"},
            timeout=(10, 30),
        )
        r.raise_for_status()
        return dict(collections.Counter(
            x["value"] for x in r.json().get("data", []) if "value" in x
        ))
    except requests.RequestException:
        return None


def load_cache():
    """image_id -> {klasse: anzahl}. Fehlversuche stehen als leeres dict drin."""
    if not CACHE_PATH.exists():
        return {}
    cache = {}
    for line in CACHE_PATH.read_text().splitlines():
        if not line.strip():
            continue
        eintrag = json.loads(line)
        cache[str(eintrag["image_id"])] = eintrag["counts"]
    return cache


def fetch_missing(image_ids, cache, token, workers):
    """Holt, was noch nicht im Cache liegt, und schreibt es fortlaufend mit."""
    fehlend = [i for i in image_ids if i not in cache]
    if not fehlend:
        print(f"Cache vollstaendig ({len(image_ids):,} Bilder), keine Abrufe noetig.")
        return 0

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    schreibsperre = threading.Lock()
    n_fehler = 0

    with CACHE_PATH.open("a") as datei, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        aufgaben = {
            pool.submit(fetch_counts, i, token, workers): i for i in fehlend
        }
        for future in tqdm(aufgaben, desc="Detections holen", total=len(aufgaben)):
            image_id = aufgaben[future]
            counts = future.result()
            if counts is None:
                n_fehler += 1
                counts = {}
            cache[image_id] = counts
            with schreibsperre:
                datei.write(json.dumps(
                    {"image_id": image_id, "counts": counts}) + "\n")
                datei.flush()

    return n_fehler


# ----------------------------------------------------------------------
# Histogramme
# ----------------------------------------------------------------------

def build_vectors(cache, image_ids, drop_transient):
    """
    Klassenzaehlungen -> normalisierte, IDF-gewichtete Vektoren.

    Ohne IDF dominieren Strasse, Vegetation und Himmel, die in praktisch
    jedem Strassenbild vorkommen -- das Histogramm waere stadtweit nahezu
    konstant. Die Gewichtung hebt den informativen Schwanz heraus.
    """
    def gefiltert(counts):
        if not drop_transient:
            return counts
        return {k: v for k, v in counts.items()
                if not k.startswith(TRANSIENT_PREFIXES)}

    zaehlungen = {i: gefiltert(cache.get(i, {})) for i in image_ids}

    dokumentfrequenz = collections.Counter()
    for counts in zaehlungen.values():
        dokumentfrequenz.update(counts.keys())

    klassen = sorted(dokumentfrequenz)
    spalte = {k: j for j, k in enumerate(klassen)}
    n = max(len(zaehlungen), 1)
    idf = np.array([math.log(n / dokumentfrequenz[k]) + 1.0 for k in klassen],
                   dtype=np.float32)

    matrix = np.zeros((len(image_ids), len(klassen)), dtype=np.float32)
    for zeile, image_id in enumerate(image_ids):
        for k, v in zaehlungen[image_id].items():
            matrix[zeile, spalte[k]] = v

    # Wurzel daempft, dass ein naher Zaun 40 Segmente liefert und ein ferner 2.
    matrix = np.sqrt(matrix) * idf
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.where(norm > 0, norm, 1.0)
    return matrix, {i: j for j, i in enumerate(image_ids)}, len(klassen)


# ----------------------------------------------------------------------
# Auswertung
# ----------------------------------------------------------------------

def paarweise_auc(scores, ist_richtig, gueltig=None):
    """
    Anteil der Paare (richtig, falsch) innerhalb einer Anfrage, bei denen der
    richtige Kandidat hoeher liegt. Gleichstand zaehlt halb.

    Bewusst nur innerhalb einer Anfrage: ueber Anfragen hinweg zu vergleichen
    wuerde messen, welche Anfrage leichter ist, nicht ob das Signal traegt.

    gueltig blendet Kandidaten ganz aus, statt sie der Gegenseite zuzuschlagen.
    """
    if gueltig is None:
        gueltig = np.ones_like(ist_richtig, dtype=bool)

    treffer = 0.0
    paare = 0
    for zeile, marke, gilt in zip(scores, ist_richtig, gueltig):
        r, f = zeile[marke & gilt], zeile[(~marke) & gilt]
        if len(r) == 0 or len(f) == 0:
            continue
        vergleich = r[:, None] - f[None, :]
        treffer += float((vergleich > 0).sum() + 0.5 * (vergleich == 0).sum())
        paare += vergleich.size
    return (treffer / paare if paare else float("nan")), paare


def z_norm(zeile):
    """Innerhalb einer Anfrage standardisieren -- Deskriptor- und
    Detection-Aehnlichkeit liegen sonst auf voellig anderen Skalen."""
    s = zeile.std()
    return (zeile - zeile.mean()) / s if s > 0 else zeile * 0.0


def main():
    args = _args()
    name = args.method if args.adapter in ("none", "None") \
        else f"{args.method}_{args.adapter}"

    metadata_path = (ROOT / "data" / "embeddings" / args.method
                     / f"{name}_metadata.parquet")
    retrieval_path = (ROOT / "results" / "retrieval" / args.method
                      / f"{name}_retrieval.npz")
    for p in (metadata_path, retrieval_path):
        if not p.exists():
            raise SystemExit(f"Fehlt: {p.relative_to(ROOT)}")

    metadata = pd.read_parquet(metadata_path)
    database = metadata[metadata["split"] == "database"].reset_index(drop=True)
    queries = metadata[metadata["split"] == "query"].reset_index(drop=True)

    retrieval = np.load(retrieval_path)
    indices = retrieval["indices"][:, :args.top_k]
    similarities = retrieval["similarities"][:, :args.top_k]
    if retrieval["indices"].shape[1] < args.top_k:
        raise SystemExit(f"Nur {retrieval['indices'].shape[1]} Treffer gespeichert.")

    # ------------------------------------------------------------------
    # Stichprobe: nur Anfragen, bei denen im Top-k richtige UND falsche
    # Kandidaten stehen -- sonst gibt es kein Paar zu vergleichen.
    # ------------------------------------------------------------------
    q_lat, q_lon = queries["lat"].to_numpy(), queries["lon"].to_numpy()
    db_lat, db_lon = database["lat"].to_numpy(), database["lon"].to_numpy()

    abstand = haversine_distance(q_lat[:, None], q_lon[:, None],
                        db_lat[indices], db_lon[indices])
    richtig = abstand <= args.radius
    brauchbar = np.flatnonzero(richtig.any(axis=1) & (~richtig).any(axis=1))

    rng = np.random.default_rng(int(CFG["vpr"]["split_seed"]))
    if len(brauchbar) > args.n_queries:
        gezogen = rng.choice(brauchbar, args.n_queries, replace=False)
    else:
        gezogen = brauchbar
    gezogen = np.sort(gezogen)

    print(f"Verfahren:            {name}")
    print(f"Anfragen gesamt:      {len(queries):,}")
    print(f"davon brauchbar:      {len(brauchbar):,}  "
          f"(richtige und falsche Kandidaten im Top-{args.top_k})")
    print(f"Stichprobe:           {len(gezogen):,}")

    q_ids = [str(x) for x in queries["image_id"].to_numpy()[gezogen]]
    kandidat_ids = [[str(x) for x in database["image_id"].to_numpy()[reihe]]
                    for reihe in indices[gezogen]]
    alle_ids = list(dict.fromkeys(q_ids + [i for reihe in kandidat_ids for i in reihe]))
    print(f"Verschiedene Bilder:  {len(alle_ids):,}")

    # ------------------------------------------------------------------
    # Detections
    # ------------------------------------------------------------------
    cache = load_cache()
    print(f"Im Cache:             {len(cache):,}")
    if not args.no_fetch:
        n_fehler = fetch_missing(alle_ids, cache, load_token(ROOT), args.workers)
        if n_fehler:
            print(f"Fehlgeschlagene Abrufe: {n_fehler:,}")

    fehlend = [i for i in alle_ids if i not in cache]
    if fehlend:
        raise SystemExit(
            f"{len(fehlend):,} Bilder fehlen im Cache. Ohne --no-fetch starten."
        )
    hat_det = {i: bool(cache[i]) for i in alle_ids}
    q_abdeckung = np.mean([hat_det[i] for i in q_ids])
    k_abdeckung = np.mean([hat_det[i] for reihe in kandidat_ids for i in reihe])
    print(f"Abdeckung Anfragen:   {q_abdeckung:.1%}")
    print(f"Abdeckung Kandidaten: {k_abdeckung:.1%}")
    print("Ohne Detections gibt es kein Signal -- das ist die Obergrenze "
          "dessen,\nwas Umsortieren ueberhaupt erreichen kann.")

    # ------------------------------------------------------------------
    # Zwei Varianten: mit und ohne die zeitabhaengigen Klassen
    # ------------------------------------------------------------------
    for drop_transient in (False, True):
        etikett = "ohne Autos/Personen" if drop_transient else "alle Klassen"
        matrix, zeile_von, n_klassen = build_vectors(cache, alle_ids, drop_transient)

        det_scores, desc_scores, marken, messbar, voll = [], [], [], [], []
        for pos, qi in enumerate(gezogen):
            q_vec = matrix[zeile_von[q_ids[pos]]]
            k_mat = matrix[[zeile_von[i] for i in kandidat_ids[pos]]]
            roh = k_mat @ q_vec
            k_hat = np.array([hat_det[i] for i in kandidat_ids[pos]])

            # Fehlende Detections neutral setzen statt auf 0: eine 0 waere
            # der schlechtestmoegliche Wert und wuerde fehlende Daten
            # bestrafen, nicht falsche Orte.
            if hat_det[q_ids[pos]] and k_hat.any():
                roh = np.where(k_hat, roh, roh[k_hat].mean())
            else:
                roh = np.zeros_like(roh)

            det_scores.append(roh)
            desc_scores.append(similarities[qi].astype(np.float64))
            marken.append(richtig[qi])
            # Nur wo Anfrage und Kandidat Detections haben, ist das Paar
            # ueberhaupt eine Aussage ueber das Signal.
            messbar.append(hat_det[q_ids[pos]] & k_hat)
            voll.append(hat_det[q_ids[pos]] and k_hat.all())

        det_scores = np.array(det_scores)
        desc_scores = np.array(desc_scores)
        marken = np.array(marken)
        messbar = np.array(messbar)
        voll = np.array(voll)

        auc_det, n_paare = paarweise_auc(det_scores, marken, messbar)
        auc_desc, _ = paarweise_auc(desc_scores, marken)

        print()
        print("=" * 62)
        print(f"{etikett}   ({n_klassen} Klassen, {n_paare:,} auswertbare Paare)")
        print("=" * 62)
        print(f"  AUC Detection-Histogramm   {auc_det:.4f}   "
              f"(nur Paare mit Detections)")
        print(f"  AUC Deskriptor (Vergleich) {auc_desc:.4f}")
        print("  0.500 = kein Signal. Der Deskriptor ist die Messlatte.")

        # Umsortieren: beide Signale je Anfrage standardisieren, dann mischen.
        z_det = np.array([z_norm(r) for r in det_scores])
        z_desc = np.array([z_norm(r) for r in desc_scores])
        print()
        print(f"  Umsortieren des Top-k    "
              f"(alle {len(marken):,} | volle Abdeckung {int(voll.sum()):,})")
        print(f"    {'Gewicht':>8}   {'R@1 alle':>9}   {'R@1 voll':>9}")
        for lam in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
            gemischt = z_desc + lam * z_det
            beste = gemischt.argmax(axis=1)
            getroffen = marken[np.arange(len(beste)), beste]
            r1 = float(getroffen.mean())
            r1_voll = float(getroffen[voll].mean()) if voll.any() else float("nan")
            markierung = "   <- Ausgangslage" if lam == 0.0 else ""
            print(f"    {lam:>8.2f}   {r1:>9.4f}   {r1_voll:>9.4f}{markierung}")

    print()
    print("Lesart: liegt die AUC bei 0.50 und faellt R@1 mit steigendem")
    print("Gewicht, traegt das Signal nichts. Die Idee ist dann sauber")
    print("widerlegt und der Befund gehoert in die Fehleranalyse.")


if __name__ == "__main__":
    main()
