"""
Konfidenzintervalle fuer Recall@k per Sequenz-Bootstrap.

Die 53.414 Anfragen sind keine unabhaengigen Stichproben: sie stammen aus
198 Fahrten, und aufeinanderfolgende Frames sehen dieselbe Strasse und
scheitern gemeinsam. Der binomiale Standardfehler tut so, als waeren es
34.112 unabhaengige Wuerfe, und unterschaetzt die Unsicherheit.

Hier wird auf Sequenz-Ebene neu gezogen: 1.000-mal die Query-Sequenzen mit
Zuruecklegen, Recall = Treffer / loesbar ueber die gezogenen Sequenzen.
Dieselben Ziehungen fuer alle Encoder, damit die Differenz zweier Encoder
gepaart ausgewertet werden kann -- das ist die Frage, die zaehlt: ist
0.507 gegen 0.484 ein Unterschied oder Rauschen?

Loesbar und Treffer je Anfrage kommen aus src/retrieval.py (einmal je
Anfrage, keine Distanzmatrix im Bootstrap); der Bootstrap selbst ist
Arithmetik auf Sequenz-Summen. Reines Nachbearbeiten der .npz aus 06;
kein Modell, keine GPU.

    python experiments/bootstrap_ci.py                  # alle Zeilen aus results/evaluation/
    python experiments/bootstrap_ci.py --n-bootstrap 200
    python compare.py --ci                              # Intervalle neben R@1

Ergebnis: experiments/results/bootstrap_ci.json
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from _common import CFG, PATHS, RESULTS, ROOT
from src.retrieval import hits_at_k, load_retrieval, localizable

EVAL_DIR = PATHS.evaluation
OUT = RESULTS / "bootstrap_ci.json"
SPLIT = "Alle Queries"

# Die Paare, die in docs/STAND.md und experiments/README.md verglichen werden.
# Differenz = b - a. Jeder Encoder gegen seine _linear-Variante kommt dazu.
PAARE = [
    ("eigenplaces", "eigenplaces_pca512"),
    ("eigenplaces", "eigenplaces_pcaw512"),
    ("eigenplaces", "eigenplaces_pcaw2048"),
    ("anyloc", "anyloc_pcaw512"),
    ("anyloc", "anyloc_pcaw4096"),
    ("megaloc", "megaloc_pca512"),
    ("megaloc", "megaloc_pcaw512"),
    ("megaloc", "eigenplaces_megaloc_concat"),
    ("eigenplaces_pcaw512", "eigenplaces_pcaw512_seq3"),
    # Rangfolge auf voller Breite und auf 512 gewhitent, je Nachbarpaar.
    ("clip", "anyloc"),
    ("anyloc", "mixvpr"),
    ("mixvpr", "eigenplaces"),
    ("eigenplaces", "megaloc"),
    ("clip", "megaloc"),
    ("clip_pcaw512", "anyloc_pcaw512"),
    ("anyloc_pcaw512", "mixvpr_pcaw512"),
    ("mixvpr_pcaw512", "eigenplaces_pcaw512"),
    ("eigenplaces_pcaw512", "megaloc_pcaw512"),
]


def _args():
    ap = argparse.ArgumentParser(
        description="Konfidenzintervalle fuer Recall@k per Sequenz-Bootstrap.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--threshold", type=float,
                    default=float(CFG["vpr"]["uncertain_radius_m"]),
                    help="Distanzschwelle in Metern (Standard: uncertain_radius_m)")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--block", type=int, default=256,
                    help="Anfragen je Distanzblock, wie in src/evaluation.py")
    ap.add_argument("--reference", choices=("database", "full"), default="database",
                    help="database = Benchmark-Protokoll; full = die fullref-Zeilen aus "
                         "experiments/full_reference.py (eigene JSON)")
    return ap.parse_args()


def load_run(record):
    """Trefferliste und Metadaten zu einer Recall-JSON aus 07 -- seq-Varianten
    tragen ihr Fenster in der JSON und werden aus der Basis nachgerechnet,
    fullref-Zeilen ihre Referenzsplits."""
    query, database, indices, _ = load_retrieval(
        ROOT, CFG, record["method"], record["adapter"],
        sequence_window=record.get("sequence_window"),
        reference_splits=record.get("reference_splits"),
    )
    return query, database, indices


def per_sequence(query, sequences, loesbar, hits):
    """Summen je Sequenz in der festen Reihenfolge `sequences`."""
    code = pd.Index(sequences).get_indexer(query["sequence_id"].to_numpy())
    if (code < 0).any():
        raise RuntimeError("Query-Sequenzen passen nicht zum ersten Encoder -- anderer Split?")
    n = len(sequences)
    return {
        "loesbar": np.bincount(code, weights=loesbar, minlength=n),
        "hits": {k: np.bincount(code, weights=h, minlength=n) for k, h in hits.items()},
    }


def draw_counts(n_sequences, n_bootstrap, rng):
    """Wie oft jede Sequenz in jeder Ziehung vorkommt: (n_bootstrap, n_sequences)."""
    gezogen = rng.integers(0, n_sequences, size=(n_bootstrap, n_sequences))
    counts = np.zeros((n_bootstrap, n_sequences), dtype=np.float64)
    np.add.at(counts, (np.arange(n_bootstrap)[:, None], gezogen), 1.0)
    return counts


def bootstrap_recall(counts, summen, k):
    """Recall@k je Ziehung -- Summe Treffer durch Summe loesbar der gezogenen Sequenzen."""
    return (counts @ summen["hits"][k]) / (counts @ summen["loesbar"])


def interval(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [float(lo), float(hi)]


def main():
    args = _args()
    t0 = time.time()
    k_values = [int(k) for k in CFG["retrieval"]["k_values"]]
    schwelle = args.threshold
    schwelle_key = str(int(schwelle)) if float(schwelle).is_integer() else str(schwelle)

    records = [json.loads(p.read_text()) for p in sorted(EVAL_DIR.glob("*.json"))]
    records = [r for r in records if "auswertungen" in r]
    voll = [r for r in records if r.get("variant") == "fullref"]
    records = voll if args.reference == "full" else [r for r in records if r not in voll]
    out = OUT.with_name("bootstrap_ci_fullref.json") if args.reference == "full" else OUT
    records.sort(key=lambda r: (r["method"], r.get("variant", r["adapter"]) != "none"))
    if not records:
        raise SystemExit(f"Keine Auswertungen in {EVAL_DIR.relative_to(ROOT)}.")

    sequences = None
    n_queries = 0
    summen = {}
    eintraege = {}
    for rec in tqdm(records, desc="Encoder"):
        name = rec["embedding_name"]
        query, database, indices = load_run(rec)
        if sequences is None:
            sequences = np.unique(query["sequence_id"].to_numpy())
            n_queries = len(query)
        loesbar = localizable(query, database, schwelle, args.block)
        hits = hits_at_k(query, database, indices, loesbar, schwelle, k_values)
        summen[name] = per_sequence(query, sequences, loesbar, hits)

        # Kontrolle gegen 07: dieselbe Rechnung muss dieselbe Zahl ergeben.
        n_loesbar = int(loesbar.sum())
        eigen = {k: float(hits[k].sum() / n_loesbar) for k in k_values}
        gespeichert = rec["auswertungen"][SPLIT]["schwellen"].get(schwelle_key)
        if gespeichert is not None:
            if gespeichert["loesbar"] != n_loesbar:
                raise RuntimeError(f"{name}: loesbar {n_loesbar} != {gespeichert['loesbar']} in 07")
            for k in k_values:
                soll = gespeichert["recall"][str(k)]
                if abs(soll - eigen[k]) > 1e-9:
                    raise RuntimeError(f"{name}: R@{k} {eigen[k]:.6f} != {soll:.6f} in 07")
        eintraege[name] = {
            "method": rec["method"],
            "adapter": rec["adapter"],
            "variant": rec.get("variant", rec["adapter"]),
            "dim": int(rec["dim"]),
            "n_loesbar": n_loesbar,
            "recall": {str(k): {"wert": eigen[k]} for k in k_values},
        }

    n_seq = len(sequences)
    n_loesbar = int(next(iter(summen.values()))["loesbar"].sum())
    for name, s in summen.items():
        if int(s["loesbar"].sum()) != n_loesbar:
            raise RuntimeError(f"{name}: andere loesbare Menge -- anderer Split?")

    rng = np.random.default_rng(int(CFG["vpr"]["split_seed"]))
    counts = draw_counts(n_seq, args.n_bootstrap, rng)

    # Je Encoder: Intervall und, zum Vergleich, die binomiale Halbbreite.
    samples = {}
    for name, s in summen.items():
        samples[name] = {k: bootstrap_recall(counts, s, k) for k in k_values}
        for k in k_values:
            e = eintraege[name]["recall"][str(k)]
            p = e["wert"]
            e["ci"] = interval(samples[name][k])
            e["halbbreite"] = float((e["ci"][1] - e["ci"][0]) / 2)
            e["halbbreite_naiv"] = float(1.96 * np.sqrt(p * (1 - p) / n_loesbar))

    # Gepaarte Differenzen: dieselben Ziehungen fuer beide Seiten.
    # Bei voller Referenz heissen die Zeilen <name>_fullref; Adapter gibt es dort nicht.
    suffix = "_fullref" if args.reference == "full" else ""
    paare = [(a + suffix, b + suffix) for a, b in PAARE]
    for name in eintraege:
        if f"{name}_linear" in eintraege:
            paare.append((name, f"{name}_linear"))
    differenzen = []
    for a, b in paare:
        if a not in eintraege or b not in eintraege:
            continue
        for k in k_values:
            diff = samples[b][k] - samples[a][k]
            ci = interval(diff)
            differenzen.append({
                "a": a,
                "b": b,
                "k": k,
                "recall_a": eintraege[a]["recall"][str(k)]["wert"],
                "recall_b": eintraege[b]["recall"][str(k)]["wert"],
                "differenz": eintraege[b]["recall"][str(k)]["wert"]
                             - eintraege[a]["recall"][str(k)]["wert"],
                "ci": ci,
                "schliesst_null_ein": bool(ci[0] <= 0.0 <= ci[1]),
            })

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "datum": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "reference": args.reference,
        "split": SPLIT,
        "threshold_m": schwelle,
        "n_bootstrap": args.n_bootstrap,
        "seed": int(CFG["vpr"]["split_seed"]),
        "n_sequences": int(n_seq),
        "n_queries": int(n_queries),
        "n_loesbar": n_loesbar,
        "encoder": eintraege,
        "paare": differenzen,
    }, indent=2))

    # ------------------------------------------------------------------
    # Tabellen
    # ------------------------------------------------------------------
    print(f"\nSequenz-Bootstrap  |  {SPLIT}  |  {schwelle:g} m  |  {n_seq} Sequenzen, "
          f"{n_loesbar:,} loesbare Anfragen, {args.n_bootstrap:,} Ziehungen, "
          f"Seed {CFG['vpr']['split_seed']}\n")
    breite = max(len(n) for n in eintraege) + 2
    kopf = (f"{'Encoder':<{breite}}{'Dim':>6}   {'R@1':>6}  {'95-%-Intervall':>16}"
            f"  {'+-boot':>7} {'+-naiv':>7}   {'R@5':>6}  {'95-%-Intervall':>16}")
    print(kopf)
    print("-" * len(kopf))
    for name, e in eintraege.items():
        r1, r5 = e["recall"]["1"], e["recall"]["5"]
        print(f"{name:<{breite}}{e['dim']:>6}   {r1['wert']:>6.3f}  "
              f"[{r1['ci'][0]:.3f}, {r1['ci'][1]:.3f}]  "
              f"{r1['halbbreite']:>7.3f} {r1['halbbreite_naiv']:>7.3f}   "
              f"{r5['wert']:>6.3f}  [{r5['ci'][0]:.3f}, {r5['ci'][1]:.3f}]")

    print(f"\nGepaarte Differenzen b - a, R@1 bei {schwelle:g} m\n")
    kopf = (f"{'a':<{breite}}{'b':<{breite}}{'R@1 a':>7} {'R@1 b':>7} {'Diff':>7}  "
            f"{'95-%-Intervall':>16}  schliesst 0 ein")
    print(kopf)
    print("-" * len(kopf))
    for d in differenzen:
        if d["k"] != 1:
            continue
        print(f"{d['a']:<{breite}}{d['b']:<{breite}}{d['recall_a']:>7.3f} {d['recall_b']:>7.3f} "
              f"{d['differenz']:>+7.3f}  [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]  "
              f"{'ja' if d['schliesst_null_ein'] else 'nein'}")

    print(f"\ngeschrieben: {out.relative_to(ROOT)}   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
