"""
Aus der Trefferliste eine Koordinate: fuenf Wege, alle gemessen, alle
unterliegen dem besten Treffer. 08 rechnet deshalb nur noch Top-1; die
Verfahren stehen hier, damit das Ergebnis nachrechenbar bleibt.

  Schwerpunkt (roh)        Mittel der Top-k, gewichtet mit der Aehnlichkeit
  Schwerpunkt (gespreizt)  dasselbe mit Softmax-Temperatur -- Cosinus-Werte
                           liegen dicht beieinander, ohne Spreizung waere es
                           ein ungewichteter Mittelwert
  Clustering               DBSCAN ueber die Top-k, Schwerpunkt der
                           gewichtet staerksten Gruppe
  Snap                     der beste echte Treffer aus der staerksten Gruppe
  Gated                    Top-1, es sei denn eine Gruppe ist sich zu
                           gate_share einig

Warum alles verliert: Fehlgriffe sind zu 44 % grobe Verwechslungen, bei
denen die Nachbarn geschlossen am falschen Ort liegen (confusion_atlas.py).
Konsens bestaetigt dann den Fehler. Parameter aus config.yaml -> localization.

    python experiments/localization_aggregation.py                  # megaloc
    python experiments/localization_aggregation.py --method eigenplaces

Ergebnis: experiments/results/localization_aggregation_<name>.{json,png}
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from _common import CFG, RESULTS, ROOT
from src.geo import to_metric_xy
from src.retrieval import load_retrieval

LOC = CFG["localization"]
EPS_M = float(LOC["eps_m"])
MIN_SAMPLES = int(LOC["min_samples"])
GATE_SHARE = float(LOC["gate_share"])
TEMPERATUR = float(LOC["softmax_temperature"])


def _args():
    ap = argparse.ArgumentParser(
        description="Aggregationsverfahren fuer die Lokalisierung, gegen Top-1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--method", default="megaloc")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--top-k", type=int, default=int(LOC["top_k"]))
    return ap.parse_args()


def gewichte(sim, art="softmax"):
    if art == "roh":
        return np.clip(sim, 0, None)
    return np.exp(TEMPERATUR * (sim - sim.max()))


def _staerkste_gruppe(punkte, sim):
    labels = DBSCAN(eps=EPS_M, min_samples=MIN_SAMPLES).fit_predict(punkte)
    gueltig = labels >= 0
    if not gueltig.any():
        return None
    w = gewichte(sim)
    beste = max(set(labels[gueltig]), key=lambda k: w[labels == k].sum())
    return labels == beste


def aggregiere(punkte, sim, methode, art="softmax"):
    """punkte: (k, 2) in Metern, sim: (k,) -> geschaetzte Position (2,)"""
    if methode == "top1":
        return punkte[0]
    if methode == "schwerpunkt":
        w = gewichte(sim, art)
        return (punkte * w[:, None]).sum(0) / w.sum()
    m = _staerkste_gruppe(punkte, sim)
    if m is None:
        return punkte[0]
    if methode == "snap":
        return punkte[np.flatnonzero(m)[0]]
    if methode == "gated" and m.mean() < GATE_SHARE:
        return punkte[0]
    w = gewichte(sim[m], art)
    return (punkte[m] * w[:, None]).sum(0) / w.sum()


VARIANTEN = [
    ("Top-1", "top1", "softmax"),
    ("Schwerpunkt (roh)", "schwerpunkt", "roh"),
    ("Schwerpunkt (gespreizt)", "schwerpunkt", "softmax"),
    ("Clustering", "cluster", "softmax"),
    ("Snap (bester Treffer der Gruppe)", "snap", "softmax"),
    ("Gated (Gruppe nur bei Einigkeit)", "gated", "softmax"),
]


def kennzahlen(e):
    return {
        "median_m": float(np.median(e)),
        "p90_m": float(np.percentile(e, 90)),
        **{f"unter_{s}m": float((e <= s).mean()) for s in (25, 100)},
    }


def main():
    args = _args()
    name = args.method if args.adapter in ("none", "None") else f"{args.method}_{args.adapter}"
    query, database, indices, sims = load_retrieval(ROOT, CFG, args.method, args.adapter)
    indices, sims = indices[:, :args.top_k], sims[:, :args.top_k]
    db_xy, crs = to_metric_xy(database["lat"].to_numpy(), database["lon"].to_numpy())
    q_xy, _ = to_metric_xy(query["lat"].to_numpy(), query["lon"].to_numpy(), crs=crs)

    fehler = {n: np.empty(len(query)) for n, _, _ in VARIANTEN}
    for qi in tqdm(range(len(query)), desc="Lokalisieren"):
        punkte, sim = db_xy[indices[qi]], sims[qi]
        for n, methode, art in VARIANTEN:
            fehler[n][qi] = np.linalg.norm(aggregiere(punkte, sim, methode, art) - q_xy[qi])

    tabelle = {n: kennzahlen(e) for n, e in fehler.items()}
    print(f"\n{name}  |  {len(query):,} Anfragen, Top-{args.top_k}\n")
    print(f"{'Verfahren':<34}{'Median':>9}{'p90':>10}{'<25 m':>8}{'<100 m':>8}")
    for n, k in tabelle.items():
        print(f"{n:<34}{k['median_m']:>8.0f}m{k['p90_m']:>9.0f}m{k['unter_25m']:>8.3f}{k['unter_100m']:>8.3f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"localization_aggregation_{name}.json"
    out.write_text(json.dumps({
        "datum": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "embedding_name": name, "top_k": args.top_k, "eps_m": EPS_M,
        "min_samples": MIN_SAMPLES, "gate_share": GATE_SHARE,
        "n_queries": int(len(query)), "verfahren": tabelle,
    }, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["figure.dpi"] = 150
    fig, ax = plt.subplots(figsize=(7, 4.5))
    schritte = np.logspace(0, 4.5, 200)
    for n, e in fehler.items():
        ax.plot(schritte, [(e <= s).mean() for s in schritte], label=n, linewidth=1.6)
    ax.axvline(25, color="0.85", linewidth=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Lokalisierungsfehler [m]")
    ax.set_ylabel("Anteil der Anfragen")
    ax.set_title(f"{name}: Aggregation gegen Top-1")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"\ngeschrieben: {out.relative_to(ROOT)} und .png")


if __name__ == "__main__":
    main()
