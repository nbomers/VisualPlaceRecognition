"""
Schwierigkeitsprofil: woran haengt der Recall je Anfrage?

Vier Eigenschaften, die nur an den Metadaten haengen, nicht am Encoder --
und deshalb sagen, welche Anfragen ein System ueberhaupt loesen kann:

  Nachbarn        Datenbankbilder im Umkreis der Schwelle (KDTree, UTM)
  Zeitabstand     Tage bis zum naechsten davon
  Blickrichtung   gibt es einen Nachbarn, der in dieselbe Richtung schaut?
  Fotograf        gibt es einen Nachbarn vom selben Fotografen am selben Tag?

R@1 je Klasse fuer die loesbaren Anfragen, als Balken mit n. Das ist der
direkte Test fuer den Dichte-Befund: faellt R@1 bei wenigen Nachbarn ein,
haengt der Recall an der Referenz vor Ort -- nicht am Stadtteil.

    python experiments/recall_by_difficulty.py                    # megaloc
    python experiments/recall_by_difficulty.py --method eigenplaces_pcaw512

Ergebnis: experiments/results/recall_by_difficulty_<name>.{json,png}
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from _common import CFG, RESULTS, ROOT
from src.geo import heading_matches, to_metric_xy
from src.retrieval import hits_at_k, load_retrieval, localizable

NACHBAR_KLASSEN = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 10**9)]
TAGE_KLASSEN = [(0, 7), (8, 30), (31, 180), (181, 365), (366, 10**9)]


def _args():
    ap = argparse.ArgumentParser(
        description="R@1 gegen Nachbarzahl, Zeitabstand, Blickrichtung und Fotograf je Anfrage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--method", default="megaloc")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--threshold", type=float, default=float(CFG["vpr"]["uncertain_radius_m"]))
    return ap.parse_args()


def query_properties(query, database, threshold, max_heading_diff):
    """Je Anfrage: Nachbarn, Tage zum naechsten, Blickrichtung passt, Fotograf/Tag gleich."""
    db_xy, crs = to_metric_xy(database["lat"].to_numpy(), database["lon"].to_numpy())
    q_xy, _ = to_metric_xy(query["lat"].to_numpy(), query["lon"].to_numpy(), crs=crs)
    # UTM-Radius mit Aufschlag, exakt zaehlt dann nichts -- fuer Klassen reicht das.
    listen = cKDTree(db_xy).query_ball_point(q_xy, r=threshold)
    laengen = np.array([len(n) for n in listen])
    qi = np.repeat(np.arange(len(query)), laengen)
    di = np.concatenate([np.asarray(n, dtype=int) for n in listen if len(n)]) if laengen.sum() else np.array([], int)

    q_t, db_t = query["captured_at"].to_numpy(), database["captured_at"].to_numpy()
    tage = np.abs(q_t[qi] - db_t[di]) / 86_400_000.0
    tage_min = np.full(len(query), np.inf)
    np.minimum.at(tage_min, qi, tage)

    blick = heading_matches(query["compass_angle"].to_numpy()[qi],
                            database["compass_angle"].to_numpy()[di], max_heading_diff)
    blick_ok = np.zeros(len(query), dtype=bool)
    np.logical_or.at(blick_ok, qi, blick)

    gleich = ((query["creator_id"].to_numpy()[qi] == database["creator_id"].to_numpy()[di])
              & (tage < 1))
    gleicher_tag = np.zeros(len(query), dtype=bool)
    np.logical_or.at(gleicher_tag, qi, gleich)
    return laengen, tage_min, blick_ok, gleicher_tag


def klassen_tabelle(werte, klassen, hit, loesbar, name):
    zeilen = []
    for lo, hi in klassen:
        m = loesbar & (werte >= lo) & (werte <= hi)
        label = f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}"
        zeilen.append({"klasse": label, "n": int(m.sum()),
                       "recall_1": float(hit[m].mean()) if m.any() else None})
    return {"merkmal": name, "klassen": zeilen}


def bool_tabelle(flag, hit, loesbar, name, ja, nein):
    zeilen = []
    for label, m in ((ja, loesbar & flag), (nein, loesbar & ~flag)):
        zeilen.append({"klasse": label, "n": int(m.sum()),
                       "recall_1": float(hit[m].mean()) if m.any() else None})
    return {"merkmal": name, "klassen": zeilen}


def main():
    args = _args()
    name = args.method if args.adapter in ("none", "None") else f"{args.method}_{args.adapter}"
    query, database, indices, _ = load_retrieval(ROOT, CFG, args.method, args.adapter)
    loesbar = localizable(query, database, args.threshold)
    hit = hits_at_k(query, database, indices, loesbar, args.threshold, [1])[1]
    nachbarn, tage, blick_ok, gleicher_tag = query_properties(
        query, database, args.threshold, float(CFG["vpr"]["max_heading_diff_deg"]))

    tabellen = [
        klassen_tabelle(nachbarn, NACHBAR_KLASSEN, hit, loesbar, "Nachbarn im Umkreis"),
        klassen_tabelle(tage, TAGE_KLASSEN, hit, loesbar, "Tage zum naechsten Referenzbild"),
        bool_tabelle(blick_ok, hit, loesbar, "Nachbar mit passender Blickrichtung", "ja", "nein"),
        bool_tabelle(gleicher_tag, hit, loesbar, "Nachbar vom selben Fotografen am selben Tag", "ja", "nein"),
    ]
    gesamt = float(hit[loesbar].mean())
    print(f"\n{name}  |  R@1 bei {args.threshold:g} m ueber {loesbar.sum():,} loesbare: {gesamt:.3f}\n")
    for t in tabellen:
        print(t["merkmal"])
        for z in t["klassen"]:
            r = f"{z['recall_1']:.3f}" if z["recall_1"] is not None else "-"
            print(f"  {z['klasse']:<10}{z['n']:>8,}   R@1 {r}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"recall_by_difficulty_{name}.json"
    out.write_text(json.dumps({
        "datum": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "embedding_name": name, "threshold_m": args.threshold,
        "n_loesbar": int(loesbar.sum()), "recall_1": gesamt, "merkmale": tabellen,
    }, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["figure.dpi"] = 150
    fig, achsen = plt.subplots(1, 4, figsize=(16, 4), gridspec_kw={"width_ratios": [6, 5, 2, 2]})
    for ax, t in zip(achsen, tabellen):
        labels = [z["klasse"] for z in t["klassen"]]
        werte = [z["recall_1"] or 0 for z in t["klassen"]]
        ax.bar(range(len(labels)), werte, color="#37474f")
        for i, z in enumerate(t["klassen"]):
            ax.text(i, (z["recall_1"] or 0) + 0.01, f"n={z['n']:,}", ha="center", fontsize=6.5)
        ax.axhline(gesamt, color="#c62828", linewidth=0.9, linestyle="--")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(t["merkmal"], fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    achsen[0].set_ylabel(f"R@1 bei {args.threshold:g} m")
    fig.suptitle(f"{name}: Recall je Schwierigkeitsklasse (rot: gesamt {gesamt:.3f})", fontsize=10)
    plt.tight_layout()
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"\ngeschrieben: {out.relative_to(ROOT)} und .png")


if __name__ == "__main__":
    main()
