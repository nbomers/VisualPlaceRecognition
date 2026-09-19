"""
Ablehnungskurve: wie gut ist die Antwort, wenn das System bei niedriger
Konfidenz "weiss ich nicht" sagen darf?

Drei Konfidenzmasse, alle aus der Trefferliste von 06, ohne neue Modelle:

  aehnlichkeit   Cosinus des besten Treffers
  marge          Cosinus Platz 1 minus Platz 2 -- das, was locate.py meldet
  geschlossenheit  Anteil der Top-k, die innerhalb der Schwelle um Platz 1
                 liegen -- die Konfidenz aus 08, ohne DBSCAN

Fuer jede Konfidenz wird die Schwelle von "alles beantworten" bis "nichts
beantworten" gesenkt; Abdeckung = Anteil beantworteter Anfragen, Praezision
= Anteil richtiger unter den beantworteten (Top-1 innerhalb der Schwelle).
Zwei Sichten: ueber die loesbaren Anfragen (das System hatte eine Chance)
und ueber alle (ohne Referenz ist jede Antwort falsch -- das ist der
Betrieb). Kennzahl: Praezision bei 80 % Abdeckung, dazu AUC ueber der
Abdeckung.

    python experiments/rejection_curve.py                      # megaloc
    python experiments/rejection_curve.py --method eigenplaces_megaloc_concat

Ergebnis: experiments/results/rejection_curve_<name>.{json,png}
"""

import argparse
import json

import numpy as np
import pandas as pd

from _common import CFG, RESULTS, ROOT
from src.geo import haversine_distance
from src.retrieval import load_retrieval, localizable, top_distances


def _args():
    ap = argparse.ArgumentParser(
        description="Praezision gegen Abdeckung bei fallender Konfidenzschwelle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--method", default="megaloc")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--threshold", type=float, default=float(CFG["vpr"]["uncertain_radius_m"]))
    ap.add_argument("--top-k", type=int, default=int(CFG["localization"]["top_k"]),
                    help="Treffer fuer die Geschlossenheit")
    return ap.parse_args()


def confidences(query, database, indices, sims, top_k, threshold):
    """Drei Konfidenzwerte je Anfrage -- nur aus der Trefferliste, ohne die Anfrageposition."""
    db_lat, db_lon = database["lat"].to_numpy(), database["lon"].to_numpy()
    d_top1 = haversine_distance(db_lat[indices[:, :1]], db_lon[indices[:, :1]],
                                db_lat[indices[:, :top_k]], db_lon[indices[:, :top_k]])
    return {
        "aehnlichkeit": sims[:, 0].astype(np.float64),
        "marge": (sims[:, 0] - sims[:, 1]).astype(np.float64),
        "geschlossenheit": (d_top1 <= threshold).mean(axis=1),
    }


def curve(konfidenz, richtig):
    """Abdeckung und Praezision bei absteigender Konfidenz, je Anfrage ein Punkt."""
    order = np.argsort(-konfidenz, kind="stable")
    richtig_sortiert = richtig[order]
    n = len(order)
    abdeckung = np.arange(1, n + 1) / n
    praezision = np.cumsum(richtig_sortiert) / np.arange(1, n + 1)
    return abdeckung, praezision


def bei_abdeckung(abdeckung, praezision, ziel):
    return float(praezision[np.searchsorted(abdeckung, ziel)])


def main():
    args = _args()
    name = args.method if args.adapter in ("none", "None") else f"{args.method}_{args.adapter}"
    query, database, indices, sims = load_retrieval(ROOT, CFG, args.method, args.adapter)
    loesbar = localizable(query, database, args.threshold)
    top1_richtig = top_distances(query, database, indices[:, :1])[:, 0] <= args.threshold
    konf = confidences(query, database, indices, sims, args.top_k, args.threshold)

    sichten = {"loesbar": loesbar, "alle": np.ones(len(query), dtype=bool)}
    ergebnis = {}
    kurven = {}
    for sicht, maske in sichten.items():
        ergebnis[sicht] = {"n": int(maske.sum()), "praezision_alle": float(top1_richtig[maske].mean())}
        for k_name, k_wert in konf.items():
            abd, prae = curve(k_wert[maske], top1_richtig[maske])
            kurven[(sicht, k_name)] = (abd, prae)
            ergebnis[sicht][k_name] = {
                "auc": float(np.trapezoid(prae, abd)),
                **{f"praezision_bei_{int(z * 100)}": bei_abdeckung(abd, prae, z)
                   for z in (0.2, 0.5, 0.8)},
            }

    print(f"\n{name}  |  Top-1 richtig bei {args.threshold:g} m  |  "
          f"loesbar {sichten['loesbar'].sum():,}, alle {len(query):,}\n")
    print(f"{'Sicht':<9}{'Konfidenz':<17}{'ohne Ablehnung':>15}{'bei 80 %':>10}{'bei 50 %':>10}"
          f"{'bei 20 %':>10}{'AUC':>7}")
    for sicht in sichten:
        for k_name in konf:
            e = ergebnis[sicht][k_name]
            print(f"{sicht:<9}{k_name:<17}{ergebnis[sicht]['praezision_alle']:>15.3f}"
                  f"{e['praezision_bei_80']:>10.3f}{e['praezision_bei_50']:>10.3f}"
                  f"{e['praezision_bei_20']:>10.3f}{e['auc']:>7.3f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"rejection_curve_{name}.json"
    out.write_text(json.dumps({
        "datum": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "embedding_name": name, "threshold_m": args.threshold, "top_k": args.top_k,
        "sichten": ergebnis,
    }, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["figure.dpi"] = 150
    fig, achsen = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, sicht in zip(achsen, sichten):
        for k_name in konf:
            abd, prae = kurven[(sicht, k_name)]
            ax.plot(abd, prae, label=k_name, linewidth=1.6)
        ax.axhline(ergebnis[sicht]["praezision_alle"], color="0.6", linewidth=0.8, linestyle="--",
                   label="ohne Ablehnung")
        ax.axvline(0.8, color="0.85", linewidth=0.8, zorder=0)
        ax.set_xlabel("Abdeckung (Anteil beantworteter Anfragen)")
        ax.set_title(f"{name}: {sicht} ({sichten[sicht].sum():,} Anfragen)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
    achsen[0].set_ylabel(f"Praezision (Top-1 innerhalb {args.threshold:g} m)")
    achsen[0].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"\ngeschrieben: {out.relative_to(ROOT)} und .png")


if __name__ == "__main__":
    main()
