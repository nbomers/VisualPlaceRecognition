"""
Wo in der Stadt scheitert das System? Recall je Stadtteil.

Jede Anfrage wird per Point-in-Polygon einem OSM-Stadtteil zugeordnet
(dieselbe Gliederung wie die Abdeckungskarte in 01, siehe src/districts.py).
Je Stadtteil: Anfragen, Anteil loesbar bei 25 m, R@1 ueber die loesbaren.
Daneben die Referenzdichte -- Datenbankbilder je km2. Zeigen beide Karten
dasselbe Muster, ist der Dichte-Befund aus database_density.py auf
Stadtteil-Ebene belegt.

Loesbar und Treffer kommen aus src/retrieval.py, wie beim Bootstrap;
Distanzen einmal je Anfrage, kein Modell, keine GPU.

    python experiments/recall_by_district.py                       # megaloc
    python experiments/recall_by_district.py --method eigenplaces_pcaw512

Ergebnis: experiments/results/recall_by_district_<name>.json und .png
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _common import CFG, PATHS, RESULTS, ROOT
from src.districts import assign_district, load_districts
from src.retrieval import hits_at_k, load_retrieval, localizable

OUT_DIR = RESULTS


def _args():
    ap = argparse.ArgumentParser(
        description="Recall je Stadtteil, daneben die Referenzdichte.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--method", default="megaloc")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--threshold", type=float,
                    default=float(CFG["vpr"]["uncertain_radius_m"]))
    ap.add_argument("--min-solvable", type=int, default=100,
                    help="Stadtteile mit weniger loesbaren Anfragen bleiben grau")
    return ap.parse_args()


def district_table(query, database, districts, loesbar, hits, k_values):
    """Je Stadtteil: Anfragen, loesbar, Recall@k, Datenbankbilder und Dichte."""
    q_bezirk = assign_district(query["lat"].to_numpy(), query["lon"].to_numpy(), districts)
    db_bezirk = assign_district(database["lat"].to_numpy(), database["lon"].to_numpy(), districts)
    flaeche = districts.set_index("name")["area_km2"]
    q_seq = query["sequence_id"].to_numpy()

    zeilen = []
    namen = list(districts["name"]) + [None]
    for name in namen:
        q_mask = pd.isna(q_bezirk) if name is None else (q_bezirk == name)
        db_mask = pd.isna(db_bezirk) if name is None else (db_bezirk == name)
        n_loesbar = int(loesbar[q_mask].sum())
        km2 = float(flaeche[name]) if name is not None else float("nan")
        zeilen.append({
            "stadtteil": name if name is not None else "(ohne Stadtteil)",
            "km2": km2,
            "n_queries": int(q_mask.sum()),
            # Wenige Fahrten je Stadtteil: die Karte zeigt dann das Schicksal
            # einer Sequenz, nicht das des Ortes -- derselbe Effekt wie im
            # Sequenz-Bootstrap.
            "n_sequences": int(len(np.unique(q_seq[q_mask]))),
            "n_loesbar": n_loesbar,
            "anteil_loesbar": float(n_loesbar / q_mask.sum()) if q_mask.sum() else float("nan"),
            "n_database": int(db_mask.sum()),
            "database_pro_km2": float(db_mask.sum() / km2) if km2 == km2 and km2 > 0 else float("nan"),
            "recall": {
                str(k): (float(hits[k][q_mask].sum() / n_loesbar) if n_loesbar else None)
                for k in k_values
            },
        })
    return zeilen


def plot(districts, city_polygon, zeilen, name, schwelle, min_solvable, ziel):
    import matplotlib
    matplotlib.use("Agg")
    import geopandas as gpd
    import matplotlib.pyplot as plt

    plt.rcParams["figure.dpi"] = 150
    werte = pd.DataFrame([z for z in zeilen if z["stadtteil"] != "(ohne Stadtteil)"])
    werte["r1"] = [z["recall"]["1"] for z in zeilen if z["stadtteil"] != "(ohne Stadtteil)"]
    werte.loc[werte["n_loesbar"] < min_solvable, "r1"] = np.nan
    karte = districts.merge(werte, left_on="name", right_on="stadtteil", how="left")
    grenze = gpd.GeoSeries([city_polygon], crs="EPSG:4326").boundary

    fig, achsen = plt.subplots(1, 2, figsize=(17, 7.5))
    spalten = [
        ("r1", "RdYlGn", f"R@1 bei {schwelle:g} m", f"{name}: R@1 je Stadtteil"),
        ("database_pro_km2", "YlOrRd", "Datenbankbilder je km²", "Referenzdichte je Stadtteil"),
    ]
    for ax, (spalte, cmap, legende, titel) in zip(achsen, spalten):
        karte.plot(column=spalte, cmap=cmap, legend=True, ax=ax,
                   edgecolor="black", linewidth=0.5,
                   legend_kwds={"label": legende, "shrink": 0.7},
                   missing_kwds={"color": "lightgrey"})
        for _, r in karte.iterrows():
            p = r.geometry.representative_point()
            wenig = r["n_loesbar"] < min_solvable
            text = f"{r['name']}\nn={int(r['n_loesbar'])}"
            if spalte == "r1" and not wenig:
                text += f"  {r['r1']:.2f}"
            ax.annotate(text, xy=(p.x, p.y), fontsize=6, ha="center", va="center",
                        color="0.45" if wenig else "black",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))
        grenze.plot(ax=ax, color="black", linewidth=1.2, linestyle="--")
        ax.set_aspect(1 / np.cos(np.radians(city_polygon.centroid.y)))
        ax.set_title(titel)
        ax.set_axis_off()
    fig.text(0.5, 0.02, f"grau: unter {min_solvable} loesbare Anfragen  |  n = loesbare Anfragen",
             ha="center", fontsize=8, color="0.3")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(ziel, bbox_inches="tight")
    plt.close(fig)


def main():
    args = _args()
    name = args.method if args.adapter in ("none", "None") else f"{args.method}_{args.adapter}"
    k_values = [int(k) for k in CFG["retrieval"]["k_values"]]

    query, database, indices, _ = load_retrieval(ROOT, CFG, args.method, args.adapter)
    loesbar = localizable(query, database, args.threshold, 256)
    hits = hits_at_k(query, database, indices, loesbar, args.threshold, k_values)

    districts, city_polygon, _ = load_districts(CFG, PATHS.cache)
    zeilen = district_table(query, database, districts, loesbar, hits, k_values)

    # Haengt R@1 an der Dichte? Rangkorrelation ueber die Stadtteile mit
    # genug loesbaren Anfragen -- eine Zahl statt zwei Karten nebeneinander.
    gut = [z for z in zeilen if z["stadtteil"] != "(ohne Stadtteil)"
           and z["n_loesbar"] >= args.min_solvable]
    rho, p = spearmanr([z["database_pro_km2"] for z in gut], [z["recall"]["1"] for z in gut])
    rho_loesbar, _ = spearmanr([z["anteil_loesbar"] for z in gut], [z["recall"]["1"] for z in gut])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"recall_by_district_{name}.json"
    out.write_text(json.dumps({
        "datum": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "embedding_name": name,
        "threshold_m": args.threshold,
        "min_solvable": args.min_solvable,
        "n_districts": int(len(districts)),
        "spearman_r1_dichte": {"rho": float(rho), "p": float(p), "n": len(gut)},
        "spearman_r1_anteil_loesbar": float(rho_loesbar),
        "stadtteile": zeilen,
    }, indent=2))
    plot(districts, city_polygon, zeilen, name, args.threshold, args.min_solvable,
         out.with_suffix(".png"))

    zeilen.sort(key=lambda z: -(z["recall"]["1"] if z["recall"]["1"] is not None else -1))
    print(f"\n{name}  |  R@1 bei {args.threshold:g} m  |  {len(districts)} Stadtteile\n")
    kopf = (f"{'Stadtteil':<28}{'n':>7}{'Seq.':>6}{'loesbar':>9}{'Anteil':>8}{'R@1':>7}"
            f"{'R@5':>7}{'DB/km2':>9}")
    print(kopf)
    print("-" * len(kopf))
    for z in zeilen:
        r1 = f"{z['recall']['1']:.3f}" if z["recall"]["1"] is not None else "-"
        r5 = f"{z['recall']['5']:.3f}" if z["recall"]["5"] is not None else "-"
        markierung = "  (grau)" if z["n_loesbar"] < args.min_solvable else ""
        dichte = f"{z['database_pro_km2']:,.0f}" if z["database_pro_km2"] == z["database_pro_km2"] else "-"
        print(f"{z['stadtteil']:<28}{z['n_queries']:>7,}{z['n_sequences']:>6}{z['n_loesbar']:>9,}"
              f"{z['anteil_loesbar']:>7.1%}{r1:>7}{r5:>7}{dichte:>9}{markierung}")
    print(f"\nSpearman R@1 gegen Datenbankbilder/km2: rho = {rho:+.2f} (p = {p:.3f}, "
          f"{len(gut)} Stadtteile); gegen Anteil loesbar: rho = {rho_loesbar:+.2f}")
    print(f"geschrieben: {out.relative_to(ROOT)} und .png")


if __name__ == "__main__":
    main()
