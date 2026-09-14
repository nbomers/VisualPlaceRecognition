"""
Verwechslungsatlas: wohin schaetzt das System, wenn es falsch liegt?

Fuer jede loesbare Anfrage, deren Top-1 weiter als die Schwelle entfernt
liegt, ein Pfeil von der echten zur geschaetzten Position auf dem
Strassennetz -- und dieselben Pfeile aggregiert nach Stadtteil-Paar
(Gliederung aus src/districts.py, wie in recall_by_district.py). Die zehn
haeufigsten Paare zwischen verschiedenen Stadtteilen als Tabelle; die
Fehlgriffe innerhalb eines Stadtteils -- meist dieselbe Strasse, knapp
jenseits der Schwelle -- stehen als Anteil daneben.

Nur loesbare Anfragen: dort hatte das System eine Referenz in Reichweite
und hat sich trotzdem fuer einen anderen Ort entschieden. Bei unloesbaren
ist jeder Top-1 zwangslaeufig falsch, das sagt nichts ueber den Encoder.

Reines Nachbearbeiten der .npz aus 06; Strassennetz und Stadtteile kommen
aus dem osmnx-Cache.

    python experiments/confusion_atlas.py                       # megaloc
    python experiments/confusion_atlas.py --method eigenplaces_pcaw512

Ergebnis: experiments/results/confusion_atlas_<name>.json und .png
"""

import argparse
import json

import numpy as np
import pandas as pd

from _common import CFG, RESULTS, ROOT
from src.districts import assign_district, load_districts
from src.geo import haversine_distance
from src.retrieval import load_retrieval, localizable

OUT_DIR = RESULTS


def _args():
    ap = argparse.ArgumentParser(
        description="Pfeile von echter zu geschaetzter Position, nach Stadtteil-Paar.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--method", default="megaloc")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--threshold", type=float,
                    default=float(CFG["vpr"]["uncertain_radius_m"]))
    ap.add_argument("--n-arrows", type=int, default=3000,
                    help="Stichprobe einzelner Pfeile in der linken Karte")
    ap.add_argument("--n-pairs", type=int, default=10,
                    help="Haeufigste Stadtteil-Paare in Tabelle und rechter Karte")
    return ap.parse_args()


def confusion_pairs(von, nach, fehler_m, n_pairs):
    """Die haeufigsten (echt, geschaetzt)-Paare verschiedener Stadtteile unter den Fehlgriffen."""
    df = pd.DataFrame({"von": von, "nach": nach, "fehler_m": fehler_m})
    df["von"] = df["von"].fillna("(ohne Stadtteil)")
    df["nach"] = df["nach"].fillna("(ohne Stadtteil)")
    zwischen = df[df["von"] != df["nach"]]
    gruppen = zwischen.groupby(["von", "nach"], sort=False)
    tabelle = gruppen.agg(n=("fehler_m", "size"), median_m=("fehler_m", "median")).reset_index()
    tabelle = tabelle.sort_values("n", ascending=False).reset_index(drop=True)
    tabelle["anteil"] = tabelle["n"] / len(df)
    return tabelle.head(n_pairs), df


_STRASSENTYP = {
    "autobahn": ("motorway", "motorway_link", "trunk", "trunk_link"),
    "hauptstrasse": ("primary", "primary_link", "secondary", "secondary_link"),
}


def road_type(G, lat, lon):
    """Strassentyp der naechsten Kante je Punkt: autobahn, hauptstrasse, sonstige."""
    import osmnx as ox

    kanten = ox.nearest_edges(G, lon, lat)
    typ = []
    for u, v, k in kanten:
        h = G.edges[u, v, k].get("highway", "")
        h = h[0] if isinstance(h, list) else h
        typ.append(next((n for n, tags in _STRASSENTYP.items() if h in tags), "sonstige"))
    return np.array(typ)


def by_road_type(typ, loesbar, falsch, fehler):
    """Scheitert das System auf der Autobahn anders als in Wohnstrassen?"""
    raus = {}
    for t in ("autobahn", "hauptstrasse", "sonstige"):
        m = loesbar & (typ == t)
        f = falsch & m
        raus[t] = {
            "n_loesbar": int(m.sum()),
            "anteil_loesbar": float(m.sum() / loesbar.sum()),
            "recall_1": float(1 - f.sum() / m.sum()) if m.any() else None,
            "median_fehler_falsch_m": float(np.median(fehler[f])) if f.any() else None,
            "anteil_an_fehlern_ueber_1km": float((typ[falsch & (fehler > 1000)] == t).mean()),
        }
    return raus


def plot(G, city_polygon, districts, df, paare, name, schwelle, n_arrows, ziel, rng):
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import osmnx as ox

    plt.rcParams["figure.dpi"] = 150
    fig, (links, rechts) = plt.subplots(1, 2, figsize=(18, 8.5))
    grenze = gpd.GeoSeries([city_polygon], crs="EPSG:4326").boundary
    for ax in (links, rechts):
        ox.plot_graph(G, ax=ax, node_size=0, edge_color="0.82", edge_linewidth=0.3,
                      bgcolor="white", show=False, close=False)
        districts.boundary.plot(ax=ax, color="0.55", linewidth=0.5, zorder=2)
        grenze.plot(ax=ax, color="black", linewidth=1.0, linestyle="--", zorder=2)
        ax.set_aspect(1 / np.cos(np.radians(city_polygon.centroid.y)))
        ax.set_axis_off()

    # Links: Stichprobe einzelner Fehlgriffe, echt -> geschaetzt.
    probe = df.sample(n=min(n_arrows, len(df)), random_state=int(rng.integers(0, 2**31)))
    links.quiver(probe["q_lon"], probe["q_lat"],
                 probe["t_lon"] - probe["q_lon"], probe["t_lat"] - probe["q_lat"],
                 angles="xy", scale_units="xy", scale=1, width=0.0012,
                 color="#c62828", alpha=0.18, zorder=3)
    links.scatter(probe["q_lon"], probe["q_lat"], s=2, color="#1565c0", alpha=0.5, zorder=4)
    links.set_title(f"{name}: {len(probe):,} von {len(df):,} Fehlgriffen (Top-1 > {schwelle:g} m)\n"
                    "blau = echte Position, Pfeil = geschaetzte", fontsize=10)

    # Rechts: aggregiert nach Stadtteil-Paar, Pfeil von Mittel zu Mittel.
    mittel = df.groupby(["von_name", "nach_name"]).agg(
        q_lon=("q_lon", "mean"), q_lat=("q_lat", "mean"),
        t_lon=("t_lon", "mean"), t_lat=("t_lat", "mean"))
    n_max = int(paare["n"].max())
    liste = []
    for nr, (_, p) in enumerate(paare.iterrows(), start=1):
        m = mittel.loc[(p["von"], p["nach"])]
        breite = 0.5 + 4.5 * p["n"] / n_max
        rechts.annotate("", xy=(m["t_lon"], m["t_lat"]), xytext=(m["q_lon"], m["q_lat"]),
                        arrowprops=dict(arrowstyle="-|>", lw=breite, color="#c62828",
                                        alpha=0.85, shrinkA=2, shrinkB=2), zorder=5)
        # Nummer am Pfeilanfang -- Paare laufen oft parallel, Text auf dem
        # Pfeil wuerde sich ueberlagern.
        rechts.annotate(str(nr), (m["q_lon"], m["q_lat"]), fontsize=6.5, ha="center",
                        va="center", color="white", weight="bold", zorder=7,
                        bbox=dict(boxstyle="circle,pad=0.25", fc="#c62828", ec="none"))
        liste.append(f"{nr:>2}  {p['von']} → {p['nach']}   {int(p['n'])}   ({p['median_m']:,.0f} m)")
    rechts.text(0.99, 0.01, "\n".join(liste), transform=rechts.transAxes, fontsize=7,
                ha="right", va="bottom", multialignment="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9, ec="0.7"))
    innerhalb = float((df["von_name"] == df["nach_name"]).mean())
    rechts.set_title(f"Die {len(paare)} haeufigsten Paare zwischen Stadtteilen (echt → geschaetzt)\n"
                     f"Pfeil von Mittel zu Mittel, Breite = Haeufigkeit  |  "
                     f"{innerhalb:.0%} der Fehlgriffe bleiben im eigenen Stadtteil",
                     fontsize=10)
    plt.tight_layout()
    fig.savefig(ziel, bbox_inches="tight")
    plt.close(fig)


def main():
    args = _args()
    name = args.method if args.adapter in ("none", "None") else f"{args.method}_{args.adapter}"
    rng = np.random.default_rng(int(CFG["vpr"]["split_seed"]))

    query, database, indices, _ = load_retrieval(ROOT, CFG, args.method, args.adapter)
    loesbar = localizable(query, database, args.threshold, 256)
    top1 = indices[:, 0]
    q_lat, q_lon = query["lat"].to_numpy(), query["lon"].to_numpy()
    t_lat, t_lon = database["lat"].to_numpy()[top1], database["lon"].to_numpy()[top1]
    fehler = haversine_distance(q_lat, q_lon, t_lat, t_lon)
    falsch = np.flatnonzero(loesbar & (fehler > args.threshold))

    districts, city_polygon, _ = load_districts(CFG, ROOT / "cache")
    von = assign_district(q_lat[falsch], q_lon[falsch], districts)
    nach = assign_district(t_lat[falsch], t_lon[falsch], districts)
    paare, df = confusion_pairs(von, nach, fehler[falsch], args.n_pairs)
    df["von_name"], df["nach_name"] = df["von"], df["nach"]
    df["q_lat"], df["q_lon"] = q_lat[falsch], q_lon[falsch]
    df["t_lat"], df["t_lon"] = t_lat[falsch], t_lon[falsch]
    innerhalb = float((df["von"] == df["nach"]).mean())
    # Knapp daneben oder grob verwechselt? Quantile des Fehlers.
    quantile = {str(q): float(np.percentile(fehler[falsch], q)) for q in (25, 50, 75, 90)}
    anteile = {"unter_100m": float((fehler[falsch] < 100).mean()),
               "100m_bis_1km": float(((fehler[falsch] >= 100) & (fehler[falsch] < 1000)).mean()),
               "ueber_1km": float((fehler[falsch] >= 1000).mean())}

    import osmnx as ox
    G = ox.graph_from_polygon(city_polygon, network_type="drive")
    falsch_mask = np.zeros(len(query), dtype=bool)
    falsch_mask[falsch] = True
    strassen = by_road_type(road_type(G, q_lat, q_lon), loesbar, falsch_mask, fehler)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"confusion_atlas_{name}.json"
    out.write_text(json.dumps({
        "datum": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "embedding_name": name,
        "threshold_m": args.threshold,
        "n_loesbar": int(loesbar.sum()),
        "n_falsch": int(len(falsch)),
        "median_fehler_m": float(np.median(fehler[falsch])),
        "fehler_quantile_m": quantile,
        "fehler_anteile": anteile,
        "anteil_innerhalb_stadtteil": innerhalb,
        "nach_strassentyp": strassen,
        "paare": [
            {"von": p["von"], "nach": p["nach"], "n": int(p["n"]), "anteil": float(p["anteil"]),
             "median_m": float(p["median_m"])}
            for _, p in paare.iterrows()
        ],
    }, indent=2))
    plot(G, city_polygon, districts, df, paare, name, args.threshold, args.n_arrows,
         out.with_suffix(".png"), rng)

    print(f"\n{name}  |  {len(falsch):,} Fehlgriffe unter {loesbar.sum():,} loesbaren "
          f"(Top-1 > {args.threshold:g} m)")
    print(f"Fehler: Quartile {quantile['25']:,.0f} / {quantile['50']:,.0f} / {quantile['75']:,.0f} m, "
          f"90 % unter {quantile['90']:,.0f} m  |  < 100 m {anteile['unter_100m']:.0%}, "
          f"100 m - 1 km {anteile['100m_bis_1km']:.0%}, > 1 km {anteile['ueber_1km']:.0%}")
    print(f"{innerhalb:.0%} bleiben im eigenen Stadtteil; die haeufigsten Paare zwischen Stadtteilen:\n")
    kopf = f"{'echt':<28}{'geschaetzt':<28}{'n':>7}{'Anteil':>8}{'Median m':>10}"
    print(kopf)
    print("-" * len(kopf))
    for _, p in paare.iterrows():
        print(f"{p['von']:<28}{p['nach']:<28}{int(p['n']):>7,}{p['anteil']:>7.1%}"
              f"{p['median_m']:>10,.0f}")
    print(f"\n{'Strassentyp':<14}{'loesbar':>9}{'Anteil':>8}{'R@1':>7}{'Med. Fehler':>13}{'an >1 km':>10}")
    for t, z in strassen.items():
        print(f"{t:<14}{z['n_loesbar']:>9,}{z['anteil_loesbar']:>7.1%}{z['recall_1']:>7.3f}"
              f"{z['median_fehler_falsch_m']:>11,.0f} m{z['anteil_an_fehlern_ueber_1km']:>9.1%}")
    print(f"\ngeschrieben: {out.relative_to(ROOT)} und .png")


if __name__ == "__main__":
    main()
