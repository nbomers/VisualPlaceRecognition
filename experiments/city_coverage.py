"""
Welche Stadt taugt als naechste? Mapillary-Abdeckung je Stadt, gemessen
wie 01 die Bilder holt (Vector Tiles, Zoom 14), aber nur gezaehlt.

Bilder je km2 sagt wenig -- eine Kampagne auf den Hauptstrassen treibt die
Zahl hoch, ohne dass die Wohnstrassen ein Bild haben. Deshalb die
Strassenabdeckung: Anteil des OSM-Fahrnetzes, das im Umkreis von 25 m ein
Bild hat, getrennt nach grossen Strassen (motorway bis secondary) und
Wohnstrassen (tertiary, residential, living_street, unclassified). Dazu
Sequenzen, Fotografen-Konzentration und der Anteil seit 2022.

    python experiments/city_coverage.py "Mainz, Germany" "Würzburg, Germany"

Ergebnis (stadtuebergreifend): experiments/results/city_coverage.json.
Braucht das Mapillary-Token (.env) und Overpass; zwischen Staedten wird
gewartet, Overpass sperrt sonst.
"""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from _common import CFG, PATHS, ROOT
from src.districts import city_boundary, configure_osmnx
from src.mapillary import load_tile, load_token, make_session, tiles_for_bounds

# Stadtuebergreifend, deshalb bewusst NICHT unter experiments/results/<stadt>.
OUT = ROOT / "experiments" / "results" / "city_coverage.json"

# Aus der config, damit hier nicht zum zweiten Mal steht, was 01 und die
# Auswertung schon festlegen.
ZOOM = int(CFG["zoom"])
RADIUS_M = float(CFG["vpr"]["uncertain_radius_m"])
SCHRITT_M = RADIUS_M
GROSS = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
         "secondary", "secondary_link"}
KLEIN = {"tertiary", "tertiary_link", "residential", "living_street", "unclassified"}


def _args():
    ap = argparse.ArgumentParser(description="Mapillary-Abdeckung je Stadt.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("staedte", nargs="+", help='z.B. "Mainz, Germany"')
    ap.add_argument("--pause", type=int, default=60, help="Sekunden zwischen Staedten (Overpass)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tiles-only", action="store_true",
                    help="Nur zaehlen (Bilder, Sequenzen, Fotografen, Jahre), kein Strassennetz -- "
                         "braucht kein Overpass, eine Minute je Stadt")
    return ap.parse_args()


def image_points(polygon, token):
    """Alle Bildpunkte der Kacheln ueber der Stadt, geschnitten auf die Grenze."""
    def eine(t):
        s = make_session(8)
        try:
            # Leere Kacheln (auch die, die Mapillary mit 404 beantwortet)
            # kommen als [] zurueck -- siehe src/mapillary.load_tile.
            return load_tile(s, token, ZOOM, *t)
        finally:
            s.close()

    pts = []
    with ThreadPoolExecutor(16) as pool:
        for job in as_completed([pool.submit(eine, t) for t in tiles_for_bounds(*polygon.bounds, ZOOM)]):
            pts.extend(job.result())
    drin = gpd.GeoSeries([Point(p["lon"], p["lat"]) for p in pts], crs="EPSG:4326").covered_by(polygon)
    return [p for p, i in zip(pts, drin.to_numpy()) if i]


def street_coverage(polygon, utm_crs, pts):
    """Anteil des Fahrnetzes (nach Laenge) mit einem Bild im Umkreis, je Strassenklasse."""
    G = ox.graph_from_polygon(polygon, network_type="drive", simplify=True)
    edges = ox.graph_to_gdfs(G, nodes=False).to_crs(utm_crs)
    xy = gpd.GeoSeries([Point(p["lon"], p["lat"]) for p in pts], crs="EPSG:4326").to_crs(utm_crs)
    baum = cKDTree(np.c_[xy.x.to_numpy(), xy.y.to_numpy()])
    laenge = Counter()
    gedeckt = Counter()
    for _, e in edges.iterrows():
        hw = e["highway"]
        hw = hw[0] if isinstance(hw, list) else hw
        klasse = "gross" if hw in GROSS else "klein" if hw in KLEIN else "sonst"
        geom = e.geometry if isinstance(e.geometry, LineString) else LineString(
            [e.geometry.coords[0], e.geometry.coords[-1]])
        n = max(1, int(geom.length // SCHRITT_M))
        proben = np.array([[p.x, p.y] for p in (geom.interpolate(d) for d in np.linspace(0, geom.length, n + 1))])
        d, _ = baum.query(proben)
        laenge[klasse] += geom.length
        gedeckt[klasse] += geom.length * float((d <= RADIUS_M).mean())
    return laenge, gedeckt


def survey(name, token, tiles_only=False):
    polygon, utm_crs = city_boundary(name)
    area = gpd.GeoSeries([polygon], crs="EPSG:4326").to_crs(utm_crs).area.iloc[0] / 1e6
    pts = image_points(polygon, token)
    laenge, gedeckt = (Counter(), Counter()) if tiles_only else street_coverage(polygon, utm_crs, pts)
    fot = Counter(p.get("creator_id") for p in pts if p.get("creator_id") is not None)
    seqs = Counter(p.get("sequence_id") for p in pts if p.get("sequence_id"))
    seit22 = sum(1 for p in pts if p.get("captured_at") and time.gmtime(p["captured_at"] / 1000).tm_year >= 2022)
    ges = sum(laenge.values())
    return {
        "stadt": name.split(",")[0], "km2": round(area, 1), "bilder": len(pts),
        "bilder_pro_km2": round(len(pts) / area), "strassen_km": round(ges / 1000),
        "abdeckung_gesamt": round(sum(gedeckt.values()) / ges, 3) if ges else None,
        "abdeckung_grosse_strassen": round(gedeckt["gross"] / laenge["gross"], 3) if laenge["gross"] else None,
        "abdeckung_wohnstrassen": round(gedeckt["klein"] / laenge["klein"], 3) if laenge["klein"] else None,
        "wohnstrassen_km": round(laenge["klein"] / 1000),
        "sequenzen": len(seqs), "median_sequenz": int(np.median(list(seqs.values()))) if seqs else 0,
        "fotografen": len(fot),
        "anteil_groesster_fotograf": round(fot.most_common(1)[0][1] / len(pts), 3) if fot else None,
        "anteil_seit_2022": round(seit22 / max(len(pts), 1), 2),
        "datum": pd.Timestamp.now().strftime("%Y-%m-%d"),
    }


def main():
    args = _args()
    # Endpunkt, Zeitlimit und Ratenbremse aus config.yaml -> osm -- dieselben
    # Einstellungen, mit denen 01 Strassennetz und Stadtteile holt.
    configure_osmnx(CFG, PATHS.cache)
    token = load_token(ROOT)
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    for i, name in enumerate(args.staedte):
        if name in done and not args.force:
            continue
        if i and not args.tiles_only:
            time.sleep(args.pause)
        t0 = time.time()
        try:
            r = survey(name, token, args.tiles_only)
            if args.tiles_only and name in done:       # Strassenwerte aus frueherem Lauf behalten
                r = {**done[name], **{k: v for k, v in r.items() if v is not None}}
            done[name] = r
        except Exception as e:
            print(f"{name}: {type(e).__name__}: {str(e)[:120]}")
            continue
        strassen = (f"Strassen gedeckt {r['abdeckung_gesamt']:.0%} (gross {r['abdeckung_grosse_strassen']:.0%}, "
                    f"Wohn {r['abdeckung_wohnstrassen']:.0%})" if r.get("abdeckung_gesamt") is not None
                    else "Strassen: nicht gemessen")
        print(f"{r['stadt']:<16} {r['bilder']:>9,} Bilder {r['bilder_pro_km2']:>6,}/km2 | {strassen} | "
              f"{r['sequenzen']:,} Seq. | {r['fotografen']} Fotografen, groesster "
              f"{r['anteil_groesster_fotograf']:.0%} | seit 2022: {r['anteil_seit_2022']:.0%}  "
              f"({time.time() - t0:.0f} s)", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(done, indent=1, ensure_ascii=False))
    if done:
        print(f"\n{'Stadt':<16}{'Bilder':>10}{'/km2':>7}{'gedeckt':>9}{'Wohn':>7}{'seit22':>8}{'Fotogr.':>9}")
        for r in sorted(done.values(), key=lambda r: -(r.get('abdeckung_gesamt') or 0)):
            ab = f"{r['abdeckung_gesamt']:>9.0%}{r['abdeckung_wohnstrassen']:>7.0%}" if r.get("abdeckung_gesamt") is not None else f"{'-':>9}{'-':>7}"
            print(f"{r['stadt']:<16}{r['bilder']:>10,}{r['bilder_pro_km2']:>7,}{ab}"
                  f"{r['anteil_seit_2022']:>8.0%}{r['anteil_groesster_fotograf']:>9.0%}")


if __name__ == "__main__":
    main()
