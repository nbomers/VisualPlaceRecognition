"""
Stadtteile aus OSM -- dieselbe Abfrage wie in 01, fuer alles, was danach
nach Stadtteil auswertet.

Eine Abfrage mit den Parametern aus config.yaml -> districts, die 01 fuer
die Abdeckungskarte und die Experimente fuer Recall je Stadtteil und den
Verwechslungsatlas nutzen. Antworten kommen aus dem osmnx-Cache unter
cache/; ohne Cache geht die Abfrage an Overpass.

    districts, city_polygon, utm_crs = load_districts(cfg, PATHS.cache)
    zuordnung = assign_district(lat, lon, districts)
"""

import geopandas as gpd
import osmnx as ox


def city_boundary(city_name):
    """Stadtgrenze als Polygon -- das erste Flaechenergebnis des Geocoders."""
    gdf = ox.geocode_to_gdf(city_name)
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        raise RuntimeError(f"Keine Polygon-Grenze fuer {city_name!r} gefunden.")
    polygon = gdf.geometry.iloc[0]
    utm_crs = gpd.GeoSeries([polygon], crs="EPSG:4326").estimate_utm_crs()
    return polygon, utm_crs


def _prepare(gdf, city_polygon, utm_crs, min_area_km2):
    """An die Stadtgrenze clippen, kleine Flaechen raus, je Name die groesste."""
    gdf = gdf.copy()
    kaputt = ~gdf.geometry.is_valid
    gdf.loc[kaputt, "geometry"] = gdf.loc[kaputt, "geometry"].buffer(0)
    gdf["geometry"] = gdf.geometry.intersection(city_polygon)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf["area_km2"] = gdf.to_crs(utm_crs).area.to_numpy() / 1e6
    gdf = gdf[gdf["area_km2"] >= min_area_km2]
    gdf = gdf.sort_values("area_km2", ascending=False).drop_duplicates("name")
    return gdf[["name", "area_km2", "geometry"]].reset_index(drop=True)


def _score(gdf, city_area_km2, utm_crs):
    """Abdeckung der Stadtflaeche minus Ueberlappung -- die Wertung aus 01."""
    metric = gdf.to_crs(utm_crs)
    summe = metric.area.sum() / 1e6
    union = metric.geometry.union_all().area / 1e6
    abdeckung = union / city_area_km2
    ueberlappung = max(0.0, 1.0 - union / summe) if summe else 0.0
    return (1.0 - abs(1.0 - abdeckung)) - ueberlappung


def load_districts(cfg, cache_dir):
    """
    Administrative Grenzen der konfigurierten admin_levels; die Ebene mit
    der besten Wertung gewinnt, bei Gleichstand die zuerst konfigurierte.
    Ohne brauchbare Ebene die place-Tags. Gibt (GeoDataFrame, Stadtpolygon,
    UTM-CRS). 01 und die Experimente rufen dieselbe Funktion.
    """
    ox.settings.cache_folder = cache_dir
    ox.settings.use_cache = True
    d = cfg["districts"]
    city_polygon, utm_crs = city_boundary(cfg["city"])
    city_area = gpd.GeoSeries([city_polygon], crs="EPSG:4326").to_crs(utm_crs).area.iloc[0] / 1e6

    roh = ox.features_from_polygon(city_polygon, tags={"boundary": "administrative"})
    roh = roh[roh.geometry.type.isin(["Polygon", "MultiPolygon"])]
    roh = roh[roh["name"].notna() & roh["admin_level"].notna()]

    beste, beste_wertung = None, float("-inf")
    for level in d.get("admin_levels", [10, 9]):
        gdf = roh[roh["admin_level"].astype(str) == str(level)]
        if gdf.empty:
            continue
        gdf = _prepare(gdf, city_polygon, utm_crs, float(d["min_area_km2"]))
        if len(gdf) < int(d.get("min_districts_required", 3)):
            continue
        wertung = _score(gdf, city_area, utm_crs)
        if wertung > beste_wertung:
            beste, beste_wertung = gdf, wertung

    # Rueckfall fuer Staedte ohne administrative Stadtteile: place-Tags.
    if beste is None:
        roh = ox.features_from_place(cfg["city"], tags={"place": list(d["place_tags"])})
        roh = roh[roh.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if "name" in roh.columns and not roh.empty:
            beste = _prepare(roh[roh["name"].notna()], city_polygon, utm_crs,
                             float(d["min_area_km2"]))
        if beste is None or len(beste) < int(d.get("min_districts_required", 3)):
            raise RuntimeError(
                f"Keine brauchbare Stadtteilgliederung fuer {cfg['city']!r} -- weder "
                f"admin_level {d.get('admin_levels')} noch place-Tags {d['place_tags']}."
            )
    return gpd.GeoDataFrame(beste, geometry="geometry", crs="EPSG:4326"), city_polygon, utm_crs


def assign_district(lat, lon, districts):
    """
    Je Punkt der Name des Stadtteils, der ihn enthaelt, sonst None.
    Bei mehreren Treffern die kleinste Flaeche -- wie in 01.
    """
    punkte = gpd.GeoDataFrame(
        {"zeile": range(len(lat))},
        geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326",
    )
    treffer = gpd.sjoin(punkte, districts[["name", "area_km2", "geometry"]],
                        how="left", predicate="covered_by")
    treffer = treffer.sort_values(["zeile", "area_km2"]).drop_duplicates("zeile")
    return treffer.set_index("zeile")["name"].reindex(range(len(lat))).to_numpy(dtype=object)
