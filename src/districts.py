"""
Stadtteile aus OSM -- dieselbe Abfrage wie in 01, fuer alles, was danach
nach Stadtteil auswertet.

Eine Abfrage mit den Parametern aus config.yaml -> districts, die 01 fuer
die Abdeckungskarte und die Experimente fuer Recall je Stadtteil und den
Verwechslungsatlas nutzen. Antworten kommen aus dem osmnx-Cache unter
cache/; ohne Cache geht die Abfrage an Overpass.

    configure_osmnx(cfg, PATHS.cache)
    districts, city_polygon, utm_crs = load_districts(cfg, PATHS.cache)
    zuordnung = assign_district(lat, lon, districts)

Overpass ist ein oeffentlicher, geteilter Dienst: Zeitueberschreitungen,
503 und Wartezeiten sind Normalbetrieb, kein Programmfehler. Deshalb steht
die Konfiguration (Endpunkt, Zeitlimit, Ratenbremse) in configure_osmnx an
einer Stelle, und die Abfrage holt nur die admin_levels, die diese Datei
auch auswertet -- nicht jede administrative Grenze im Stadtgebiet.
"""

import os

import geopandas as gpd
import osmnx as ox

# osmnx 1.x und 2.x benennen dieselben Einstellungen verschieden. Wir setzen
# den ersten Namen, den es in der installierten Version gibt.
_EINSTELLUNGEN = {
    "overpass_url": ("overpass_url", "overpass_endpoint"),
    "timeout": ("requests_timeout", "timeout"),
    "rate_limit": ("overpass_rate_limit",),
}

# osmnx meldet "nichts gefunden" je nach Version als eigene Ausnahme statt
# mit einem leeren GeoDataFrame. Das ist kein Fehler, sondern eine Antwort.
try:                                              # osmnx >= 2
    from osmnx._errors import InsufficientResponseError as _LeereAntwort
except ImportError:
    try:                                          # osmnx 1.x
        from osmnx._errors import EmptyOverpassResponse as _LeereAntwort
    except ImportError:
        class _LeereAntwort(Exception):
            pass


def _setze(gruppe, wert):
    for name in _EINSTELLUNGEN[gruppe]:
        if hasattr(ox.settings, name):
            setattr(ox.settings, name, wert)
            return name
    return None


def configure_osmnx(cfg=None, cache_dir=None):
    """
    Cache und Overpass-Einstellungen aus config.yaml -> osm.

    Der Cache ist der wichtigste Teil: eine wiederholte Abfrage geht dann
    gar nicht mehr ans Netz. Auf einem frisch geklonten Rechner ist er leer
    (cache/ ist gitignored), dort laufen alle Abfragen das erste Mal
    wirklich.

    VPR_OVERPASS_URL sticht osm.overpass_url -- ein Spiegel laesst sich so
    je Rechner setzen, ohne die versionierte Datei zu aendern:

        VPR_OVERPASS_URL="https://overpass.kumi.systems/api" python run.py
    """
    if cache_dir is not None:
        ox.settings.cache_folder = cache_dir
    ox.settings.use_cache = True

    osm = (cfg or {}).get("osm") or {}
    # or statt get(..., default): eine leer gesetzte Variable soll auf die
    # Datei zurueckfallen, nicht den Endpunkt auf "" setzen.
    url = (os.environ.get("VPR_OVERPASS_URL") or osm.get("overpass_url") or "").strip()
    if url:
        _setze("overpass_url", url.rstrip("/"))
    if osm.get("timeout_s"):
        # Deckt beides ab: das HTTP-Zeitlimit der Anfrage und das
        # [timeout:...] im Overpass-Skript, das osmnx daraus bildet.
        _setze("timeout", int(osm["timeout_s"]))
    if "rate_limit" in osm:
        _setze("rate_limit", bool(osm["rate_limit"]))


def overpass_url():
    """Der Endpunkt, den osmnx gerade verwendet -- fuer die Ausgabe in 01."""
    for name in _EINSTELLUNGEN["overpass_url"]:
        if hasattr(ox.settings, name):
            return getattr(ox.settings, name)
    return "unbekannt"


def geocode_city(city_name):
    """
    Stadtgrenze, UTM-CRS und die Geocoder-Zeile dazu.

    Der Geocoder liefert je nach Suchbegriff mehrere Objekte (Punkt, Stadt,
    Landkreis); zuerst auf Flaechen filtern, dann das erste nehmen. Die
    zurueckgegebene Zeile nennt osm_type/osm_id -- damit laesst sich bei
    einer neuen Stadt in einem Blick pruefen, ob wirklich die Stadt und
    nicht der gleichnamige Landkreis getroffen wurde.
    """
    gdf = ox.geocode_to_gdf(city_name)
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        raise RuntimeError(f"Keine Polygon-Grenze fuer {city_name!r} gefunden.")
    zeile = gdf.iloc[0]
    polygon = zeile["geometry"]
    utm_crs = gpd.GeoSeries([polygon], crs="EPSG:4326").estimate_utm_crs()
    return polygon, utm_crs, zeile


def city_boundary(city_name):
    """Stadtgrenze als Polygon -- das erste Flaechenergebnis des Geocoders."""
    polygon, utm_crs, _ = geocode_city(city_name)
    return polygon, utm_crs


def _features(polygon, tags):
    """features_from_polygon, aber "nichts gefunden" ist leer, kein Fehler."""
    try:
        return ox.features_from_polygon(polygon, tags=tags)
    except _LeereAntwort:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


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


def load_districts(cfg, cache_dir, city_polygon=None, utm_crs=None):
    """
    Administrative Grenzen der konfigurierten admin_levels; die Ebene mit
    der besten Wertung gewinnt, bei Gleichstand die zuerst konfigurierte.
    Ohne brauchbare Ebene die place-Tags. Gibt (GeoDataFrame, Stadtpolygon,
    UTM-CRS). 01 und die Experimente rufen dieselbe Funktion.

    city_polygon/utm_crs koennen uebergeben werden, wenn der Aufrufer die
    Stadtgrenze schon hat (01 hat sie). Spart einen zweiten Geocode und
    stellt sicher, dass Stadtteile an genau dieselbe Grenze geclippt
    werden, gegen die auch die Bilder gefiltert wurden.
    """
    configure_osmnx(cfg, cache_dir)
    d = cfg["districts"]
    if city_polygon is None or utm_crs is None:
        city_polygon, utm_crs = city_boundary(cfg["city"])
    city_area = gpd.GeoSeries([city_polygon], crs="EPSG:4326").to_crs(utm_crs).area.iloc[0] / 1e6

    # Nur die konfigurierten Ebenen abfragen. boundary=administrative allein
    # holt jede Grenze, die das Stadtgebiet beruehrt -- Gemeinde, Kreis,
    # Bundesland -- und Overpass laedt zu jeder Relation ueber die Rekursion
    # alle Mitgliedswege und -knoten mit. Ein Vielfaches an Daten fuer
    # Ebenen, die die Schleife unten ohnehin verwirft.
    levels = [str(x) for x in d.get("admin_levels", [10, 9])]
    roh = _features(city_polygon, tags={"admin_level": levels})
    roh = roh[roh.geometry.type.isin(["Polygon", "MultiPolygon"])]
    # admin_level haengt selten auch an anderem als einer administrativen
    # Grenze; die Bedingung stand vorher in der Abfrage und bleibt erhalten.
    if "boundary" in roh.columns:
        roh = roh[roh["boundary"] == "administrative"]
    if {"name", "admin_level"} <= set(roh.columns):
        roh = roh[roh["name"].notna() & roh["admin_level"].notna()]
    else:
        roh = roh.iloc[0:0]

    beste, beste_wertung = None, float("-inf")
    for level in levels:
        gdf = roh[roh["admin_level"].astype(str) == level] if len(roh) else roh
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
        roh = _features(city_polygon, tags={"place": list(d["place_tags"])})
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
