"""
Abstaende und Projektionen -- ein Ort fuer das, was 05, 07, 08 und die
Experimente bisher jeweils selbst definiert haben.
"""

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Grosskreisabstand in Metern. Broadcastet wie numpy: eine Anfrage gegen
    alle Datenbankbilder ist `haversine_distance(q_lat, q_lon, db_lat, db_lon)`,
    ein Block von Anfragen gegen alle ist dasselbe mit `[:, None]` links und
    `[None, :]` rechts.
    """
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return EARTH_RADIUS_M * 2 * np.arcsin(np.sqrt(a))


def heading_difference(a, b):
    """Zyklische Differenz zweier Kompasswinkel in Grad, 0 bis 180."""
    diff = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 360.0
    return np.minimum(diff, 360.0 - diff)


def to_metric_xy(lat, lon, crs=None):
    """
    Lat/Lon -> UTM-Meter. DBSCAN und Mittelwerte brauchen ein metrisches
    System; in Grad waere ein Radius von 25 m je nach Breitengrad
    unterschiedlich gross. Gibt (xy, crs) zurueck, damit ein zweiter Aufruf
    dasselbe crs verwenden kann.
    """
    import geopandas as gpd  # nur hier gebraucht, haelt das Modul leicht

    gs = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326")
    if crs is None:
        crs = gs.estimate_utm_crs()
    gs = gs.to_crs(crs)
    return np.c_[gs.x.to_numpy(), gs.y.to_numpy()], crs
