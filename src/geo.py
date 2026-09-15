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


def normalize_heading(werte):
    """
    Mapillary-Kompasswinkel in [0, 360) -- oder NaN, wenn unbekannt.

    Die Quelle kodiert eine unbekannte Blickrichtung als -1 und liefert
    vereinzelt Werte knapp ueber 360. Beides hier abfangen, nicht in jedem
    Aufrufer: -1 roh weiterzureichen hiesse, es als 359 Grad zu lesen, und
    das ist kein fehlender Wert, sondern ein falscher.
    """
    a = np.asarray(werte, dtype=float)
    return np.where(a < 0, np.nan, a % 360.0)


def heading_difference(a, b):
    """
    Zyklische Differenz zweier Kompasswinkel in Grad, 0 bis 180.
    Ist eine der beiden Richtungen unbekannt, kommt NaN heraus.
    """
    a, b = normalize_heading(a), normalize_heading(b)
    diff = np.abs(a - b) % 360.0
    return np.minimum(diff, 360.0 - diff)


def heading_matches(a, b, max_diff_deg):
    """
    Schauen zwei Bilder in aehnliche Richtung?

    Unbekannte Blickrichtung schliesst NICHT aus. Der Test soll Paare
    verwerfen, von denen man WEISS, dass sie auseinanderschauen -- nicht
    solche, ueber die die Metadaten nichts sagen. Sonst bestraft die
    Auswertung fehlende Daten statt falscher Orte, und zwar unsichtbar:
    je nach Stadt sind das einzelne Bilder oder ein spuerbarer Anteil.
    """
    d = heading_difference(a, b)
    return np.isnan(d) | (d <= float(max_diff_deg))


def utm_crs_for(lat, lon):
    """EPSG-Code der UTM-Zone, in der der Schwerpunkt der Punkte liegt."""
    lat0, lon0 = float(np.mean(lat)), float(np.mean(lon))
    zone = int((lon0 + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat0 >= 0 else 32700 + zone}"


def to_metric_xy(lat, lon, crs=None):
    """
    Lat/Lon -> UTM-Meter. DBSCAN, KDTree und Mittelwerte brauchen ein
    metrisches System; in Grad waere ein Radius von 25 m je nach
    Breitengrad unterschiedlich gross. Gibt (xy, crs) zurueck, damit ein
    zweiter Aufruf dasselbe crs verwenden kann.
    """
    from pyproj import Transformer  # nur hier gebraucht, haelt das Modul leicht

    if crs is None:
        crs = utm_crs_for(lat, lon)
    x, y = Transformer.from_crs("EPSG:4326", str(crs), always_xy=True).transform(
        np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    )
    return np.c_[x, y], str(crs)
