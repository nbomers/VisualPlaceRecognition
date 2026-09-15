"""
Zugang zu Mapillary: Token aus .env, Sessions mit Wiederholung, und die
Vector Tiles, aus denen 01 die Bildpositionen holt.

Warum Vector Tiles und nicht die Bbox-Suche der Graph-API: die Kacheln
liefern alle Bildpunkte einer Stadt in rund 150 Anfragen, die Bbox-Suche
kappt bei 2.000 Treffern je Aufruf.
"""

import math
import os
import threading

import requests
from requests.adapters import HTTPAdapter, Retry

TILE_URL = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}"

# Nur diese Felder liefern die Kacheln; Kameratyp und dergleichen muesste
# die Graph-API nachliefern.
TILE_PROPS = {
    "id": "image_id",
    "sequence_id": "sequence_id",
    "captured_at": "captured_at",
    "compass_angle": "compass_angle",
    "is_pano": "is_pano",
    "creator_id": "creator_id",
}

_thread_local = threading.local()


def load_token(root):
    """
    MAPILLARY_TOKEN aus der Umgebung, sonst aus <root>/.env.

    Die .env ist gitignored; das Token faellt nie in ein Notebook oder ein
    Ergebnis.
    """
    env_file = root / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    token = os.environ.get("MAPILLARY_TOKEN", "")
    if not token.startswith("MLY|"):
        raise RuntimeError(
            "Kein Mapillary-Token. Datei .env im Projektwurzelverzeichnis anlegen:\n"
            "  MAPILLARY_TOKEN=MLY|dein|token"
        )
    return token


def make_session(pool_size=16):
    """
    Session mit Wiederholung bei 429 und 5xx. Bei 429 haelt sie sich an
    Retry-After, sonst exponentiell.
    """
    session = requests.Session()
    retry = Retry(
        total=5, connect=5, read=5, status=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=True,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry,
                          pool_connections=pool_size, pool_maxsize=pool_size)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_session(pool_size=16):
    """Eine Session je Thread -- requests.Session ist nicht threadsicher."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = make_session(pool_size)
    return _thread_local.session


# ----------------------------------------------------------------------
# Vector Tiles
# ----------------------------------------------------------------------


def deg2tile(lon, lat, zoom):
    """Web-Mercator-Kachel (x, y), in der ein Punkt liegt."""
    n = 2 ** zoom
    return (int((lon + 180) / 360 * n),
            int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n))


def px2deg(tx, ty, px, py, extent, zoom):
    """
    Pixel innerhalb einer Kachel -> Lon/Lat. Die Dekodierbibliothek legt
    den Ursprung unten links, die Kachelspezifikation oben links -- gemessen
    gegen die Graph-API: so 0,3 m Abweichung, andersherum 699 m.
    """
    n = 2 ** zoom
    py = extent - py
    wx, wy = (tx + px / extent) / n, (ty + py / extent) / n
    return (wx * 360 - 180,
            math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * wy)))))


def load_tile(session, token, zoom, tx, ty):
    """
    Alle Bildpunkte einer Kachel als Liste von dicts (lon, lat, Metadaten).

    Eine Kachel ohne Inhalt ist kein Fehler: Mapillary antwortet darauf mal
    mit einer leeren Kachel, mal mit 404. Beides ergibt hier [] -- sonst
    zaehlt eine Stadt mit Wasser- oder Waldkacheln lauter "gescheiterte"
    Kacheln, die in Wahrheit nur leer sind.
    """
    import mapbox_vector_tile  # nur hier gebraucht

    r = session.get(TILE_URL.format(z=zoom, x=tx, y=ty),
                    params={"access_token": token}, timeout=60)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    layer = mapbox_vector_tile.decode(r.content).get("image")
    if layer is None:
        return []
    extent = layer.get("extent", 4096)
    out = []
    for f in layer.get("features", []):
        geom = f.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        lon, lat = px2deg(tx, ty, *geom["coordinates"], extent, zoom)
        row = {"lon": lon, "lat": lat}
        row.update({TILE_PROPS[k]: v for k, v in (f.get("properties") or {}).items()
                    if k in TILE_PROPS})
        out.append(row)
    return out


def tiles_for_bounds(lon_min, lat_min, lon_max, lat_max, zoom):
    """Alle Kacheln, die einen Kartenausschnitt abdecken."""
    x0, y0 = deg2tile(lon_min, lat_max, zoom)
    x1, y1 = deg2tile(lon_max, lat_min, zoom)
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
