"""
Die osmnx-Einstellungen aus config.yaml -> osm landen wirklich in osmnx.

Alles, was Overpass oder Nominatim wirklich anfragt, steht hier nicht drin:
Netz im Test ist ein Test, der irgendwann ohne eigenes Zutun rot wird. Was
sich pruefen laesst, ist die Verdrahtung -- und die hat zwei Fallen: osmnx
1.x und 2.x benennen dieselben Einstellungen verschieden, und VPR_OVERPASS_URL
muss die Datei stechen, sonst laeuft ein Rechner mit Spiegel doch wieder
gegen den ueberlasteten Standardendpunkt.

Ohne osmnx (so laeuft die CI) wird die Datei uebersprungen.
"""

import pytest

ox = pytest.importorskip("osmnx")

from src.districts import configure_osmnx, overpass_url  # noqa: E402

SPIEGEL = "https://overpass.kumi.systems/api"


@pytest.fixture(autouse=True)
def _einstellungen_zuruecksetzen():
    """osmnx-Einstellungen sind global -- nach jedem Test wieder wie vorher."""
    namen = ("overpass_url", "overpass_endpoint", "requests_timeout", "timeout",
             "overpass_rate_limit", "use_cache", "cache_folder")
    vorher = {n: getattr(ox.settings, n) for n in namen if hasattr(ox.settings, n)}
    yield
    for n, wert in vorher.items():
        setattr(ox.settings, n, wert)


def test_endpunkt_zeitlimit_und_bremse(tmp_path):
    configure_osmnx(
        {"osm": {"overpass_url": SPIEGEL + "/", "timeout_s": 300, "rate_limit": False}},
        tmp_path,
    )
    # Der abschliessende Schraegstrich muss weg: osmnx haengt /interpreter an.
    assert overpass_url() == SPIEGEL
    assert getattr(ox.settings, "requests_timeout", None) == 300 or ox.settings.timeout == 300
    assert ox.settings.overpass_rate_limit is False
    assert ox.settings.use_cache is True
    assert str(ox.settings.cache_folder) == str(tmp_path)


def test_umgebung_sticht_die_datei(tmp_path, monkeypatch):
    monkeypatch.setenv("VPR_OVERPASS_URL", SPIEGEL)
    configure_osmnx({"osm": {"overpass_url": "https://overpass-api.de/api"}}, tmp_path)
    assert overpass_url() == SPIEGEL


def test_ohne_osm_block_bleibt_der_standard(tmp_path):
    vorher = overpass_url()
    configure_osmnx({}, tmp_path)
    assert overpass_url() == vorher
    assert ox.settings.use_cache is True
