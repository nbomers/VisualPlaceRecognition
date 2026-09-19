"""
Abgebrochene Downloads erkennen.

Der Test steht hier, weil die Pruefung in 03 vorher `Image.verify()` benutzte
und damit nichts fand: verify() liest den Kopf, nicht die Bilddaten. Ein bei
50 % abgeschnittenes JPEG -- also genau das, was ein Strg-C hinterlaesst --
ging als heil durch.
"""

import io

import pytest

# Erst die Abhaengigkeiten, dann der Import. src.mapillary zieht auf
# Modulebene `requests` nach (es ist das HTTP-Modul des Projekts), und
# bild_ist_heil braucht Pillow. Fehlt eines, soll dieser Test uebersprungen
# werden -- nicht die ganze Sammlung abbrechen, wie es passiert, wenn der
# Import oben ohne Schutz steht.
pytest.importorskip("requests")
Image = pytest.importorskip("PIL.Image")

from src.mapillary import bild_ist_heil          # noqa: E402


def _jpeg_bytes(groesse=(400, 300)):
    puffer = io.BytesIO()
    Image.new("RGB", groesse, (120, 90, 60)).save(puffer, "JPEG", quality=90)
    return puffer.getvalue()


def test_vollstaendiges_jpeg_ist_heil(tmp_path):
    p = tmp_path / "ganz.jpg"
    p.write_bytes(_jpeg_bytes())
    assert bild_ist_heil(p)


@pytest.mark.parametrize("anteil", [0.1, 0.5, 0.9, 0.99])
def test_abgeschnittenes_jpeg_faellt_durch(tmp_path, anteil):
    """
    Ab 50 % haelt Image.verify() die Datei fuer heil -- deshalb prueft
    bild_ist_heil() mit load(). Der Test faellt um, wenn jemand zurueck
    auf verify() stellt.
    """
    roh = _jpeg_bytes()
    p = tmp_path / f"halb_{int(anteil * 100)}.jpg"
    p.write_bytes(roh[: int(len(roh) * anteil)])
    assert not bild_ist_heil(p)


def test_leere_und_fehlende_datei(tmp_path):
    leer = tmp_path / "leer.jpg"
    leer.write_bytes(b"")
    assert not bild_ist_heil(leer)
    assert not bild_ist_heil(tmp_path / "gibtsnicht.jpg")


def test_verify_allein_wuerde_es_nicht_finden(tmp_path):
    """Die Begruendung des Wechsels, als Test festgehalten."""
    roh = _jpeg_bytes()
    p = tmp_path / "halb.jpg"
    p.write_bytes(roh[: len(roh) // 2])
    with Image.open(p) as im:
        im.verify()            # wirft NICHT -- genau das war das Problem
    assert not bild_ist_heil(p)
