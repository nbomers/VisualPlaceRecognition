"""
Die Schutzmechanismen selbst -- bisher prueften Tests, was sie schuetzen,
aber nicht, ob sie ausloesen.

Zwei Wege, auf denen ein Artefakt aus einem fremden Lauf durchrutschen kann:

  1. Gleiche Datei, andere config      -> require_fingerprint
  2. Gleiche config, andere Stadt      -> require_city_match

Der zweite Weg ist der stillere: der Fingerabdruck beschreibt die Metadaten,
und die Metadaten liegen neben den Embeddings und wandern beim Kopieren mit.
Er kann sich also gar nicht widersprechen.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.run_guard import (
    embedding_fingerprint,
    require_city_match,
    require_fingerprint,
    write_fingerprint,
)

CFG = {
    "city": "Testheim, Germany",
    "image_root": "~/nicht/verwendet",
    "vpr": {
        "models": {"clip": "openai/clip-vit-base-patch32"},
        "clip": {"revision": "abc123"},
        "split_seed": 42,
        "train_fraction": 0.70,
        "database_fraction": 0.15,
        "query_fraction": 0.15,
    },
}


def _metadaten(erste_id=1000, n=30):
    return pd.DataFrame({
        "image_id": np.arange(erste_id, erste_id + n, dtype=np.int64),
        "split": ["train"] * (n - 10) + ["database"] * 5 + ["query"] * 5,
    })


def _artefakt(tmp_path, cfg, metadata, name="clip_embeddings.npy"):
    pfad = tmp_path / name
    pfad.write_bytes(b"nicht wirklich ein Array")
    write_fingerprint(pfad, embedding_fingerprint(cfg, "clip", "none", metadata))
    return pfad


# ---------------------------------------------------------------- Fingerabdruck


def test_passender_fingerabdruck_laesst_durch(tmp_path):
    md = _metadaten()
    pfad = _artefakt(tmp_path, CFG, md)
    gespeichert = require_fingerprint(pfad, embedding_fingerprint(CFG, "clip", "none", md))
    assert gespeichert["hash"]


@pytest.mark.parametrize("schluessel, wert", [
    ("split_seed", 4242),
    ("train_fraction", 0.60),
])
def test_geaenderte_config_bricht_ab_und_nennt_den_schluessel(tmp_path, schluessel, wert):
    md = _metadaten()
    pfad = _artefakt(tmp_path, CFG, md)

    anders = json.loads(json.dumps(CFG))
    anders["vpr"][schluessel] = wert
    with pytest.raises(RuntimeError) as fehler:
        require_fingerprint(pfad, embedding_fingerprint(anders, "clip", "none", md))

    text = str(fehler.value)
    assert schluessel in text, text            # sagt, WAS abweicht
    assert str(wert) in text, text
    assert "run.py --from 04" in text, text    # und was zu tun ist


def test_geaenderte_modellkonfiguration_bricht_ab(tmp_path):
    md = _metadaten()
    pfad = _artefakt(tmp_path, CFG, md)
    anders = json.loads(json.dumps(CFG))
    anders["vpr"]["clip"]["revision"] = "eine-andere-revision"
    with pytest.raises(RuntimeError, match="revision"):
        require_fingerprint(pfad, embedding_fingerprint(anders, "clip", "none", md))


def test_anderer_split_in_den_metadaten_bricht_ab(tmp_path):
    """metadata_digest haengt an image_id UND split -- ein verschobenes Bild zaehlt."""
    md = _metadaten()
    pfad = _artefakt(tmp_path, CFG, md)
    verschoben = md.copy()
    verschoben.loc[0, "split"] = "query"
    with pytest.raises(RuntimeError, match="metadata_digest"):
        require_fingerprint(pfad, embedding_fingerprint(CFG, "clip", "none", verschoben))


def test_fehlende_datei_und_fehlender_fingerabdruck_sind_verschiedene_meldungen(tmp_path):
    md = _metadaten()
    fp = embedding_fingerprint(CFG, "clip", "none", md)

    with pytest.raises(FileNotFoundError, match="fehlt"):
        require_fingerprint(tmp_path / "clip_embeddings.npy", fp)

    (tmp_path / "clip_embeddings.npy").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="keinen Fingerabdruck"):
        require_fingerprint(tmp_path / "clip_embeddings.npy", fp)


# ---------------------------------------------------------------- Stadt


def _stadt_anlegen(tmp_path, metadata):
    processed = tmp_path / "data" / "testheim" / "processed"
    processed.mkdir(parents=True)
    metadata.to_parquet(processed / "metadata.parquet", index=False)
    return tmp_path


def test_eigene_stadt_geht_durch(tmp_path):
    md = _metadaten()
    root = _stadt_anlegen(tmp_path, md)
    require_city_match(root, CFG, md)


def test_fehlende_bilder_sind_erlaubt(tmp_path):
    """Ein Encoder darf Bilder fehlen -- in Osnabrueck ist ein Download
    gescheitert, bei MegaLoc zwei. Er darf nur keine fremden enthalten."""
    md = _metadaten()
    root = _stadt_anlegen(tmp_path, md)
    require_city_match(root, CFG, md.iloc[2:].reset_index(drop=True))


def test_fremde_stadt_bricht_ab(tmp_path):
    root = _stadt_anlegen(tmp_path, _metadaten(erste_id=1000))
    fremd = _metadaten(erste_id=9_000_000)
    with pytest.raises(RuntimeError) as fehler:
        require_city_match(root, CFG, fremd, what="Encoder 'clip'")
    text = str(fehler.value)
    assert "Testheim, Germany" in text, text
    assert "embeddings" in text, text


def test_ohne_stadtmetadaten_passiert_nichts(tmp_path):
    """01 noch nicht gelaufen -- es gibt nichts, wogegen man pruefen koennte."""
    require_city_match(tmp_path, CFG, _metadaten(erste_id=9_000_000))
