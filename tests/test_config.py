"""validate_config nimmt die echte config.yaml an und lehnt typische Fehler ab."""

import copy
from pathlib import Path

import pytest

from src.config import load_config
from src.run_guard import validate_config

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg():
    return load_config(ROOT)


def test_echte_config_ist_gueltig(cfg):
    validate_config(cfg)


def test_tippfehler_in_source(cfg):
    kaputt = copy.deepcopy(cfg)
    kaputt["vpr"]["clip_pca512"]["source"] = "clpi"
    with pytest.raises(ValueError, match="clpi"):
        validate_config(kaputt)


def test_unbekannter_adapter(cfg):
    kaputt = copy.deepcopy(cfg)
    kaputt["vpr"]["adapter"] = "mlp"
    with pytest.raises(ValueError, match="adapter"):
        validate_config(kaputt)


def test_localization_top_k_groesser_als_retrieval(cfg):
    kaputt = copy.deepcopy(cfg)
    kaputt["localization"]["top_k"] = kaputt["retrieval"]["top_k"] + 1
    with pytest.raises(ValueError, match="localization.top_k"):
        validate_config(kaputt)


def test_abgeleitete_bloecke_vollstaendig(cfg):
    """Die YAML-Anker muessen in jedem abgeleiteten Block dieselben Felder ergeben."""
    for name, block in cfg["vpr"].items():
        if isinstance(block, dict) and "source" in block:
            assert {"source", "pca_dim", "fit_images", "whiten"} <= set(block), name
