"""Die Lesart der Adapter-Diagnose -- ohne Torch, ohne Embeddings."""

import math
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import adapter_diagnose as ad  # noqa: E402


def test_schon_auf_val_schlechter():
    # Die Identitaet schlaegt die beste Epoche schon auf val: 05 haette nicht trainieren sollen.
    assert ad.lesart(0.70, 0.65, 0.568, 0.442) == "schon auf val schlechter"
    assert ad.lesart(0.70, 0.70, 0.568, 0.442) == "schon auf val schlechter"


def test_val_steigt_test_faellt():
    assert ad.lesart(0.65, 0.70, 0.568, 0.442) == "val steigt, test faellt"


def test_adapter_hilft_im_test():
    # CLIP: der Adapter hilft -- dann gibt es keinen Schaden zu erklaeren.
    assert ad.lesart(0.30, 0.35, 0.073, 0.123) == "adapter hilft im test"


def test_fehlende_werte():
    assert ad.lesart(0.70, None, 0.568, 0.442) == "unvollstaendig"
    assert ad.lesart(float("nan"), 0.65, 0.568, 0.442) == "unvollstaendig"


def test_nan_wird_none():
    # NaN ist kein gueltiges JSON (tests/test_ergebnis_json.py).
    assert ad._zahl(float("nan")) is None
    assert ad._zahl(None) is None
    assert math.isclose(ad._zahl(0.5), 0.5)
