"""Die Auswahlregel des Adapter-Rasters -- ohne Torch, ohne Embeddings."""

import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import adapter_sweep as sweep  # noqa: E402


def test_identitaet_bleibt_ohne_klaren_gewinn():
    # Nur ein Zuwachs ueber min_delta zaehlt -- sonst entschiede Rauschen.
    assert sweep.waehle(0.198, 0.132, 0.001) == "identitaet"
    assert sweep.waehle(0.198, 0.1985, 0.001) == "identitaet"


def test_adapter_bei_klarem_gewinn():
    assert sweep.waehle(0.050, 0.194, 0.001) == "adapter"
