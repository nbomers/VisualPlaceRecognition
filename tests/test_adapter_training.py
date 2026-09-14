"""Triplet-Dataset und Trainingsschleife auf synthetischen Daten -- CPU, Sekunden."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.adapter_training import (  # noqa: E402
    TripletDataset,
    make_loader,
    split_fit_val,
    train_adapter,
    val_halves,
    val_recall_at_1,
)
from src.geo import haversine_distance  # noqa: E402
from src.models.adapter import LinearAdapter  # noqa: E402

LAT0, LON0, M_PER_DEG = 52.0, 8.0, 111_320.0


def _daten(n_seq=12, je=20, seed=0):
    """Fahrten als Ketten von Punkten, 3 m Abstand, alle in train."""
    rng = np.random.default_rng(seed)
    zeilen = []
    for s in range(n_seq):
        x0, y0 = rng.uniform(0, 2000, 2)
        for i in range(je):
            zeilen.append({"image_id": s * 1000 + i, "sequence_id": f"s{s}", "split": "train",
                           "lat": LAT0 + (y0 + 3 * i) / M_PER_DEG, "lon": LON0 + x0 / M_PER_DEG,
                           "compass_angle": 0.0, "captured_at": 0, "creator_id": 1, "is_pano": False})
    meta = pd.DataFrame(zeilen)
    emb = rng.standard_normal((len(meta), 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return meta, emb


def test_negative_liegen_ausserhalb_der_unsicherheitszone():
    meta, emb = _daten()
    # Paare: benachbarte Frames verschiedener "Fahrten" -- hier konstruiert
    # als (i, i+1) innerhalb derselben Kette, das Dataset filtert nicht
    paare = [(i, i + 1, 3.0) for i in range(0, len(meta) - 1, 7)]
    ds = TripletDataset(emb, paare, meta, positive_radius_m=10.0, uncertain_radius_m=25.0,
                        hard_negative_min_m=25.0, hard_negative_max_m=100.0)
    lat, lon = meta["lat"].to_numpy(), meta["lon"].to_numpy()
    np.random.seed(1)
    for a, _, _ in paare[:10]:
        neg = [ds.sample_negative(a) for _ in range(50)]
        d = haversine_distance(lat[a], lon[a], lat[neg], lon[neg])
        assert d.min() > 25.0
    a, p, n = ds[0]
    assert a.shape == p.shape == n.shape == (16,)


def test_split_und_training_laufen():
    meta, emb = _daten()
    fit, val = split_fit_val(meta, val_fraction=0.25, seed=42)
    assert fit.sum() + val.sum() == len(meta) and not (fit & val).any()
    seq = meta["sequence_id"]
    assert not (set(seq[fit]) & set(seq[val]))

    val_meta = meta[val].reset_index(drop=True)
    db_m, q_m = val_halves(val_meta)
    cfg = {"vpr": {"split_seed": 0}, "retrieval": {"thresholds": [25], "k_values": [1]}}
    adapter = LinearAdapter(16)
    recall, n = val_recall_at_1(adapter, emb[val][db_m], emb[val][q_m],
                                val_meta[db_m].reset_index(drop=True),
                                val_meta[q_m].reset_index(drop=True), cfg, 25.0, "cpu")
    assert n >= 0 and (np.isnan(recall) or 0.0 <= recall <= 1.0)

    fit_meta = meta[fit].reset_index(drop=True)
    paare = [(i, i + 1, 3.0) for i in range(0, len(fit_meta) - 1, 5)]
    ds = TripletDataset(emb[fit], paare, fit_meta, 10.0, 25.0, 25.0, 100.0)
    loader = make_loader(ds, batch_size=8, max_pairs_per_epoch=16)
    state, epoch, best, _ = train_adapter(
        adapter, loader, lambda m: (0.5, 10), epochs=2, learning_rate=1e-3,
        margin=0.2, device="cpu", patience=None, log=lambda *_: None,
    )
    assert epoch == 1 and best == 0.5 and "linear.weight" in state
