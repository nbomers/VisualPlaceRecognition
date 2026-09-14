"""Paare aus den Metadaten: Sequenz- und Blickrichtungsfilter, Radius."""

import pandas as pd

from src.pairs import query_database_pairs, train_positive_pairs

LAT0, LON0 = 52.0, 8.0
M_PER_DEG = 111_320.0


def _lat(meter):
    return LAT0 + meter / M_PER_DEG


def test_train_positive_pairs_filter():
    meta = pd.DataFrame({
        "image_id": [1, 2, 3, 4, 5],
        "split": ["train"] * 5,
        "sequence_id": ["a", "b", "a", "c", "c"],
        "lat": [_lat(0), _lat(5), _lat(6), _lat(8), _lat(500)],
        "lon": [LON0] * 5,
        "compass_angle": [0.0, 10.0, 0.0, 200.0, 0.0],
    })
    paare = train_positive_pairs(meta, positive_radius_m=10.0, max_heading_diff_deg=90.0)
    gefunden = {tuple(sorted(p)) for p in zip(paare["anchor_image_id"], paare["positive_image_id"])}
    # 1-3 gleiche Sequenz -> raus; 1-4, 2-4, 3-4 Blickrichtung 200 Grad -> raus;
    # 5 ist 500 m entfernt. Bleiben 1-2 (5 m) und 2-3 (1 m).
    assert gefunden == {(1, 2), (2, 3)}
    assert paare["distance_m"].between(0.9, 5.1).all()


def test_train_positive_pairs_unbekannter_kompass_bleibt():
    meta = pd.DataFrame({
        "image_id": [1, 2],
        "split": ["train", "train"],
        "sequence_id": ["a", "b"],
        "lat": [_lat(0), _lat(3)],
        "lon": [LON0, LON0],
        "compass_angle": [-1.0, 90.0],
    })
    assert len(train_positive_pairs(meta, 10.0, 90.0)) == 1


def test_query_database_pairs_radius():
    meta = pd.DataFrame({
        "image_id": [10, 11, 20, 21],
        "split": ["database", "database", "query", "query"],
        "sequence_id": ["d", "d", "q", "q"],
        "lat": [_lat(0), _lat(100), _lat(4), _lat(60)],
        "lon": [LON0] * 4,
        "compass_angle": [0.0] * 4,
    })
    paare = query_database_pairs(meta, radius_m=25.0)
    assert list(zip(paare["query_image_id"], paare["database_image_id"])) == [(20, 10)]
    assert 3.9 < paare["distance_m"].iloc[0] < 4.1
    assert query_database_pairs(meta, radius_m=50.0).query_image_id.tolist() == [20, 21]
