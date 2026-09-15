"""
Geografische Bildpaare aus den Metadaten -- in Sekunden ableitbar, deshalb
nie als Datei versioniert.

Zwei Sorten:
  train_positive_pairs    Anchor/Positive fuer das Adaptertraining in 05:
                          zwei train-Bilder innerhalb positive_radius_m,
                          aus verschiedenen Sequenzen, Blickrichtung passt
  query_database_pairs    Query -> Datenbankbild innerhalb eines Radius,
                          fuer den Datensatz-Audit in 02

Abstaende in UTM-Metern (cKDTree), nicht Haversine: ein 10-m-Radius ist so
exakt genug, und die Baumsuche ist um Groessenordnungen schneller als eine
Distanzmatrix.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import heading_matches, to_metric_xy


def train_positive_pairs(metadata, positive_radius_m, max_heading_diff_deg):
    """
    DataFrame mit anchor_image_id, positive_image_id, distance_m.

    Paare aus derselben Sequenz sind ausgeschlossen: aufeinanderfolgende
    Frames sind Beinahe-Duplikate, der Adapter lernte sonst nur, dass ein
    Bild sich selbst aehnelt.
    """
    trn = metadata[metadata["split"] == "train"].reset_index(drop=True)
    xy, _ = to_metric_xy(trn["lat"].to_numpy(), trn["lon"].to_numpy())
    seq = trn["sequence_id"].to_numpy()
    ids = trn["image_id"].to_numpy()

    pairs = cKDTree(xy).query_pairs(r=float(positive_radius_m), output_type="ndarray")
    pairs = pairs[seq[pairs[:, 0]] != seq[pairs[:, 1]]]

    # Unbekannte Blickrichtung verwirft kein Paar -- siehe geo.heading_matches.
    ang = trn["compass_angle"].to_numpy(dtype=float)
    pairs = pairs[heading_matches(ang[pairs[:, 0]], ang[pairs[:, 1]],
                                  max_heading_diff_deg)]

    dist = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
    return pd.DataFrame({
        "anchor_image_id": ids[pairs[:, 0]],
        "positive_image_id": ids[pairs[:, 1]],
        "distance_m": dist.round(3),
    })


def query_database_pairs(metadata, radius_m):
    """DataFrame mit query_image_id, database_image_id, distance_m fuer alle Paare im Radius."""
    db = metadata[metadata["split"] == "database"].reset_index(drop=True)
    q = metadata[metadata["split"] == "query"].reset_index(drop=True)
    db_xy, crs = to_metric_xy(db["lat"].to_numpy(), db["lon"].to_numpy())
    q_xy, _ = to_metric_xy(q["lat"].to_numpy(), q["lon"].to_numpy(), crs=crs)

    listen = cKDTree(db_xy).query_ball_point(q_xy, r=float(radius_m))
    qi = np.repeat(np.arange(len(q)), [len(n) for n in listen])
    di = np.concatenate([np.asarray(n, dtype=int) for n in listen]) if len(qi) else np.array([], int)
    dist = np.linalg.norm(q_xy[qi] - db_xy[di], axis=1)
    return pd.DataFrame({
        "query_image_id": q["image_id"].to_numpy()[qi],
        "database_image_id": db["image_id"].to_numpy()[di],
        "distance_m": dist.round(3),
    })
