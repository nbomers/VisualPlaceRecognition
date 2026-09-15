"""
Trefferlisten aus 06 laden und je Anfrage bewerten -- die Bausteine, die
Bootstrap, Stadtteil-Auswertung und Verwechslungsatlas teilen.

    query, database, indices, sims = load_retrieval(ROOT, cfg, "megaloc")
    loesbar = localizable(query, database, 25.0)          # (n_query,) bool
    hits = hits_at_k(query, database, indices, loesbar, 25.0, [1, 5])

Loesbar haengt nur an den Koordinaten, nicht am Encoder, und wird deshalb
einmal je Split gerechnet und ueber image_id wiederverwendet -- die
Metadaten verschiedener Encoder halten dieselben Bilder in anderer
Zeilenreihenfolge. Treffer brauchen nur die Distanzen an den gespeicherten
Kandidaten, nicht die volle Matrix.
"""

import hashlib

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from .geo import haversine_distance, to_metric_xy
from .paths import Paths
from .run_guard import embedding_fingerprint, require_fingerprint


def retrieval_inputs(root, cfg, method, adapter="none", reference_splits=None):
    """
    Die beiden Dateien, die load_retrieval oeffnen wird: Metadaten und
    Trefferliste.

    Getrennt, damit ein Aufrufer vorab pruefen kann, ob ein Encoder lokal
    vorliegt, ohne die Namensbildung nachzubauen. Genau daran ist die
    Vorabpruefung in experiments/bootstrap_ci.py einmal gescheitert: die
    fullref-Zeilen haben KEINE eigene Metadatendatei -- sie nutzen die des
    Basis-Encoders, und nur die .npz traegt das _fullref-Suffix. Dasselbe
    gilt fuer die seq-Varianten, die ueberhaupt keine eigene .npz haben.
    """
    name = method if adapter in ("none", "None") else f"{method}_{adapter}"
    pfade = Paths(cfg, root)
    meta_datei = pfade.metadata_file(name, method)
    if reference_splits:
        name = f"{name}_fullref"
    return meta_datei, pfade.retrieval_file(name, method)


def load_retrieval(root, cfg, method, adapter="none", sequence_window=None,
                   reference_splits=None):
    """
    Trefferliste (indices, similarities) und Query-/Datenbank-Metadaten,
    Fingerabdruck geprueft. Mit sequence_window wird die Aggregation ueber
    Nachbarframes aus experiments/sequence_retrieval.py nachgerechnet -- die
    seq-Varianten haben keine eigene .npz. Mit reference_splits (etwa
    ["database", "train"]) kommt die Trefferliste aus
    experiments/full_reference.py, und "database" meint alle Referenzzeilen.
    """
    meta_datei, npz = retrieval_inputs(root, cfg, method, adapter,
                                       reference_splits=reference_splits)
    name = method if adapter in ("none", "None") else f"{method}_{adapter}"
    if not meta_datei.exists():
        # Embeddings sind gitignored und liegen je nach Rechner verteilt.
        # Die nackte FileNotFoundError von pandas sagt das nicht.
        raise FileNotFoundError(
            f"Keine Embeddings fuer {name!r} unter {meta_datei.parent}.\n"
            "Entweder wurde der Encoder hier nie gerechnet, oder er liegt auf "
            "einem anderen Rechner (Embeddings und Trefferlisten sind "
            "gitignored).\n"
            "  python run.py --bestand        zeigt, was hier vollstaendig ist\n"
            f"  python run.py --method {method}   rechnet ihn hier"
        )
    meta = pd.read_parquet(meta_datei)
    fingerprint = embedding_fingerprint(cfg, method, adapter, meta)
    if reference_splits:
        fingerprint = {**fingerprint, "reference_splits": list(reference_splits)}
    require_fingerprint(npz, fingerprint, "Retrieval-Ergebnis")
    r = np.load(npz)
    indices, similarities = r["indices"], r["similarities"]
    query = meta[meta.split == "query"].reset_index(drop=True)
    referenz = list(reference_splits) if reference_splits else ["database"]
    database = meta[meta.split.isin(referenz)].reset_index(drop=True)
    if sequence_window:
        indices, similarities = aggregate_sequence(
            indices, similarities, sequence_windows(query), int(sequence_window),
            indices.shape[1],
        )
    return query, database, indices, similarities


def _id_digest(ids):
    return hashlib.sha256(np.sort(np.asarray(ids, dtype=np.int64)).tobytes()).hexdigest()


_LOESBAR = {}


def localizable(query, database, threshold, block=4096):
    """
    Je Anfrage: liegt mindestens ein Datenbankbild innerhalb der Schwelle?

    Wie in src/evaluation.py: Kandidaten kommen aus einem KDTree in
    UTM-Metern, die exakte Haversine-Distanz wird nur fuer diese Paare
    gerechnet. Der Aufschlag auf den Suchradius (UTM und Haversine weichen um
    unter 0,3 % voneinander ab) haelt jeden Haversine-Treffer unter der
    Schwelle im Kandidatenkreis -- das Ergebnis ist identisch mit der vollen
    Distanzmatrix, bei 279k Referenzbildern aber Sekunden statt Minuten.
    """
    key = (_id_digest(query["image_id"]), _id_digest(database["image_id"]), float(threshold))
    if key not in _LOESBAR:
        q_lat, q_lon = query["lat"].to_numpy(), query["lon"].to_numpy()
        db_lat, db_lon = database["lat"].to_numpy(), database["lon"].to_numpy()
        db_xy, crs = to_metric_xy(db_lat, db_lon)
        q_xy, _ = to_metric_xy(q_lat, q_lon, crs=crs)
        baum = cKDTree(db_xy)
        radius = float(threshold) * 1.01 + 2.0

        loesbar = np.zeros(len(query), dtype=bool)
        for start in tqdm(range(0, len(query), block), desc="loesbar", leave=False):
            qi = np.arange(start, min(start + block, len(query)))
            listen = baum.query_ball_point(q_xy[qi], r=radius)
            laengen = np.fromiter((len(n) for n in listen), dtype=np.int64, count=len(listen))
            if not laengen.sum():
                continue
            lokal = np.repeat(qi, laengen)
            di = np.concatenate([np.asarray(n, dtype=np.int64) for n in listen if len(n)])
            d = haversine_distance(q_lat[lokal], q_lon[lokal], db_lat[di], db_lon[di])
            loesbar[lokal[d <= threshold]] = True
        _LOESBAR[key] = pd.Series(loesbar, index=query["image_id"].to_numpy())
    return _LOESBAR[key].reindex(query["image_id"].to_numpy()).to_numpy()


def top_distances(query, database, indices):
    """Distanz in Metern von jeder Anfrage zu jedem ihrer gespeicherten Treffer."""
    q_lat, q_lon = query["lat"].to_numpy(), query["lon"].to_numpy()
    db_lat, db_lon = database["lat"].to_numpy(), database["lon"].to_numpy()
    return haversine_distance(q_lat[:, None], q_lon[:, None], db_lat[indices], db_lon[indices])


def hits_at_k(query, database, indices, loesbar, threshold, k_values):
    """Je Anfrage und k: ist ein Treffer innerhalb der Schwelle unter den ersten k?"""
    d = top_distances(query, database, indices)
    return {k: (d[:, :k] <= threshold).any(axis=1) & loesbar for k in k_values}


# ----------------------------------------------------------------------
# Sequenz-Aggregation: die Trefferlisten benachbarter Frames stuetzen sich
# gegenseitig. Gemessen und unterlegen (experiments/README.md), bleibt aber
# nachrechenbar, weil die seq-Zeile in compare.py steht.
# ----------------------------------------------------------------------


def sequence_windows(query_metadata):
    """Je Query-Zeile: die Zeilen derselben Sequenz in zeitlicher Reihenfolge
    und die eigene Position darin."""
    order = np.lexsort((query_metadata["captured_at"].to_numpy(),
                        query_metadata["sequence_id"].to_numpy()))
    seq = query_metadata["sequence_id"].to_numpy()[order]
    grenzen = np.flatnonzero(np.r_[True, seq[1:] != seq[:-1], True])
    fenster = [None] * len(query_metadata)
    for a, b in zip(grenzen[:-1], grenzen[1:]):
        zeilen = order[a:b]
        for pos, zeile in enumerate(zeilen):
            fenster[zeile] = (zeilen, pos)
    return fenster


def aggregate_sequence(indices, similarities, fenster, w, top_k):
    """Neue Trefferliste je Query aus den dreiecksgewichteten Listen der +-w Nachbarn."""
    n, k_max = indices.shape
    neu_idx = np.empty((n, top_k), dtype=indices.dtype)
    neu_sim = np.empty((n, top_k), dtype=np.float32)
    gewichte = 1.0 - np.abs(np.arange(-w, w + 1)) / (w + 1)   # Dreieck, Mitte = 1

    for qi in tqdm(range(n), desc=f"Fenster +-{w}", leave=False):
        zeilen, pos = fenster[qi]
        lo, hi = max(0, pos - w), min(len(zeilen), pos + w + 1)
        nachbarn = zeilen[lo:hi]
        gw = gewichte[(lo - pos + w):(hi - pos + w)]

        kandidaten = indices[nachbarn].ravel()
        scores = (similarities[nachbarn] * gw[:, None]).ravel()
        uniq, inv = np.unique(kandidaten, return_inverse=True)
        summe = np.zeros(len(uniq), dtype=np.float64)
        np.add.at(summe, inv, scores)

        if len(uniq) > top_k:
            beste = np.argpartition(-summe, top_k - 1)[:top_k]
        else:
            beste = np.arange(len(uniq))
        beste = beste[np.argsort(-summe[beste])]
        m = len(beste)
        neu_idx[qi, :m] = uniq[beste]
        neu_sim[qi, :m] = summe[beste]
        if m < top_k:                      # kurze Sequenzen liefern weniger Kandidaten
            neu_idx[qi, m:] = uniq[beste[-1]]
            neu_sim[qi, m:] = -np.inf
    return neu_idx, neu_sim
