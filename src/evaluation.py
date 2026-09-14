"""
Recall-Auswertung einer Trefferliste -- die Rechnung aus 07, als Funktion.

Jedes Verfahren, das eine Trefferliste erzeugt -- 06, aber auch ein
Experiment, das Trefferlisten umsortiert oder ueber Sequenzen aufsummiert --
muss genau gleich bewertet werden, sonst sind die Zeilen in compare.py nicht
vergleichbar. Deshalb steht die Auswertung hier und nicht mehr in einer
Notebook-Zelle.

    befunde = standard_evaluations(indices, query_meta, db_meta, cfg)
    write_evaluation(pfad, cfg, name, dim, len(db_meta), befunde)

Vier Auswertungen, jede eine andere Ground Truth:
  Alle Queries          Treffer zaehlt, wenn er innerhalb der Schwelle liegt
  Nicht-Panorama        nur wenn der Datensatz Panoramen enthaelt
  Hard                  Treffer nur von anderem Fotografen oder > N Tage entfernt
  Blickrichtung         Treffer nur, wenn auch der Kompass passt
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from scipy.spatial import cKDTree

from .geo import haversine_distance, heading_difference, to_metric_xy
from .run_guard import code_version, short_hash


def evaluate_retrieval(
    retrieved_indices,
    query_metadata,
    database_metadata,
    cfg,
    label="Alle Queries",
    query_filter=None,
    gt_filter=None,
    rng=None,
    block=256,
    verbose=True,
):
    """
    Recall@k je Distanzschwelle fuer eine Trefferliste.

    retrieved_indices : (n_query, k_max) Zeilennummern in database_metadata
    query_filter      : bool-Array ueber Queries -- welche zaehlen mit
    gt_filter         : Funktion(qi, di) -> bool-Array ueber Paare (Query-Zeile,
                        DB-Zeile), zusaetzliche Bedingung dafuer, dass ein
                        DB-Bild als Referenz zaehlt
    rng               : fuer die Zufallsbasis; wird geteilt, wenn mehrere
                        Auswertungen nacheinander laufen

    Distanzen nur dort, wo sie zaehlen: Kandidaten innerhalb der groessten
    Schwelle kommen aus einem KDTree in UTM-Metern (mit Sicherheitsaufschlag,
    damit kein Haversine-Treffer verloren geht), die exakte Haversine-Distanz
    dann nur fuer diese Paare und fuer die Trefferliste. Das Ergebnis ist
    dasselbe wie mit der vollen Distanzmatrix, bei 279k Referenzbildern aber
    Sekunden statt einer Viertelstunde.
    """
    thresholds = cfg["retrieval"]["thresholds"]
    k_values = cfg["retrieval"]["k_values"]
    k_max = max(k_values)
    if k_max > retrieved_indices.shape[1]:
        raise ValueError(
            f"Nur {retrieved_indices.shape[1]} Treffer je Anfrage, "
            f"Recall@{k_max} unmoeglich"
        )
    if rng is None:
        rng = np.random.default_rng(int(cfg["vpr"]["split_seed"]))

    db_lat = database_metadata["lat"].to_numpy()
    db_lon = database_metadata["lon"].to_numpy()
    q_lat = query_metadata["lat"].to_numpy()
    q_lon = query_metadata["lon"].to_numpy()
    n_db = len(database_metadata)

    db_xy, crs = to_metric_xy(db_lat, db_lon)
    q_xy, _ = to_metric_xy(q_lat, q_lon, crs=crs)
    baum = cKDTree(db_xy)
    # UTM und Haversine weichen hier um unter 0,3 % voneinander ab; der
    # Aufschlag haelt jeden Haversine-Treffer unter der Schwelle im Kandidatenkreis.
    radius = max(thresholds) * 1.01 + 2.0

    auswahl = (
        np.flatnonzero(query_filter)
        if query_filter is not None
        else np.arange(len(query_metadata))
    )

    n_localizable = {t: 0 for t in thresholds}
    hits = {(t, k): 0 for t in thresholds for k in k_values}
    zufall = {(t, k): 0 for t in thresholds for k in k_values}

    def distanz(qi_paare, di_paare):
        d = haversine_distance(q_lat[qi_paare], q_lon[qi_paare], db_lat[di_paare], db_lon[di_paare])
        if gt_filter is not None:
            # Ausgeschlossene Referenzen auf unendlich: sie fallen aus jeder Schwelle.
            d = np.where(gt_filter(qi_paare, di_paare), d, np.inf)
        return d

    for start in tqdm(range(0, len(auswahl), block), desc=label,
                      leave=False, disable=not verbose):
        qi = auswahl[start : start + block]

        # Naechste Referenz je Anfrage, exakt ueber die Kandidaten aus dem Baum.
        listen = baum.query_ball_point(q_xy[qi], r=radius)
        laengen = np.fromiter((len(n) for n in listen), dtype=np.int64, count=len(listen))
        d_min = np.full(len(qi), np.inf)
        if laengen.sum():
            lokal = np.repeat(np.arange(len(qi)), laengen)
            di = np.concatenate([np.asarray(n, dtype=np.int64) for n in listen if len(n)])
            np.minimum.at(d_min, lokal, distanz(qi[lokal], di))

        idx = retrieved_indices[qi, :k_max]
        d_top = distanz(np.repeat(qi, k_max), idx.ravel()).reshape(len(qi), k_max)
        # Zufallsbasis: dieselbe Rechnung mit blind gezogenen Datenbankbildern.
        idx_z = rng.integers(0, n_db, size=(len(qi), k_max))
        d_zufall = distanz(np.repeat(qi, k_max), idx_z.ravel()).reshape(len(qi), k_max)

        for t in thresholds:
            loesbar = d_min <= t
            n_localizable[t] += int(loesbar.sum())
            for k in k_values:
                hits[(t, k)] += int(((d_top[:, :k] <= t).any(axis=1) & loesbar).sum())
                zufall[(t, k)] += int(((d_zufall[:, :k] <= t).any(axis=1) & loesbar).sum())

    n_queries = len(auswahl)
    befund = {
        "n_queries": n_queries,
        "schwellen": {
            str(t): {
                "loesbar": n_localizable[t],
                "recall": {
                    str(k): (hits[(t, k)] / n_localizable[t] if n_localizable[t] else None)
                    for k in k_values
                },
                "zufall": {
                    str(k): (zufall[(t, k)] / n_localizable[t] if n_localizable[t] else None)
                    for k in k_values
                },
            }
            for t in thresholds
        },
    }

    if verbose:
        print(f"\n{label}   (Queries: {n_queries:,})")
        print(
            f"{'Schwelle':>10} {'loesbar':>10} {'Anteil':>8} "
            + " ".join(f"R@{k:<5}" for k in k_values)
        )
        for t in thresholds:
            n = n_localizable[t]
            frac = n / n_queries * 100 if n_queries else 0.0
            vals = " ".join(
                f"{hits[(t, k)] / n:<7.3f}" if n else f"{'-':<7}" for k in k_values
            )
            print(f"{t:>8} m {n:>10,} {frac:>7.1f}% {vals}")
        mitte = thresholds[len(thresholds) // 2]
        z = " ".join(
            f"{zufall[(mitte, k)] / n_localizable[mitte]:<7.4f}"
            if n_localizable[mitte] else f"{'-':<7}"
            for k in k_values
        )
        print(f"{'Zufall':>10} {'':>10} {'':>8} {z}   (bei {mitte} m)")

    return befund


def standard_evaluations(retrieved_indices, query_metadata, database_metadata, cfg,
                         verbose=True):
    """
    Die vier Auswertungen aus 07, in fester Reihenfolge und mit geteiltem
    Zufallsgenerator -- damit dieselbe Trefferliste dieselbe JSON ergibt.
    """
    q_pano = query_metadata["is_pano"].to_numpy().astype(bool)
    q_creator = query_metadata["creator_id"].to_numpy()
    q_time = query_metadata["captured_at"].to_numpy().astype("int64")
    q_heading = query_metadata["compass_angle"].to_numpy().astype("float64")
    db_creator = database_metadata["creator_id"].to_numpy()
    db_time = database_metadata["captured_at"].to_numpy().astype("int64")
    db_heading = database_metadata["compass_angle"].to_numpy().astype("float64")

    min_days_apart = float(cfg["retrieval"]["min_days_apart"])
    max_heading_diff = float(cfg["vpr"]["max_heading_diff_deg"])
    rng = np.random.default_rng(int(cfg["vpr"]["split_seed"]))

    def run(label, **kw):
        return evaluate_retrieval(retrieved_indices, query_metadata, database_metadata,
                                  cfg, label=label, rng=rng, verbose=verbose, **kw)

    def disjoint(qi, di):
        dt_days = np.abs(q_time[qi] - db_time[di]) / 86_400_000.0
        return (db_creator[di] != q_creator[qi]) | (dt_days > min_days_apart)

    # Zwei Bilder 5 m auseinander, die in entgegengesetzte Richtungen schauen,
    # haben keinen gemeinsamen Bildinhalt -- geometrisch "richtig", visuell
    # unmoeglich. Faellt "loesbar" gegenueber der Standardauswertung, sind das
    # die Anfragen, die kein Encoder loesen kann.
    def heading_ok(qi, di):
        return heading_difference(q_heading[qi], db_heading[di]) <= max_heading_diff

    befunde = {}
    befunde["Alle Queries"] = run("Alle Queries")
    if q_pano.any():
        befunde["Nur Nicht-Panorama-Queries"] = run(
            "Nur Nicht-Panorama-Queries", query_filter=~q_pano
        )
    elif verbose:
        print("\nKeine Panorama-Queries im Datensatz -- Aufteilung entfaellt.")
    label = f"Hard: anderer creator_id ODER > {min_days_apart:g} Tage Abstand"
    befunde[label] = run(label, gt_filter=disjoint)
    label = f"Blickrichtung: Treffer nur bei <= {max_heading_diff:g} Grad Abweichung"
    befunde[label] = run(label, gt_filter=heading_ok)
    return befunde


def write_evaluation(path, cfg, embedding_name, dim, n_database, befunde,
                     variant=None, fingerprint=None, root=None, **extra):
    """
    JSON im Format von 07 -- das, was compare.py liest.

    variant      die Zeile in compare.py: "none", "linear", "seq3", ...;
                 Standard ist der Adapter. Der Adapter selbst bleibt getrennt.
    fingerprint  der Embedding-Fingerabdruck; sein Hash und die Code-Kennung
                 sagen run.py, ob die JSON noch zur Trefferliste passt.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adapter = cfg["vpr"].get("adapter", "none")
    payload = {
        "datum": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "method": cfg["vpr"]["method"],
        "adapter": adapter,
        "variant": variant or adapter,
        "embedding_name": embedding_name,
        "dim": int(dim),
        "n_database": int(n_database),
        "fingerprint_hash": short_hash(fingerprint) if fingerprint is not None else None,
        "code_version": code_version(root) if root is not None else None,
        **extra,
        "auswertungen": befunde,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
