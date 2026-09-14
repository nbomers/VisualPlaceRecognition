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

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .geo import haversine_distance, heading_difference


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
    gt_filter         : Funktion(qi_block) -> bool-Matrix (len(block), n_database),
                        zusaetzliche Bedingung dafuer, dass ein DB-Bild zaehlt
    rng               : fuer die Zufallsbasis; wird geteilt, wenn mehrere
                        Auswertungen nacheinander laufen

    Blockweise statt je Query: die Distanzmatrix eines Blocks entsteht in
    einem numpy-Aufruf. Bei 256 Queries x 48k Datenbankbildern sind das rund
    100 MB je Block.
    """
    thresholds = cfg["retrieval"]["thresholds"]
    k_values = cfg["retrieval"]["k_values"]
    if max(k_values) > retrieved_indices.shape[1]:
        raise ValueError(
            f"Nur {retrieved_indices.shape[1]} Treffer je Anfrage, "
            f"Recall@{max(k_values)} unmoeglich"
        )
    if rng is None:
        rng = np.random.default_rng(int(cfg["vpr"]["split_seed"]))

    db_lat = database_metadata["lat"].to_numpy()
    db_lon = database_metadata["lon"].to_numpy()
    q_lat = query_metadata["lat"].to_numpy()
    q_lon = query_metadata["lon"].to_numpy()

    auswahl = (
        np.flatnonzero(query_filter)
        if query_filter is not None
        else np.arange(len(query_metadata))
    )

    n_localizable = {t: 0 for t in thresholds}
    hits = {(t, k): 0 for t in thresholds for k in k_values}
    zufall = {(t, k): 0 for t in thresholds for k in k_values}

    for start in tqdm(range(0, len(auswahl), block), desc=label,
                      leave=False, disable=not verbose):
        qi = auswahl[start : start + block]

        d = haversine_distance(
            q_lat[qi, None], q_lon[qi, None], db_lat[None, :], db_lon[None, :]
        )
        if gt_filter is not None:
            # Ausgeschlossene Treffer auf unendlich setzen: sie fallen damit
            # aus jeder Schwelle heraus, ohne dass eine zweite Maske noetig ist.
            d = np.where(gt_filter(qi), d, np.inf)

        d_top = np.take_along_axis(d, retrieved_indices[qi], axis=1)
        # Zufallsbasis: dieselbe Rechnung mit blind gezogenen Datenbankbildern.
        d_zufall = np.take_along_axis(
            d, rng.integers(0, d.shape[1], size=(len(qi), max(k_values))), axis=1
        )

        for t in thresholds:
            loesbar = (d <= t).any(axis=1)
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

    def disjoint(qi):
        dt_days = np.abs(q_time[qi][:, None] - db_time[None, :]) / 86_400_000.0
        return (db_creator[None, :] != q_creator[qi][:, None]) | (dt_days > min_days_apart)

    # Zwei Bilder 5 m auseinander, die in entgegengesetzte Richtungen schauen,
    # haben keinen gemeinsamen Bildinhalt -- geometrisch "richtig", visuell
    # unmoeglich. Faellt "loesbar" gegenueber der Standardauswertung, sind das
    # die Anfragen, die kein Encoder loesen kann.
    def heading_ok(qi):
        return heading_difference(q_heading[qi][:, None], db_heading[None, :]) <= max_heading_diff

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


def write_evaluation(path, cfg, embedding_name, dim, n_database, befunde, **extra):
    """JSON im Format von 07 -- das, was compare.py liest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "datum": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "method": cfg["vpr"]["method"],
        "adapter": cfg["vpr"].get("adapter", "none"),
        "embedding_name": embedding_name,
        "dim": int(dim),
        "n_database": int(n_database),
        **extra,
        "auswertungen": befunde,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
