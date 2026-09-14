"""
Sequenzbasierter Split -- die wichtigste Entscheidung des Benchmarks, als
Funktion statt als Notebook-Zelle, damit ein Test sie festnageln kann.

Die Split-Einheit ist die Sequenz (eine Fahrt), nie das Einzelbild:
aufeinanderfolgende Frames derselben Fahrt in database UND query waeren
Leakage. Liegen die drei Listen unter data/processed/ schon vor, werden sie
uebernommen statt neu gewuerfelt -- Mapillary aendert sich ueber die Zeit,
derselbe Seed ergaebe sonst einen anderen Split und alle Embeddings waeren
wertlos.

    train, database, query, herkunft = split_sequences(seq_ids, processed_dir, cfg)
    write_split_lists(processed_dir, train, database, query)
"""

import random

LISTEN = ("train_sequences.txt", "database_sequences.txt", "query_sequences.txt")


def draw_split(sequence_ids, seed, train_fraction, database_fraction):
    """
    Sequenzen mit festem Seed in train / database / query wuerfeln.
    Query bekommt den Rest, damit sich die gerundeten Anteile exakt zu allen
    Sequenzen ergaenzen.
    """
    seq_ids = sorted(str(s) for s in sequence_ids)
    if len(seq_ids) < 3:
        raise RuntimeError(
            f"Fuer train/database/query werden mindestens 3 Sequenzen benoetigt; "
            f"gefunden: {len(seq_ids)}"
        )
    rng = random.Random(seed)
    rng.shuffle(seq_ids)
    n = len(seq_ids)
    n_train = max(1, round(n * train_fraction))
    n_database = max(1, round(n * database_fraction))
    if n_train + n_database >= n:
        n_database = max(1, n - n_train - 1)
    return (seq_ids[:n_train],
            seq_ids[n_train:n_train + n_database],
            seq_ids[n_train + n_database:])


def read_split_lists(processed_dir):
    """Die drei Listen, oder None, wenn eine fehlt."""
    pfade = [processed_dir / name for name in LISTEN]
    if not all(p.exists() for p in pfade):
        return None
    return tuple(p.read_text().split() for p in pfade)


def write_split_lists(processed_dir, train, database, query):
    for name, liste in zip(LISTEN, (train, database, query)):
        (processed_dir / name).write_text("\n".join(liste) + "\n")


def split_sequences(sequence_ids, processed_dir, cfg, min_overlap=0.5):
    """
    Vorhandene Listen uebernehmen, sonst wuerfeln. Gibt (train, database,
    query, herkunft) zurueck; herkunft ist "uebernommen" oder "gewuerfelt".

    Die Schranke min_overlap faengt Listen aus einem anderen Datensatz ab --
    etwa nach einem Wechsel von city: sie wuerden sonst alle Sequenzen
    wegfiltern und einen leeren Datensatz erzeugen, ohne dass etwas auffaellt.
    """
    bekannt = set(str(s) for s in sequence_ids)
    vorhanden = read_split_lists(processed_dir)
    if vorhanden is None:
        vpr = cfg["vpr"]
        train, database, query = draw_split(
            bekannt, int(vpr["split_seed"]),
            float(vpr["train_fraction"]), float(vpr["database_fraction"]),
        )
        return train, database, query, "gewuerfelt"

    uebernommen = {s for gruppe in vorhanden for s in gruppe}
    anteil = len(uebernommen & bekannt) / max(len(uebernommen), 1)
    if anteil < min_overlap:
        raise RuntimeError(
            f"Nur {anteil:.0%} der {len(uebernommen):,} gespeicherten Sequenzen kommen "
            f"in den geladenen Daten vor. Die Listen in {processed_dir} gehoeren "
            "offenbar zu einem anderen Datensatz (andere Stadt?). Fuer einen "
            "bewussten Neuaufbau die drei *_sequences.txt loeschen."
        )
    train, database, query = ([s for s in gruppe if s in bekannt] for gruppe in vorhanden)
    return train, database, query, "uebernommen"


def split_column(sequence_ids, train, database, query):
    """Je Eintrag "train" / "database" / "query", None fuer unbekannte Sequenzen."""
    zuordnung = ({s: "train" for s in train}
                 | {s: "database" for s in database}
                 | {s: "query" for s in query})
    return [zuordnung.get(str(s)) for s in sequence_ids]
