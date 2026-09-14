"""Der Split ist die wichtigste Entscheidung des Benchmarks -- hier wird sie festgenagelt."""

from pathlib import Path

import pandas as pd
import pytest

from src.split import draw_split, read_split_lists, split_column, split_sequences, write_split_lists

from src.config import load_config, paths  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = paths(load_config(ROOT), ROOT).processed


def _cfg():
    return {"vpr": {"split_seed": 42, "train_fraction": 0.7, "database_fraction": 0.15}}


def test_draw_split_deterministisch_und_disjunkt():
    seqs = [f"s{i}" for i in range(100)]
    a = draw_split(seqs, 42, 0.7, 0.15)
    b = draw_split(reversed(seqs), 42, 0.7, 0.15)      # Reihenfolge der Eingabe egal
    assert a == b
    train, database, query = a
    assert (len(train), len(database), len(query)) == (70, 15, 15)
    assert not (set(train) & set(database) | set(train) & set(query) | set(database) & set(query))
    assert draw_split(seqs, 43, 0.7, 0.15) != a


def test_split_sequences_wuerfelt_ohne_listen(tmp_path):
    train, database, query, herkunft = split_sequences([f"s{i}" for i in range(20)], tmp_path, _cfg())
    assert herkunft == "gewuerfelt"
    assert len(train) + len(database) + len(query) == 20


def test_split_sequences_uebernimmt_listen(tmp_path):
    write_split_lists(tmp_path, ["a", "b"], ["c"], ["d", "e"])
    train, database, query, herkunft = split_sequences(["a", "b", "c", "d", "e", "neu"], tmp_path, _cfg())
    assert herkunft == "uebernommen"
    assert (train, database, query) == (["a", "b"], ["c"], ["d", "e"])
    assert split_column(["a", "d", "neu"], train, database, query) == ["train", "query", None]
    assert read_split_lists(tmp_path) == (["a", "b"], ["c"], ["d", "e"])


def test_split_sequences_lehnt_fremde_listen_ab(tmp_path):
    write_split_lists(tmp_path, ["x1", "x2"], ["x3"], ["x4"])
    with pytest.raises(RuntimeError, match="anderen Datensatz"):
        split_sequences(["a", "b", "c"], tmp_path, _cfg())


@pytest.mark.skipif(not (PROCESSED / "metadata.parquet").exists(), reason="Metadaten fehlen")
def test_versionierter_split_reproduziert_sich():
    """Die drei Listen im Git entstehen aus metadata.parquet und Seed 42 -- exakt."""
    meta = pd.read_parquet(PROCESSED / "metadata.parquet")
    neu = draw_split(meta["sequence_id"].astype(str).unique(), 42, 0.7, 0.15)
    alt = read_split_lists(PROCESSED)
    assert alt is not None
    assert [list(x) for x in neu] == [list(x) for x in alt]
    zuordnung = dict(zip(meta["sequence_id"].astype(str), meta["split"]))
    for name, gruppe in zip(("train", "database", "query"), alt):
        assert all(zuordnung[s] == name for s in gruppe)
