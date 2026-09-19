"""
Fehlende Split-Listen muessen sagen, welche Datei fehlt und wer sie erzeugt.

Vorher entpackte 02 das None aus read_split_lists direkt und meldete
`TypeError: cannot unpack non-iterable NoneType object` -- eine Meldung, aus
der niemand ableitet, dass 01 noch laufen muss.
"""

import pytest

from src.split import LISTEN, read_split_lists, require_split_lists, write_split_lists


def test_read_gibt_none_wenn_eine_liste_fehlt(tmp_path):
    write_split_lists(tmp_path, ["a"], ["b"], ["c"])
    assert read_split_lists(tmp_path) == (["a"], ["b"], ["c"])
    (tmp_path / LISTEN[1]).unlink()
    assert read_split_lists(tmp_path) is None


def test_require_nennt_die_fehlenden_dateien_und_den_befehl(tmp_path):
    write_split_lists(tmp_path, ["a"], ["b"], ["c"])
    (tmp_path / LISTEN[1]).unlink()
    with pytest.raises(FileNotFoundError) as fehler:
        require_split_lists(tmp_path)
    text = str(fehler.value)
    assert LISTEN[1] in text, text
    assert LISTEN[0] not in text, text          # die vorhandene nicht melden
    assert "run.py --from 01" in text, text


def test_require_gibt_die_listen_zurueck_wenn_alle_da_sind(tmp_path):
    write_split_lists(tmp_path, ["a", "b"], ["c"], ["d"])
    assert require_split_lists(tmp_path) == (["a", "b"], ["c"], ["d"])
