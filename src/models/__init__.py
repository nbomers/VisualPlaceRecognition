"""
Encoder und Adapter.

`LinearAdapter` wird bewusst erst beim Zugriff importiert: er haengt an torch,
und `src/models/derived.py` (PCA-/Whitening-Projektion, reines numpy) soll
ohne torch ladbar bleiben -- sonst zieht jeder Import aus diesem Paket die
Torch-Laufzeit nach und `pytest tests/` laeuft nicht mehr ohne sie.
"""

__all__ = ["LinearAdapter"]


def __getattr__(name):
    if name == "LinearAdapter":
        from .adapter import LinearAdapter

        return LinearAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
