"""
Die faiss-Pruefungen fuer blockwise_search -- als eigenes Skript, damit sie
in einem eigenen Prozess laufen.

faiss und torch bringen auf macOS jeweils ihr eigenes OpenMP mit. Beide in
denselben Prozess zu laden, beendet ihn mit einem Segmentation Fault. In der
Testsuite laedt tests/test_adapter_training.py torch, und der erste
faiss-Test danach riss alles mit -- ohne Traceback, ohne Testnamen. In der
Pipeline faellt das nie auf: 04 laedt torch, 06 laedt faiss, das sind zwei
Prozesse. Genau das stellt dieses Skript wieder her.

Von Hand aufrufbar:  python tests/blockwise_check.py

Rueckgabewert 0 = alles gut, 77 = faiss fehlt (der Test ueberspringt dann),
alles andere = Fehler, Meldung auf stderr.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _normiert(rng, n, dim):
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def pruefe_gegen_einen_index(faiss):
    """Geblockt muss dasselbe herauskommen wie mit einem Index ueber alles."""
    from src.retrieval import blockwise_search

    rng = np.random.default_rng(0)
    emb = _normiert(rng, 2000, 16)
    ref_rows = np.sort(rng.choice(2000, 1200, replace=False))
    q_rows = np.sort(rng.choice(2000, 150, replace=False))
    top_k = 10

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.ascontiguousarray(emb[ref_rows]))
    sim_ref, idx_ref = index.search(np.ascontiguousarray(emb[q_rows]), top_k)

    for block_ref in (37, 256, 4096):     # viele, wenige, ein einziger Block
        idx, sim = blockwise_search(emb, ref_rows, q_rows, top_k,
                                    block_ref=block_ref, block_q=11)
        assert np.allclose(sim, sim_ref, atol=1e-5), f"Aehnlichkeit weicht ab (Block {block_ref})"
        # Ueber die Zeilen vergleichen, nicht ueber die Positionen: bei
        # gleicher Aehnlichkeit darf die Reihenfolge abweichen.
        gleich = (ref_rows[idx] == ref_rows[idx_ref]).mean()
        assert gleich > 0.999, f"Treffer weichen ab (Block {block_ref}, {gleich:.4f})"


def pruefe_kleine_referenz(faiss):
    """Weniger Referenzbilder als top_k: keine Platzhalter im Ergebnis."""
    from src.retrieval import blockwise_search

    rng = np.random.default_rng(1)
    emb = _normiert(rng, 200, 8)
    ref_rows = np.array([3, 17, 42, 99])
    q_rows = np.arange(20)

    idx, sim = blockwise_search(emb, ref_rows, q_rows, top_k=10)

    # Frueher standen die uebrigen Spalten auf Aehnlichkeit -inf, aber Index 0
    # -- ein Treffer auf ref_rows[0], den der Aufrufer in den Recall gezaehlt
    # haette. Jetzt ist das Ergebnis so breit wie es Referenzbilder gibt.
    assert idx.shape == (len(q_rows), len(ref_rows)), f"Form {idx.shape}"
    assert np.isfinite(sim).all(), "es steht noch -inf im Ergebnis"
    assert np.array_equal(np.sort(ref_rows[idx], axis=1),
                          np.tile(np.sort(ref_rows), (len(q_rows), 1)))


def main():
    try:
        import faiss
    except ImportError:
        print("faiss nicht installiert", file=sys.stderr)
        return 77

    pruefe_gegen_einen_index(faiss)
    pruefe_kleine_referenz(faiss)
    print("blockwise_search: beide Pruefungen bestanden")
    return 0


if __name__ == "__main__":
    sys.exit(main())
