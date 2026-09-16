"""
Sequenzbasiertes Retrieval: die Trefferlisten benachbarter Frames stuetzen
sich gegenseitig.

Die Anfragen sind keine Einzelbilder, sondern Fahrten -- 0,17 s und 3,3 m
zwischen aufeinanderfolgenden Frames. Fuer jede Anfrage werden die
Trefferlisten ihrer +-W Nachbarn in derselben Sequenz aufsummiert, mit
Dreiecksgewicht (der eigene Frame zaehlt voll, der W-te Nachbar fast nichts).
Ein Datenbankbild, das bei mehreren Nachbarn auftaucht, steigt; eines, das
nur bei einem auftaucht, faellt.

Das ist eine ANDERE Aufgabe als Einzelbild-Retrieval -- Sequenzlokalisierung
-- und wird deshalb als eigene Zeile berichtet, nicht als bessere Version
derselben. Bewertet wird trotzdem exakt wie in 07: gegen die echte Position
des mittleren Frames, mit denselben vier Ground-Truth-Varianten.

Reines Nachbearbeiten der .npz aus 06; kein Modell, keine GPU.

    python experiments/sequence_retrieval.py --method eigenplaces_pcaw512
    python experiments/sequence_retrieval.py --method megaloc --window 5
    python experiments/sequence_retrieval.py --method megaloc --windows 1,2,3,5,10

Fenstergroesse gegen Schwelle: bei 3,3 m je Frame reicht +-3 knapp 10 m
weit, +-5 gut 16 m -- beides unter den 25 m der Ground Truth. +-10 (33 m)
liegt darueber; dort misst man nicht mehr denselben Ort.
"""

import argparse

import pandas as pd

from _common import CFG, PATHS, ROOT
from src.evaluation import standard_evaluations, write_evaluation
from src.retrieval import aggregate_sequence, descriptor_dim, load_retrieval, sequence_windows
from src.run_guard import embedding_fingerprint


def _args():
    ap = argparse.ArgumentParser(
        description="Trefferlisten ueber benachbarte Frames einer Sequenz aufsummieren.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--method", default="eigenplaces_pcaw512")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--windows", default="3",
                    help="Kommaliste von Fenstergroessen W (+-W Frames), Standard 3")
    ap.add_argument("--quiet", action="store_true", help="Keine Zwischentabellen")
    return ap.parse_args()


def main():
    args = _args()
    method, adapter = args.method, args.adapter
    name = method if adapter in ("none", "None") else f"{method}_{adapter}"
    CFG["vpr"]["method"], CFG["vpr"]["adapter"] = method, adapter

    query, database, indices, similarities = load_retrieval(ROOT, CFG, method, adapter)
    top_k = indices.shape[1]
    emb_dir = PATHS.embedding_dir(method)
    dim = descriptor_dim(ROOT, CFG, method, adapter)
    fingerprint = embedding_fingerprint(
        CFG, method, adapter, pd.read_parquet(emb_dir / f"{name}_metadata.parquet"))
    fenster = sequence_windows(query)

    n_seq = query["sequence_id"].nunique()
    print(f"{name}: {len(query):,} Anfragen in {n_seq:,} Sequenzen, Top-{top_k} je Anfrage")

    basis = standard_evaluations(indices, query, database, CFG, verbose=False)
    b25 = basis["Alle Queries"]["schwellen"]["25"]["recall"]
    print(f"{'Fenster':>8}   {'R@1':>6} {'R@5':>6} {'R@10':>6}   {'dR@1':>6}")
    print(f"{'einzeln':>8}   {b25['1']:>6.3f} {b25['5']:>6.3f} {b25['10']:>6.3f}")

    fenstergroessen = [int(x) for x in args.windows.split(",") if x.strip()]
    # w = 0 wuerde eine seq0-Zeile erzeugen, die identisch zur Baseline ist --
    # kein Fehler, aber eine Zeile in compare.py, die nichts aussagt.
    if any(w < 1 for w in fenstergroessen):
        raise SystemExit(
            f"--windows braucht Werte ab 1, bekommen: {args.windows!r}.\n"
            "Ein Fenster von 0 Nachbarn ist die Einzelbild-Auswertung -- "
            "die steht schon als Baseline in compare.py."
        )
    if not fenstergroessen:
        raise SystemExit("--windows ist leer.")

    for w in fenstergroessen:
        neu_idx, neu_sim = aggregate_sequence(indices, similarities, fenster, w, top_k)
        befunde = standard_evaluations(neu_idx, query, database, CFG, verbose=not args.quiet)
        r25 = befunde["Alle Queries"]["schwellen"]["25"]["recall"]
        print(f"{'+-' + str(w):>8}   {r25['1']:>6.3f} {r25['5']:>6.3f} {r25['10']:>6.3f}   "
              f"{r25['1'] - b25['1']:>+6.3f}")

        # Als eigene Variante in results/<stadt>/evaluation: compare.py zeigt sie in
        # der Spalte "Variante" neben none und linear.
        variante = f"{adapter}+seq{w}" if adapter not in ("none", "None") else f"seq{w}"
        pfad = write_evaluation(
            PATHS.evaluation / f"{name}_seq{w}.json",
            CFG, f"{name}_seq{w}", dim, len(database), befunde,
            variant=variante, fingerprint=fingerprint, root=ROOT,
            sequence_window=w, sequence_weighting="dreieck",
        )
        print(f"           -> {pfad.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
