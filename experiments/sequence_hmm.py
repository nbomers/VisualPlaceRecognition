"""
Sequenz-HMM: die Fahrt als Pfad, nicht als Folge von Einzelentscheidungen.

`sequence_retrieval.py` summiert die Trefferlisten der Nachbarframes auf und
hilft in Osnabrueck nicht (R@1 0.507 einzeln, 0.498 bei +-3). Der Grund steht
im Befund: benachbarte Frames sehen dieselbe Strasse und machen denselben
Fehler. Der Mechanismus hat aber noch eine zweite Schwaeche -- er braucht
DASSELBE Datenbankbild in mehreren Listen, und bei einer duennen Datenbank
kommt das selten vor.

Dieses Skript trennt die beiden Gruende. Das HMM laesst je Frame andere
Kandidaten zu und fragt nur, ob sie geometrisch zueinander passen: Zustaende
sind die Top-k eines Frames, ein Uebergang bewertet den Abstand zweier
Kandidaten gegen die Zeit zwischen den Frames. Ein Kandidat sechs Kilometer
abseits faellt dann, weil man in 0,17 s keine sechs Kilometer faehrt -- nicht,
weil der Nachbarframe ihn nicht auch gefunden haette.

Bleibt der Gewinn trotzdem aus, ist der Befund bestaetigt: die Fehler sind
kohaerent, und keine Sequenzmethode holt sie zurueck. Genau deshalb steht das
Skript hier auch dann, wenn es nichts bringt.

    Emission   E[t, j] = beta * Aehnlichkeit(t, j)
    Uebergang  A[i, j] = -|d(i, j) - v * dt| / sigma
    Ergebnis   Posterior je Frame (Forward-Backward), Pfad (Viterbi)

Die Geschwindigkeit v kommt aus den DATENBANKsequenzen, nicht aus den
Anfragen -- deren Positionen sind die Ground Truth und gehen nirgends ein.

Reines Nachbearbeiten der .npz aus 06; kein Modell, keine GPU, keine Bilder.
Wie die geometrische Verifikation sortiert es die Top-k nur um: R@k fuer das
volle k bleibt, was die Suche geliefert hat.

    python experiments/sequence_hmm.py --method eigenplaces_pcaw512
    python experiments/sequence_hmm.py --method megaloc --beta 10,30,100
    python experiments/sequence_hmm.py --method megaloc --sigma 10,25,50

ZU DEN PARAMETERN: beta und sigma sind Hyperparameter, und die Tabelle unten
wird auf den Anfragen ausgewertet. Die beste Zeile herauszugreifen und als
Ergebnis zu berichten, waere Tuning auf der Testmenge. Das Skript druckt
deshalb den ganzen Durchlauf -- wer eine Zahl braucht, legt die Parameter
vorher fest (die Voreinstellung beta=30, sigma=25 ist die Schwelle der Ground
Truth und eine Aehnlichkeitsskala, die Top-1 und Top-20 rund 2 bis 4 nat
auseinanderlegt) und berichtet diese eine Zeile.
"""

import argparse
import itertools

import pandas as pd

from _common import CFG, PATHS, ROOT
from src.evaluation import standard_evaluations, write_evaluation
from src.retrieval import descriptor_dim, load_retrieval, localizable, top_distances
from src.run_guard import embedding_fingerprint
from src.sequence_hmm import hmm_rerank, schaetze_geschwindigkeit


def _args():
    ap = argparse.ArgumentParser(
        description="Trefferlisten einer Fahrt ueber ein HMM umsortieren.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--method", default="eigenplaces_pcaw512")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--beta", default="30",
                    help="Kommaliste: Gewicht der Aehnlichkeit gegen die Geometrie")
    ap.add_argument("--sigma", default="25",
                    help="Kommaliste: Toleranz der Streckenabweichung in Metern")
    ap.add_argument("--speed", type=float, default=None,
                    help="Geschwindigkeit in m/s; ohne Angabe aus der Datenbank geschaetzt")
    ap.add_argument("--quiet", action="store_true", help="Keine Zwischentabellen")
    return ap.parse_args()


def _liste(text, name):
    werte = [float(x) for x in text.split(",") if x.strip()]
    if not werte:
        raise SystemExit(f"--{name} ist leer.")
    if any(w <= 0 for w in werte):
        raise SystemExit(f"--{name} braucht Werte groesser 0, bekommen: {text!r}.")
    return werte


def main():
    args = _args()
    method, adapter = args.method, args.adapter
    name = method if adapter in ("none", "None") else f"{method}_{adapter}"
    CFG["vpr"]["method"], CFG["vpr"]["adapter"] = method, adapter

    betas = _liste(args.beta, "beta")
    sigmas = _liste(args.sigma, "sigma")

    query, database, indices, similarities = load_retrieval(ROOT, CFG, method, adapter)
    top_k = indices.shape[1]
    emb_dir = PATHS.embedding_dir(method)
    dim = descriptor_dim(ROOT, CFG, method, adapter)
    fingerprint = embedding_fingerprint(
        CFG, method, adapter, pd.read_parquet(emb_dir / f"{name}_metadata.parquet"))

    v_hat = args.speed if args.speed else schaetze_geschwindigkeit(database)
    herkunft = "vorgegeben" if args.speed else "aus den Datenbanksequenzen"
    n_seq = query["sequence_id"].nunique()
    print(f"{name}: {len(query):,} Anfragen in {n_seq:,} Sequenzen, Top-{top_k} je Anfrage")
    print(f"Geschwindigkeit {v_hat:.1f} m/s ({v_hat * 3.6:.0f} km/h), {herkunft}")
    print()

    # Dieselbe Berichtsschwelle wie in src/evaluation.py: die mittlere aus
    # retrieval.thresholds, mit der Voreinstellung also 25 m.
    schwellen = CFG["retrieval"]["thresholds"]
    schluessel = str(schwellen[len(schwellen) // 2])
    schwelle = float(schluessel)
    loesbar = localizable(query, database, schwelle)
    basis = standard_evaluations(indices, query, database, CFG, verbose=False)
    b = basis["Alle Queries"]["schwellen"][schluessel]["recall"]

    print(f"{'beta':>6} {'sigma':>6}   {'R@1':>6} {'R@5':>6} {'R@10':>6}   "
          f"{'dR@1':>6}   {'Viterbi':>7}")
    print(f"{'--':>6} {'einzeln':>6}   {b['1']:>6.3f} {b['5']:>6.3f} {b['10']:>6.3f}"
          f"   {'':>6}   {'':>7}")

    for beta, sigma in itertools.product(betas, sigmas):
        neu_idx, _, viterbi = hmm_rerank(indices, similarities, query, database,
                                         v_hat, beta=beta, sigma_m=sigma)
        befunde = standard_evaluations(neu_idx, query, database, CFG,
                                       verbose=not args.quiet)
        r = befunde["Alle Queries"]["schwellen"][schluessel]["recall"]

        # Der Viterbi-Pfad waehlt EINE Zeile je Frame -- das ist eine andere
        # Entscheidung als "Platz 1 der umsortierten Liste" und wird deshalb
        # getrennt ausgewiesen, nicht als R@1 verkauft.
        d_vit = top_distances(query, database, viterbi[:, None])[:, 0]
        r_vit = float(((d_vit <= schwelle) & loesbar).sum() / loesbar.sum())

        print(f"{beta:>6.0f} {sigma:>6.0f}   {r['1']:>6.3f} {r['5']:>6.3f} {r['10']:>6.3f}"
              f"   {r['1'] - b['1']:>+6.3f}   {r_vit:>7.3f}")

        marke = f"hmm{beta:g}-{sigma:g}"
        variante = f"{adapter}+{marke}" if adapter not in ("none", "None") else marke
        pfad = write_evaluation(
            PATHS.evaluation / f"{name}_{marke}.json",
            CFG, f"{name}_{marke}", dim, len(database), befunde,
            variant=variante, fingerprint=fingerprint, root=ROOT,
            hmm_beta=beta, hmm_sigma_m=sigma, hmm_speed_ms=v_hat,
            hmm_viterbi_recall=r_vit,
        )
        print(f"                 -> {pfad.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
