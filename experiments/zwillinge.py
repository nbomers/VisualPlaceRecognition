"""
Zwillingsfahrten: dieselbe Fahrt, zweimal hochgeladen.

Der Split trennt nach sequence_id. Mapillary fuehrt dieselbe Fahrt aber
manchmal als zwei Sequenzen -- selbes Konto, Zeitstempel Sekundenbruchteile
auseinander, dieselben Koordinaten. Landet eine Kopie in query und die andere
in database oder train, liegt zur Anfrage ein fast identisches Referenzbild
bereit: 0 m daneben, cos um 0.99. Das ist Leakage, die der Sequenz-Split
nicht sieht.

Das Skript misst, wie oft das vorkommt, und was R@1 ohne diese Zwillinge
waere -- so gerechnet, als waere die zweite Kopie nie hochgeladen worden:
Zwillinge fallen aus der Trefferliste (die naechsten Kandidaten ruecken auf)
und zaehlen nicht als Referenz fuer "loesbar". Gezaehlt wird mit
src/evaluation.py, derselben Auswertung wie in 07; die Standardzeile muss
deshalb die Zahl aus 07 treffen, sonst bricht das Skript ab.

Zwilling heisst: selbes Konto (creator_id) und hoechstens --fenster-s Sekunden
Abstand. Ein Konto fotografiert nicht an zwei Orten zugleich; was zeitlich so
nah liegt, stammt aus derselben Fahrt. Anders als die Hard-Ground-Truth
(selbes Konto bis 180 Tage) trifft das nur die Kopie, nicht jede andere Fahrt
desselben Fotografen.

    python experiments/zwillinge.py
    python experiments/zwillinge.py --methods megaloc,eigenplaces --fenster-s 60

Ergebnis: experiments/results/<stadt>/zwillinge.json
"""

import argparse
import json

import numpy as np

from _common import CFG, PATHS, RESULTS, ROOT
from src.evaluation import evaluate_retrieval
from src.geo import haversine_distance
from src.retrieval import load_retrieval

PROTOKOLLE = {"benchmark": None, "voll": ["database", "train"]}


def _args():
    ap = argparse.ArgumentParser(
        description="Wie oft Anfragen eine Zwillingsfahrt in der Referenz haben, und "
                    "was R@1 ohne sie waere.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--methods", default="megaloc,eigenplaces", help="Kommaliste")
    ap.add_argument("--fenster-s", type=float, default=60.0,
                    help="hoechster Zeitabstand desselben Kontos, der als Zwilling gilt")
    return ap.parse_args()


def zwilling_fn(query, referenz, fenster_ms):
    """(qi, di) -> bool-Array: ist Referenzbild di ein Zwilling von Anfrage qi?"""
    q_konto = query["creator_id"].to_numpy()
    r_konto = referenz["creator_id"].to_numpy()
    q_zeit = query["captured_at"].to_numpy().astype("int64")
    r_zeit = referenz["captured_at"].to_numpy().astype("int64")

    def f(qi, di):
        return (r_konto[di] == q_konto[qi]) & (np.abs(q_zeit[qi] - r_zeit[di]) <= fenster_ms)
    return f


def anteil_mit_zwilling(query, referenz, fenster_ms, radius_m):
    """Je Anfrage: liegt ein Zwilling im Umkreis von radius_m? Das sind die
    Anfragen, die er allein loesbar machen kann."""
    hat = np.zeros(len(query), dtype=bool)
    q_zeit = query["captured_at"].to_numpy().astype("int64")
    q_lat, q_lon = query["lat"].to_numpy(), query["lon"].to_numpy()
    gruppen = {}
    for konto, zeilen in referenz.groupby("creator_id").indices.items():
        ordnung = zeilen[np.argsort(referenz["captured_at"].to_numpy()[zeilen], kind="stable")]
        gruppen[konto] = (referenz["captured_at"].to_numpy().astype("int64")[ordnung],
                          referenz["lat"].to_numpy()[ordnung], referenz["lon"].to_numpy()[ordnung])
    for konto, zeilen in query.groupby("creator_id").indices.items():
        if konto not in gruppen:
            continue
        t, lat, lon = gruppen[konto]
        a = np.searchsorted(t, q_zeit[zeilen] - fenster_ms, side="left")
        b = np.searchsorted(t, q_zeit[zeilen] + fenster_ms, side="right")
        for z, von, bis in zip(zeilen, a, b):
            if bis > von:
                d = haversine_distance(q_lat[z], q_lon[z], lat[von:bis], lon[von:bis])
                hat[z] = bool((d <= radius_m).any())
    return hat


def ohne_zwillinge(indices, zwilling):
    """Zwillinge je Zeile ans Ende schieben, Reihenfolge der uebrigen bleibt."""
    zeilen = np.repeat(np.arange(len(indices))[:, None], indices.shape[1], axis=1)
    maske = zwilling(zeilen, indices)
    ordnung = np.argsort(maske, axis=1, kind="stable")
    return np.take_along_axis(indices, ordnung, axis=1), maske


def auswerten(query, referenz, indices, cfg, fenster_ms):
    schwelle = str(int(float(cfg["vpr"]["uncertain_radius_m"])))
    zwilling = zwilling_fn(query, referenz, fenster_ms)
    idx_ohne, maske = ohne_zwillinge(indices, zwilling)
    std = evaluate_retrieval(indices, query, referenz, cfg, label="Standard", verbose=False)
    ohne = evaluate_retrieval(idx_ohne, query, referenz, cfg, label="ohne Zwillinge",
                              gt_filter=lambda qi, di: ~zwilling(qi, di), verbose=False)
    s, o = std["schwellen"][schwelle], ohne["schwellen"][schwelle]
    return {
        "anteil_anfragen_mit_zwilling": float(
            anteil_mit_zwilling(query, referenz, fenster_ms, float(schwelle)).mean()),
        "anteil_top1_ist_zwilling": float(maske[:, 0].mean()),
        "loesbar": s["loesbar"], "loesbar_ohne": o["loesbar"],
        "recall": s["recall"], "recall_ohne": o["recall"],
    }


def main():
    args = _args()
    fenster_ms = int(args.fenster_s * 1000)
    schwelle = str(int(float(CFG["vpr"]["uncertain_radius_m"])))
    raus = {"fenster_s": args.fenster_s, "schwelle_m": float(schwelle), "encoder": {}}

    print(f"Zwilling = selbes Konto, hoechstens {args.fenster_s:g} s Abstand; R@1 bei {schwelle} m\n")
    print(f"{'Encoder':<13}{'Protokoll':<11}{'Zwilling 25 m':>14}{'Top-1 Zwilling':>16}"
          f"{'R@1':>8}{'ohne':>8}{'Diff':>8}{'loesbar':>10}{'ohne':>9}")
    print("-" * 97)
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        raus["encoder"][method] = {}
        for protokoll, splits in PROTOKOLLE.items():
            try:
                query, referenz, indices, _ = load_retrieval(ROOT, CFG, method,
                                                             reference_splits=splits)
            except FileNotFoundError as e:
                print(f"{method:<13}{protokoll:<11}fehlt hier: {str(e).splitlines()[0]}")
                continue
            z = auswerten(query, referenz, indices, CFG, fenster_ms)

            # Kontrolle: die Standardzeile ist die aus 07 bzw. full_reference.py.
            name = method + ("_fullref" if splits else "")
            soll = json.loads((PATHS.evaluation / f"{name}.json").read_text(encoding="utf-8"))
            soll = soll["auswertungen"]["Alle Queries"]["schwellen"][schwelle]["recall"]["1"]
            if abs(z["recall"]["1"] - soll) > 1e-9:
                raise SystemExit(f"{name}: Standard-R@1 {z['recall']['1']:.6f} trifft "
                                 f"{soll:.6f} aus der Auswertungs-JSON nicht -- nicht deuten.")

            r1, r1o = z["recall"]["1"], z["recall_ohne"]["1"]
            print(f"{method:<13}{protokoll:<11}{z['anteil_anfragen_mit_zwilling']:>14.1%}"
                  f"{z['anteil_top1_ist_zwilling']:>16.1%}{r1:>8.3f}{r1o:>8.3f}{r1o - r1:>+8.3f}"
                  f"{z['loesbar']:>10,}{z['loesbar_ohne']:>9,}")
            raus["encoder"][method][protokoll] = z

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "zwillinge.json"
    out.write_text(json.dumps(raus, indent=2), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
