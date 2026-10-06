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

Nicht jeder Zwilling ist eine Kopie. Eine Kamera-Anordnung mit mehreren
Blickrichtungen (vorn, hinten, seitlich) laedt ebenfalls zeitgleiche Sequenzen
desselben Kontos hoch -- nur zeigen sie in eine andere Richtung. Solche
Zwillinge machen eine Anfrage nach der Ground Truth "loesbar", ohne dass ein
Encoder sie finden koennte. Das Skript trennt deshalb: Kopie heisst, die
Blickrichtung weicht hoechstens KOPIE_GRAD ab (oder beide Bilder sind Panoramen).

    python experiments/zwillinge.py
    python experiments/zwillinge.py --methods megaloc,eigenplaces --fenster-s 60
    python experiments/zwillinge.py --nur-anteile     # nur Metadaten, ohne Trefferlisten

Ergebnis: experiments/results/<stadt>/zwillinge.json
"""

import argparse
import json

import numpy as np
import pandas as pd

from _common import CFG, PATHS, RESULTS, ROOT
from src.evaluation import evaluate_retrieval
from src.geo import haversine_distance
from src.retrieval import load_retrieval

PROTOKOLLE = {"benchmark": None, "voll": ["database", "train"]}
KOPIE_GRAD = 30.0


def _args():
    ap = argparse.ArgumentParser(
        description="Wie oft Anfragen eine Zwillingsfahrt in der Referenz haben, und "
                    "was R@1 ohne sie waere.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--methods", default="megaloc,eigenplaces", help="Kommaliste")
    ap.add_argument("--fenster-s", type=float, default=60.0,
                    help="hoechster Zeitabstand desselben Kontos, der als Zwilling gilt")
    ap.add_argument("--nur-anteile", action="store_true",
                    help="nur die Anteile aus den Metadaten, ohne Trefferlisten und R@1")
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


def zwillinge_je_anfrage(query, referenz, fenster_ms, radius_m):
    """Je Anfrage zwei Masken: liegt ein Zwilling im Umkreis von radius_m, und
    liegt darunter eine Kopie (dieselbe Blickrichtung)? Die erste Maske sind
    die Anfragen, die ein Zwilling allein loesbar machen kann."""
    hat = np.zeros(len(query), dtype=bool)
    kopie = np.zeros(len(query), dtype=bool)
    q_zeit = query["captured_at"].to_numpy().astype("int64")
    q_lat, q_lon = query["lat"].to_numpy(), query["lon"].to_numpy()
    q_blick = _spalte(query, "compass_angle", np.nan).astype(float)
    q_pano = _spalte(query, "is_pano", False).astype(bool)
    r_zeit_alle = referenz["captured_at"].to_numpy().astype("int64")
    r_blick_alle = _spalte(referenz, "compass_angle", np.nan).astype(float)
    r_pano_alle = _spalte(referenz, "is_pano", False).astype(bool)
    gruppen = {}
    for konto, zeilen in referenz.groupby("creator_id").indices.items():
        o = zeilen[np.argsort(r_zeit_alle[zeilen], kind="stable")]
        gruppen[konto] = (r_zeit_alle[o], referenz["lat"].to_numpy()[o],
                          referenz["lon"].to_numpy()[o], r_blick_alle[o], r_pano_alle[o])
    for konto, zeilen in query.groupby("creator_id").indices.items():
        if konto not in gruppen:
            continue
        t, lat, lon, blick, pano = gruppen[konto]
        a = np.searchsorted(t, q_zeit[zeilen] - fenster_ms, side="left")
        b = np.searchsorted(t, q_zeit[zeilen] + fenster_ms, side="right")
        for z, von, bis in zip(zeilen, a, b):
            if bis <= von:
                continue
            nah = haversine_distance(q_lat[z], q_lon[z], lat[von:bis], lon[von:bis]) <= radius_m
            if not nah.any():
                continue
            hat[z] = True
            dw = np.abs((blick[von:bis] - q_blick[z] + 180.0) % 360.0 - 180.0)
            gleich = (dw <= KOPIE_GRAD) | (pano[von:bis] & q_pano[z])
            kopie[z] = bool((nah & gleich).any())
    return hat, kopie


def _spalte(df, name, ersatz):
    return df[name].to_numpy() if name in df.columns else np.full(len(df), ersatz)


def anteil_mit_zwilling(query, referenz, fenster_ms, radius_m):
    """Je Anfrage: liegt ein Zwilling im Umkreis von radius_m?"""
    return zwillinge_je_anfrage(query, referenz, fenster_ms, radius_m)[0]


def anteile(fenster_ms, radius_m):
    """Anteile je Protokoll, nur aus den Metadaten -- braucht keine Trefferliste."""
    meta = pd.read_parquet(PATHS.processed / "metadata.parquet")
    query = meta[meta["split"] == "query"].reset_index(drop=True)
    aus = {}
    for protokoll, splits in PROTOKOLLE.items():
        referenz = meta[meta["split"].isin(splits or ["database"])].reset_index(drop=True)
        hat, kopie = zwillinge_je_anfrage(query, referenz, fenster_ms, radius_m)
        aus[protokoll] = {"n_anfragen": int(len(query)),
                          "anteil_anfragen_mit_zwilling": float(hat.mean()),
                          "anteil_anfragen_mit_kopie": float(kopie.mean()),
                          "anteil_kopie_unter_zwillingen": float(kopie.sum() / max(hat.sum(), 1))}
    return aus


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
    out = RESULTS / "zwillinge.json"
    # Die Trefferlisten einer Stadt koennen auf zwei Rechnern liegen (Benchmark
    # hier, volle Referenz dort). Ein zweiter Lauf ergaenzt dann, statt den
    # ersten zu ueberschreiben -- solange Fenster und Schwelle dieselben sind.
    if out.exists():
        alt = json.loads(out.read_text(encoding="utf-8"))
        if (alt.get("fenster_s"), alt.get("schwelle_m")) == (raus["fenster_s"], raus["schwelle_m"]):
            raus["encoder"] = alt.get("encoder", {})

    print(f"Zwilling = selbes Konto, hoechstens {args.fenster_s:g} s Abstand; R@1 bei {schwelle} m\n")
    raus["anteile"] = anteile(fenster_ms, float(schwelle))
    for protokoll, a in raus["anteile"].items():
        print(f"{protokoll:<11}Anfragen mit Zwilling {a['anteil_anfragen_mit_zwilling']:>6.1%}"
              f"   davon Kopie (Blick <= {KOPIE_GRAD:g} Grad) {a['anteil_kopie_unter_zwillingen']:>6.1%}")
    print()
    if args.nur_anteile:
        args.methods = ""
    print(f"{'Encoder':<13}{'Protokoll':<11}{'Zwilling 25 m':>14}{'Top-1 Zwilling':>16}"
          f"{'R@1':>8}{'ohne':>8}{'Diff':>8}{'loesbar':>10}{'ohne':>9}")
    print("-" * 97)
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        raus["encoder"].setdefault(method, {})
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
    out.write_text(json.dumps(raus, indent=2), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
