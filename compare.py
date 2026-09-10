"""
Stellt die gespeicherten Auswertungen aller Verfahren nebeneinander.

07_evaluation legt je Lauf eine Datei unter results/evaluation/ ab. Dieses
Skript liest sie und druckt die Vergleichstabelle.

    python compare.py                      # R@k bei 25 m
    python compare.py --threshold 5        # andere Schwelle
    python compare.py --split "Hard: anderer creator_id ODER > 180 Tage Abstand"
    python compare.py --list               # welche Auswertungen liegen vor
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent
EVAL_DIR = ROOT / "results" / "evaluation"


def load():
    if not EVAL_DIR.exists():
        return []
    laeufe = [json.loads(p.read_text()) for p in sorted(EVAL_DIR.glob("*.json"))]
    # In die Recall-Tabelle gehoeren nur Auswertungen aus 07. Ueber den Inhalt
    # filtern, nicht ueber den Dateinamen -- haelt auch fuer alles Weitere,
    # was spaeter einmal in dem Verzeichnis landet.
    laeufe = [r for r in laeufe if "auswertungen" in r]
    # Baseline vor Adapter, sonst alphabetisch nach Verfahren
    return sorted(laeufe, key=lambda r: (r["method"], r["adapter"] != "none"))


def main():
    ap = argparse.ArgumentParser(
        description="Vergleichstabelle aus results/evaluation/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--threshold", type=int, default=25,
                    help="Distanzschwelle in Metern (Standard: 25)")
    ap.add_argument("--split", default="Alle Queries",
                    help='Welche Auswertung (Standard: "Alle Queries")')
    ap.add_argument("--list", action="store_true",
                    help="Vorhandene Laeufe und Auswertungen anzeigen, sonst nichts")
    args = ap.parse_args()

    laeufe = load()
    if not laeufe:
        raise SystemExit(
            f"Keine Auswertungen in {EVAL_DIR.relative_to(ROOT)}.\n"
            "07_evaluation je Verfahren einmal laufen lassen."
        )

    if args.list:
        for r in laeufe:
            print(f"{r['embedding_name']:24s} {r['datum']}  {r['dim']:>5}d")
            for name, a in r["auswertungen"].items():
                print(f"    {name}  ({a['n_queries']:,} Queries)")
        return

    t = str(args.threshold)
    ks = None
    zeilen = []
    for r in laeufe:
        a = r["auswertungen"].get(args.split)
        if a is None or t not in a["schwellen"]:
            continue
        eintrag = a["schwellen"][t]
        ks = ks or sorted(eintrag["recall"], key=int)
        zeilen.append((r, eintrag))

    if not zeilen:
        vorhanden = {n for r in laeufe for n in r["auswertungen"]}
        raise SystemExit(
            f'Keine Daten fuer --split "{args.split}" bei {args.threshold} m.\n'
            f"Vorhanden: {sorted(vorhanden)}"
        )

    loesbar = zeilen[0][1]["loesbar"]
    print(f'{args.split}  |  Schwelle {args.threshold} m  |  {loesbar:,} loesbare Queries\n')
    kopf = f"{'Encoder':<14}{'Adapter':<10}{'Dim':>6}   " + "".join(f"{'R@'+k:>8}" for k in ks)
    print(kopf)
    print("-" * len(kopf))
    for r, eintrag in zeilen:
        werte = "".join(
            f"{eintrag['recall'][k]:>8.3f}" if eintrag["recall"][k] is not None else f"{'-':>8}"
            for k in ks
        )
        print(f"{r['method']:<14}{r['adapter']:<10}{r['dim']:>6}   {werte}")

    if len({z[1]["loesbar"] for z in zeilen}) > 1:
        print("\nAchtung: unterschiedlich viele loesbare Queries -- die Laeufe "
              "beruhen nicht auf demselben Split.")


if __name__ == "__main__":
    main()
