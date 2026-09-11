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


FIGURE_DIR = ROOT / "results" / "figures" / "evaluation"


def _reihen(laeufe, split, schwelle, k):
    """(name, dim, adapter, recall) fuer alle Laeufe, die das hergeben."""
    raus = []
    for r in laeufe:
        a = r["auswertungen"].get(split)
        if a is None or str(schwelle) not in a["schwellen"]:
            continue
        wert = a["schwellen"][str(schwelle)]["recall"].get(str(k))
        if wert is not None:
            raus.append((r["method"], int(r["dim"]), r["adapter"], float(wert)))
    return raus


def plot(laeufe, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["figure.dpi"] = 150
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    split, schwelle = args.split, args.threshold
    geschrieben = []

    # ------------------------------------------------------------------
    # 1. Der Kernbefund: Baseline gegen Adapter, nach Dimension sortiert.
    #    Das Vorzeichen der Differenz ist die eigentliche Aussage, deshalb
    #    steht es als eigene Achse darunter statt als Beschriftung daneben.
    # ------------------------------------------------------------------
    reihen = _reihen(laeufe, split, schwelle, 1)
    basis = {n: (d, v) for n, d, a, v in reihen if a in ("none", "None")}
    adapt = {n: v for n, d, a, v in reihen if a not in ("none", "None")}
    encoder = sorted(basis, key=lambda n: (basis[n][0], n))

    if encoder:
        x = range(len(encoder))
        b = [basis[n][1] for n in encoder]
        a = [adapt.get(n) for n in encoder]
        etiketten = [f"{n}\n{basis[n][0]}d" for n in encoder]

        fig, (oben, unten) = plt.subplots(
            2, 1, figsize=(1.7 * len(encoder) + 2, 7),
            gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True,
        )
        breite = 0.38
        oben.bar([i - breite / 2 for i in x], b, breite,
                 label="Baseline", color="#37474f")
        oben.bar([i + breite / 2 for i in x],
                 [v if v is not None else 0 for v in a], breite,
                 label="+ linearer Adapter", color="#90a4ae")
        for i, (bv, av) in enumerate(zip(b, a)):
            oben.text(i - breite / 2, bv + 0.008, f"{bv:.3f}",
                      ha="center", fontsize=7)
            if av is not None:
                oben.text(i + breite / 2, av + 0.008, f"{av:.3f}",
                          ha="center", fontsize=7)
        oben.set_ylabel(f"R@1 bei {schwelle} m")
        oben.set_title(f"{split}  |  Schwelle {schwelle} m  |  "
                       "nach Deskriptorbreite sortiert")
        oben.legend(fontsize=8)
        oben.grid(axis="y", alpha=0.3)
        oben.set_axisbelow(True)

        delta = [(av - bv) if av is not None else 0.0 for bv, av in zip(b, a)]
        farben = ["#2e7d32" if d > 0 else "#c62828" for d in delta]
        unten.bar(x, delta, 0.55, color=farben)
        unten.axhline(0, color="0.3", linewidth=0.8)
        for i, d in enumerate(delta):
            if d:
                unten.text(i, d + (0.004 if d > 0 else -0.010), f"{d:+.3f}",
                           ha="center", fontsize=8)
        unten.set_ylabel("Differenz durch\nden Adapter")
        unten.set_xticks(list(x))
        unten.set_xticklabels(etiketten, fontsize=8)
        unten.grid(axis="y", alpha=0.3)
        unten.set_axisbelow(True)

        plt.tight_layout()
        ziel = FIGURE_DIR / f"vergleich_adapter_{schwelle}m.png"
        fig.savefig(ziel, bbox_inches="tight")
        plt.close(fig)
        geschrieben.append(ziel)

    # ------------------------------------------------------------------
    # 2. Recall ueber k -- die uebliche VPR-Kurve, nur die Baselines.
    # ------------------------------------------------------------------
    ks = sorted({int(k)
                 for r in laeufe
                 for a in [r["auswertungen"].get(split)] if a
                 for e in a["schwellen"].values()
                 for k in e["recall"]})
    if ks and encoder:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for r in sorted(laeufe, key=lambda r: -r["dim"]):
            a = r["auswertungen"].get(split)
            if a is None or str(schwelle) not in a["schwellen"]:
                continue
            recall = a["schwellen"][str(schwelle)]["recall"]
            y = [recall.get(str(k)) for k in ks]
            if any(v is None for v in y):
                continue
            stil = "-" if r["adapter"] in ("none", "None") else "--"
            ax.plot(ks, y, stil, marker="o", markersize=3.5, linewidth=1.5,
                    label=f"{r['embedding_name']} ({r['dim']}d)")
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("k")
        ax.set_ylabel(f"Recall@k bei {schwelle} m")
        ax.set_title(f"{split}  |  durchgezogen = Baseline, gestrichelt = Adapter")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        ziel = FIGURE_DIR / f"vergleich_recall_k_{schwelle}m.png"
        fig.savefig(ziel, bbox_inches="tight")
        plt.close(fig)
        geschrieben.append(ziel)

    # ------------------------------------------------------------------
    # 3. R@1 ueber die Distanzschwelle -- zeigt, wie streng die 25 m sind.
    # ------------------------------------------------------------------
    schwellen = sorted({int(s)
                        for r in laeufe
                        for a in [r["auswertungen"].get(split)] if a
                        for s in a["schwellen"]})
    if schwellen:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for r in sorted(laeufe, key=lambda r: -r["dim"]):
            a = r["auswertungen"].get(split)
            if a is None:
                continue
            y = [a["schwellen"].get(str(s), {}).get("recall", {}).get("1")
                 for s in schwellen]
            if any(v is None for v in y):
                continue
            stil = "-" if r["adapter"] in ("none", "None") else "--"
            ax.plot(schwellen, y, stil, marker="o", markersize=3.5,
                    linewidth=1.5, label=f"{r['embedding_name']}")
        ax.axvline(schwelle, color="0.8", linewidth=1.0, zorder=0)
        ax.set_xscale("log")
        ax.set_xticks(schwellen)
        ax.set_xticklabels([str(s) for s in schwellen])
        ax.set_xlabel("Distanzschwelle [m]")
        ax.set_ylabel("R@1")
        ax.set_title(f"{split}  |  wie streng ist die Ground Truth?")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        ziel = FIGURE_DIR / "vergleich_schwellen.png"
        fig.savefig(ziel, bbox_inches="tight")
        plt.close(fig)
        geschrieben.append(ziel)

    for z in geschrieben:
        print(f"geschrieben: {z.relative_to(ROOT)}")


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
    ap.add_argument("--plot", action="store_true",
                    help="Vergleichsabbildungen nach results/figures/evaluation/ "
                         "schreiben, sonst nichts")
    args = ap.parse_args()

    laeufe = load()
    if not laeufe:
        raise SystemExit(
            f"Keine Auswertungen in {EVAL_DIR.relative_to(ROOT)}.\n"
            "07_evaluation je Verfahren einmal laufen lassen."
        )

    if args.plot:
        plot(laeufe, args)
        return

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
