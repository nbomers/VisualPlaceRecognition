"""
Stellt die gespeicherten Auswertungen aller Verfahren nebeneinander.

07_evaluation legt je Lauf eine Datei unter results/evaluation/ ab. Dieses
Skript liest sie und druckt die Vergleichstabelle.

    python compare.py                      # R@k bei 25 m
    python compare.py --derived            # dazu PCA-, Whitening-, Verkettungs- und Sequenz-Zeilen
    python compare.py --threshold 5        # andere Schwelle
    python compare.py --split "Hard: anderer creator_id ODER > 180 Tage Abstand"
    python compare.py --localization       # 08: Koordinate statt Trefferliste
    python compare.py --ci                 # 95-%-Intervall neben R@1 (experiments/bootstrap_ci.py)
    python compare.py --reference full     # database + train als Referenz (experiments/full_reference.py)
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent


def _paths():
    from src.config import load_config, paths
    return paths(load_config(ROOT), ROOT)


PATHS = _paths()
EVAL_DIR = PATHS.evaluation


def load(reference="database"):
    if not EVAL_DIR.exists():
        return []
    laeufe = [json.loads(p.read_text()) for p in sorted(EVAL_DIR.glob("*.json"))]
    # In die Recall-Tabelle gehoeren nur Auswertungen aus 07. Ueber den Inhalt
    # filtern, nicht ueber den Dateinamen -- haelt auch fuer alles Weitere,
    # was spaeter einmal in dem Verzeichnis landet.
    laeufe = [r for r in laeufe if "auswertungen" in r]
    # Zwei Protokolle, die nicht in eine Tabelle gehoeren: die Benchmark-
    # Referenz (database) und die volle (database + train, Variante fullref).
    voll = [r for r in laeufe if r.get("variant") == "fullref"]
    laeufe = voll if reference == "full" else [r for r in laeufe if r not in voll]
    # Aeltere JSONs tragen die Variante im Adapter-Feld.
    for r in laeufe:
        r.setdefault("variant", r["adapter"])
    # Baseline vor Adapter, sonst alphabetisch nach Verfahren
    return sorted(laeufe, key=lambda r: (r["method"], r["variant"] != "none"))


FIGURE_DIR = PATHS.figures / "evaluation"


def _abgeleitet(method):
    """Hat der Encoder in der config einen source-Eintrag, ist er aus einem
    anderen gerechnet -- die PCA- und Whitening-Varianten."""
    from src.config import load_config
    block = load_config(ROOT)["vpr"].get(method)
    return isinstance(block, dict) and ("source" in block or "sources" in block)


def _reihen(laeufe, split, schwelle, k):
    """(name, dim, adapter, recall) fuer alle Laeufe, die das hergeben."""
    raus = []
    for r in laeufe:
        a = r["auswertungen"].get(split)
        if a is None or str(schwelle) not in a["schwellen"]:
            continue
        wert = a["schwellen"][str(schwelle)]["recall"].get(str(k))
        if wert is not None:
            raus.append((r["method"], int(r["dim"]), r["variant"], float(wert)))
    return raus


LOC_DIR = PATHS.localization
CI_DIR = PATHS.experiments


def bootstrap_intervals(args, still=False):
    """
    95-%-Intervalle aus dem Sequenz-Bootstrap, je embedding_name das Intervall
    fuer R@1. Leer, wenn die JSON fehlt oder zu einer anderen Schwelle gehoert.
    """
    ci_path = CI_DIR / ("bootstrap_ci_fullref.json" if args.reference == "full" else "bootstrap_ci.json")
    if not ci_path.exists():
        if not still:
            print(f"Keine Intervalle: {ci_path.relative_to(ROOT)} fehlt "
                  f"-> python experiments/bootstrap_ci.py --reference {args.reference}\n")
        return {}
    ci = json.loads(ci_path.read_text())
    if int(ci["threshold_m"]) != args.threshold or ci["split"] != args.split:
        if not still:
            print(f"Keine Intervalle: {ci_path.relative_to(ROOT)} gilt fuer "
                  f"\"{ci['split']}\" bei {ci['threshold_m']:g} m\n")
        return {}
    return {name: e["recall"]["1"]["ci"] for name, e in ci["encoder"].items()}


def localization_table(args):
    """
    08 macht aus der Trefferliste eine Koordinate -- auf drei Wegen. Hier
    steht je Encoder, welcher Weg wie oft unter der Schwelle landet und wie
    weit der Median danebenliegt. Bezogen auf ALLE Anfragen, nicht nur die
    loesbaren: eine Koordinate wird immer geschaetzt.
    """
    if not LOC_DIR.exists():
        raise SystemExit(f"Keine Lokalisierung in {LOC_DIR.relative_to(ROOT)}. 08 laufen lassen.")
    laeufe = [json.loads(p.read_text()) for p in sorted(LOC_DIR.glob("*.json"))]
    laeufe = [r for r in laeufe if "verfahren" in r]
    if not args.derived:
        laeufe = [r for r in laeufe if not _abgeleitet(r["method"])]
    laeufe.sort(key=lambda r: (r["method"], r["adapter"] != "none"))
    if not laeufe:
        raise SystemExit("Keine Lokalisierungsergebnisse.")

    schluessel = f"unter_{args.threshold}m"
    # Alle Verfahren, die 08 geschrieben hat -- die zwei Vergleichswerte
    # (Zufall, Stadtmitte) kommen als Fussnote.
    vergleich = ("Zufaelliges DB-Bild", "Stadtmittelpunkt")
    wege = []
    for r in laeufe:
        for w in r["verfahren"]:
            if w not in vergleich and w not in wege:
                wege.append(w)
    kurz = {"Top-1": "Top-1", "Schwerpunkt (roh)": "Schwp roh",
            "Schwerpunkt (gespreizt)": "Schwerp.", "Clustering": "Cluster",
            "Snap (bester Treffer der Gruppe)": "Snap",
            "Gated (Gruppe nur bei Einigkeit)": "Gated"}
    namen = [kurz.get(w, w[:9]) for w in wege]
    breite = max(14, max(len(r["embedding_name"]) for r in laeufe) + 2)

    print(f"Lokalisierung  |  Anteil unter {args.threshold} m  |  "
          f"{laeufe[0]['n_queries']:,} Anfragen, Top-{laeufe[0]['top_k']} je Anfrage")
    print("Median des Fehlers in Klammern.\n")
    kopf = f"{'Encoder':<{breite}}" + "".join(f"{n:>17}" for n in namen)
    print(kopf)
    print("-" * len(kopf))
    for r in laeufe:
        v = r["verfahren"]
        zellen = []
        for w in wege:
            e = v.get(w)
            zellen.append(f"{e[schluessel]:>6.3f} ({e['median_m']:>6,.0f} m)"
                          if e and schluessel in e else f"{'-':>17}")
        print(f"{r['embedding_name']:<{breite}}" + "".join(f"{z:>17}" for z in zellen))
    z = laeufe[0]["verfahren"].get("Zufaelliges DB-Bild")
    if z and schluessel in z:
        print("-" * len(kopf))
        print(f"{'Zufall (DB-Bild)':<{breite}}{z[schluessel]:>6.3f} ({z['median_m']:>6,.0f} m)")
    print()
    print("Lesart: liegt Top-1 vorn, sind die Nachbartreffer zu oft falsch, als")
    print("dass Mitteln oder Clustern helfen koennte.")


def plot(laeufe, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["figure.dpi"] = 150
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    split, schwelle = args.split, args.threshold
    suffix = "_derived" if args.derived else ""
    geschrieben = []

    # ------------------------------------------------------------------
    # 1. R@1 je Zeile mit dem Bootstrap-Intervall, nach Deskriptorbreite
    #    sortiert. Der Adapter ist dabei eine Variante wie jede andere.
    # ------------------------------------------------------------------
    reihen = _reihen(laeufe, split, schwelle, 1)
    intervalle = bootstrap_intervals(args, still=True)
    reihen.sort(key=lambda r: (r[1], r[0], r[2] != "none"))
    encoder = [n for n, _, a, _ in reihen if a in ("none", "None")]

    if reihen:
        namen = [f"{n}" + ("" if a in ("none", "None") else f" +{a}") for n, _, a, _ in reihen]
        werte = [v for _, _, _, v in reihen]
        fehler = [
            [v - intervalle[f"{n}{'' if a in ('none', 'None') else '_' + a}"][0],
             intervalle[f"{n}{'' if a in ('none', 'None') else '_' + a}"][1] - v]
            if f"{n}{'' if a in ('none', 'None') else '_' + a}" in intervalle else [0, 0]
            for n, _, a, v in reihen
        ]
        farben = ["#37474f" if a in ("none", "None") else "#90a4ae" for _, _, a, _ in reihen]
        fig, ax = plt.subplots(figsize=(max(7, 0.42 * len(reihen) + 2), 4.8))
        x = range(len(reihen))
        ax.bar(x, werte, 0.7, color=farben,
               yerr=[[e[0] for e in fehler], [e[1] for e in fehler]] if intervalle else None,
               capsize=2, error_kw={"linewidth": 0.8, "color": "0.3"})
        ax.set_xticks(list(x))
        ax.set_xticklabels(namen, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(f"R@1 bei {schwelle} m")
        ax.set_title(f"{split}  |  Schwelle {schwelle} m  |  nach Deskriptorbreite sortiert"
                     + ("  |  95-%-Intervall aus dem Sequenz-Bootstrap" if intervalle else ""))
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        ziel = FIGURE_DIR / f"vergleich_r1_{schwelle}m{suffix}.png"
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
            stil = "-" if r["variant"] in ("none", "None") else "--"
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
        ziel = FIGURE_DIR / f"vergleich_recall_k_{schwelle}m{suffix}.png"
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
            stil = "-" if r["variant"] in ("none", "None") else "--"
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
        ziel = FIGURE_DIR / f"vergleich_schwellen{suffix}.png"
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
    ap.add_argument("--plot", action="store_true",
                    help="Vergleichsabbildungen nach results/figures/evaluation/ "
                         "schreiben, sonst nichts")
    ap.add_argument("--localization", action="store_true",
                    help="Statt Recall die Lokalisierung aus 08 vergleichen: "
                         "Anteil unter --threshold Metern und Median je Verfahren")
    ap.add_argument("--derived", action="store_true",
                    help="Auch die abgeleiteten Varianten (PCA, Whitening, Verkettung, "
                         "Sequenz) in Tabelle und Abbildungen -- standardmaessig nur "
                         "die echten Encoder")
    ap.add_argument("--ci", action="store_true",
                    help="95-%%-Intervall des Sequenz-Bootstraps neben R@1, "
                         "wenn experiments/results/bootstrap_ci.json vorliegt")
    ap.add_argument("--reference", choices=("database", "full"), default="database",
                    help="database = Benchmark-Protokoll (15 %% der Sequenzen als Referenz); "
                         "full = database + train als Referenz (experiments/full_reference.py)")
    args = ap.parse_args()

    if args.localization:
        localization_table(args)
        return

    laeufe = load(args.reference)
    if not laeufe:
        raise SystemExit(
            f"Keine Auswertungen in {EVAL_DIR.relative_to(ROOT)}.\n"
            "07_evaluation je Verfahren einmal laufen lassen"
            + (" -- fuer --reference full: python experiments/full_reference.py"
               if args.reference == "full" else "") + "."
        )

    # Standard: die echten Encoder und ihre Adapter. Die abgeleiteten
    # Varianten (PCA, Whitening, Verkettung, Sequenz) verdreifachen die
    # Tabelle -- --derived holt sie dazu.
    if not args.derived:
        laeufe = [r for r in laeufe if not _abgeleitet(r["method"])]

    if args.plot:
        plot(laeufe, args)
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
    referenz = zeilen[0][0]["n_database"]
    print(f'{args.split}  |  Schwelle {args.threshold} m  |  {loesbar:,} loesbare Queries  |  '
          f'Referenz: {referenz:,} Bilder'
          + (' (database + train)' if args.reference == "full" else ' (database)') + '\n')
    intervalle = bootstrap_intervals(args) if args.ci else {}
    # Spaltenbreite am laengsten Namen ausrichten -- die PCA-Varianten sind
    # laenger als die urspruenglichen Encoder.
    breite = max(14, max(len(r["method"]) for r, _ in zeilen) + 2)
    # Das Intervall steht direkt hinter R@1 -- der Spalte, um die es geht.
    spalten = {k: f"{'R@'+k:>8}" for k in ks}
    if intervalle:
        spalten["1"] += f"{'95-%-KI':>17}"
    kopf = f"{'Encoder':<{breite}}{'Variante':<10}{'Dim':>6}   " + "".join(spalten[k] for k in ks)
    print(kopf)
    print("-" * len(kopf))
    for r, eintrag in zeilen:
        werte = ""
        for k in ks:
            wert = eintrag["recall"][k]
            werte += f"{wert:>8.3f}" if wert is not None else f"{'-':>8}"
            if intervalle and k == "1":
                ci = intervalle.get(r["embedding_name"])
                werte += f"  [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else f"{'-':>17}"
        variante = "none" if r["variant"] == "fullref" else r["variant"]
        print(f"{r['method']:<{breite}}{variante:<10}{r['dim']:>6}   {werte}")

    # Zufallsbasis als Fussnote: was blindes Raten erreicht. Haengt nicht
    # vom Encoder ab, steht deshalb in jeder JSON gleich -- die erste reicht.
    # Fehlt sie, stammt die Auswertung von vor dieser Ergaenzung.
    zufall = next((e.get("zufall") for _, e in zeilen if e.get("zufall")), None)
    if zufall:
        werte = ""
        for k in ks:
            werte += f"{zufall[k]:>8.4f}" if zufall.get(k) is not None else f"{'-':>8}"
            if intervalle and k == "1":
                werte += f"{'':>17}"
        print("-" * len(kopf))
        print(f"{'Zufall':<{breite}}{'(Raten)':<10}{'':>6}   {werte}")

    if intervalle:
        print("\nIntervall: Sequenz-Bootstrap, 2,5- und 97,5-Perzentil "
              "(experiments/bootstrap_ci.py). Fuer den Vergleich zweier Zeilen "
              "gilt die gepaarte Differenz dort, nicht die Ueberlappung.")



if __name__ == "__main__":
    main()
