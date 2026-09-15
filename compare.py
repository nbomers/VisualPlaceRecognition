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


def _config():
    from src.config import load_config, paths
    cfg = load_config(ROOT)
    return cfg, paths(cfg, ROOT)


CFG, PATHS = _config()
EVAL_DIR = PATHS.evaluation

# Die Schwelle der Ground Truth steht in der config, nicht zweimal. Frueher
# war die 25 hier ein argparse-Literal -- wer vpr.uncertain_radius_m aenderte,
# aenderte die Tabelle nicht mit.
STANDARD_SCHWELLE = int(float(CFG["vpr"]["uncertain_radius_m"]))


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
    block = CFG["vpr"].get(method)
    return isinstance(block, dict) and ("source" in block or "sources" in block)


def _basis_encoder(method):
    """
    Der echte Encoder hinter einem Namen: megaloc_pcaw512 -> megaloc,
    eigenplaces_megaloc_concat -> eigenplaces (erste Quelle). Die Kette wird
    verfolgt, bis ein Name ohne source/sources erreicht ist.
    """
    gesehen = set()
    while _abgeleitet(method) and method not in gesehen:
        gesehen.add(method)
        block = CFG["vpr"][method]
        method = block["source"] if "source" in block else block["sources"][0]
    return method


def _variantenfamilie(method):
    """
    Was die Ableitung mit dem Deskriptor gemacht hat: "" (Basis), "pca512",
    "pcaw512", "concat", ... -- der Namensteil hinter dem Basis-Encoder.
    Dieselbe Familie heisst bei jedem Encoder gleich und bekommt deshalb
    ueber alle Facetten hinweg dieselbe Farbe.
    """
    basis = _basis_encoder(method)
    if method == basis:
        return ""
    rest = method[len(basis):].lstrip("_") if method.startswith(basis) else method
    # eigenplaces_megaloc_concat -> "concat", nicht "megaloc_concat".
    return "concat" if rest.endswith("concat") else (rest or "")


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


# ----------------------------------------------------------------------
# Abbildungen
#
# Farbe traegt Identitaet, nie Rang: dieselbe Variantenfamilie hat in jeder
# Facette dieselbe Farbe, derselbe Encoder in jeder Abbildung. Ueber acht
# Serien in einem Achsenpaar gibt es nicht -- darueber wird facettiert statt
# weitere Farben zu erfinden. Der Adapter ist keine eigene Farbe, sondern der
# Linienstil: zwei Kanaele fuer zwei unabhaengige Fragen.
# ----------------------------------------------------------------------

# Validierte kategoriale Reihenfolge (helle Flaeche), feste Reihenfolge, nie
# zyklisch weitergedreht. Gilt fuer die Encoder: in Balken und in der einen
# Kurvenachse liegen nur benachbarte Paare nebeneinander, dafuer traegt diese
# Reihenfolge (schlechtestes benachbartes Paar CVD dE 9.1, Normalsicht 19.6).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# In den kleinen Vielfachen steht jede Farbe gegen jede andere (die Legende
# zeigt alle zugleich), und dafuer traegt die volle Reihenfolge nicht: Gruen
# gegen Orange liegt bei CVD dE 3.2. Diese sechs sind die Auswahl daraus, die
# auch ueber ALLE Paare besteht. Dazu je Familie eine eigene Markerform --
# die Identitaet haengt damit nicht an der Farbe allein.
FAMILIEN_PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
FAMILIEN_MARKER = ["o", "s", "^", "D", "v", "P"]
INK = "#1c1c1c"
INK_LEISE = "#5c5c5c"
GITTER = "#d9d9d6"

# Linienstil je Auswertungsvariante -- gilt in allen Abbildungen gleich.
STILE = {"none": ("-", "Baseline"), "None": ("-", "Baseline"),
         "linear": ("--", "+ linearer Adapter"), "seq3": (":", "+ Sequenz ±3")}

# Sprechende Namen der Deskriptorvarianten, in fester Reihenfolge.
FAMILIEN = [("", "Basis"), ("pca512", "PCA 512"), ("pcaw512", "PCA+Whitening 512"),
            ("pcaw2048", "Whitening 2048"), ("pcaw4096", "Whitening 4096"),
            ("concat", "Verkettung")]
FAMILIE_LABEL = dict(FAMILIEN)


def _stil(variant):
    return STILE.get(variant, ("-.", variant))


def _achse_aufraeumen(ax):
    """Recessives Gitter, keine Rahmen oben und rechts."""
    ax.grid(color=GITTER, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for kante in ("top", "right"):
        ax.spines[kante].set_visible(False)
    for kante in ("left", "bottom"):
        ax.spines[kante].set_color(GITTER)
    ax.tick_params(colors=INK_LEISE, labelsize=8, length=0)


def _kurven(laeufe, split, x_werte, wert_von):
    """(name, method, variant, dim, y-Liste) je Lauf, der alle x-Werte hergibt."""
    raus = []
    for r in sorted(laeufe, key=lambda r: -r["dim"]):
        a = r["auswertungen"].get(split)
        if a is None:
            continue
        y = [wert_von(a, x) for x in x_werte]
        if any(v is None for v in y):
            continue
        raus.append((r["embedding_name"], r["method"], r["variant"], r["dim"], y))
    return raus


def _zeichne_kurven(ax, kurven, farbe_von, label_von, marker_von=None):
    for name, method, variant, dim, y in kurven:
        stil, _ = _stil(variant)
        ax.plot(range(len(y)), y, stil, color=farbe_von(method), linewidth=2.2,
                marker=marker_von(method) if marker_von else "o", markersize=5.5,
                markeredgecolor="white", markeredgewidth=0.8,
                label=label_von(name, method, dim), solid_capstyle="round")


def _facetten(laeufe, split, x_werte, wert_von, x_label, y_label, titel, ziel, args):
    """
    Eine Kurvenabbildung. Bis acht Serien in einem Achsenpaar mit Legende
    daneben; darueber (--derived) kleine Vielfache je Encoder -- Farbe ist
    dann die Deskriptorvariante, die in jeder Facette dasselbe bedeutet.
    """
    import matplotlib.pyplot as plt

    kurven = _kurven(laeufe, split, x_werte, wert_von)
    if not kurven:
        return None

    def achse_beschriften(ax, mit_x=True, mit_y=True):
        ax.set_xticks(range(len(x_werte)))
        ax.set_xticklabels([str(x) for x in x_werte])
        if mit_x:
            ax.set_xlabel(x_label, color=INK_LEISE, fontsize=9)
        if mit_y:
            ax.set_ylabel(y_label, color=INK_LEISE, fontsize=9)
        _achse_aufraeumen(ax)

    if not args.derived:
        # Eine Achse, Farbe = Encoder.
        encoder = sorted({_basis_encoder(m) for _, m, _, _, _ in kurven})
        farben = {e: PALETTE[i % len(PALETTE)] for i, e in enumerate(encoder)}
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        _zeichne_kurven(ax, kurven,
                        lambda m: farben[_basis_encoder(m)],
                        lambda n, m, d: f"{n}  ({d}d)")
        achse_beschriften(ax)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
                  fontsize=8, labelcolor=INK, handlelength=2.4)
        fig.suptitle(titel, color=INK, fontsize=11, x=0.02, ha="left")
        ax.set_title("durchgezogen = Baseline, gestrichelt = Adapter",
                     color=INK_LEISE, fontsize=8.5, loc="left")
    else:
        # Kleine Vielfache: ein Feld je Encoder, Farbe = Deskriptorvariante.
        encoder = sorted({_basis_encoder(m) for _, m, _, _, _ in kurven})
        familien = [f for f, _ in FAMILIEN
                    if f in {_variantenfamilie(m) for _, m, _, _, _ in kurven}]
        farben = {f: FAMILIEN_PALETTE[i % len(FAMILIEN_PALETTE)]
                  for i, f in enumerate(familien)}
        marker = {f: FAMILIEN_MARKER[i % len(FAMILIEN_MARKER)]
                  for i, f in enumerate(familien)}
        spalten = min(3, len(encoder))
        zeilen = -(-len(encoder) // spalten)
        fig, achsen = plt.subplots(zeilen, spalten, figsize=(4.0 * spalten, 3.1 * zeilen),
                                   sharey=True, squeeze=False)
        alle = [a for reihe in achsen for a in reihe]
        for i, (ax, enc) in enumerate(zip(alle, encoder)):
            teil = [k for k in kurven if _basis_encoder(k[1]) == enc]
            _zeichne_kurven(ax, teil,
                            lambda m: farben[_variantenfamilie(m)],
                            lambda n, m, d: None,
                            lambda m: marker[_variantenfamilie(m)])
            ax.set_title(enc, color=INK, fontsize=10, loc="left")
            # Achsen nur beschriften, wo es nicht dreimal dasselbe waere:
            # x unten in jeder Spalte, y links in jeder Zeile.
            unterste = i + spalten >= len(encoder)
            achse_beschriften(ax, mit_x=unterste, mit_y=(i % spalten == 0))
        frei = alle[len(encoder):]
        for ax in frei:
            ax.set_visible(False)

        from matplotlib.lines import Line2D
        eintraege = [Line2D([], [], color=farben[f], linewidth=2.2,
                            marker=marker[f], markersize=5.5,
                            markeredgecolor="white", markeredgewidth=0.8,
                            label=FAMILIE_LABEL.get(f, f or "Basis"))
                     for f in familien]
        varianten = sorted({v for _, _, v, _, _ in kurven}, key=lambda v: v != "none")
        eintraege += [Line2D([], [], color=INK_LEISE, linewidth=1.6,
                             linestyle=_stil(v)[0], label=_stil(v)[1]) for v in varianten]
        if frei:
            # Bleibt ein Feld im Raster leer, gehoert die Legende dorthin --
            # kein Platz unter der Abbildung, keine Kollision mit der Achse.
            legende_ax = frei[0]
            legende_ax.set_visible(True)
            legende_ax.axis("off")
            legende_ax.legend(handles=eintraege, loc="center left", frameon=False,
                              fontsize=9, labelcolor=INK, handlelength=2.6,
                              borderaxespad=0.0)
        else:
            fig.legend(handles=eintraege, loc="upper center", ncol=min(5, len(eintraege)),
                       frameon=False, fontsize=8.5, labelcolor=INK,
                       bbox_to_anchor=(0.5, 0.035))
            fig.subplots_adjust(bottom=0.16)
        fig.suptitle(titel, color=INK, fontsize=11.5, x=0.01, ha="left")

    fig.tight_layout()
    fig.savefig(ziel, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ziel


def plot(laeufe, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["font.size"] = 9
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    split, schwelle = args.split, args.threshold
    suffix = "_derived" if args.derived else ""
    geschrieben = []

    # ------------------------------------------------------------------
    # 1. R@1 je Zeile mit dem Bootstrap-Intervall.
    #    Liegende Balken, nach Wert sortiert: 37 Namen lesen sich waagerecht,
    #    um 60 Grad gedreht nicht. Farbe = Encoder, der Variantenname steht
    #    ohnehin daneben.
    # ------------------------------------------------------------------
    reihen = _reihen(laeufe, split, schwelle, 1)
    intervalle = bootstrap_intervals(args, still=True)
    reihen.sort(key=lambda r: r[3])          # kleinster Wert unten
    encoder_liste = sorted({_basis_encoder(n) for n, _, _, _ in reihen})
    farben_enc = {e: PALETTE[i % len(PALETTE)] for i, e in enumerate(encoder_liste)}

    if reihen:
        def voller_name(n, a):
            return n if a in ("none", "None") else f"{n}_{a}"

        namen = [f"{n}" + ("" if a in ("none", "None") else f"  +{a}")
                 for n, _, a, _ in reihen]
        werte = [v for _, _, _, v in reihen]
        farben = [farben_enc[_basis_encoder(n)] for n, _, _, _ in reihen]
        unten, oben = [], []
        for n, _, a, v in reihen:
            ci = intervalle.get(voller_name(n, a))
            unten.append(v - ci[0] if ci else 0.0)
            oben.append(ci[1] - v if ci else 0.0)

        hoehe = max(3.2, 0.26 * len(reihen) + 1.4)
        fig, ax = plt.subplots(figsize=(8.4, hoehe))
        y = range(len(reihen))
        ax.barh(y, werte, 0.68, color=farben,
                xerr=[unten, oben] if intervalle else None,
                capsize=2, error_kw={"linewidth": 0.9, "ecolor": INK_LEISE})
        ax.set_yticks(list(y))
        ax.set_yticklabels(namen, fontsize=8, color=INK)
        ax.set_xlabel(f"R@1 bei {schwelle} m", color=INK_LEISE, fontsize=9)
        ax.set_xlim(0, min(1.0, max(werte + [0.1]) * 1.18))
        _achse_aufraeumen(ax)
        ax.grid(axis="y", visible=False)
        # Direkte Werte statt einer zweiten Ableseachse.
        for yi, (v, o) in enumerate(zip(werte, oben)):
            ax.text(v + o + 0.012, yi, f"{v:.3f}", va="center", fontsize=7.5,
                    color=INK_LEISE)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=farben_enc[e], label=e) for e in encoder_liste],
                  loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
        fig.suptitle(f"{split}  |  Schwelle {schwelle} m", color=INK,
                     fontsize=11.5, x=0.01, ha="left")
        ax.set_title("Fehlerbalken: 95-%-Intervall aus dem Sequenz-Bootstrap"
                     if intervalle else "ohne Intervalle -- python experiments/bootstrap_ci.py",
                     color=INK_LEISE, fontsize=8.5, loc="left")
        fig.tight_layout()
        ziel = FIGURE_DIR / f"vergleich_r1_{schwelle}m{suffix}.png"
        fig.savefig(ziel, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        geschrieben.append(ziel)

    # ------------------------------------------------------------------
    # 2. Recall ueber k -- die uebliche VPR-Kurve.
    # ------------------------------------------------------------------
    ks = sorted({int(k)
                 for r in laeufe
                 for a in [r["auswertungen"].get(split)] if a
                 for e in a["schwellen"].values()
                 for k in e["recall"]})
    if ks:
        ziel = _facetten(
            laeufe, split, ks,
            lambda a, k: a["schwellen"].get(str(schwelle), {}).get("recall", {}).get(str(k)),
            "k", f"Recall@k bei {schwelle} m",
            f"{split}  |  Recall@k bei {schwelle} m",
            FIGURE_DIR / f"vergleich_recall_k_{schwelle}m{suffix}.png", args)
        if ziel:
            geschrieben.append(ziel)

    # ------------------------------------------------------------------
    # 3. R@1 ueber die Distanzschwelle -- zeigt, wie streng die 25 m sind.
    # ------------------------------------------------------------------
    schwellen = sorted({int(s)
                        for r in laeufe
                        for a in [r["auswertungen"].get(split)] if a
                        for s in a["schwellen"]})
    if schwellen:
        ziel = _facetten(
            laeufe, split, schwellen,
            lambda a, t: a["schwellen"].get(str(t), {}).get("recall", {}).get("1"),
            "Distanzschwelle [m]", "R@1",
            f"{split}  |  wie streng ist die Ground Truth?",
            FIGURE_DIR / f"vergleich_schwellen{suffix}.png", args)
        if ziel:
            geschrieben.append(ziel)

    for z in geschrieben:
        print(f"geschrieben: {z.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(
        description="Vergleichstabelle aus results/evaluation/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--threshold", type=int, default=STANDARD_SCHWELLE,
                    help=f"Distanzschwelle in Metern (Standard: {STANDARD_SCHWELLE}, "
                         "aus config.yaml -> vpr.uncertain_radius_m)")
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
