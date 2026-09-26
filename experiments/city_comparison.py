"""
Dieselbe Pipeline, mehrere Staedte -- was sich uebertraegt und was nicht.

compare.py stellt Encoder INNERHALB einer Stadt gegenueber, gepaart ueber
dieselben Anfragen. Dieses Skript stellt STAEDTE gegenueber, und das ist eine
andere Rechnung: zwei Staedte haben disjunkte Anfragen, es gibt nichts zu
paaren. Uebrig bleibt der Vergleich unabhaengiger Schaetzer, jeder mit seinem
eigenen, breiten Intervall.

Gelesen wird ausschliesslich Versioniertes:

    results/<stadt>/evaluation/<encoder>.json          R@1 je Auswertung
    results/<stadt>/evaluation/<encoder>_fullref.json  dasselbe, volle Referenz
    experiments/results/<stadt>/bootstrap_ci*.json     Intervalle
    experiments/results/<stadt>/recall_by_difficulty_<encoder>.json
                                                      Herkunft des Top-1-Treffers
    results/<stadt>/evaluation/<encoder>_gv<k>.json    geometrische Verifikation
    experiments/results/city_coverage.json             Abdeckung, Panorama, Jahre

Embeddings und Trefferlisten braucht es nicht. Das Skript laeuft also in
jedem frischen Klon und auf jedem Rechner -- anders als alles andere unter
experiments/.

    python experiments/city_comparison.py
    python experiments/city_comparison.py --method eigenplaces --plot

ZU DEN P-WERTEN: scipy.stats.spearmanr liefert bei perfekter Monotonie einen
p-Wert von 0 -- seine t-Naeherung dividiert durch eine Varianz, die dort
verschwindet. Bei vier Staedten ist der exakte Wert 0,083, also NICHT
signifikant. Dieses Skript zaehlt deshalb die Permutationen durch.
"""

import argparse
import json
from itertools import permutations
from math import factorial

import numpy as np

from _common import CFG, RESULTS, ROOT

OUT = RESULTS.parent / "city_comparison.json"


def _args():
    ap = argparse.ArgumentParser(
        description="Recall, Intervalle und Datensatzeigenschaften ueber Staedte.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--method", default="megaloc", help="Encoder (Standard: megaloc)")
    ap.add_argument("--plot", action="store_true", help="Streudiagramm dazu schreiben")
    return ap.parse_args()


# -- Rangkorrelation mit exaktem p ----------------------------------------

def _rho(x, y):
    """Spearman ueber Raenge -- ohne scipy, damit die Permutation schnell bleibt."""
    n = len(x)
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    d = rx - ry
    return 1 - 6 * float((d * d).sum()) / (n * (n * n - 1))


def rangkorrelation(x, y, max_n=8):
    """
    (rho, p, art). p ist exakt, solange n! handhabbar bleibt: der Anteil der
    Permutationen mit mindestens so starkem |rho|. Darueber None -- lieber
    keine Zahl als eine falsche.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3:
        return None, None, "zu wenige Punkte"
    r = _rho(x, y)
    if n > max_n:
        return r, None, f"n={n} zu gross fuer den exakten Test"
    basis = np.arange(n)
    treffer = sum(1 for q in permutations(basis) if abs(_rho(basis, np.array(q))) >= abs(r) - 1e-12)
    return r, treffer / factorial(n), "exakt"


# -- Einlesen --------------------------------------------------------------

def _schwelle():
    t = CFG["retrieval"]["thresholds"]
    return str(t[len(t) // 2])


def _recall(block, schwelle):
    return block["schwellen"][schwelle]["recall"]["1"]


def _stadt(slug, method, schwelle):
    """Alles, was zu einer Stadt versioniert vorliegt. Fehlendes bleibt None."""
    ev = ROOT / "results" / slug / "evaluation"
    bo = ROOT / "experiments" / "results" / slug
    d = {"stadt": slug}

    haupt = ev / f"{method}.json"
    if not haupt.exists():
        return None
    a = json.loads(haupt.read_text(encoding="utf-8"))["auswertungen"]
    d["alle"] = _recall(a["Alle Queries"], schwelle)
    # Der Anteil der Anfragen mit ueberhaupt einem Datenbankbild im Umkreis.
    # Das ist die Groesse, die abdeckung_gesamt zu messen VORGIBT: jene zaehlt
    # den Gesamtbestand gegen das Strassennetz, die Datenbank sind aber 15 %
    # der Sequenzen. Eine Strasse, die nur eine Fahrt kennt, liegt zu 70 % in
    # train -- abgedeckt laut Kennzahl, in der Datenbank nicht vorhanden.
    ganz = a["Alle Queries"]
    d["loesbar_n"] = ganz["schwellen"][schwelle]["loesbar"]
    if ganz.get("n_queries"):
        d["loesbar_anteil"] = d["loesbar_n"] / ganz["n_queries"]
    hard = next((v for k, v in a.items() if k.startswith("Hard")), None)
    d["hard"] = _recall(hard, schwelle) if hard else None
    if hard:
        # Der Nenner der Hard-Auswertung: Anfragen mit mindestens einem
        # hard-konformen Datenbankbild im Umkreis. Er schrumpft mit, und
        # genau das trennt den Abschlag vom Dublettenanteil.
        d["loesbar_hard"] = hard["schwellen"][schwelle]["loesbar"]
    pano = a.get("Nur Nicht-Panorama-Queries")
    if pano:
        d["ohne_panorama"] = _recall(pano, schwelle)
        # Mit dieser Zahl laesst sich R@1 der Panorama-Anfragen exakt
        # zurueckrechnen -- 07 wertet sie nicht eigens aus.
        d["loesbar_ohne_panorama"] = pano["schwellen"][schwelle]["loesbar"]

    voll = ev / f"{method}_fullref.json"
    if voll.exists():
        d["voll"] = _recall(json.loads(voll.read_text(encoding="utf-8"))["auswertungen"]["Alle Queries"], schwelle)

    # Herkunft des Top-1-Treffers, falls recall_by_difficulty.py gelaufen ist.
    # Das ist die direkte Messung des Dubletteneffekts: wieviele korrekte
    # Treffer stammen vom selben Konto aus demselben Zeitfenster -- also aus
    # derselben Befahrung, nur in einer anderen Sequenz.
    hk = bo / f"recall_by_difficulty_{method}.json"
    if hk.exists():
        h = json.loads(hk.read_text(encoding="utf-8")).get("herkunft_top1")
        if h:
            d["dublette"] = h["anteil_dublette"]
            d["median_tage"] = h["median_tage"]
            d["n_korrekt"] = h["n_korrekt"]

    for datei, feld in (("bootstrap_ci.json", "halbbreite"),
                        ("bootstrap_ci_fullref.json", "halbbreite_voll")):
        p = bo / datei
        if not p.exists():
            continue
        roh = json.loads(p.read_text(encoding="utf-8"))
        name = method if datei.endswith("ci.json") else f"{method}_fullref"
        eintrag = roh["encoder"].get(name)
        if eintrag:
            d[feld] = eintrag["recall"]["1"]["halbbreite"]
            d["n_sequenzen"] = roh["n_sequences"]
    return d


def _eigenschaften():
    """Abdeckung, Panoramaanteil und Jahre aus city_coverage.json, nach Slug."""
    from src.paths import city_slug
    p = ROOT / "experiments" / "results" / "city_coverage.json"
    if not p.exists():
        return {}
    return {city_slug(name): s for name, s in json.loads(p.read_text(encoding="utf-8")).items()}


# -- Ausgabe ---------------------------------------------------------------

def _zelle(wert, form, breite):
    """Zahl oder Gedankenstrich -- fehlende Werte sollen die Spalte nicht sprengen."""
    return form.format(wert) if wert is not None else f"{'—':>{breite}}"


def _tabelle(zeilen):
    print(f"{'Stadt':<16}{'Abd':>6}{'loesbar':>9}{'Pano':>7}{'Dubl':>7}{'Tage':>7}"
          f"{'Alle':>7}{'Hard':>8}{'voll':>7}{'±boot':>8}")
    print("-" * 82)
    for z in zeilen:
        hard = z["hard"] - z["alle"] if z.get("hard") is not None else None
        print(f"{z['stadt']:<16}"
              + _zelle(z.get("abdeckung"), "{:>6.2f}", 6)
              + _zelle(z.get("loesbar_anteil"), "{:>9.1%}", 9)
              + _zelle(z.get("panorama"), "{:>7.1%}", 7)
              + _zelle(z.get("dublette"), "{:>7.1%}", 7)
              + _zelle(z.get("median_tage"), "{:>7.0f}", 7)
              + _zelle(z.get("alle"), "{:>7.3f}", 7)
              + _zelle(hard, "{:>+8.3f}", 8)
              + _zelle(z.get("voll"), "{:>7.3f}", 7)
              + _zelle(z.get("halbbreite"), "{:>8.3f}", 8))


def zerlegung(zeile):
    """
    Den Hard-Abschlag aus der Herkunftsmessung ausrechnen -- exakt, nicht
    geschaetzt.

    Der Hard-Filter aendert nicht die Trefferliste, sondern die Ground Truth:
    ein Datenbankbild zaehlt nur, wenn es von einem anderen Konto stammt ODER
    mehr als min_days_apart entfernt ist (src/evaluation.py, `disjoint`).
    Daraus folgt beides:

      Zaehler   Ein korrekter Top-1-Treffer faellt genau dann weg, wenn er
                eine Dublette ist -- das ist die Negation von `disjoint` und
                damit exakt `anteil_dublette` aus treffer_herkunft.
      Nenner    Wer einen korrekten, nicht-dublettigen Treffer hat, ist
                automatisch auch hard-loesbar (dieses Bild ist ja selbst eine
                gueltige Referenz im Umkreis). Es faellt also nichts aus dem
                Zaehler, was nicht schon gezaehlt waere -- aber der Nenner
                schrumpft eigenstaendig, um Anfragen, deren einzige Referenz
                eine Dublette war.

        R@1_hard = n_korrekt * (1 - anteil_dublette) / loesbar_hard

    Damit ist der Abschlag kein Raetsel mehr, sondern ein Bruch aus zwei
    gemessenen Groessen: dem Dublettenanteil und r = loesbar_hard/loesbar.
    Er verschwindet genau dann, wenn r = 1 - anteil_dublette -- wenn also der
    Nenner im selben Mass schrumpft wie der Zaehler.

    Gibt None zurueck, solange eine der vier Zahlen fehlt.
    """
    noetig = ("n_korrekt", "dublette", "loesbar_hard", "loesbar_n", "hard", "alle")
    if any(zeile.get(k) is None for k in noetig):
        return None
    r = zeile["loesbar_hard"] / zeile["loesbar_n"]
    erwartet = zeile["n_korrekt"] * (1 - zeile["dublette"]) / zeile["loesbar_hard"]
    return {"r": r, "erwartet": erwartet, "gemessen": zeile["hard"],
            "rest": zeile["hard"] - erwartet,
            "erwarteter_abschlag": erwartet - zeile["alle"],
            # Zwei Masse, die dasselbe Wort "Dublette" tragen und doch
            # verschiedene Dinge zaehlen:
            #   d     Anteil der korrekten TREFFER, die Dubletten sind
            #   1-r   Anteil der loesbaren ANFRAGEN, deren einzige Referenz
            #         eine Dublette war -- die strukturelle Abhaengigkeit
            # Ihre Differenz ist der Abschlag: rel = -(d - (1-r)) / r.
            # Sie verschwindet, wenn das System Dubletten nur in dem Mass
            # nutzt, in dem der Datensatz sie erzwingt.
            "strukturell": 1 - r,
            "uebernutzung": zeile["dublette"] - (1 - r),
            "rel_abschlag": (zeile["hard"] - zeile["alle"]) / zeile["alle"]}


def _zerlegungstabelle(zeilen):
    """
    Die Identitaet je Stadt gegen die gemessene Zahl halten.

    Das ist kein Befund ueber die Welt, sondern ein Test: stimmt die Spalte
    `Rest` nicht auf drei Nachkommastellen mit 0 ueberein, messen die beiden
    Skripte nicht dasselbe, und dann taugt der Dublettenbefund nichts.
    """
    mit = [(z, zerlegung(z)) for z in zeilen]
    mit = [(z, x) for z, x in mit if x]
    if not mit:
        print("\nHard-Abschlag zerlegt: keine Stadt mit Herkunftsmessung "
              "(experiments/recall_by_difficulty.py).")
        return {}
    print("\nHard-Abschlag zerlegt   rel. Abschlag = -(d - (1-r)) / r")
    print("  d   = Anteil der korrekten Treffer, die Dubletten sind")
    print("  1-r = Anteil der loesbaren Anfragen, die NUR durch Dubletten loesbar waren")
    print(f"{'Stadt':<16}{'d':>7}{'1-r':>7}{'d-(1-r)':>10}{'rel.Absch':>11}{'Rest':>8}")
    print("-" * 59)
    for z, x in mit:
        print(f"{z['stadt']:<16}{z['dublette']:>7.1%}{x['strukturell']:>7.1%}"
              f"{x['uebernutzung']:>+10.3f}{x['rel_abschlag']:>+11.3f}{x['rest']:>+8.3f}")
    groesster = max(abs(x["rest"]) for _, x in mit)
    if groesster > 0.002:
        print(f"  ACHTUNG: groesster Rest {groesster:+.4f}. Die Identitaet gilt "
              f"nur, wenn beide Skripte dieselbe Schwelle, dasselbe "
              f"min_days_apart und dieselbe Trefferliste benutzt haben.")
    else:
        print(f"  Groesster Rest {groesster:.4f} -- die Zerlegung traegt.")
    return {z["stadt"]: x for z, x in mit}


def panorama(zeile):
    """
    R@1 der Panorama-Anfragen -- exakt, obwohl 07 sie nicht eigens auswertet.

    07 rechnet "Alle Queries" und "Nur Nicht-Panorama-Queries". Die Differenz
    der Trefferzahlen ist die Trefferzahl AUF den Panoramen, die Differenz der
    loesbaren Anfragen ihre Zahl:

        R@1_pano = (R_alle * L - R_ohne * L_ohne) / (L - L_ohne)

    Das ist der direkte Wert, nicht die bisher berichtete Hochrechnung
    "Differenz geteilt durch Panoramaanteil" -- und er sagt dasselbe, nur
    ohne Umweg.
    """
    noetig = ("alle", "loesbar_n", "ohne_panorama", "loesbar_ohne_panorama")
    if any(zeile.get(k) is None for k in noetig):
        return None
    n = zeile["loesbar_n"] - zeile["loesbar_ohne_panorama"]
    if n < 1:
        return None
    treffer = (zeile["alle"] * zeile["loesbar_n"]
               - zeile["ohne_panorama"] * zeile["loesbar_ohne_panorama"])
    r = treffer / n
    return {"n_pano": n, "recall_pano": r, "recall_ohne": zeile["ohne_panorama"],
            "strafe": zeile["ohne_panorama"] - r,
            # Binomialer Standardfehler, optimistisch (Sequenzen sind korreliert),
            # aber er zeigt schon, wann n zu klein ist, um etwas zu behaupten.
            "se_binomial": (r * (1 - r) / n) ** 0.5}


def _panoramatabelle(zeilen):
    mit = [(z, panorama(z)) for z in zeilen]
    mit = [(z, x) for z, x in mit if x]
    if not mit:
        print("\nPanorama-Anfragen: keine Stadt mit getrennter Auswertung.")
        return {}
    print("\nPanorama-Anfragen   R@1 getrennt, exakt aus beiden Auswertungen von 07")
    print(f"{'Stadt':<16}{'n':>8}{'R@1 Pano':>10}{'R@1 ohne':>10}{'Strafe':>9}{'+-SE':>8}")
    print("-" * 61)
    for z, x in sorted(mit, key=lambda t: -t[1]["n_pano"]):
        print(f"{z['stadt']:<16}{x['n_pano']:>8,}{x['recall_pano']:>10.3f}"
              f"{x['recall_ohne']:>10.3f}{x['strafe']:>9.3f}{x['se_binomial']:>8.3f}")
    gross = [x for _, x in mit if x["n_pano"] >= 1000]
    if len(gross) >= 2:
        sp = [x["strafe"] for x in gross]
        print(f"  Ueber die {len(gross)} Staedte mit mindestens 1.000 Panorama-Anfragen: "
              f"Strafe {min(sp):.3f} bis {max(sp):.3f}")
    return {z["stadt"]: x for z, x in mit}


def geometrische_verifikation(slug, method, schwelle):
    """
    Die gv-Zeile einer Stadt (experiments/geometric_verification.py
    --n-queries 0) mit gepaarter Differenz aus bootstrap_ci.json, oder None.
    Die Differenz gibt es erst, wenn der Bootstrap nach dem gv-Lauf lief.
    """
    ev = ROOT / "results" / slug / "evaluation"
    laeufe = sorted(ev.glob(f"{method}_gv*.json"))
    if not laeufe:
        return None
    r = json.loads(laeufe[0].read_text(encoding="utf-8"))
    x = {"name": r["embedding_name"], "top_k": r.get("verification_top_k"),
         "gv": _recall(r["auswertungen"]["Alle Queries"], schwelle),
         "anteil_verifiziert": r.get("anteil_verifiziert"),
         "n_verifiziert": r.get("n_verifiziert"),
         "stunden": r["dauer_s"] / 3600 if r.get("dauer_s") else None,
         "differenz": None, "ci": None}
    boot = ROOT / "experiments" / "results" / slug / "bootstrap_ci.json"
    if boot.exists():
        roh = json.loads(boot.read_text(encoding="utf-8"))
        paar = next((q for q in roh.get("paare", []) if q["a"] == method
                     and q["b"] == x["name"] and q["k"] == 1), None)
        if paar:
            x["differenz"], x["ci"] = paar["differenz"], paar["ci"]
    return x


def _gv_tabelle(zeilen, method, schwelle):
    mit = [(z, geometrische_verifikation(z["stadt"], method, schwelle)) for z in zeilen]
    if not any(x for _, x in mit):
        print("\nGeometrische Verifikation: in keiner Stadt ueber alle Anfragen gerechnet.")
        return {}
    print("\nGeometrische Verifikation   Top-k nach SuperPoint + LightGlue umsortiert, "
          "Differenz gepaart")
    print(f"{'Stadt':<16}{'R@1':>7}{'+GV':>7}{'dR@1':>8}{'95-%-Intervall':>19}"
          f"{'verif.':>8}{'Anfragen':>10}{'h':>6}")
    print("-" * 81)
    for z, x in mit:
        x = x or {}
        ci = x.get("ci")
        print(f"{z['stadt']:<16}"
              + _zelle(z.get("alle"), "{:>7.3f}", 7)
              + _zelle(x.get("gv"), "{:>7.3f}", 7)
              + _zelle(x.get("differenz"), "{:>+8.3f}", 8)
              + (f"   [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else f"{'—':>19}")
              + _zelle(x.get("anteil_verifiziert"), "{:>8.1%}", 8)
              + _zelle(x.get("n_verifiziert"), "{:>10,}", 10)
              + _zelle(x.get("stunden"), "{:>6.1f}", 6))
    ohne_ci = [z["stadt"] for z, x in mit if x and x["ci"] is None]
    if ohne_ci:
        print(f"  Ohne Intervall: {', '.join(ohne_ci)} -- dort bootstrap_ci.py nach dem gv-Lauf starten.")
    return {z["stadt"]: x for z, x in mit if x}


def _zusammenhang(titel, zeilen, a, b, deutung):
    paare = [(z[a], z[b]) for z in zeilen if z.get(a) is not None and z.get(b) is not None]
    if len(paare) < 3:
        print(f"\n{titel}: zu wenige Staedte ({len(paare)})")
        return None
    x, y = zip(*paare)
    rho, p, art = rangkorrelation(x, y)
    pt = f"p = {p:.4f} ({art})" if p is not None else art
    print(f"\n{titel}")
    print(f"  n = {len(paare)}   Spearman rho = {rho:+.2f}   {pt}")
    print(f"  {deutung}")
    if p is not None and p > 0.05:
        # Die Schranke nach unten: perfekte Monotonie hat p = 2/n! (beide
        # Richtungen). Bei vier Staedten sind das 0,083 -- da ist ueberhaupt
        # nichts signifikant zu bekommen; ab fuenf (0,017) schon.
        n = len(paare)
        if 2 / factorial(n) > 0.05:
            print(f"  -> NICHT signifikant, und bei n = {n} auch nicht erreichbar: "
                  f"selbst perfekte Monotonie ergaebe p = {2/factorial(n):.3f}. "
                  f"Dafuer braucht es fuenf Staedte.")
        else:
            print(f"  -> NICHT signifikant. Bei n = {n} reicht nur perfekte Monotonie "
                  f"(p = {2/factorial(n):.3f}); so eng ist dieser Zusammenhang nicht.")
    return {"n": len(paare), "rho": rho, "p": p, "art": art}


def main():
    args = _args()
    schwelle = _schwelle()
    eig = _eigenschaften()

    zeilen = []
    for ordner in sorted((ROOT / "results").glob("*/evaluation")):
        slug = ordner.parent.name
        z = _stadt(slug, args.method, schwelle)
        if z is None:
            continue
        s = eig.get(slug, {})
        z["abdeckung"] = s.get("abdeckung_gesamt")
        z["panorama"] = s.get("anteil_panorama")
        z["seit_2022"] = s.get("anteil_seit_2022")
        zeilen.append(z)

    if not zeilen:
        raise SystemExit(f"Keine Stadt mit results/<stadt>/evaluation/{args.method}.json gefunden.")

    zeilen.sort(key=lambda z: z.get("abdeckung") if z.get("abdeckung") is not None else 9)
    print(f"Encoder {args.method}  |  R@1 bei {schwelle} m  |  {len(zeilen)} Staedte\n")
    _tabelle(zeilen)

    # Die drei Befunde, jeder mit seiner eigenen Stichprobe.
    befunde = {}
    befunde["abdeckung_intervall"] = _zusammenhang(
        "Abdeckung gegen Intervallbreite", zeilen, "abdeckung", "halbbreite",
        "Loechrige Abdeckung -> manche Fahrten laufen ins Leere -> grosse Streuung zwischen Fahrten.")
    for z in zeilen:
        if z.get("hard") is not None:
            z["hard_abschlag"] = z["hard"] - z["alle"]
    befunde["loesbar_intervall"] = _zusammenhang(
        "Loesbarer Anteil gegen Intervallbreite", zeilen, "loesbar_anteil", "halbbreite",
        "Misst, was die Abdeckung zu messen vorgibt: wieviele Anfragen ueberhaupt "
        "ein Datenbankbild haben.")
    befunde["dublette_hard"] = _zusammenhang(
        "Dublettenanteil gegen Hard-Abschlag", zeilen, "dublette", "hard_abschlag",
        "Die schwache Fassung der Zerlegung oben: sie ignoriert, dass mit dem Zaehler "
        "auch der Nenner schrumpft, und kann deshalb selbst dann flach sein, wenn die "
        "Identitaet auf drei Stellen aufgeht.")
    zerlegt = _zerlegungstabelle(zeilen)
    for z in zeilen:
        x = zerlegt.get(z["stadt"])
        if x:
            z["uebernutzung"] = x["uebernutzung"]
            z["rel_abschlag"] = x["rel_abschlag"]
    befunde["uebernutzung_abschlag"] = _zusammenhang(
        "Uebernutzung d-(1-r) gegen relativen Hard-Abschlag", zeilen,
        "uebernutzung", "rel_abschlag",
        "Die starke Fassung: nicht der Dublettenanteil allein, sondern sein Ueberschuss "
        "ueber die strukturelle Abhaengigkeit. Analytisch erzwungen (rel = -(d-(1-r))/r), "
        "die Rangkorrelation prueft nur, ob beide Skripte dasselbe messen.")
    panoramen = _panoramatabelle(zeilen)
    befunde["jahre_hard"] = _zusammenhang(
        "Anteil seit 2022 gegen Hard-Abschlag", zeilen, "seit_2022", "hard_abschlag",
        "Wenige alte Kampagnen -> Anfrage und Treffer aus derselben Befahrung -> der Filter greift hart.")

    verifikation = _gv_tabelle(zeilen, args.method, schwelle)

    absolut = [z["alle"] for z in zeilen]
    print(f"\nSpanne des absoluten R@1: {min(absolut):.3f} bis {max(absolut):.3f} "
          f"({max(absolut)-min(absolut):.3f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"method": args.method, "schwelle_m": float(schwelle),
                               "staedte": zeilen, "befunde": befunde,
                               "zerlegung": zerlegt,
                               "panorama": panoramen,
                               "geometrische_verifikation": verifikation},
                              indent=2), encoding="utf-8")
    print(f"\ngeschrieben: {OUT.relative_to(ROOT)}")

    if args.plot:
        _abbildung(zeilen, args.method)


def _abbildung(zeilen, method):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = [(z["abdeckung"], z["halbbreite"], z["stadt"]) for z in zeilen
         if z.get("abdeckung") is not None and z.get("halbbreite") is not None]
    if len(p) < 2:
        print("Abbildung uebersprungen: zu wenige Staedte mit beidem.")
        return
    x, y, namen = zip(*p)
    plt.rcParams["figure.dpi"] = 150
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(x, y, s=40, zorder=3)
    for xi, yi, n in p:
        ax.annotate(n, (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=8)
    if len(p) >= 3:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 50)
        ax.plot(xs, m * xs + b, "--", linewidth=1, alpha=0.7,
                label=f"±boot = {m:.3f}·Abdeckung + {b:.3f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Strassenabdeckung")
    ax.set_ylabel("95-%-Halbbreite von R@1 (Sequenz-Bootstrap)")
    ax.set_title(f"{method}: Abdeckung bestimmt die Messunsicherheit")
    ax.grid(alpha=0.3)
    ziel = OUT.with_suffix(".png")
    fig.savefig(ziel, bbox_inches="tight")
    print(f"geschrieben: {ziel.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
