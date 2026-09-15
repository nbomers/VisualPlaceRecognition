"""
Fuehrt die Notebooks der Reihe nach aus.

Eine Stufe wird uebersprungen, wenn ihr Ergebnis vorliegt UND laut
Fingerabdruck zur aktuellen config.yaml passt. Aendert man etwas an der
Config, laufen genau die betroffenen Stufen neu.

01 bis 03 werden nur auf Existenz ihres Ergebnisses geprueft: 01 wuerfelt
sonst den Split neu und entwertet damit alle vorhandenen Embeddings, 02 und
03 haengen nicht an der config. 05 entfaellt, solange vpr.adapter auf "none"
steht. Welche Stufe woran erkannt wird, steht an einer Stelle: _stages().

--method und --adapter nehmen auch Listen ("clip,mixvpr") oder "all" und
rechnen dann eine Kombination nach der anderen.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import nbformat
import pandas as pd
import yaml
from nbclient import NotebookClient

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.yaml"
CONFIG_ORIGINAL = CONFIG_PATH.read_text()
BASIS_CFG = yaml.safe_load(CONFIG_ORIGINAL)

sys.path.insert(0, str(ROOT))
from src.config import paths  # noqa: E402
from src.run_guard import (  # noqa: E402
    code_version,
    embedding_fingerprint,
    require_fingerprint,
    short_hash,
    validate_config,
)

# Diese Stufen haengen nicht am Verfahren und laufen bei einem Durchgang
# ueber mehrere Encoder nur einmal.
VERFAHRENSUNABHAENGIG = ("01", "02", "03")


def _args():
    ap = argparse.ArgumentParser(
        description="Fuehrt die Notebooks der Reihe nach aus und ueberspringt, "
                    "was bereits zur config.yaml passt.",
        epilog="Beispiele:\n"
               "  python run.py                                alles, was noetig ist\n"
               "  python run.py --method mixvpr                anderer Encoder\n"
               "  python run.py --method clip --adapter linear\n"
               "  python run.py --method all                   jeden Encoder nacheinander\n"
               "  python run.py --method derived               die PCA-Varianten\n"
               "  python run.py --method all --adapter all     dazu je Baseline und Adapter\n"
               "  python run.py --method clip,mixvpr           nur diese beiden\n"
               "  python run.py --from 06                      ab dem Retrieval, erzwungen\n"
               "  python run.py --force                        alles neu rechnen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--force", action="store_true",
                    help="Auch Stufen ausfuehren, deren Ergebnis schon passt")
    ap.add_argument("--from", dest="start", metavar="NOTEBOOK",
                    help="Erst ab diesem Notebook beginnen, z.B. 06. Ab dort wird "
                         "alles ausgefuehrt, auch wenn es schon vorliegt.")
    ap.add_argument("--method", metavar="LISTE",
                    help='vpr.method ueberschreiben. Mehrere durch Komma, "all" '
                         'nimmt die echten Encoder, "derived" die abgeleiteten '
                         "(PCA, Whitening, Verkettung aus experiments/).")
    ap.add_argument("--adapter", metavar="LISTE",
                    help='vpr.adapter ueberschreiben. Mehrere durch Komma, "all" '
                         "entspricht none,linear.")
    return ap.parse_args()


def _abgeleitet(cfg, name):
    """Hat der Encoder einen source- oder sources-Eintrag, ist er aus anderen
    gerechnet -- PCA-Varianten aus experiments/pca_reduce.py, Verkettungen
    aus experiments/concat_embeddings.py."""
    block = cfg["vpr"].get(name)
    return isinstance(block, dict) and ("source" in block or "sources" in block)


def _modelle(cfg, abgeleitet):
    return [n for n in cfg["vpr"]["models"] if _abgeleitet(cfg, n) == abgeleitet]


def _liste(wert, alle, standard, abgeleitet=()):
    """Kommaliste, "all", "derived" oder None -> Liste der zu rechnenden Werte.

    "all" nimmt bewusst nur die echten Encoder. Die abgeleiteten Varianten
    entstehen nicht in 04, sondern in experiments/pca_reduce.py -- sie
    stillschweigend mitzurechnen wuerde bei fehlender Quelle nur abbrechen.
    """
    if wert is None:
        return [standard]
    if wert.strip() == "all":
        return list(alle)
    if wert.strip() == "derived":
        return list(abgeleitet)
    return [t.strip() for t in wert.split(",") if t.strip()]


def _stages(cfg, method, adapter):
    p = paths(cfg, ROOT)
    emb = p.embedding_dir(method)
    name = method if adapter in ("none", "None") else f"{method}_{adapter}"

    def gate(dateiname, variante):
        def fingerprint():
            datei = emb / f"{dateiname}_metadata.parquet"
            if not datei.exists():
                raise FileNotFoundError(datei)
            return embedding_fingerprint(cfg, method, variante, pd.read_parquet(datei))
        return fingerprint

    # Je Stufe: Notebook, das Artefakt, an dem man erkennt, dass sie fertig
    # ist, und optional ein Fingerabdruck, der prueft, ob es noch zur config
    # passt. Ohne Fingerabdruck zaehlt nur die Existenz.
    return [
        ("01_mapillary_coverage.ipynb",
         p.processed / "metadata.parquet", None),
        ("02_dataset_audit.ipynb",
         p.dataset_audit, None),
        # 03 wird ueber den Bildbestand beurteilt, nicht ueber eine Datei --
        # siehe _download_stand(); deshalb hier kein Artefakt.
        ("03_image_download.ipynb", None, None),
        ("04_embeddings.ipynb",
         emb / f"{method}_embeddings.npy", gate(method, "none")),
        ("05_adapter.ipynb",
         emb / f"{method}_linear_embeddings.npy", gate(f"{method}_linear", "linear")),
        ("06_retrieval.ipynb",
         p.retrieval_file(name, method),
         gate(name, adapter if adapter not in ("none", "None") else "none")),
        ("07_evaluation.ipynb", None, None),
        ("08_localization.ipynb", None, None),
    ]


def _download_stand(cfg):
    """
    Was 03 zuletzt hinterlassen hat: erwartete, vorhandene und fehlende
    Bilder. None, wenn 03 nie gelaufen ist.

    03 nur auf die Existenz seiner Ausgabedatei zu pruefen war die stillste
    Luecke der Pipeline: laeuft der Token mitten im Download ab, liegt die
    Datei trotzdem da und 03 gilt als erledigt. Beurteilt wird deshalb der
    Bestand.
    """
    pfad = paths(cfg, ROOT).processed / "image_download.json"
    if not pfad.exists():
        return None
    try:
        return json.loads(pfad.read_text())
    except ValueError:
        return None


def _result_current(ergebnis, treffer, cfg, method, adapter):
    """Passt eine 07-/08-JSON noch zu Trefferliste und Auswertungscode?"""
    if not (ergebnis.exists() and treffer.exists()):
        return False
    try:
        json_inhalt = json.loads(ergebnis.read_text())
    except ValueError:
        return False
    if json_inhalt.get("fingerprint_hash") and json_inhalt.get("code_version"):
        name = method if adapter in ("none", "None") else f"{method}_{adapter}"
        meta = pd.read_parquet(paths(cfg, ROOT).metadata_file(name, method))
        erwartet = short_hash(embedding_fingerprint(cfg, method, adapter, meta))
        return (json_inhalt["fingerprint_hash"] == erwartet
                and json_inhalt["code_version"].get("evaluation") == code_version(ROOT)["evaluation"])
    return ergebnis.stat().st_mtime >= treffer.stat().st_mtime


def _is_valid(pfad, fingerprint):
    """
    Ist das Artefakt vorhanden UND passt es zur aktuellen config.yaml?

    Nur auf Existenz zu pruefen liesse eine Stufe ueberspringen, deren
    Ergebnis zu einer anderen Konfiguration gehoert -- der Fehler faellt dann
    erst zwei Stufen spaeter auf.
    """
    if not pfad.exists():
        return False
    try:
        require_fingerprint(pfad, fingerprint(), "")
        return True
    except Exception as e:
        print(f"  ({pfad.name}: {type(e).__name__}: {str(e).splitlines()[0][:90]})")
        return False


class DurchreichenderClient(NotebookClient):
    """
    NotebookClient sammelt Zellenausgaben nur ins Notebook-Objekt. Bei einem
    Encoder-Lauf ueber Stunden sieht man dadurch gar nichts.

    Durchgereicht werden nur Fortschrittsbalken: tqdm setzt einen
    Wagenruecklauf, um seine Zeile zu ueberschreiben. Prints und Warnungen
    enden dagegen mit Zeilenumbruch und bleiben im Notebook-Objekt.
    """

    def output(self, outs, msg, display_id, cell_index):
        if msg["msg_type"] == "stream":
            text = msg["content"].get("text", "")
            if "\r" in text:
                sys.stderr.write(text)
                sys.stderr.flush()
        return super().output(outs, msg, display_id, cell_index)


def run_notebook(notebook):
    nb = nbformat.read(ROOT / "notebooks" / notebook, as_version=4)
    DurchreichenderClient(nb, timeout=None, kernel_name="python3").execute()


def _durchlauf(cfg, method, adapter, args, erledigt):
    """Eine Kombination rechnen. Gibt die Zeiten je Stufe zurueck."""
    zeiten = []
    started = args.start is None

    for notebook, artifact, fingerprint in _stages(cfg, method, adapter):
        if not started:
            if notebook.startswith(args.start):
                started = True
            else:
                continue

        # 01 bis 03 haengen nicht am Verfahren -- bei mehreren Kombinationen
        # waere jede Wiederholung verlorene Zeit.
        if notebook[:2] in VERFAHRENSUNABHAENGIG and notebook in erledigt:
            continue

        force = args.force or args.start is not None

        # Ohne konfigurierten Adapter wertet 06/07 die Baseline aus -- ein
        # Training waere Zeit und Speicher fuer ein Ergebnis, das niemand
        # anfasst. --force oder --from 05 fuehrt es trotzdem aus.
        if notebook.startswith("05") and adapter in ("none", "None") and not force:
            print(f"uebersprungen (kein Adapter): {notebook}")
            continue

        # 03 ist fertig, wenn die Bilder da sind -- nicht, wenn eine Datei
        # existiert. Fehlen mehr als max_missing_images_frac, laeuft es neu
        # und holt genau die fehlenden nach.
        if notebook.startswith("03") and not force:
            stand = _download_stand(cfg)
            if stand is not None:
                schranke = float(stand.get("max_missing_images_frac",
                                           cfg.get("max_missing_images_frac", 0.0)))
                if float(stand.get("anteil_fehlend", 1.0)) <= schranke:
                    print(f"uebersprungen (vollstaendig):{notebook}  ->  "
                          f"{stand['vorhanden']:,} von {stand['erwartet']:,} Bildern")
                    erledigt.add(notebook)
                    continue
                print(f"neu zu rechnen:              {notebook}  ->  "
                      f"{stand['fehlend']:,} Bilder fehlen "
                      f"({float(stand['anteil_fehlend']):.2%} > {schranke:.2%})")

        # 07 und 08 schreiben JSONs ohne eigene Sidecar-Datei. Aktuell sind
        # sie, wenn sie den Fingerabdruck der Trefferliste UND die Kennung
        # des Auswertungscodes tragen. Aeltere JSONs ohne diese Felder werden
        # ueber die Dateizeit beurteilt -- bis sie einmal neu gerechnet sind.
        if notebook[:2] in ("07", "08") and not force:
            name = method if adapter in ("none", "None") else f"{method}_{adapter}"
            p = paths(cfg, ROOT)
            ergebnis = (p.evaluation if notebook.startswith("07") else p.localization) / f"{name}.json"
            treffer = p.retrieval_file(name, method)
            if _result_current(ergebnis, treffer, cfg, method, adapter):
                print(f"uebersprungen (aktuell):     {notebook}  ->  {ergebnis.name}")
                continue

        if not force and artifact is not None:
            if fingerprint is None:
                if artifact.exists():
                    print(f"uebersprungen (liegt vor):   {notebook}  ->  {artifact.name}")
                    erledigt.add(notebook)
                    continue
            elif _is_valid(artifact, fingerprint):
                print(f"uebersprungen (passt):       {notebook}  ->  {artifact.name}")
                continue
            elif artifact.exists():
                print(f"neu zu rechnen:              {notebook}  ->  {artifact.name} "
                      f"passt nicht zur config.yaml")

        print("=" * 60)
        print(f"Starte: {notebook}")
        print("=" * 60)
        t0 = time.time()
        run_notebook(notebook)
        dauer = (time.time() - t0) / 60
        zeiten.append((notebook, dauer))
        erledigt.add(notebook)
        print(f"{notebook} fertig in {dauer:.2f} min")

    return zeiten


def main():
    args = _args()

    methoden = _liste(args.method,
                      _modelle(BASIS_CFG, abgeleitet=False),
                      BASIS_CFG["vpr"]["method"],
                      _modelle(BASIS_CFG, abgeleitet=True))
    adapter = _liste(args.adapter, ("none", "linear"),
                     BASIS_CFG["vpr"].get("adapter", "none"))
    kombinationen = [(m, a) for m in methoden for a in adapter]

    if len(kombinationen) > 1:
        print(f"{len(kombinationen)} Kombinationen:")
        for m, a in kombinationen:
            print(f"  {m} / {a}")
        print()

    erledigt = set()
    ergebnisse = []
    gesamt = time.time()

    for method, adapterwert in kombinationen:
        cfg = yaml.safe_load(CONFIG_ORIGINAL)
        cfg["vpr"]["method"] = method
        cfg["vpr"]["adapter"] = adapterwert
        validate_config(cfg)

        # Die Notebooks lesen Verfahren und Adapter aus der Umgebung, wenn
        # sie gesetzt sind. Frueher wurde dafuer config.yaml ueberschrieben --
        # eine versionierte Datei als Zustandsspeicher, die nach jedem Lauf
        # als geaendert dastand und bei jedem git pull im Weg war.
        os.environ["VPR_METHOD"] = method
        os.environ["VPR_ADAPTER"] = adapterwert

        print("#" * 60)
        print(f"# {method}  /  adapter={adapterwert}")
        print("#" * 60)
        try:
            zeiten = _durchlauf(cfg, method, adapterwert, args, erledigt)
            ergebnisse.append((method, adapterwert, sum(d for _, d in zeiten), None))
        except Exception as e:
            # Ein gescheiterter Encoder soll die uebrigen nicht mitreissen --
            # sonst ist eine Nacht Rechenzeit wegen eines OOM verloren.
            ergebnisse.append((method, adapterwert, 0.0, f"{type(e).__name__}: {e}"))
            print(f"\nABGEBROCHEN: {method}/{adapterwert} -- {type(e).__name__}")
            if len(kombinationen) == 1:
                raise
            print("weiter mit der naechsten Kombination\n")

    print("\n" + "=" * 60)
    for method, adapterwert, dauer, fehler in ergebnisse:
        stand = "abgebrochen" if fehler else f"{dauer:>7.2f} min"
        print(f"  {method + ' / ' + adapterwert:<32s} {stand}")
    print("  " + "-" * 42)
    print(f"  {'gesamt':<32s} {(time.time() - gesamt) / 60:>7.2f} min")
    print("=" * 60)

    gescheitert = [(m, a, f) for m, a, _, f in ergebnisse if f]
    if gescheitert:
        print("\nFehlgeschlagen:")
        for m, a, f in gescheitert:
            print(f"  {m}/{a}: {f.splitlines()[0][:100]}")
        # Sonst sieht ein Durchgang, bei dem die Haelfte abgebrochen ist, fuer
        # jedes aufrufende Skript wie ein Erfolg aus.
        raise SystemExit(1)


if __name__ == "__main__":
    main()
