"""
Fuehrt die Notebooks der Reihe nach aus.

Eine Stufe wird uebersprungen, wenn ihr Ergebnis vorliegt UND laut
Fingerabdruck zur aktuellen config.yaml passt. Aendert man etwas an der
Config, laufen genau die betroffenen Stufen neu.

01 ist davon ausgenommen und wird nur auf Existenz geprueft: es wuerfelt
sonst den Split neu und entwertet damit alle vorhandenen Embeddings. 05
entfaellt, solange vpr.adapter auf "none" steht.

--method und --adapter nehmen auch Listen ("clip,mixvpr") oder "all" und
rechnen dann eine Kombination nach der anderen.
"""

import argparse
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
from src.run_guard import (  # noqa: E402
    adapter_fingerprint,
    embedding_fingerprint,
    require_fingerprint,
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
                         "nimmt jeden Eintrag aus vpr.models.")
    ap.add_argument("--adapter", metavar="LISTE",
                    help='vpr.adapter ueberschreiben. Mehrere durch Komma, "all" '
                         "entspricht none,linear.")
    return ap.parse_args()


def _liste(wert, alle, standard):
    """Kommaliste, "all" oder None -> Liste der zu rechnenden Werte."""
    if wert is None:
        return [standard]
    if wert.strip() == "all":
        return list(alle)
    return [t.strip() for t in wert.split(",") if t.strip()]


def _stages(cfg, method, adapter):
    emb = ROOT / "data" / "embeddings" / method
    name = method if adapter in ("none", "None") else f"{method}_{adapter}"

    def gate(dateiname, variante):
        def fingerprint():
            datei = emb / f"{dateiname}_metadata.parquet"
            if not datei.exists():
                raise FileNotFoundError(datei)
            return embedding_fingerprint(cfg, method, variante, pd.read_parquet(datei))
        return fingerprint

    return [
        ("01_mapillary_coverage.ipynb",
         ROOT / "data" / "processed" / "metadata.parquet", None),
        ("02_dataset_audit.ipynb", None, None),
        ("03_image_download.ipynb", None, None),
        ("04_embeddings.ipynb",
         emb / f"{method}_embeddings.npy", gate(method, "none")),
        ("05_adapter.ipynb",
         emb / f"{method}_linear_embeddings.npy", gate(f"{method}_linear", "linear")),
        ("06_retrieval.ipynb",
         ROOT / "results" / "retrieval" / method / f"{name}_retrieval.npz",
         gate(name, adapter if adapter not in ("none", "None") else "none")),
        ("07_evaluation.ipynb", None, None),
        ("08_localization.ipynb", None, None),
    ]


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
    except Exception:
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

        # 07 hat keinen Fingerabdruck, seine Auswertung haengt aber allein an
        # der Retrieval-Datei. Ist sie aelter als das Ergebnis, gibt es nichts
        # neu zu rechnen.
        if notebook[:2] in ("07", "08") and not force:
            name = method if adapter in ("none", "None") else f"{method}_{adapter}"
            anhang = "" if notebook.startswith("07") else "_localization"
            ergebnis = ROOT / "results" / "evaluation" / f"{name}{anhang}.json"
            treffer = ROOT / "results" / "retrieval" / method / f"{name}_retrieval.npz"
            if (ergebnis.exists() and treffer.exists()
                    and ergebnis.stat().st_mtime >= treffer.stat().st_mtime):
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

    methoden = _liste(args.method, BASIS_CFG["vpr"]["models"], BASIS_CFG["vpr"]["method"])
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

        # Die Notebooks lesen config.yaml von der Platte -- der Wert muss also
        # wirklich dorthin, nicht nur in dieses Skript.
        CONFIG_PATH.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

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


if __name__ == "__main__":
    try:
        main()
    finally:
        if CONFIG_PATH.read_text() != CONFIG_ORIGINAL:
            CONFIG_PATH.write_text(CONFIG_ORIGINAL)
            print("config.yaml zurueckgesetzt.")
