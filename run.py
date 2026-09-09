"""
Fuehrt die Pipeline der Reihe nach aus.

Eine Stufe wird uebersprungen, wenn ihr Ergebnis vorliegt UND laut
Fingerabdruck zur aktuellen config.yaml passt. Aendert man etwas an der
Config, laufen genau die betroffenen Stufen neu.

01 ist davon ausgenommen und wird nur auf Existenz geprueft: es wuerfelt
sonst den Split neu und entwertet damit alle vorhandenen Embeddings. 05
entfaellt, solange vpr.adapter auf "none" steht.

--force rechnet alles, --from beginnt bei einer bestimmten Stufe.
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


def _args():
    ap = argparse.ArgumentParser(
        description="Fuehrt die Notebooks der Reihe nach aus und ueberspringt, "
                    "was bereits zur config.yaml passt.",
        epilog="Beispiele:\n"
               "  python run.py                              alles, was noetig ist\n"
               "  python run.py --method mixvpr              anderer Encoder, ohne Config zu aendern\n"
               "  python run.py --method clip --adapter linear\n"
               "  python run.py --from 06                    ab dem Retrieval, erzwungen\n"
               "  python run.py --force                      alles neu rechnen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--force", action="store_true",
                    help="Auch Stufen ausfuehren, deren Ergebnis schon passt")
    ap.add_argument("--from", dest="start", metavar="NOTEBOOK",
                    help="Erst ab diesem Notebook beginnen, z.B. 06. Ab dort wird "
                         "alles ausgefuehrt, auch wenn es schon vorliegt.")
    ap.add_argument("--method", metavar="VERFAHREN",
                    help="vpr.method fuer diesen Lauf ueberschreiben, ohne die "
                         "config.yaml zu aendern")
    ap.add_argument("--adapter", metavar="none|linear",
                    help="vpr.adapter fuer diesen Lauf ueberschreiben")
    return ap.parse_args()


ARGS = _args()
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())

# Die Notebooks lesen config.yaml von der Platte. Ein Override muss also
# wirklich in die Datei, sonst liefe run.py mit anderen Werten als die
# Stufen, die es startet. Nach dem Lauf wird der Originaltext zurueckgelegt.
CONFIG_PATH = ROOT / "config.yaml"
CONFIG_ORIGINAL = CONFIG_PATH.read_text()
UEBERSCHRIEBEN = {}
for schluessel, wert in (("method", ARGS.method), ("adapter", ARGS.adapter)):
    if wert is not None and wert != CFG["vpr"].get(schluessel):
        UEBERSCHRIEBEN[schluessel] = wert
        CFG["vpr"][schluessel] = wert

METHOD = CFG["vpr"]["method"]
ADAPTER = CFG["vpr"].get("adapter", "none")
EMBEDDING_NAME = METHOD if ADAPTER in ("none", "None") else f"{METHOD}_{ADAPTER}"

EMB = ROOT / "data" / "embeddings" / METHOD

sys.path.insert(0, str(ROOT))
from src.run_guard import (  # noqa: E402
    adapter_fingerprint,
    embedding_fingerprint,
    require_fingerprint,
    validate_config,
)

validate_config(CFG)


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


def _load_metadata(name):
    datei = EMB / f"{name}_metadata.parquet"
    return pd.read_parquet(datei) if datei.exists() else None


def _embedding_gate(name, adapter):
    def fingerprint():
        meta = _load_metadata(name)
        if meta is None:
            raise FileNotFoundError(name)
        return embedding_fingerprint(CFG, METHOD, adapter, meta)

    return fingerprint


def _adapter_gate():
    def fingerprint():
        meta = _load_metadata(METHOD)
        if meta is None:
            raise FileNotFoundError(METHOD)
        return adapter_fingerprint(
            CFG, METHOD, embedding_fingerprint(CFG, METHOD, "none", meta)
        )

    return fingerprint


# Notebook -> (Artefakt, Fingerabdruck oder None fuer reine Existenzpruefung)
STAGES = [
    ("01_mapillary_coverage.ipynb",
     ROOT / "data" / "processed" / "metadata.parquet", None),
    ("02_dataset_audit.ipynb", None, None),
    ("03_image_download.ipynb", None, None),
    ("04_embeddings.ipynb",
     EMB / f"{METHOD}_embeddings.npy", _embedding_gate(METHOD, "none")),
    ("05_adapter.ipynb",
     EMB / f"{METHOD}_linear_embeddings.npy", _embedding_gate(f"{METHOD}_linear", "linear")),
    ("06_retrieval.ipynb",
     ROOT / "results" / "retrieval" / METHOD / f"{EMBEDDING_NAME}_retrieval.npz",
     _embedding_gate(EMBEDDING_NAME, ADAPTER if ADAPTER not in ("none", "None") else "none")),
    ("07_evaluation.ipynb", None, None),
]


def run_notebook(notebook):
    path = ROOT / "notebooks" / notebook
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=None, kernel_name="python3").execute()


def main():
    args = ARGS
    print(f"method={METHOD}  adapter={ADAPTER}")
    if UEBERSCHRIEBEN:
        werte = ", ".join(f"{k}={v}" for k, v in UEBERSCHRIEBEN.items())
        print(f"per Kommandozeile ueberschrieben: {werte}")
        print(f"config.yaml wird dafuer vorruebergehend angepasst und danach "
              f"zurueckgesetzt")
    print()

    zeiten = []
    gesamt = time.time()
    started = args.start is None

    for notebook, artifact, fingerprint in STAGES:
        if not started:
            if notebook.startswith(args.start):
                started = True
            else:
                print(f"uebersprungen (vor --from): {notebook}")
                continue

        force = args.force or args.start is not None

        # Ohne konfigurierten Adapter wertet 06/07 die Baseline aus -- ein
        # Training waere Zeit und Speicher fuer ein Ergebnis, das niemand
        # anfasst. --force oder --from 05 fuehrt es trotzdem aus.
        if notebook.startswith("05") and ADAPTER in ("none", "None") and not force:
            print(f"uebersprungen (kein Adapter): {notebook}")
            continue

        if not force and artifact is not None:
            if fingerprint is None:
                if artifact.exists():
                    print(f"uebersprungen (liegt vor):   {notebook}  ->  {artifact.name}")
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
        print(f"{notebook} fertig in {dauer:.2f} min")

    print("\n" + "=" * 60)
    if zeiten:
        for notebook, dauer in zeiten:
            print(f"  {notebook:<32s} {dauer:>7.2f} min")
        print("  " + "-" * 42)
    print(f"  {'gesamt':<32s} {(time.time() - gesamt) / 60:>7.2f} min")
    print("=" * 60)


if __name__ == "__main__":
    if UEBERSCHRIEBEN:
        CONFIG_PATH.write_text(
            yaml.safe_dump(CFG, allow_unicode=True, sort_keys=False)
        )
    try:
        main()
    finally:
        if UEBERSCHRIEBEN:
            CONFIG_PATH.write_text(CONFIG_ORIGINAL)
            print("config.yaml zurueckgesetzt.")
