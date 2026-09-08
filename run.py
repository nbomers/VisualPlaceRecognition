"""
Fuehrt die Pipeline der Reihe nach aus.

Eine Stufe wird uebersprungen, wenn ihr Ergebnis vorliegt UND laut
Fingerabdruck zur aktuellen config.yaml passt. Aendert man etwas an der
Config, laufen genau die betroffenen Stufen neu.

01 ist davon ausgenommen und wird nur auf Existenz geprueft: es wuerfelt
sonst den Split neu und entwertet damit alle vorhandenen Embeddings.

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
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
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


def _gueltig(pfad, fingerprint):
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


def _metadaten(name):
    datei = EMB / f"{name}_metadata.parquet"
    return pd.read_parquet(datei) if datei.exists() else None


def _embedding_gate(name, adapter):
    def fingerprint():
        meta = _metadaten(name)
        if meta is None:
            raise FileNotFoundError(name)
        return embedding_fingerprint(CFG, METHOD, adapter, meta)

    return fingerprint


def _adapter_gate():
    def fingerprint():
        meta = _metadaten(METHOD)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Auch Stufen ausfuehren, deren Ergebnis schon vorliegt")
    ap.add_argument("--from", dest="start", metavar="NOTEBOOK",
                    help="Erst ab diesem Notebook beginnen, z.B. 06. Ab dort "
                         "wird alles ausgefuehrt, auch wenn es schon vorliegt.")
    args = ap.parse_args()

    print(f"method={METHOD}  adapter={ADAPTER}\n")

    started = args.start is None
    for notebook, artifact, fingerprint in STAGES:
        if not started:
            if notebook.startswith(args.start):
                started = True
            else:
                print(f"uebersprungen (vor --from): {notebook}")
                continue

        force = args.force or args.start is not None
        if not force and artifact is not None:
            if fingerprint is None:
                if artifact.exists():
                    print(f"uebersprungen (liegt vor):   {notebook}  ->  {artifact.name}")
                    continue
            elif _gueltig(artifact, fingerprint):
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
        print(f"{notebook} fertig in {(time.time() - t0) / 60:.2f} min")

    print("\nPipeline abgeschlossen.")


if __name__ == "__main__":
    main()
