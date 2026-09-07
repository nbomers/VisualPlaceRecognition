"""
Fuehrt die Pipeline der Reihe nach aus.

Stufen, deren Ergebnis schon vorliegt, werden uebersprungen. Das betrifft
vor allem 01 und 03: 01 wuerde den Split neu wuerfeln, was saemtliche
vorhandenen Embeddings entwertet, und 03 laedt sonst erneut alle Bilder.
Mit --force laeuft alles.
"""

import argparse
import time
from pathlib import Path

import nbformat
import yaml
from nbclient import NotebookClient

ROOT = Path(__file__).parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
METHOD = CFG["vpr"]["method"]
ADAPTER = CFG["vpr"].get("adapter", "none")
EMBEDDING_NAME = METHOD if ADAPTER in ("none", "None") else f"{METHOD}_{ADAPTER}"

EMB = ROOT / "data" / "embeddings" / METHOD

# Notebook -> Datei, deren Existenz die Stufe ueberfluessig macht.
STAGES = [
    ("01_mapillary_coverage.ipynb", ROOT / "data" / "processed" / "metadata.parquet"),
    ("02_dataset_audit.ipynb", None),
    ("03_image_download.ipynb", None),
    ("04_embeddings.ipynb", EMB / f"{METHOD}_embeddings.npy"),
    # Gate auf die adaptierten Embeddings, nicht auf die .pt: 05 liefert
    # beides, und 06 braucht die Embeddings.
    ("05_adapter.ipynb", EMB / f"{METHOD}_linear_embeddings.npy"),
    ("06_retrieval.ipynb",
     ROOT / "results" / "retrieval" / METHOD / f"{EMBEDDING_NAME}_retrieval.npz"),
    ("07_evaluation.ipynb", None),
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
    for notebook, artifact in STAGES:
        if not started:
            if notebook.startswith(args.start):
                started = True
            else:
                print(f"uebersprungen (vor --from): {notebook}")
                continue

        force = args.force or args.start is not None
        if not force and artifact is not None and artifact.exists():
            print(f"uebersprungen (liegt vor):   {notebook}  ->  {artifact.name}")
            continue

        print("=" * 60)
        print(f"Starte: {notebook}")
        print("=" * 60)
        t0 = time.time()
        run_notebook(notebook)
        print(f"{notebook} fertig in {(time.time() - t0) / 60:.2f} min")

    print("\nPipeline abgeschlossen.")


if __name__ == "__main__":
    main()
