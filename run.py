from pathlib import Path
import nbformat
from nbclient import NotebookClient


NOTEBOOKS = [
    "01_mapillary_coverage.ipynb",
    "02_dataset_audit.ipynb",
    "03_image_download.ipynb",
    "04_embeddings.ipynb",
    "05_retrieval.ipynb",
    "06_evaluation.ipynb",
]


def run_notebook(notebook):
    path = Path("src") / notebook

    print("=" * 50)
    print(f"Starte: {notebook}")
    print("=" * 50)

    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(
        nb,
        timeout=None,
        kernel_name="python3",
    )

    client.execute()

    print(f"{notebook} abgeschlossen")


for notebook in NOTEBOOKS:
    run_notebook(notebook)

print("\nPipeline vollständig abgeschlossen.")