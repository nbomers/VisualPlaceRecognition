"""
Holt den Mapillary-Status (Map Matched / Best Capture) für Bilder.
Speichert das Ergebnis lokal als JSON, um API-Limits zu schonen und
die Auswertung vom Netzwerk zu entkoppeln.
"""

import json
import os
import requests
import numpy as np  # Neu hinzugefügt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from _common import CFG, PATHS, ROOT, RESULTS
from src.retrieval import load_retrieval


def load_token(root):
    """
    MAPILLARY_TOKEN aus der Umgebung, sonst aus <root>/.env.

    Die .env ist gitignored; das Token faellt nie in ein Notebook oder ein
    Ergebnis.
    """
    env_file = root / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    token = os.environ.get("MAPILLARY_TOKEN", "")
    if not token.startswith("MLY|"):
        raise RuntimeError(
            "Kein Mapillary-Token. Datei .env im Projektwurzelverzeichnis anlegen:\n"
            "  MAPILLARY_TOKEN=MLY|dein|token"
        )
    return token


def fetch_status(img_id, token):
    """Prüft, ob Mapillary eine computed_geometry (SfM) berechnen konnte."""
    url = f"https://graph.mapillary.com/{img_id}?access_token={token}&fields=id,computed_geometry"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return img_id, "computed_geometry" in res.json()
    except Exception:
        pass
    return img_id, False

def main():
    token = load_token(ROOT)

    method = CFG["vpr"]["method"]
    adapter = CFG["vpr"]["adapter"]
    query, _, _, _ = load_retrieval(ROOT, CFG, method, adapter)
    image_ids_full = query["image_id"].to_numpy()

    # Exakt gleiche Stichproben-Logik wie in geometric_verification.py
    n_queries = 2000
    rng = np.random.default_rng(int(CFG["vpr"]["split_seed"]))
    auswahl = np.sort(rng.choice(len(query), n_queries, replace=False))

    # Nur die 2.000 ausgewaehlten IDs verwenden
    image_ids = image_ids_full[auswahl].tolist()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS / "mapillary_status.json"

    status_cache = {}
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            status_cache = json.load(f)

    fehlende_ids = [str(i) for i in image_ids if str(i) not in status_cache]

    if not fehlende_ids:
        print(f"Alle {len(image_ids)} Stichproben-IDs sind bereits lokal in {out_file} gecached.")
        return

    print(f"Hole Mapillary-Status für {len(fehlende_ids)} neue Queries...")

    with ThreadPoolExecutor(16) as pool:
        jobs = [pool.submit(fetch_status, img_id, token) for img_id in fehlende_ids]
        for job in tqdm(as_completed(jobs), total=len(jobs), desc="API Abfrage"):
            img_id, has_computed = job.result()
            status_cache[str(img_id)] = has_computed

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(status_cache, f, indent=2)

    print(f"-> Cache aktualisiert: {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()