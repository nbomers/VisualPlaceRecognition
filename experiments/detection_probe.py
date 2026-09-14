"""
Wie gut deckt Mapillary die Bilder mit Detections ab?

War urspruenglich die letzte Zelle in 03_image_download und ist von dort
herausgeloest worden: eine einmalige Erhebung, die nichts mit dem Download zu
tun hat und bei jedem Pipeline-Lauf nur im Weg stand.

Laedt keine Bilder, nur eine kleine JSON-Antwort je Bild. Das Ergebnis steht
in detections_probe.json daneben; ein zweiter Aufruf gibt es nur aus.

    python experiments/detection_probe.py
    python experiments/detection_probe.py --n 2000 --neu
"""

import argparse
import collections
import json
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from _common import CFG, PATHS, ROOT
from src.mapillary import load_token, make_session
PROBE_PATH = Path(__file__).resolve().parent / "detections_probe.json"


def _args():
    ap = argparse.ArgumentParser(description="Detection-Abdeckung erheben.")
    ap.add_argument("--n", type=int, default=500,
                    help="Wieviele Bilder pruefen (Standard: 500)")
    ap.add_argument("--split", default="database",
                    help="Aus welchem Split ziehen (Standard: database)")
    ap.add_argument("--neu", action="store_true",
                    help="Neu messen, auch wenn ein Ergebnis vorliegt")
    return ap.parse_args()


def show_summary(befund):
    print("=" * 58)
    print("DETECTION-ABDECKUNG")
    print("=" * 58)
    print(f"Geprueft:                   {befund['n_geprueft']:,}")
    print(f"Bilder mit Detections:      {befund['anteil_mit_detections'] * 100:.1f} %")
    print(f"Detections je Bild:         Median {befund['median_detections']:.0f}")
    print(f"Verschiedene Klassen:       {befund['n_klassen']:,}")
    print()
    print("Haeufigste Klassen:")
    for k, v in list(befund["haeufigste"].items())[:15]:
        print(f"  {k:<45s} {v:>6,}")
    print("=" * 58)
    print("Faustregel: unter 50 % Abdeckung ODER Median < 3 Detections/Bild")
    print("-> Re-Ranking lohnt nicht, Befund als Absatz in die Fehleranalyse.")


def messen(args, token):
    metadata = pd.read_parquet(PATHS.processed / "metadata.parquet")
    ids = metadata.loc[metadata["split"] == args.split, "image_id"].sample(
        args.n, random_state=int(CFG["vpr"]["split_seed"])
    )

    session = make_session()
    n_with = 0
    n_detections = []
    klassen = collections.Counter()
    n_fehler = 0

    for image_id in tqdm(ids, desc="Detections pruefen"):
        try:
            r = session.get(
                f"https://graph.mapillary.com/{image_id}/detections",
                params={"access_token": token, "fields": "value"},
                timeout=30,
            )
            r.raise_for_status()
            d = r.json().get("data", [])
        except requests.RequestException:
            n_fehler += 1
            continue

        n_detections.append(len(d))
        if d:
            n_with += 1
            klassen.update(x["value"] for x in d)

    n_ok = len(n_detections)
    return {
        "datum": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "split": args.split,
        "n_geprueft": n_ok,
        "n_fehler": n_fehler,
        "anteil_mit_detections": n_with / max(n_ok, 1),
        "median_detections": float(pd.Series(n_detections).median()) if n_detections else 0.0,
        "mittel_detections": float(pd.Series(n_detections).mean()) if n_detections else 0.0,
        "n_klassen": len(klassen),
        "haeufigste": dict(klassen.most_common(30)),
    }


def main():
    args = _args()

    if PROBE_PATH.exists() and not args.neu:
        befund = json.loads(PROBE_PATH.read_text())
        print(f"Gespeicherter Befund vom {befund['datum']} "
              f"({PROBE_PATH.name}), --neu misst erneut.\n")
        show_summary(befund)
        return

    befund = messen(args, load_token(ROOT))
    PROBE_PATH.write_text(json.dumps(befund, indent=2))
    show_summary(befund)
    print(f"\nGespeichert unter {PROBE_PATH}")


if __name__ == "__main__":
    main()
