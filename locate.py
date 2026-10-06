"""
Wo ist das? Ein Foto durch den konfigurierten Encoder schicken und die
Koordinate des aehnlichsten Datenbankbildes ausgeben.

    python locate.py foto.jpg
    python locate.py foto.jpg --method eigenplaces_megaloc_concat --k 5
    python locate.py foto.jpg --json                # maschinenlesbar
    python locate.py ~/Downloads/mapillary/test     # alle Bilder in einem Ordner
    python locate.py foto.jpg --referenz alle       # gegen alle 332.868 Bilder suchen

Der Encoder kommt aus der Factory (auch PCA-, Whitening- und
Verkettungsvarianten), die Datenbank aus data/<stadt>/embeddings/, die Suche
exakt mit numpy statt FAISS -- Torch und FAISS passen auf macOS nicht in einen
Prozess (src/locate.py). Sonst derselbe Weg wie 04 bis 06, fuer ein Bild.
AnyLoc geht ebenfalls: 04 legt die angepasste PCA als anyloc_pca.npz neben
die Embeddings, die Factory laedt sie (src/models/factory.py).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.config import load_config, paths  # noqa: E402
from src.locate import REFERENZEN, Locator  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Ein Foto in der konfigurierten Stadt verorten "
                                "(config.yaml -> city).")
    ap.add_argument("bild", nargs="+", help="Bilddatei(en) oder ein Ordner voller Bilder")
    ap.add_argument("--method", help="Encoder aus vpr.models (Standard: config.yaml)")
    ap.add_argument("--adapter", help="none | linear (Standard: config.yaml)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--referenz", choices=sorted(REFERENZEN), default="database",
                    help="wogegen gesucht wird: database (wie 07), voll (+ train) oder alle "
                         "(+ query). Fuer eigene Fotos ist alles erlaubt; die cos-Eichung "
                         "ist nur fuer database gemessen.")
    ap.add_argument("--json", action="store_true", help="Ergebnis als JSON ausgeben")
    args = ap.parse_args()
    if args.k < 2:
        ap.error("--k muss mindestens 2 sein: die Marge vergleicht Platz 1 mit Platz 2.")

    cfg = load_config(ROOT)
    # Ein Ordner steht fuer alle Bilder darin -- der Testordner aus config.yaml.
    bilder = []
    for b in args.bild:
        pfad = Path(b).expanduser()
        if pfad.is_dir():
            bilder += sorted(str(x) for x in pfad.iterdir()
                             if x.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
        elif pfad.exists():
            bilder.append(str(pfad))
        else:
            # Sonst faellt der Tippfehler erst auf, nachdem der Encoder
            # geladen und die Datenbank im Speicher ist -- und dann mit einer
            # Meldung ueber ein fehlendes Artefakt statt ueber die Datei.
            raise SystemExit(
                f"Kein Bild und kein Ordner: {pfad}\n"
                f"Eigene Fotos gehoeren nach {paths(cfg, ROOT).own_images}"
            )
    if not bilder:
        raise SystemExit(f"Keine Bilder in {args.bild} -- Standardordner: {paths(cfg, ROOT).own_images}")
    locator = Locator(cfg, ROOT, args.method, args.adapter, verbose=not args.json,
                      referenz=args.referenz)
    antworten, fehler = locator.locate_many(bilder, k=args.k)
    for bild, e in fehler.items():
        print(f"{bild}: uebersprungen ({e})", file=sys.stderr)
    for bild, a in antworten.items():
        if args.json:
            continue
        print(f"\n{bild}")
        print(f"  Position:    {a['lat']:.6f}, {a['lon']:.6f}   "
              f"https://www.openstreetmap.org/?mlat={a['lat']:.6f}&mlon={a['lon']:.6f}#map=17/{a['lat']:.5f}/{a['lon']:.5f}")
        print(f"  Konfidenz (cos) {a['konfidenz']:.3f}, Marge zu Platz 2 {a['marge']:.3f}, "
              f"Streuung der Top-{args.k} {a['streuung_m']:,.0f} m")
        for platz, t in enumerate(a["treffer"], start=1):
            print(f"  #{platz:<2} cos {t['aehnlichkeit']:.3f}  {t['lat']:.5f}, {t['lon']:.5f}  "
                  f"({t['abstand_zu_top1_m']:,.0f} m zu #1)  image_id {t['image_id']} ({t['split']})")
        if a["konfidenz"] > 0.95:
            print("  Hinweis: fast identisch mit einem Referenzbild -- liegt das Foto selbst im Datensatz?")
    if args.json:
        print(json.dumps(antworten, indent=2))
    locator.close()


if __name__ == "__main__":
    main()
