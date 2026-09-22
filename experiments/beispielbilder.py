"""
Beispielaufnahmen aus dem Datensatz fuer das README.

Vier Bilder nebeneinander, die zeigen, womit das System arbeitet. Gewaehlt
wird ueber die Metadaten, nicht ueber den Bildinhalt: vier verschiedene
Fotografen aus moeglichst verschiedenen Jahren, dazu ein Panorama, falls die
Stadt eins hat (Osnabrueck hat keins). Wer lieber selbst aussucht, laesst sich
erst einen Kontaktabzug mit IDs zeigen und nennt dann vier davon.

    python experiments/beispielbilder.py --auswahl          # 12 Kandidaten mit ID -> cache/
    python experiments/beispielbilder.py --ids A B C D      # genau diese vier
    python experiments/beispielbilder.py                    # automatisch

Ergebnis: results/<stadt>/figures/demo/beispielbilder.png und auf der
Konsole der Block fuer das README, mit einem Link je Bild auf seine Seite
bei Mapillary. Die creator_id steht bewusst NICHT in der Abbildung
(NOTICE.md: pseudonym, aber zusammen mit Ort und Zeit eine Aufnahmespur);
die Namensnennung laeuft ueber die Bildseite, dort nennt Mapillary den
Urheber. Der Kontaktabzug landet unter cache/ und damit nicht im Git.
"""

import argparse

import numpy as np
import pandas as pd

from _common import CFG, PATHS, ROOT

ZIEL = PATHS.figures / "demo" / "beispielbilder.png"
KONTAKT = PATHS.cache / "beispielbilder_auswahl.png"
# Einheitliches Seitenverhaeltnis, sonst stehen Panoramen (2:1) neben
# Hochformat-Fotos und die Reihe zerfaellt.
FORMAT = 4 / 3
# So viele Zeilen werden hoechstens nach vorhandenen Bildern durchsucht.
SUCHTIEFE = 50_000


def _args():
    ap = argparse.ArgumentParser(
        description="Beispielaufnahmen aus dem Datensatz fuer das README.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ids", nargs="+", metavar="ID",
                    help="genau diese Bilder, in dieser Reihenfolge")
    ap.add_argument("--auswahl", type=int, nargs="?", const=12, metavar="N",
                    help="Kontaktabzug mit N Kandidaten und ihren IDs (Standard 12)")
    ap.add_argument("--n", type=int, default=4, help="Bilder in der Abbildung")
    return ap.parse_args()


def automatisch(meta, n, rng):
    """
    n Bilder, die auf diesem Rechner liegen, in drei Stufen: erst ein
    Panorama, dann neue Fotografen aus neuen Jahren, zuletzt nur neue
    Fotografen. Eine Stufe, die nichts findet, faellt einfach aus.
    """
    reihe = meta.iloc[rng.permutation(len(meta))[:SUCHTIEFE]]
    jahre = pd.to_datetime(reihe["captured_at"], unit="ms").dt.year.to_numpy()
    gewaehlt = []

    def passt(z, jahr, stufe):
        if stufe == 1:
            return bool(z.is_pano)
        if z.is_pano or any(z.creator_id == g.creator_id for g, _ in gewaehlt):
            return False
        return stufe == 3 or all(jahr != j for _, j in gewaehlt)

    for stufe, soll in ((1, 1), (2, n), (3, n)):
        for z, jahr in zip(reihe.itertuples(), jahre):
            if len(gewaehlt) >= soll:
                break
            if passt(z, jahr, stufe) and PATHS.image_file(z.image_id).exists():
                gewaehlt.append((z, jahr))

    if not gewaehlt:
        raise SystemExit(
            f"Kein einziges Bild unter {PATHS.images} gefunden.\n"
            "Bilder liegen ausserhalb des Repos; VPR_IMAGE_ROOT oder VPR_IMAGE_PATH setzen."
        )
    ids = [z.image_id for z, _ in sorted(gewaehlt, key=lambda g: g[1])]
    return meta.set_index("image_id").loc[ids].reset_index()


def nach_ids(meta, ids):
    """Die genannten Bilder in der genannten Reihenfolge -- oder ein klarer Abbruch."""
    als_text = meta["image_id"].astype(str)
    unbekannt = [i for i in ids if i not in set(als_text)]
    if unbekannt:
        raise SystemExit(f"Nicht in den Metadaten dieser Stadt: {', '.join(unbekannt)}")
    zeilen = meta[als_text.isin(ids)].set_index(als_text[als_text.isin(ids)]).loc[ids]
    fehlt = [str(i) for i in zeilen["image_id"] if not PATHS.image_file(i).exists()]
    if fehlt:
        raise SystemExit(f"Kein Bild auf diesem Rechner fuer: {', '.join(fehlt)}")
    return zeilen.reset_index(drop=True)


def lade(image_id, hoehe=480):
    """Bild mittig auf FORMAT zugeschnitten, feste Hoehe."""
    from PIL import Image

    with Image.open(PATHS.image_file(image_id)) as im:
        im.draft("RGB", (hoehe * 2, hoehe * 2))
        im = im.convert("RGB")
    w, h = im.size
    if w / h > FORMAT:
        neu = int(h * FORMAT)
        im = im.crop(((w - neu) // 2, 0, (w - neu) // 2 + neu, h))
    else:
        neu = int(w / FORMAT)
        im = im.crop((0, (h - neu) // 2, w, (h - neu) // 2 + neu))
    return im.resize((int(hoehe * FORMAT), hoehe))


def zeichne(zeilen, ziel, spalten, mit_id=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reihen = -(-len(zeilen) // spalten)
    # Mit ID steht der Titel zweizeilig -- ohne mehr Hoehe laeuft er in das
    # Bild darueber.
    hoehe = 3.75 if mit_id else 3.3
    fig, achsen = plt.subplots(reihen, spalten, figsize=(4 * spalten, hoehe * reihen),
                               squeeze=False)
    for ax in achsen.flat:
        ax.set_axis_off()
    for ax, z in zip(achsen.flat, zeilen.itertuples()):
        ax.imshow(lade(z.image_id))
        titel = str(pd.Timestamp(int(z.captured_at), unit="ms").year)
        if z.is_pano:
            titel += "  ·  Panorama"
        if mit_id:
            titel += f"\n{z.image_id}"
        ax.set_title(titel, fontsize=10)
    fig.text(0.5, 0.005, "Bilder von Mapillary (mapillary.com), CC BY-SA 4.0",
             ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    ziel.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ziel, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    args = _args()
    meta = pd.read_parquet(PATHS.processed / "metadata.parquet")
    rng = np.random.default_rng(int(CFG["vpr"]["split_seed"]))

    if args.auswahl:
        zeilen = automatisch(meta, args.auswahl, rng)
        zeichne(zeilen, KONTAKT, spalten=4, mit_id=True)
        print(f"Kontaktabzug: {KONTAKT.relative_to(ROOT)}  ({len(zeilen)} Kandidaten)")
        print("Vier aussuchen, dann:")
        print("  python experiments/beispielbilder.py --ids " + " ".join(
            str(i) for i in zeilen["image_id"][: args.n]))
        return

    zeilen = nach_ids(meta, args.ids) if args.ids else automatisch(meta, args.n, rng)
    zeichne(zeilen, ZIEL, spalten=len(zeilen))
    pfad = ZIEL.relative_to(ROOT).as_posix()
    links = " ·\n".join(
        f"[{nr}](https://www.mapillary.com/app/?pKey={i})"
        for nr, i in enumerate(zeilen["image_id"], start=1))
    print(f"geschrieben: {pfad}\n")
    print("Fuer das README -- ersetzt den Block BILD-PLATZHALTER 2:\n")
    print(f"![Beispielaufnahmen aus dem Datensatz]({pfad})\n")
    print(f"<sub>Bilder von [Mapillary](https://www.mapillary.com), CC BY-SA 4.0:\n{links}</sub>")


if __name__ == "__main__":
    main()
