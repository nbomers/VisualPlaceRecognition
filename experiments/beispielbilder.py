"""
Beispielaufnahmen aus dem Datensatz fuer das README.

Zwei Fragen, zwei Wege:

  Was ist TYPISCH?  Automatisch oder --auswahl: vier Fotografen aus
                    moeglichst verschiedenen Jahren, zufaellig gezogen. Weil
                    ein Konto fast die Haelfte der Bilder stellt und die
                    meisten Fahrten Strassen abfahren, sieht das aus wie
                    eine Dashcam -- und das ist ehrlich.
  Was ist MOEGLICH? --szenen: je Szenentyp Kandidaten, bestimmt ueber
                    OpenStreetMap statt ueber den Bildinhalt -- Fussgaenger-
                    zone, Sehenswuerdigkeit (Kamera schaut darauf), Park,
                    Ufer, Wohnstrasse, Hauptstrasse, Autobahn. Dazu der
                    Anteil jedes Typs am Datensatz, damit die Bildunter-
                    schrift nicht mehr Vielfalt verspricht, als da ist.

    python experiments/beispielbilder.py --szenen                 # Kontaktabzug je Szene -> cache/
    python experiments/beispielbilder.py --ids A:Altstadt B:Park  # genau diese, mit Beschriftung
    python experiments/beispielbilder.py --auswahl                # 12 typische Kandidaten -> cache/
    python experiments/beispielbilder.py                          # vier typische, automatisch

Ergebnis: results/<stadt>/figures/demo/beispielbilder.png, der Eintrag in
QUELLEN.md daneben (src/quellen.py) und auf der Konsole der Block fuer das
README. Die creator_id steht bewusst NICHT in der Abbildung (NOTICE.md:
pseudonym, aber zusammen mit Ort und Zeit eine Aufnahmespur); die
Namensnennung laeuft ueber die Bildseite, dort nennt Mapillary den Urheber.
Kontaktabzuege landen unter cache/ und damit nicht im Git.
"""

import argparse

import numpy as np
import pandas as pd

from _common import CFG, PATHS, ROOT
from src.quellen import LINK, notiere_quellen

ZIEL = PATHS.figures / "demo" / "beispielbilder.png"
KONTAKT = PATHS.cache / "beispielbilder_auswahl.png"
KONTAKT_SZENEN = PATHS.cache / "beispielbilder_szenen.png"
# Einheitliches Seitenverhaeltnis, sonst stehen Panoramen (2:1) neben
# Hochformat-Fotos und die Reihe zerfaellt.
FORMAT = 4 / 3
# So viele Zeilen werden hoechstens nach vorhandenen Bildern durchsucht.
SUCHTIEFE = 50_000

# Szenentypen: Beschriftung, OSM-Tags, Regel. Die Regeln sind bewusst grob --
# sie schlagen Kandidaten vor, ausgesucht wird im Kontaktabzug mit Augen.
#   nahe   Bild hoechstens so viele Meter vom Objekt entfernt
#   innen  Bild liegt in der Flaeche
#   blick  Objekt 15 m bis so viele Meter entfernt UND die Kamera schaut
#          darauf (Kompass +-BLICKWINKEL) -- sonst steht man nur daneben
SZENEN = {
    "fussgaenger": ("Fußgängerzone", {"highway": ["pedestrian", "living_street"]}, ("nahe", 12)),
    "wahrzeichen": ("Sehenswürdigkeit", {"historic": True, "tourism": ["attraction", "museum"],
                                         "amenity": ["place_of_worship"]}, ("blick", 120)),
    "park": ("Park", {"leisure": ["park", "garden"],
                      "landuse": ["forest", "recreation_ground"]}, ("innen", 0)),
    "wasser": ("Am Wasser", {"waterway": ["river", "canal"], "natural": ["water"]}, ("nahe", 40)),
    "wohnen": ("Wohnstraße", {"highway": ["residential"]}, ("nahe", 8)),
    "haupt": ("Hauptstraße", {"highway": ["primary", "secondary"]}, ("nahe", 8)),
    "autobahn": ("Autobahn", {"highway": ["motorway"]}, ("nahe", 15)),
}
BLICKWINKEL = 25
# Tageslicht in Ortszeit -- Nachtaufnahmen sind echte Daten, aber als
# Beispielbild nur schwarz.
TAGSTUNDEN = range(9, 18)


def _args():
    ap = argparse.ArgumentParser(
        description="Beispielaufnahmen aus dem Datensatz fuer das README.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ids", nargs="+", metavar="ID[:TEXT]",
                    help="genau diese Bilder, in dieser Reihenfolge; TEXT ersetzt die "
                         "Jahreszahl als Beschriftung")
    ap.add_argument("--auswahl", type=int, nargs="?", const=12, metavar="N",
                    help="Kontaktabzug mit N typischen Kandidaten und ihren IDs (Standard 12)")
    ap.add_argument("--szenen", type=int, nargs="?", const=6, metavar="N",
                    help="Kontaktabzug mit N Kandidaten je Szenentyp (Standard 6), ueber OSM")
    ap.add_argument("--n", type=int, default=4, help="Bilder in der automatischen Abbildung")
    return ap.parse_args()


# -- Typisch: nach Fotograf und Jahr -----------------------------------------

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


# -- Moeglich: nach Szenentyp ueber OSM --------------------------------------

def _alle_tags():
    """Die Tags aller Szenen in EINER Overpass-Abfrage; True schlaegt Listen."""
    tags = {}
    for _, t, _ in SZENEN.values():
        for k, v in t.items():
            if v is True or tags.get(k) is True:
                tags[k] = True
            else:
                tags[k] = sorted(set(tags.get(k, [])) | set(v))
    return tags


def osm_objekte():
    """
    OSM-Objekte im Stadtgebiet, UTM. Laeuft ueber den osmnx-Cache unter
    cache/ -- ein zweiter Aufruf geht nicht mehr ans Netz.
    """
    import geopandas as gpd

    from src.districts import _features, city_boundary, configure_osmnx

    configure_osmnx(CFG, PATHS.cache)
    polygon, utm = city_boundary(CFG["city"])
    roh = _features(polygon, _alle_tags())
    if roh.empty:
        raise SystemExit("OSM lieferte fuer das Stadtgebiet keine passenden Objekte.")
    return gpd.GeoDataFrame(roh.reset_index(drop=True), geometry="geometry",
                            crs="EPSG:4326").to_crs(utm), utm


def _passt_tag(objekte, tags):
    """Zeilen, auf die eines der Tags der Szene passt."""
    maske = np.zeros(len(objekte), dtype=bool)
    for k, v in tags.items():
        if k not in objekte.columns:
            continue
        spalte = objekte[k]
        maske |= spalte.notna().to_numpy() if v is True else spalte.isin(v).to_numpy()
    return maske


def szenen_zuordnen(meta, objekte, utm):
    """
    Je Szene: boolesche Maske ueber meta und, fuer Sehenswuerdigkeiten, der
    Name des Objekts, auf das die Kamera schaut.
    """
    import geopandas as gpd

    punkte = gpd.GeoDataFrame(
        {"zeile": np.arange(len(meta))},
        geometry=gpd.points_from_xy(meta["lon"], meta["lat"]), crs="EPSG:4326",
    ).to_crs(utm)
    masken, namen = {}, pd.Series("", index=np.arange(len(meta)), dtype=object)

    for schluessel, (_, tags, (regel, meter)) in SZENEN.items():
        teil = objekte[_passt_tag(objekte, tags)]
        maske = np.zeros(len(meta), dtype=bool)
        if regel == "innen":
            teil = teil[teil.geom_type.isin(["Polygon", "MultiPolygon"])]
            if len(teil):
                treffer = gpd.sjoin(punkte, teil[["geometry"]], predicate="within")
                maske[treffer["zeile"].to_numpy()] = True
        elif regel == "nahe" and len(teil):
            treffer = gpd.sjoin_nearest(punkte, teil[["geometry"]], max_distance=meter)
            maske[treffer["zeile"].to_numpy()] = True
        elif regel == "blick":
            # Nur benannte Objekte -- ein namenloser Grenzstein ist keine
            # Sehenswuerdigkeit, und die Beschriftung braucht den Namen.
            if "name" in teil.columns:
                teil = teil[teil["name"].notna()]
            if len(teil):
                ziel = gpd.GeoDataFrame({"name": teil["name"].to_numpy()},
                                        geometry=teil.geometry.representative_point().to_numpy(),
                                        crs=utm)
                paar = gpd.sjoin_nearest(punkte, ziel, max_distance=meter, distance_col="abstand")
                paar = paar[~paar.index.duplicated()]
                ziel_xy = ziel.geometry.loc[paar["index_right"]]
                dx = ziel_xy.x.to_numpy() - paar.geometry.x.to_numpy()
                dy = ziel_xy.y.to_numpy() - paar.geometry.y.to_numpy()
                peilung = np.degrees(np.arctan2(dx, dy)) % 360
                kompass = meta["compass_angle"].to_numpy()[paar["zeile"].to_numpy()]
                abweichung = np.abs((peilung - kompass + 180) % 360 - 180)
                # compass_angle -1 heisst unbekannt, nicht 359 Grad.
                ok = (kompass >= 0) & (abweichung <= BLICKWINKEL) & (paar["abstand"].to_numpy() >= 15)
                zeilen = paar["zeile"].to_numpy()[ok]
                maske[zeilen] = True
                namen.iloc[zeilen] = paar["name"].to_numpy()[ok]
        masken[schluessel] = maske
    return masken, namen


def kandidaten(meta, maske, n, rng, objekt=None):
    """
    Bis zu n Zeilen aus der Maske: bei Tageslicht, auf diesem Rechner
    vorhanden, je Fotograf und je Fahrt hoechstens eine -- und mit `objekt`
    (Name je Zeile) auch je Sehenswuerdigkeit nur eine.
    """
    stunde = (pd.to_datetime(meta["captured_at"], unit="ms", utc=True)
              .dt.tz_convert("Europe/Berlin").dt.hour.to_numpy())
    zeilen = np.flatnonzero(maske & np.isin(stunde, TAGSTUNDEN) & ~meta["is_pano"].to_numpy())
    konten, fahrten, objekte, gewaehlt = set(), set(), set(), []
    for z in rng.permutation(zeilen)[:SUCHTIEFE]:
        k, f = meta["creator_id"].iat[z], meta["sequence_id"].iat[z]
        o = objekt.iat[z] if objekt is not None else None
        if k in konten or f in fahrten or (o and o in objekte):
            continue
        if not PATHS.image_file(meta["image_id"].iat[z]).exists():
            continue
        gewaehlt.append(z)
        konten.add(k)
        fahrten.add(f)
        objekte.add(o)
        if len(gewaehlt) == n:
            break
    return gewaehlt


def szenen(meta, n, rng, objekte=None, utm=None):
    """Kontaktabzug je Szene, Anteile am Datensatz, Vorschlag fuer --ids."""
    if objekte is None:
        objekte, utm = osm_objekte()
    masken, namen = szenen_zuordnen(meta, objekte, utm)
    ohne_pano = ~meta["is_pano"].to_numpy()

    print(f"{'Szene':<18}{'Anteil am Datensatz':>21}{'Kandidaten':>12}")
    zeilen, titel, vorschlag = [], [], []
    for schluessel, (label, _, _) in SZENEN.items():
        m = masken[schluessel]
        wahl = kandidaten(meta, m, n, rng, namen if schluessel == "wahrzeichen" else None)
        print(f"{label:<18}{(m & ohne_pano).sum() / max(ohne_pano.sum(), 1):>20.1%}{len(wahl):>12}")
        for z in wahl:
            text = namen.iat[z] or label
            zeilen.append(z)
            name = f"{text}  ·  " if text != label else ""
            titel.append(f"{label}\n{name}{meta['image_id'].iat[z]}")
        if wahl:
            text = namen.iat[wahl[0]] or label
            vorschlag.append(f'"{meta["image_id"].iat[wahl[0]]}:{text}"')
    print("Szenen ueberlappen: ein Bild in der Fussgaengerzone vor dem Dom zaehlt zweimal.")

    if not zeilen:
        raise SystemExit(
            f"Keine Kandidaten -- liegen die Bilder unter {PATHS.images}?\n"
            "VPR_IMAGE_ROOT oder VPR_IMAGE_PATH setzen.")
    auswahl = meta.iloc[zeilen].reset_index(drop=True)
    zeichne(auswahl, KONTAKT_SZENEN, spalten=n, titel=titel)
    print(f"\nKontaktabzug: {KONTAKT_SZENEN.relative_to(ROOT)}  ({len(zeilen)} Kandidaten)")
    print("Aussuchen, Text nach dem Doppelpunkt anpassen, dann zum Beispiel:")
    print("  python experiments/beispielbilder.py --ids " + " ".join(vorschlag[:6]))


# -- Abbildung ----------------------------------------------------------------

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


def zeichne(zeilen, ziel, spalten, mit_id=False, titel=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reihen = -(-len(zeilen) // spalten)
    # Zweizeilige Titel brauchen mehr Hoehe, sonst laufen sie ins Bild darueber.
    zweizeilig = mit_id or (titel and any("\n" in t for t in titel))
    hoehe = 3.75 if zweizeilig else 3.3
    fig, achsen = plt.subplots(reihen, spalten, figsize=(4 * spalten, hoehe * reihen),
                               squeeze=False)
    for ax in achsen.flat:
        ax.set_axis_off()
    for nr, (ax, z) in enumerate(zip(achsen.flat, zeilen.itertuples())):
        ax.imshow(lade(z.image_id))
        if titel:
            text = titel[nr]
        else:
            text = str(pd.Timestamp(int(z.captured_at), unit="ms").year)
            if z.is_pano:
                text += "  ·  Panorama"
            if mit_id:
                text += f"\n{z.image_id}"
        ax.set_title(text, fontsize=10)
    fig.text(0.5, 0.005, "Bilder von Mapillary (mapillary.com), CC BY-SA 4.0",
             ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    ziel.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ziel, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _ids_und_texte(eintraege):
    """'123:Dom St. Peter' -> ('123', 'Dom St. Peter'); ohne Doppelpunkt kein Text."""
    ids, texte = [], []
    for e in eintraege:
        i, _, t = e.partition(":")
        ids.append(i.strip())
        texte.append(t.strip())
    return ids, texte


def main():
    args = _args()
    meta = pd.read_parquet(PATHS.processed / "metadata.parquet")
    rng = np.random.default_rng(int(CFG["vpr"]["split_seed"]))

    if args.szenen:
        szenen(meta, args.szenen, rng)
        return
    if args.auswahl:
        zeilen = automatisch(meta, args.auswahl, rng)
        zeichne(zeilen, KONTAKT, spalten=4, mit_id=True)
        print(f"Kontaktabzug: {KONTAKT.relative_to(ROOT)}  ({len(zeilen)} Kandidaten)")
        print("Vier aussuchen, dann:")
        print("  python experiments/beispielbilder.py --ids " + " ".join(
            str(i) for i in zeilen["image_id"][: args.n]))
        return

    texte = None
    if args.ids:
        ids, texte = _ids_und_texte(args.ids)
        zeilen = nach_ids(meta, ids)
    else:
        zeilen = automatisch(meta, args.n, rng)
    # Mit Beschriftung: "Text · Jahr"; ohne: nur das Jahr, wie bisher.
    if texte and any(texte):
        jahre = pd.to_datetime(zeilen["captured_at"], unit="ms").dt.year
        titel = [f"{t}  ·  {j}" if t else str(j) for t, j in zip(texte, jahre)]
    else:
        titel = None
    # Bis vier Bilder in einer Reihe, darueber drei je Reihe.
    zeichne(zeilen, ZIEL, spalten=len(zeilen) if len(zeilen) <= 4 else 3, titel=titel)
    notiere_quellen(ZIEL.parent, ZIEL.name, zeilen["image_id"])

    pfad = ZIEL.relative_to(ROOT).as_posix()
    namen = [t or str(nr) for nr, t in enumerate(texte or [""] * len(zeilen), start=1)]
    links = " ·\n".join(f"[{n}]({LINK.format(i)})" for n, i in zip(namen, zeilen["image_id"]))
    print(f"geschrieben: {pfad} und QUELLEN.md daneben\n")
    print("Fuer das README -- ersetzt die Abbildung im Abschnitt Daten:\n")
    print(f"![Beispielaufnahmen aus dem Datensatz]({pfad})\n")
    print(f"<sub>Bilder von [Mapillary](https://www.mapillary.com), CC BY-SA 4.0:\n{links}</sub>")


if __name__ == "__main__":
    main()
