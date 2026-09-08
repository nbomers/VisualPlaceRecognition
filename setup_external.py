"""
Holt die Fremd-Repos, die nicht mit ins Git-Repo gehoeren.

Nur AnyLoc und MixVPR werden als Klon gebraucht -- EigenPlaces und MegaLoc
kommen ueber torch.hub und laden sich beim ersten Lauf selbst nach.

Die Commits sind festgenagelt: aendert eines der Projekte seine helper.py,
bricht sonst irgendwann ein Aufbau, der monatelang lief.

    python setup_external.py
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())

REPOS = {
    "anyloc": {
        "url": "https://github.com/AnyLoc/AnyLoc.git",
        "commit": "a9fda68",
        "pfad": CFG["vpr"]["anyloc"]["repo_path"],
        "pruefdatei": "utilities.py",
    },
    "mixvpr": {
        "url": "https://github.com/amaralibey/MixVPR.git",
        "commit": "4043915",
        "pfad": CFG["vpr"]["mixvpr"]["repo_path"],
        "pruefdatei": "models/helper.py",
    },
}


def git(*args, cwd=None):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def hole(name, spec):
    ziel = ROOT / spec["pfad"]

    if (ziel / spec["pruefdatei"]).exists():
        try:
            ist = git("rev-parse", "--short", "HEAD", cwd=ziel)
        except subprocess.CalledProcessError:
            print(f"  {name}: liegt vor (kein git-Repo, Commit unbekannt)")
            return True
        if ist.startswith(spec["commit"]) or spec["commit"].startswith(ist):
            print(f"  {name}: liegt vor auf {ist}")
        else:
            print(
                f"  {name}: liegt vor auf {ist}, erwartet wird {spec['commit']}.\n"
                f"      Absichtlich? Sonst: git -C {spec['pfad']} checkout {spec['commit']}"
            )
        return True

    print(f"  {name}: klone nach {spec['pfad']} ...")
    ziel.parent.mkdir(parents=True, exist_ok=True)
    try:
        git("clone", spec["url"], str(ziel))
        git("checkout", "--quiet", spec["commit"], cwd=ziel)
    except subprocess.CalledProcessError as e:
        print(f"      fehlgeschlagen: {e.stderr.strip().splitlines()[-1:]}")
        return False
    print(f"      fertig auf {spec['commit']}")
    return True


# Ein Google-Drive-Download kann statt der Datei eine HTML-Fehlerseite
# liefern. Die Pruefsumme faengt das ab, bevor die Datei an ihren Platz kommt.
MIXVPR_GEWICHTE = {
    "gdrive_id": "1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L",
    "sha256": "97528606773e9920e93ca4d211daeabf1c7312480f38b45379ee5551a1b4dd24",
    "bytes": 43_747_109,
    "quelle": "https://github.com/amaralibey/MixVPR#weights",
}

# Einzeldatei-Link aus external/AnyLoc/demo/utilities.py (od_down_links).
# Der Ordnerlink im Haupt-README taugt dafuer nicht.
ANYLOC_CACHE_LINK = (
    "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/"
    "avneesh_mishra_research_iiit_ac_in/EW-ZqUeWWexNhbLEQvsCk2wBeucxNlhEpsfeUHHOreyLag"
)


def sha256(pfad, block=1 << 20):
    h = hashlib.sha256()
    with open(pfad, "rb") as f:
        for stueck in iter(lambda: f.read(block), b""):
            h.update(stueck)
    return h.hexdigest()


def hole_mixvpr_gewichte():
    ziel = ROOT / CFG["vpr"]["mixvpr"]["weights"]

    if ziel.exists():
        if sha256(ziel) == MIXVPR_GEWICHTE["sha256"]:
            print(f"  MixVPR-Gewichte: {ziel.name} (Pruefsumme stimmt)")
        else:
            print(
                f"  MixVPR-Gewichte: {ziel.name} liegt vor, aber die Pruefsumme "
                f"weicht ab.\n      Andere Variante als erwartet? Erwartet wird "
                f"ResNet50 mit 4096 Dimensionen."
            )
        return []

    try:
        import gdown
    except ImportError:
        return [
            "MixVPR-Gewichte fehlen und gdown ist nicht installiert.\n"
            "      pip install gdown   und dieses Skript erneut starten,\n"
            f"      oder von Hand nach {ziel.relative_to(ROOT)}:\n"
            f"      {MIXVPR_GEWICHTE['quelle']}"
        ]

    print(f"  MixVPR-Gewichte: lade von Google Drive ({MIXVPR_GEWICHTE['bytes'] / 1e6:.0f} MB) ...")
    ziel.parent.mkdir(parents=True, exist_ok=True)
    teil = ziel.with_suffix(ziel.suffix + ".part")
    try:
        gdown.download(id=MIXVPR_GEWICHTE["gdrive_id"], output=str(teil), quiet=False)
    except Exception as e:
        teil.unlink(missing_ok=True)
        return [
            f"MixVPR-Download fehlgeschlagen: {e}\n"
            f"      Google Drive drosselt bei zu vielen Zugriffen. Von Hand:\n"
            f"      {MIXVPR_GEWICHTE['quelle']}  ->  {ziel.relative_to(ROOT)}"
        ]

    if not teil.exists() or sha256(teil) != MIXVPR_GEWICHTE["sha256"]:
        teil.unlink(missing_ok=True)
        return [
            "MixVPR-Download hat nicht die erwartete Datei geliefert "
            "(Pruefsumme falsch).\n"
            f"      Meist eine Drive-Fehlerseite. Von Hand: {MIXVPR_GEWICHTE['quelle']}"
        ]

    teil.replace(ziel)
    print(f"      fertig: {ziel.relative_to(ROOT)}")
    return []


def anyloc_vokabular_pfad():
    a = CFG["vpr"]["anyloc"]
    ordner = (
        ROOT
        / a["repo_path"]
        / "cache"
        / "vocabulary"
        / CFG["vpr"]["models"]["anyloc"]
        / f"l{a['desc_layer']}_{a['desc_facet']}_c{a['num_clusters']}"
        / a["vocabulary_domain"]
    )
    vorhanden = [n for n in ("c_center.pt", "c_centers.pt") if (ordner / n).exists()]
    return ordner, vorhanden


def hole_anyloc_vokabular():
    ordner, vorhanden = anyloc_vokabular_pfad()
    if vorhanden:
        print(f"  AnyLoc-Vokabular: {CFG['vpr']['anyloc']['vocabulary_domain']} "
              f"({vorhanden[0]})")
        return []

    try:
        from onedrivedownloader import download
    except ImportError:
        return [
            "AnyLoc-Vokabular fehlt und onedrivedownloader ist nicht installiert.\n"
            "      pip install onedrivedownloader   und dieses Skript erneut starten,\n"
            f"      oder cache.zip von Hand nach {ordner.parents[4].relative_to(ROOT)}\n"
            f"      entpacken: {ANYLOC_CACHE_LINK}"
        ]

    ziel = ROOT / CFG["vpr"]["anyloc"]["repo_path"]
    print("  AnyLoc-Vokabular: lade cache.zip von OneDrive ...")
    try:
        download(
            ANYLOC_CACHE_LINK,
            filename=str(ziel / "cache.zip"),
            unzip=True,
            unzip_path=str(ziel),
        )
    except Exception as e:
        return [
            f"AnyLoc-Download fehlgeschlagen: {e}\n"
            f"      Von Hand: {ANYLOC_CACHE_LINK}\n"
            f"      cache.zip nach {ziel.relative_to(ROOT)} entpacken."
        ]

    ordner, vorhanden = anyloc_vokabular_pfad()
    if not vorhanden:
        return [
            "cache.zip wurde geladen, enthaelt aber kein Vokabular fuer die\n"
            f"      aktuelle Konfiguration. Erwartet unter:\n"
            f"      {ordner.relative_to(ROOT)}\n"
            "      Passen vocabulary_domain, num_clusters, desc_layer und\n"
            "      desc_facet in der config.yaml zum Inhalt des Caches?"
        ]

    print(f"      fertig: {(ordner / vorhanden[0]).relative_to(ROOT)}")
    return []


def main():
    print("Fremd-Repos (EigenPlaces und MegaLoc kommen ueber torch.hub):")
    ok = all([hole(name, spec) for name, spec in REPOS.items()])

    print("\nZusatzdateien:")
    offen = hole_mixvpr_gewichte() + hole_anyloc_vokabular()

    if offen:
        print("\nNoch zu beschaffen:")
        for eintrag in offen:
            print(f"  - {eintrag}")
        print("\nNur noetig, wenn du das jeweilige Verfahren benutzen willst.")

    if not ok:
        sys.exit(1)
    print("\nBereit.")


if __name__ == "__main__":
    main()
