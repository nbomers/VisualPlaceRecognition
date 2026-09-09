"""
Holt die Fremd-Repos, die nicht mit ins Git-Repo gehoeren.

Nur AnyLoc und MixVPR werden als Klon gebraucht -- EigenPlaces und MegaLoc
kommen ueber torch.hub und laden sich beim ersten Lauf selbst nach.

Die Commits sind festgenagelt: aendert eines der Projekte seine helper.py,
bricht sonst irgendwann ein Aufbau, der monatelang lief.

    python setup_external.py
"""

import hashlib
import shutil
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


def fetch_repo(name, spec):
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


# Google Drive liefert bei Ueberlastung eine HTML-Fehlerseite mit Status 200.
# Die Pruefsumme faengt das ab, bevor die Datei an ihren Platz kommt.
MIXVPR_GEWICHTE = {
    "gdrive_id": "1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L",
    "sha256": "97528606773e9920e93ca4d211daeabf1c7312480f38b45379ee5551a1b4dd24",
    "bytes": 43_747_109,
    "quelle": "https://github.com/amaralibey/MixVPR#weights",
}

# AnyLocs eigene OneDrive-Links sind tot (Konto migriert, Share gibt 404).
# Die HuggingFace-Space haelt dieselben Vokabulare als Einzeldateien.
ANYLOC_HF_SPACE = "TheProjectsGuy/AnyLoc"


def sha256(pfad, block=1 << 20):
    h = hashlib.sha256()
    with open(pfad, "rb") as f:
        for stueck in iter(lambda: f.read(block), b""):
            h.update(stueck)
    return h.hexdigest()


def fetch_mixvpr_weights():
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


def anyloc_vocabulary_path():
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


def fetch_anyloc_vocabulary():
    ordner, vorhanden = anyloc_vocabulary_path()
    a = CFG["vpr"]["anyloc"]

    if vorhanden:
        print(f"  AnyLoc-Vokabular: {a['vocabulary_domain']} ({vorhanden[0]})")
        return []

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return [
            "AnyLoc-Vokabular fehlt und huggingface_hub ist nicht installiert.\n"
            "      pip install huggingface_hub   und erneut starten."
        ]

    relativ = (
        f"cache/vocabulary/{CFG['vpr']['models']['anyloc']}/"
        f"l{a['desc_layer']}_{a['desc_facet']}_c{a['num_clusters']}/"
        f"{a['vocabulary_domain']}/c_centers.pt"
    )
    print(f"  AnyLoc-Vokabular: lade {relativ} aus {ANYLOC_HF_SPACE} ...")
    try:
        quelle = hf_hub_download(
            repo_id=ANYLOC_HF_SPACE, repo_type="space", filename=relativ
        )
    except Exception as e:
        return [
            f"AnyLoc-Vokabular nicht ladbar: {type(e).__name__}: {e}\n"
            f"      Erwartet wurde {relativ}\n"
            f"      in https://huggingface.co/spaces/{ANYLOC_HF_SPACE}/tree/main/cache\n"
            "      Gibt es die Kombination aus vocabulary_domain, num_clusters,\n"
            "      desc_layer und desc_facet dort ueberhaupt?"
        ]

    ordner.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(quelle, ordner / "c_centers.pt")
    print(f"      fertig: {(ordner / 'c_centers.pt').relative_to(ROOT)}")
    return []


def main():
    print("Fremd-Repos (EigenPlaces und MegaLoc kommen ueber torch.hub):")
    ok = all([fetch_repo(name, spec) for name, spec in REPOS.items()])

    print("\nZusatzdateien:")
    offen = fetch_mixvpr_weights() + fetch_anyloc_vocabulary()

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
