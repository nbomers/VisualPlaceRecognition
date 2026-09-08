"""
Holt die Fremd-Repos, die nicht mit ins Git-Repo gehoeren.

Nur AnyLoc und MixVPR werden als Klon gebraucht -- EigenPlaces und MegaLoc
kommen ueber torch.hub und laden sich beim ersten Lauf selbst nach.

Die Commits sind festgenagelt: aendert eines der Projekte seine helper.py,
bricht sonst irgendwann ein Aufbau, der monatelang lief.

    python setup_external.py
"""

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


def pruefe_zusatzdateien():
    """Gewichte und Vokabular liegen hinter Downloads, nicht hinter git."""
    offen = []

    ckpt = ROOT / CFG["vpr"]["mixvpr"]["weights"]
    if ckpt.exists():
        print(f"  MixVPR-Gewichte: {ckpt.name}")
    else:
        offen.append(
            f"MixVPR-Gewichte fehlen: {ckpt}\n"
            f"      Download-Link steht im README von {REPOS['mixvpr']['url']}"
        )

    a = CFG["vpr"]["anyloc"]
    vok = (
        ROOT
        / a["repo_path"]
        / "cache"
        / "vocabulary"
        / CFG["vpr"]["models"]["anyloc"]
        / f"l{a['desc_layer']}_{a['desc_facet']}_c{a['num_clusters']}"
        / a["vocabulary_domain"]
        / "c_center.pt"
    )
    if vok.exists():
        print(f"  AnyLoc-Vokabular: {a['vocabulary_domain']}")
    else:
        offen.append(
            f"AnyLoc-Vokabular fehlt: {vok}\n"
            f"      Ohne das fittet AnyLoc die Cluster-Zentren selbst auf den\n"
            f"      Trainingsbildern -- laeuft, ist dann aber nicht mehr mit den\n"
            f"      Zahlen aus dem Paper vergleichbar. Offizielles Vokabular:\n"
            f"      cache.zip aus den AnyLoc-Public-Data entpacken."
        )
    return offen


def main():
    print("Fremd-Repos (EigenPlaces und MegaLoc kommen ueber torch.hub):")
    ok = all([hole(name, spec) for name, spec in REPOS.items()])

    print("\nZusatzdateien:")
    offen = pruefe_zusatzdateien()

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
