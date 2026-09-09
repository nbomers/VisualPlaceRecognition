"""
Schutz gegen stille Verwechslung von Artefakten zwischen Laeufen.

Problem: Dateinamen kodieren nur METHOD und ADAPTER. Aendert man in der
config.yaml etwas anderes -- Modell, image_size, split_seed -- entsteht eine
Datei mit identischem Namen und anderem Inhalt. Nichts faellt auf.

Loesung: Neben jedes Artefakt kommt ein Fingerabdruck aller Einstellungen,
die seinen Inhalt bestimmen. Wer das Artefakt spaeter laedt, prueft ihn und
bricht bei Abweichung ab, statt falsche Zahlen zu berechnen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def metadata_digest(metadata):
    """Inhaltshash ueber image_id + split -- erkennt einen geaenderten Split."""
    cols = [c for c in ("image_id", "split") if c in metadata.columns]
    h = pd.util.hash_pandas_object(metadata[cols], index=False)
    return hashlib.sha256(h.to_numpy().tobytes()).hexdigest()[:16]


# Rein lokale Angaben. Ein verschobener Repo-Klon aendert das Modell nicht,
# darf also auch nicht die Embeddings entwerten.
_LOKALE_SCHLUESSEL = ("repo_path",)


def _model_config(block):
    """Methodenblock ohne Eintraege, die nur den Ablageort beschreiben."""
    if not isinstance(block, dict):
        return block
    gefiltert = {k: v for k, v in block.items() if k not in _LOKALE_SCHLUESSEL}
    # Gewichtsdatei ueber den Namen identifizieren, nicht ueber den Pfad.
    if "weights" in gefiltert:
        gefiltert["weights"] = Path(str(gefiltert["weights"])).name
    return gefiltert


def embedding_fingerprint(cfg, method, adapter, metadata):
    """Alles, was den Inhalt einer Embedding-Datei bestimmt.

    Bewusst nur aus config.yaml + Metadaten abgeleitet, damit jede Stufe den
    Fingerabdruck unabhaengig nachrechnen kann.
    """
    vpr = cfg["vpr"]
    fp = {
        "method": method,
        "model_id": vpr["models"][method],
        "adapter": adapter,
        "method_config": _model_config(vpr.get(method)),
        "split": {
            k: vpr.get(k)
            for k in ("split_seed", "train_fraction", "database_fraction", "query_fraction")
        },
        "metadata_digest": metadata_digest(metadata),
        "n_rows": int(len(metadata)),
    }
    return fp


# Abbruchkriterien bestimmen nur, wann das Training endet -- gespeichert wird
# ohnehin die beste Epoche.
_ABBRUCHKRITERIEN = ("patience", "min_delta")


def adapter_fingerprint(cfg, method, base_fingerprint):
    """Fingerabdruck der Adapter-Gewichte: Basis-Embeddings + Trainingsparameter."""
    training = {
        k: v
        for k, v in cfg["vpr"]["adapter_training"].items()
        if k not in _ABBRUCHKRITERIEN
    }
    return {
        "trained_on": {**base_fingerprint, "adapter": "none"},
        "adapter_training": training,
        "val_fraction": cfg["vpr"].get("val_fraction"),
        "val_radius_m": cfg["vpr"].get("val_radius_m"),
        "positive_radius_m": cfg["vpr"]["positive_radius_m"],
        "uncertain_radius_m": cfg["vpr"]["uncertain_radius_m"],
        "hard_negative_min_m": cfg["vpr"]["hard_negative_min_m"],
        "hard_negative_max_m": cfg["vpr"]["hard_negative_max_m"],
        "max_heading_diff_deg": cfg["vpr"]["max_heading_diff_deg"],
    }


def short_hash(fingerprint):
    blob = json.dumps(fingerprint, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:10]


# Welche Stufe erzeugt welches Artefakt -- fuer die Fehlermeldung.
_ERZEUGER = (
    ("_retrieval.npz", "06"),
    ("_linear.pt", "05"),
    ("_linear_embeddings.npy", "05"),
    ("_embeddings.npy", "04"),
)


def _hinweis(artifact_path):
    name = Path(artifact_path).name
    for endung, stufe in _ERZEUGER:
        if name.endswith(endung):
            return f"-> python run.py --from {stufe}"
    return "-> die erzeugende Stufe mit dieser config.yaml neu laufen lassen"


def _sidecar(artifact_path):
    artifact_path = Path(artifact_path)
    return artifact_path.with_name(artifact_path.name + ".fingerprint.json")


def write_fingerprint(artifact_path, fingerprint, **extra):
    """Legt <artefakt>.fingerprint.json ab. Nach dem Schreiben aufrufen."""
    payload = {"hash": short_hash(fingerprint), "fingerprint": fingerprint, **extra}
    _sidecar(artifact_path).write_text(json.dumps(payload, indent=2, default=str))
    return payload["hash"]


def require_fingerprint(artifact_path, fingerprint, what="Artefakt"):
    """
    Prueft den Fingerabdruck eines Artefakts vor dem Laden.
    Wirft mit einer Diff-Ausgabe, statt falsche Daten durchzulassen.
    """
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"{what} fehlt: {artifact_path}\n"
            f"{_hinweis(artifact_path)}"
        )

    side = _sidecar(artifact_path)
    if not side.exists():
        raise FileNotFoundError(
            f"{what} hat keinen Fingerabdruck: {side.name} fehlt.\n"
            f"{_hinweis(artifact_path)}"
        )

    stored = json.loads(side.read_text())
    want = short_hash(fingerprint)
    if stored.get("hash") == want:
        return stored

    diff = _diff(stored.get("fingerprint", {}), fingerprint)
    raise RuntimeError(
        f"{what} passt nicht zur aktuellen config.yaml.\n"
        f"  Datei:   {artifact_path.name}\n"
        f"  gespeichert {stored.get('hash')}  erwartet {want}\n"
        f"  Abweichungen:\n{diff}\n"
        f"  {_hinweis(artifact_path)}"
    )


def _diff(stored, current, prefix=""):
    lines = []
    for key in sorted(set(stored) | set(current)):
        s, c = stored.get(key), current.get(key)
        if isinstance(s, dict) and isinstance(c, dict):
            lines.append(_diff(s, c, prefix + key + "."))
        elif s != c:
            lines.append(f"    {prefix}{key}: gespeichert={s!r}  aktuell={c!r}")
    return "\n".join(x for x in lines if x)


# ----------------------------------------------------------------------
# Config-Pruefung: laeuft beim Laden, nicht erst hinter stundenlangen Stufen.
# ----------------------------------------------------------------------


# Schluessel, ohne die eine Stufe nicht laufen kann. Fehlen sie, ist die
# config.yaml aelter als der Code -- meist ein Rechner, der den Commit der
# Config nicht mitbekommen hat.
_PFLICHT = (
    "tile_workers",
    "download_workers",
    "download_image_size",
    "vpr.method",
    "vpr.models",
    "vpr.val_fraction",
    "vpr.val_radius_m",
    "vpr.embed_batch_size",
    "vpr.train_fraction",
    "vpr.database_fraction",
    "vpr.query_fraction",
    "vpr.positive_radius_m",
    "vpr.uncertain_radius_m",
    "vpr.hard_negative_min_m",
    "vpr.hard_negative_max_m",
    "vpr.max_heading_diff_deg",
    "retrieval.method",
    "retrieval.top_k",
    "retrieval.k_values",
    "retrieval.thresholds",
    "retrieval.min_days_apart",
)


def _fehlende_schluessel(cfg):
    fehlend = []
    for pfad in _PFLICHT:
        knoten = cfg
        for teil in pfad.split("."):
            if not isinstance(knoten, dict) or teil not in knoten:
                fehlend.append(pfad)
                break
            knoten = knoten[teil]
    return fehlend


def validate_config(cfg):
    fehlend = _fehlende_schluessel(cfg)
    if fehlend:
        raise ValueError(
            "config.yaml passt nicht zum Code -- diese Schluessel fehlen:\n"
            + "\n".join(f"  - {k}" for k in fehlend)
            + "\n\nMeist steht auf diesem Rechner eine aeltere config.yaml. "
            "Auf dem Rechner, auf dem sie aktuell ist, committen und hier "
            "git pull."
        )

    vpr = cfg["vpr"]
    ret = cfg["retrieval"]
    fehler = []

    if vpr["method"] not in vpr["models"]:
        fehler.append(
            f"vpr.method={vpr['method']!r} hat keinen Eintrag unter vpr.models "
            f"({sorted(vpr['models'])})"
        )

    if vpr.get("adapter", "none") not in ("none", "None", "linear"):
        fehler.append(f"vpr.adapter={vpr['adapter']!r} ist unbekannt (none oder linear)")

    anteile = sum(
        float(vpr[k]) for k in ("train_fraction", "database_fraction", "query_fraction")
    )
    if abs(anteile - 1.0) > 1e-6:
        fehler.append(f"train/database/query_fraction ergeben {anteile}, nicht 1.0")

    val = float(vpr.get("val_fraction", 0.1))
    if not 0.0 < val < 1.0:
        fehler.append(f"vpr.val_fraction={val} muss zwischen 0 und 1 liegen")

    # Hard Negatives duerfen nicht in die Unsicherheitszone reichen, sonst
    # trainiert man gegen die eigene Ground Truth.
    p_r, u_r = float(vpr["positive_radius_m"]), float(vpr["uncertain_radius_m"])
    hn_min, hn_max = float(vpr["hard_negative_min_m"]), float(vpr["hard_negative_max_m"])
    if not p_r <= u_r <= hn_min < hn_max:
        fehler.append(
            f"Radien muessen aufsteigend sein: positive({p_r}) <= uncertain({u_r}) "
            f"<= hard_negative_min({hn_min}) < hard_negative_max({hn_max})"
        )

    if not 0 < float(vpr["max_heading_diff_deg"]) <= 180:
        fehler.append("vpr.max_heading_diff_deg muss in (0, 180] liegen")

    # Faellt sonst erst in 07 auf, nachdem 06 schon gelaufen ist.
    if max(ret["k_values"]) > ret["top_k"]:
        fehler.append(
            f"retrieval.k_values geht bis {max(ret['k_values'])}, "
            f"gespeichert werden aber nur top_k={ret['top_k']} Nachbarn"
        )

    if not ret["thresholds"] or min(ret["thresholds"]) <= 0:
        fehler.append("retrieval.thresholds muss positive Werte enthalten")

    if ret["method"] not in ("faiss", "numpy"):
        fehler.append(f"retrieval.method={ret['method']!r} ist unbekannt")

    if fehler:
        raise ValueError(
            "config.yaml ist widerspruechlich:\n"
            + "\n".join(f"  - {f}" for f in fehler)
        )


def print_run_header(cfg, stage, device=None, run_hash=None):
    """Einheitliche Kopfzeile, damit man sich in einem langen Log zurechtfindet."""
    vpr = cfg["vpr"]
    teile = [f"[{stage}]", f"method={vpr['method']}", f"adapter={vpr.get('adapter', 'none')}"]
    if device:
        teile.append(f"device={device}")
    if run_hash:
        teile.append(f"run={run_hash}")
    zeile = "  ".join(teile)
    print(zeile)
    print("-" * len(zeile))
