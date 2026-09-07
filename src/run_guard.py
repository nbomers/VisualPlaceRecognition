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
# so oder so die beste Epoche. Sie gehoeren damit nicht zur Identitaet der
# Gewichte.
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
        "positive_radius_m": cfg["vpr"]["positive_radius_m"],
        "uncertain_radius_m": cfg["vpr"]["uncertain_radius_m"],
        "hard_negative_min_m": cfg["vpr"]["hard_negative_min_m"],
        "hard_negative_max_m": cfg["vpr"]["hard_negative_max_m"],
        "max_heading_diff_deg": cfg["vpr"]["max_heading_diff_deg"],
    }


def short_hash(fingerprint):
    blob = json.dumps(fingerprint, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:10]


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
            f"Die erzeugende Stufe mit der aktuellen config.yaml laufen lassen."
        )

    side = _sidecar(artifact_path)
    if not side.exists():
        raise FileNotFoundError(
            f"{what} hat keinen Fingerabdruck: {side.name} fehlt.\n"
            f"Die Datei stammt aus einem Lauf vor Einfuehrung der Pruefung. "
            f"Die erzeugende Stufe neu laufen lassen."
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
        f"  -> Die erzeugende Stufe mit dieser config.yaml neu laufen lassen."
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
