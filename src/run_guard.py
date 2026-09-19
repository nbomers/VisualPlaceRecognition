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
import subprocess
from pathlib import Path

import pandas as pd


def metadata_digest(metadata):
    """Inhaltshash ueber image_id + split -- erkennt einen geaenderten Split."""
    cols = [c for c in ("image_id", "split") if c in metadata.columns]
    h = pd.util.hash_pandas_object(metadata[cols], index=False)
    return hashlib.sha256(h.to_numpy().tobytes()).hexdigest()[:16]


# Aendert das Ergebnis nicht: ein verschobener Repo-Klon so wenig wie eine
# andere Batchgroesse. Beides darf die Embeddings nicht entwerten.
_LOKALE_SCHLUESSEL = ("repo_path", "batch_size")


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


# Der Fingerabdruck deckt die Config ab, nicht den Code. Aendert sich die
# Auswertung selbst, sagt erst diese Kennung, dass eine JSON veraltet ist.
# retrieval.py gehoert dazu: die Sequenz-Aggregation und "loesbar" gehen in
# die seq-Zeilen und in jede Experiment-Zahl ein.
_CODE_DATEIEN = ("src/evaluation.py", "src/geo.py", "src/retrieval.py")


def code_version(root):
    """Git-Commit (falls vorhanden) und Hash der Auswertungsquellen."""
    root = Path(root)
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    h = hashlib.sha256()
    for rel in _CODE_DATEIEN:
        h.update((root / rel).read_bytes())
    return {"commit": commit, "evaluation": h.hexdigest()[:10]}


# Welche Stufe erzeugt welches Artefakt -- fuer die Fehlermeldung.
# Reihenfolge zaehlt: der speziellere Namensteil steht vorn.
_ERZEUGER = (
    ("_retrieval.npz", "python run.py --from 06"),
    ("_linear.pt", "python run.py --from 05"),
    ("_linear_embeddings.npy", "python run.py --from 05"),
    ("_embeddings.npy", "python run.py --from 04"),
)

# Abgeleitete Encoder entstehen nicht in 04, sondern aus einem anderen
# Encoder. Am Namen erkannt -- Konvention, passend zu den source-Eintraegen
# in config.yaml und zu experiments/pca_reduce.py.
_ABGELEITET_MARKER = ("_pca", "_concat")


def _hinweis(artifact_path):
    name = Path(artifact_path).name
    for endung, befehl in _ERZEUGER:
        if not name.endswith(endung):
            continue
        if endung == "_embeddings.npy" and any(m in name for m in _ABGELEITET_MARKER):
            skript = "concat_embeddings.py" if "_concat" in name else "pca_reduce.py"
            return f"-> python experiments/{skript}"
        return f"-> {befehl}"
    return "-> die erzeugende Stufe mit dieser config.yaml neu laufen lassen"


def require_city_match(root, cfg, metadata, what="Artefakt", min_overlap=0.5):
    """
    Gehoert dieses Encoder-Artefakt ueberhaupt zur konfigurierten Stadt?

    Die Luecke, die das schliesst: der Fingerabdruck beschreibt Verfahren,
    Modellkonfiguration, Split-Parameter und einen Hash der Metadaten -- aber
    nicht die Stadt. Die Metadaten liegen NEBEN den Embeddings und wandern
    beim Kopieren mit. Landet ein ganzer data/<stadt>/embeddings/-Baum beim
    rsync zwischen den beiden Projektrechnern im falschen Stadt-Zweig, passt
    er zu sich selbst, und jede Stufe rechnet bereitwillig weiter -- mit den
    Bildern der einen Stadt und den Ergebnisordnern der anderen.

    Deshalb hier eine Pruefung gegen etwas, das NICHT mitwandert: die
    versionierte data/<stadt>/processed/metadata.parquet. Verglichen werden
    die image_id-Mengen, nicht ihre Hashes -- ein Encoder darf Bilder fehlen
    haben (in Osnabrueck ist ein Download gescheitert, bei MegaLoc zwei), er
    darf nur keine FREMDEN enthalten.

    Dasselbe Muster wie min_overlap in src/split.py, das dort Split-Listen
    aus einer anderen Stadt abfaengt.

    Ohne data/<stadt>/processed/metadata.parquet (01 noch nicht gelaufen)
    passiert nichts -- es gibt dann nichts, wogegen man pruefen koennte.
    """
    from .paths import Paths

    pfad = Paths(cfg, root).processed / "metadata.parquet"
    if not pfad.exists():
        return
    stadt = set(pd.read_parquet(pfad, columns=["image_id"])["image_id"].to_numpy().tolist())
    eigene = set(pd.Index(metadata["image_id"]).to_numpy().tolist())
    if not eigene:
        return
    anteil = len(eigene & stadt) / len(eigene)
    if anteil >= min_overlap:
        return
    raise RuntimeError(
        f"{what} gehoert nicht zu {cfg['city']!r}.\n"
        f"  Nur {anteil:.0%} seiner {len(eigene):,} Bilder kommen in\n"
        f"  {pfad} vor ({len(stadt):,} Bilder).\n"
        "Die Metadaten eines Encoders liegen neben seinen Embeddings und "
        "wandern beim\nKopieren mit -- der Fingerabdruck allein kann das "
        "nicht bemerken.\n"
        "-> Liegt der Encoder im falschen data/<stadt>/embeddings/-Zweig? "
        "Oder ist\n   VPR_CITY / config.yaml -> city auf die falsche Stadt "
        "gesetzt?"
    )


def _sidecar(artifact_path):
    artifact_path = Path(artifact_path)
    return artifact_path.with_name(artifact_path.name + ".fingerprint.json")


def write_fingerprint(artifact_path, fingerprint, **extra):
    """Legt <artefakt>.fingerprint.json ab. Nach dem Schreiben aufrufen."""
    payload = {"hash": short_hash(fingerprint), "fingerprint": fingerprint, **extra}
    _sidecar(artifact_path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
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

    stored = json.loads(side.read_text(encoding="utf-8"))
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
    "city",
    "image_root",
    "tile_workers",
    "download_workers",
    "download_image_size",
    "max_missing_images_frac",
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

    # Abgeleitete Encoder (PCA-Varianten aus experiments/pca_reduce.py) zeigen
    # ueber source auf einen echten. Ein Tippfehler dort soll hier auffallen,
    # nicht erst beim Rechnen.
    for name in vpr["models"]:
        block = vpr.get(name)
        if isinstance(block, dict) and "sources" in block:
            # Verkettung: jede Quelle muss ein Modell sein, abgeleitete sind erlaubt.
            if not isinstance(block["sources"], list) or len(block["sources"]) < 2:
                fehler.append(f"vpr.{name}.sources braucht mindestens zwei Eintraege")
            for q in block.get("sources", []) if isinstance(block.get("sources"), list) else []:
                if q not in vpr["models"]:
                    fehler.append(f"vpr.{name}.sources: {q!r} steht nicht unter vpr.models")
            continue
        if not (isinstance(block, dict) and "source" in block):
            continue
        quelle = block["source"]
        quell_block = vpr.get(quelle)
        if quelle not in vpr["models"]:
            fehler.append(f"vpr.{name}.source={quelle!r} steht nicht unter vpr.models")
        elif isinstance(quell_block, dict) and "source" in quell_block:
            fehler.append(f"vpr.{name}.source={quelle!r} ist selbst abgeleitet")
        if "pca_dim" not in block or int(block["pca_dim"]) <= 0:
            fehler.append(f"vpr.{name}: pca_dim fehlt oder ist nicht positiv")

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

    # 08 liest hoechstens so viele Treffer, wie 06 gespeichert hat.
    loc = cfg.get("localization") or {}
    if int(loc.get("top_k", 10)) > int(ret["top_k"]):
        fehler.append(
            f"localization.top_k={loc.get('top_k')} ist groesser als retrieval.top_k={ret['top_k']}"
        )

    training = vpr.get("adapter_training") or {}
    for k in ("batch_size", "epochs"):
        if int(training.get(k, 1)) <= 0:
            fehler.append(f"vpr.adapter_training.{k} muss positiv sein")
    if not 0.0 <= float(training.get("hard_negative_probability", 0.5)) <= 1.0:
        fehler.append("vpr.adapter_training.hard_negative_probability muss in [0, 1] liegen")

    fehlend_frac = float(cfg.get("max_missing_images_frac", 0.0))
    if not 0.0 <= fehlend_frac < 1.0:
        fehler.append(
            f"max_missing_images_frac={fehlend_frac} muss in [0, 1) liegen "
            "(0 = kein fehlendes Bild erlaubt)"
        )

    max_images = vpr.get("max_images")
    if max_images is not None and int(max_images) <= 0:
        fehler.append("vpr.max_images muss null oder positiv sein")

    bezirke = cfg.get("districts") or {}
    if bezirke.get("enabled", True) and not bezirke.get("admin_levels"):
        fehler.append("districts.admin_levels fehlt")

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
