"""
Laufzeit und Speicher je Encoder: Encodierdurchsatz, Suchzeit, Indexgroesse.

Das README nennt eigenplaces_pcaw512 das effizienteste Modell. Hier stehen
die Zahlen dazu -- drei Messungen, die verschiedene Dinge kosten:

  Encodieren   Bilder/s des Rueckgrats, 200 feste Bilder aus dem Query-Split
               (Seed), inklusive Laden und Dekodieren wie in 04. Ein
               Aufwaermlauf vorweg, der nicht zaehlt. Nur fuer die fuenf
               Basis-Encoder -- eine PCA-Variante encodiert mit demselben
               Netz und projiziert danach, das kostet nichts Messbares.
  Suche        ms je Anfrage im FAISS-Flat-Index ueber die Datenbankzeilen,
               1.000 Anfragen in Bloecken von 256, Median aus fuenf Runden,
               fuer alle 18 Encoder mit .npy -- auch die abgeleiteten.
               Waechst linear mit der Breite.
  Index        n_database x dim x 4 Byte, dazu die tatsaechliche Groesse der
               .npy (die haelt alle 332k Zeilen, nicht nur die Datenbank).

Die JSON ist mergefaehig: je Encoder ein Eintrag mit Hostname und Geraet,
Eintraege anderer Rechner bleiben stehen. AnyLoc und MegaLoc gehoeren auf
den GPU-Rechner:

    python experiments/timing.py                            # alles, was hier baubar ist
    python experiments/timing.py --methods anyloc,megaloc   # auf dem GPU-Rechner
    python experiments/timing.py --skip-encode              # nur Suche und Index
    python experiments/timing.py --skip-search              # nur Encodieren

Ergebnis: experiments/results/<stadt>/timing.json
"""

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import CFG, PATHS, RESULTS, ROOT
from src.device import pick_device

OUT = RESULTS / "timing.json"
EMB_DIR = PATHS.embeddings
BASIS = ("clip", "anyloc", "mixvpr", "eigenplaces", "megaloc")


def _args():
    ap = argparse.ArgumentParser(
        description="Encodierdurchsatz, Suchzeit und Indexgroesse je Encoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--methods", default=",".join(BASIS),
                    help="Basis-Encoder fuers Encodieren, Kommaliste")
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--n-queries", type=int, default=1000)
    ap.add_argument("--block", type=int, default=256, help="Anfragen je FAISS-Aufruf")
    ap.add_argument("--repeats", type=int, default=5,
                    help="Wiederholungen der Suche, berichtet wird der Median")
    ap.add_argument("--skip-encode", action="store_true")
    ap.add_argument("--skip-search", action="store_true")
    return ap.parse_args()


def _stamp():
    return {"datum": pd.Timestamp.now().strftime("%Y-%m-%d"), "host": platform.node()}


# ----------------------------------------------------------------------
# Encodieren
# ----------------------------------------------------------------------


def sample_images(n_images, n_warmup):
    """Feste Bilder aus dem Query-Split -- auf jedem Rechner dieselben."""
    meta = pd.read_parquet(PATHS.processed / "metadata.parquet")
    ids = np.sort(meta.loc[meta["split"] == "query", "image_id"].to_numpy())
    rng = np.random.default_rng(int(CFG["vpr"]["split_seed"]))
    gewaehlt = rng.choice(ids, size=n_images + n_warmup, replace=False)
    pfade = [PATHS.image_file(i) for i in gewaehlt]
    return pfade[:n_images], pfade[n_images:]


def _buildable(method):
    """Grund, warum ein Encoder hier nicht gebaut werden soll -- oder None."""
    if method == "anyloc":
        # torch.hub wuerde sonst 4,5 GB DINOv2-Gewichte nachladen.
        gewichte = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        if not any(gewichte.glob("dinov2_vitg14*")):
            return "DINOv2-Gewichte nicht im torch.hub-Cache -- auf dem GPU-Rechner messen"
    return None


def time_encoder(method, device, bilder, warmup):
    """Bilder/s eines Encoders, Aufwaermlauf ausgeschlossen."""
    import torch
    from src.models.factory import build_embedder

    embedder = build_embedder(method, CFG, device, ROOT)
    batch_size = (CFG["vpr"].get(method) or {}).get("batch_size", CFG["vpr"]["embed_batch_size"])
    eintrag = {
        **_stamp(),
        "device": device,
        "batch_size": int(batch_size),
        "n_images": len(bilder),
        "image_size": int(embedder.image_size),
    }
    # AnyLoc: die PCA-Projektion ist ein Matrixprodukt hinter ViT-G und VLAD,
    # fuer den Durchsatz ohne Belang. Ohne angepasste PCA wird roh gemessen.
    if method == "anyloc" and embedder.pca_dim is not None:
        embedder.pca_dim = None
        embedder.embedding_dim = embedder.vlad_dim
        eintrag["hinweis"] = "ohne PCA-Projektion gemessen"
    eintrag["dim"] = int(embedder.embedding_dim)

    embedder.embed_images(warmup, batch_size=batch_size)
    t0 = time.perf_counter()
    out = embedder.embed_images(bilder, batch_size=batch_size)
    sekunden = time.perf_counter() - t0
    assert len(out) == len(bilder)
    eintrag["sekunden"] = float(sekunden)
    eintrag["bilder_pro_s"] = float(len(bilder) / sekunden)

    embedder.close()
    del embedder
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    return eintrag


def encode_all(args):
    ergebnis = {}
    try:
        bilder, warmup = sample_images(args.n_images, 32)
        fehlend = [p for p in bilder + warmup if not p.exists()]
    except PermissionError as e:
        print(f"Encodieren uebersprungen: Bildordner nicht lesbar ({e})")
        return ergebnis
    if fehlend:
        print(f"Encodieren uebersprungen: {len(fehlend)} von {len(bilder) + len(warmup)} "
              f"Bildern fehlen unter {fehlend[0].parent}")
        return ergebnis

    device = pick_device()
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        if method not in BASIS:
            print(f"{method}: kein Basis-Encoder, uebersprungen")
            continue
        grund = _buildable(method)
        if grund:
            print(f"{method}: {grund}")
            continue
        try:
            eintrag = time_encoder(method, device, bilder, warmup)
        except Exception as e:  # ein fehlendes Fremd-Repo soll den Rest nicht stoppen
            print(f"{method}: nicht baubar ({type(e).__name__}: {str(e).splitlines()[0]})")
            continue
        ergebnis[method] = eintrag
        print(f"{method}: {eintrag['bilder_pro_s']:.1f} Bilder/s auf {device} "
              f"(Batch {eintrag['batch_size']}, {eintrag['n_images']} Bilder)")
    return ergebnis


# ----------------------------------------------------------------------
# Suche und Index
# ----------------------------------------------------------------------


def encoders_with_npy():
    """Alle Basis- und abgeleiteten Encoder, deren Embeddings vorliegen."""
    namen = []
    for method in CFG["vpr"]["models"]:
        if (EMB_DIR / method / f"{method}_embeddings.npy").exists():
            namen.append(method)
    return namen


def time_search(method, n_queries, block, top_k, repeats, rng):
    import faiss

    npy = EMB_DIR / method / f"{method}_embeddings.npy"
    meta = pd.read_parquet(EMB_DIR / method / f"{method}_metadata.parquet")
    emb = np.load(npy, mmap_mode="r")
    assert len(emb) == len(meta)
    split = meta["split"].to_numpy()
    db_rows = np.flatnonzero(split == "database")
    q_rows = np.sort(rng.choice(np.flatnonzero(split == "query"), size=n_queries, replace=False))
    dim = int(emb.shape[1])

    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(dim)
    for start in range(0, len(db_rows), 65536):
        index.add(np.ascontiguousarray(emb[db_rows[start:start + 65536]], dtype=np.float32))
    aufbau = time.perf_counter() - t0

    anfragen = np.ascontiguousarray(emb[q_rows], dtype=np.float32)
    index.search(anfragen[:block], top_k)          # Aufwaermen
    # Eine Runde dauert bei 512 Dimensionen unter einer Sekunde und streut
    # um +-20 % -- der Median ueber mehrere Runden ist belastbar.
    zeiten = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for start in range(0, n_queries, block):
            index.search(anfragen[start:start + block], top_k)
        zeiten.append(time.perf_counter() - t0)
    suche = float(np.median(zeiten))

    return {
        **_stamp(),
        "dim": dim,
        "n_database": int(len(db_rows)),
        "top_k": int(top_k),
        "n_queries": int(n_queries),
        "block": int(block),
        "repeats": int(repeats),
        "faiss_threads": int(faiss.omp_get_max_threads()),
        "ms_pro_anfrage": float(suche / n_queries * 1000),
        "index_aufbau_s": float(aufbau),
        "index_mb": float(len(db_rows) * dim * 4 / 2**20),
        "npy_mb": float(npy.stat().st_size / 2**20),
        "npy_rows": int(len(emb)),
    }


def search_all(args):
    import faiss

    ergebnis = {}
    top_k = int(CFG["retrieval"]["top_k"])
    # FAISS startet seinen Threadpool beim ersten Aufruf -- das soll nicht
    # dem ersten Encoder angelastet werden.
    probe = np.random.default_rng(0).standard_normal((4096, 512), dtype=np.float32)
    index = faiss.IndexFlatIP(512)
    index.add(probe)
    index.search(probe[:args.block], top_k)
    for method in encoders_with_npy():
        rng = np.random.default_rng(int(CFG["vpr"]["split_seed"]))
        eintrag = time_search(method, args.n_queries, args.block, top_k, args.repeats, rng)
        ergebnis[method] = eintrag
        print(f"{method:<28} {eintrag['dim']:>5}d  {eintrag['ms_pro_anfrage']:>7.2f} ms/Anfrage  "
              f"Index {eintrag['index_mb']:>7.0f} MB  .npy {eintrag['npy_mb']:>7.0f} MB")
    return ergebnis


# ----------------------------------------------------------------------
# Zusammenfuehren und Tabelle
# ----------------------------------------------------------------------


def merge(neu_encode, neu_search):
    alt = json.loads(OUT.read_text()) if OUT.exists() else {}
    alt.setdefault("encodieren", {}).update(neu_encode)
    alt.setdefault("suche", {}).update(neu_search)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(alt, indent=2))
    return alt


def print_table(daten):
    enc, such = daten.get("encodieren", {}), daten.get("suche", {})
    namen = sorted(set(enc) | set(such), key=lambda n: (such.get(n, {}).get("dim", 0), n))
    breite = max(len(n) for n in namen) + 2
    kopf = (f"{'Encoder':<{breite}}{'Dim':>6}  {'Bilder/s':>9}  {'Geraet':<14}"
            f"{'ms/Anfrage':>11}  {'Index MB':>9}  {'.npy MB':>8}")
    print()
    print(kopf)
    print("-" * len(kopf))
    for n in namen:
        e, s = enc.get(n), such.get(n)
        dim = (s or e or {}).get("dim", "")
        bilder = f"{e['bilder_pro_s']:>9.1f}" if e else f"{'-':>9}"
        geraet = f"{e['device']} ({e['host'][:8]})" if e else ""
        ms = f"{s['ms_pro_anfrage']:>11.2f}" if s else f"{'-':>11}"
        idx = f"{s['index_mb']:>9.0f}" if s else f"{'-':>9}"
        npy = f"{s['npy_mb']:>8.0f}" if s else f"{'-':>8}"
        print(f"{n:<{breite}}{dim:>6}  {bilder}  {geraet:<14}{ms}  {idx}  {npy}")
    hinweise = {n: e["hinweis"] for n, e in enc.items() if "hinweis" in e}
    for n, h in hinweise.items():
        print(f"  {n}: {h}")


def main():
    args = _args()
    neu_encode = {} if args.skip_encode else encode_all(args)
    neu_search = {} if args.skip_search else search_all(args)
    daten = merge(neu_encode, neu_search)
    print_table(daten)
    print(f"\ngeschrieben: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
