"""
Warum schadet der Adapter den VPR-Encodern? Eine Nachanalyse ohne neues
Training.

05 waehlt die beste Epoche nach val-Recall@1 -- verglichen wird dabei nur
unter trainierten Epochen: train_adapter startet mit recall = -1 und misst
zum ersten Mal nach Epoche 1 (src/adapter_training.py). Der untrainierte
Adapter, als Identitaet gestartet und damit exakt die Baseline, stand nie
zur Wahl. Dieses Skript misst ihn nach, auf demselben val-Split wie 05, und
legt vier Zahlen nebeneinander:

  val  ohne Training    hier gemessen (Identitaet)
  val  beste Epoche     aus dem Fingerabdruck der Adapter-Gewichte, den 05
                        schreibt -- und hier nachgemessen: mit den Gewichten,
                        oder, wenn die fehlen, mit den adaptierten Embeddings
                        aus 05 (<method>_linear_embeddings.npy). Die sind
                        genau die normierte Ausgabe des Adapters, also
                        dieselbe Zahl. Stimmen gespeichert und gemessen, ist
                        der val-Split derselbe wie in 05
  test ohne / mit       aus results/<stadt>/evaluation/<method>{,_linear}.json

Lesart:
  val ohne >= val mit   Training verschlechtert schon auf val. 05 konnte
                        "nicht trainieren" nur nicht waehlen.
  val ohne <  val mit,  val und test widersprechen sich: was auf den
  test faellt           train-Fahrten hilft, uebertraegt sich nicht auf
                        database/query.

Braucht die Basis-Embeddings (nur die val-Zeilen werden gelesen), also den
Rechner, der den Encoder gerechnet hat -- CLIP, MixVPR, EigenPlaces auf dem
einen, MegaLoc und AnyLoc auf dem anderen. Fehlt etwas, wird der Encoder
uebersprungen.

    python experiments/adapter_diagnose.py                    # megaloc
    python experiments/adapter_diagnose.py --method clip mixvpr eigenplaces

Ergebnis: experiments/results/<stadt>/adapter_diagnose_<method>.json
"""

import argparse
import json

import numpy as np
import pandas as pd

from _common import CFG, PATHS, RESULTS


def _args():
    ap = argparse.ArgumentParser(
        description="val-Recall@1 des untrainierten Adapters gegen die beste Epoche aus 05.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--method", nargs="+", default=["megaloc"],
                    help="ein oder mehrere Basis-Encoder ohne _linear (Standard: megaloc)")
    return ap.parse_args()


def lesart(val_ohne, val_mit, test_ohne, test_mit):
    """Welcher der zwei Faelle aus dem Docstring vorliegt -- oder keiner."""
    if any(x is None or np.isnan(x) for x in (val_ohne, val_mit, test_ohne, test_mit)):
        return "unvollstaendig"
    if test_mit >= test_ohne:
        return "adapter hilft im test"
    if val_ohne >= val_mit:
        return "schon auf val schlechter"
    return "val steigt, test faellt"


def _test_recall(method, suffix, schwelle):
    datei = PATHS.evaluation / f"{method}{suffix}.json"
    if not datei.exists():
        return None
    eintrag = json.loads(datei.read_text(encoding="utf-8"))["auswertungen"]["Alle Queries"]
    return eintrag["schwellen"][f"{schwelle:g}"]["recall"]["1"]


def diagnose(method, device):
    import torch

    from src.adapter_training import split_fit_val, val_halves, val_recall_at_1
    from src.models.adapter import LinearAdapter
    from src.run_guard import adapter_fingerprint, embedding_fingerprint, require_fingerprint

    vpr = CFG["vpr"]
    emb_dir = PATHS.embedding_dir(method)
    npy = emb_dir / f"{method}_embeddings.npy"
    meta_datei = emb_dir / f"{method}_metadata.parquet"
    gewichte = PATHS.adapter_file(method, "linear")
    if not npy.exists() or not meta_datei.exists():
        print(f"{method}: keine Basis-Embeddings unter {emb_dir} -- uebersprungen")
        return None

    meta = pd.read_parquet(meta_datei)
    basis = embedding_fingerprint(CFG, method, "none", meta)
    require_fingerprint(npy, basis, what="Basis-Embeddings")

    # Genau die Schritte aus 05, Zelle "fit / val": derselbe Seed, dieselben Haelften.
    _, val_mask = split_fit_val(meta, float(vpr["val_fraction"]), int(vpr["split_seed"]))
    val_meta = meta[val_mask].reset_index(drop=True)
    val_emb = np.ascontiguousarray(np.load(npy, mmap_mode="r")[np.flatnonzero(val_mask)],
                                   dtype=np.float32)
    db_mask, q_mask = val_halves(val_meta)
    radius = float(vpr["val_radius_m"])
    args = (val_emb[db_mask], val_emb[q_mask], val_meta[db_mask].reset_index(drop=True),
            val_meta[q_mask].reset_index(drop=True), CFG, radius, device)

    identitaet = LinearAdapter(embedding_dim=val_emb.shape[1]).to(device)
    with torch.no_grad():
        val_ohne, n_val = val_recall_at_1(identitaet, *args)

    val_mit_gespeichert = beste_epoche = val_mit_gemessen = None
    fp_datei = gewichte.with_name(gewichte.name + ".fingerprint.json")
    if fp_datei.exists():
        gespeichert = require_fingerprint(gewichte, adapter_fingerprint(CFG, method, basis),
                                          what="Adapter")
        val_mit_gespeichert = gespeichert.get("best_val_recall_at_1")
        beste_epoche = gespeichert.get("best_epoch")
    if gewichte.exists():
        trainiert = LinearAdapter(embedding_dim=val_emb.shape[1]).to(device)
        trainiert.load_state_dict(torch.load(gewichte, map_location=device))
        with torch.no_grad():
            val_mit_gemessen, _ = val_recall_at_1(trainiert, *args)

    quelle = "gewichte" if val_mit_gemessen is not None else None
    if val_mit_gemessen is None:
        # Ohne Gewichte: die adaptierten Embeddings aus 05 SIND die Ausgabe des
        # trainierten Adapters (apply_adapter, normiert). Die Identitaet darauf
        # rechnet also exakt, was der Adapter auf val lieferte. Die beste Epoche
        # steht in ihrem Fingerabdruck.
        lin_npy = PATHS.embedding_file(f"{method}_linear", method)
        lin_meta_datei = PATHS.metadata_file(f"{method}_linear", method)
        if lin_npy.exists() and lin_meta_datei.exists():
            lin_meta = pd.read_parquet(lin_meta_datei)
            fp = require_fingerprint(lin_npy, embedding_fingerprint(CFG, method, "linear", lin_meta),
                                     what="Adaptierte Embeddings")
            beste_epoche = beste_epoche if beste_epoche is not None else fp.get("best_epoch")
            # Zeilen ueber image_id zuordnen, nicht ueber die Position.
            zeile = pd.Series(np.arange(len(lin_meta)), index=lin_meta["image_id"].to_numpy())
            lin_rows = zeile.reindex(val_meta["image_id"].to_numpy()).to_numpy()
            if np.isnan(lin_rows).any():
                raise RuntimeError(f"{method}_linear: val-Bilder fehlen in den adaptierten Embeddings.")
            lin_emb = np.ascontiguousarray(
                np.load(lin_npy, mmap_mode="r")[lin_rows.astype(np.int64)], dtype=np.float32)
            lin_args = (lin_emb[db_mask], lin_emb[q_mask]) + args[2:]
            with torch.no_grad():
                val_mit_gemessen, _ = val_recall_at_1(identitaet, *lin_args)
            quelle = "adaptierte_embeddings"

    val_mit = val_mit_gemessen if val_mit_gemessen is not None else val_mit_gespeichert
    schwelle = float(vpr["uncertain_radius_m"])
    test_ohne, test_mit = _test_recall(method, "", schwelle), _test_recall(method, "_linear", schwelle)
    return {
        "method": method,
        "n_val_queries_loesbar": int(n_val),
        "val_radius_m": radius,
        "val_recall_1_ohne_training": _zahl(val_ohne),
        "val_recall_1_beste_epoche_gespeichert": _zahl(val_mit_gespeichert),
        "val_recall_1_beste_epoche_gemessen": _zahl(val_mit_gemessen),
        "val_mit_quelle": quelle,
        "beste_epoche": beste_epoche,
        "test_threshold_m": schwelle,
        "test_recall_1_ohne": test_ohne,
        "test_recall_1_mit": test_mit,
        "lesart": lesart(val_ohne, val_mit, test_ohne, test_mit),
    }


def _zahl(x):
    """None statt NaN -- NaN ist kein gueltiges JSON (tests/test_ergebnis_json.py)."""
    return None if x is None or np.isnan(x) else float(x)


def _fmt(x):
    return "     -" if x is None else f"{x:6.3f}"


def main():
    args = _args()
    from src.device import pick_device

    device = pick_device()
    zeilen = [z for z in (diagnose(m, device) for m in args.method) if z]
    if not zeilen:
        raise SystemExit("Kein Encoder auswertbar -- Basis-Embeddings liegen auf einem anderen Rechner.")

    print("\nAdapter-Diagnose  |  val = Mini-Retrieval auf den val-Fahrten aus 05, "
          f"test = 07 bei {zeilen[0]['test_threshold_m']:g} m")
    print(f"{'Encoder':<14}{'val ohne':>9}{'val mit':>9}{'Epoche':>7}{'test ohne':>10}"
          f"{'test mit':>9}   Lesart")
    print("-" * 82)
    for z in zeilen:
        val_mit = z["val_recall_1_beste_epoche_gemessen"]
        if val_mit is None:
            val_mit = z["val_recall_1_beste_epoche_gespeichert"]
        epoche = "-" if z["beste_epoche"] is None else str(z["beste_epoche"])
        print(f"{z['method']:<14}   {_fmt(z['val_recall_1_ohne_training'])}   {_fmt(val_mit)}"
              f"{epoche:>7}    {_fmt(z['test_recall_1_ohne'])}   {_fmt(z['test_recall_1_mit'])}"
              f"   {z['lesart']}")
        gesp, gem = z["val_recall_1_beste_epoche_gespeichert"], z["val_recall_1_beste_epoche_gemessen"]
        if gesp is not None and gem is not None and abs(gesp - gem) > 0.002:
            print(f"  Achtung: val mit Adapter gemessen {gem:.4f}, in 05 gespeichert {gesp:.4f} -- "
                  "der val-Split hier ist nicht derselbe wie in 05, die Zeile nicht deuten.")
        elif gesp is not None and gem is not None and gesp != gem:
            print(f"  val mit Adapter gemessen {gem:.4f}, in 05 gespeichert {gesp:.4f} -- "
                  "Rundung zwischen Geraeten, derselbe Split.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    datum = pd.Timestamp.now().strftime("%Y-%m-%d")
    for z in zeilen:
        out = RESULTS / f"adapter_diagnose_{z['method']}.json"
        out.write_text(json.dumps({"datum": datum, **z}, indent=2), encoding="utf-8")
        print(f"-> {out}")


if __name__ == "__main__":
    main()
