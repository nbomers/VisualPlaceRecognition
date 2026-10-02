"""
Liegt der Adapterschaden an Marge und Lernrate? Ein Raster ueber beide,
ohne die Pipeline anzufassen.

Die Adapter-Diagnose (adapter_diagnose.py) zeigt: bei den VPR-Encodern ist
schon auf val jede trainierte Epoche schlechter als keine. Eine Vermutung
dazu: die Marge von 0.2 auf dem Cosinus-Abstand entspricht 0.4 auf dem
quadrierten L2-Abstand (|a-b|^2 = 2(1-cos)), NetVLAD nimmt dort 0.1. Mit
Lernrate 1e-3 auf einer d x d-Matrix zieht das den Raum weit weg von dem,
was der Encoder gelernt hat.

Das Skript trainiert je Kombination einen Adapter genau wie 05 (dieselben
Bausteine aus src/adapter_training.py, derselbe fit/val-Split, derselbe
Seed), haelt ihn aber nur im Speicher. Anders als 05 steht die Identitaet
als Epoche 0 zur Wahl: gewaehlt wird nach val, und schlaegt keine Epoche die
Identitaet um min_delta, bleibt es bei ihr. Die Wahl sieht den test-Split
nicht; test wird fuer jede Kombination nur berichtet -- einmal fuer die
beste trainierte Epoche, einmal fuer die Wahl.

Zur Kontrolle rechnet das Skript die Identitaet auf test mit; sie muss die
Zahl aus 07 treffen. Die aktuelle Einstellung aus config.yaml gehoert ins
Raster, damit der Vergleich denselben Code sieht.

Nichts hier schreibt nach results/ oder data/ -- die Adapter-Zeilen der
Pipeline bleiben, wie sie sind.

    python experiments/adapter_sweep.py --method eigenplaces
    python experiments/adapter_sweep.py --method megaloc --margins 0.2 0.05 --lrs 1e-3 1e-4

Ergebnis: experiments/results/<stadt>/adapter_sweep_<method>.json
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from _common import CFG, PATHS, RESULTS


def _args():
    ap = argparse.ArgumentParser(
        description="Adapter mit mehreren Margen und Lernraten trainieren, nach val waehlen, "
                    "test berichten -- ohne Pipeline-Artefakte zu schreiben.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--method", default="eigenplaces", help="Basis-Encoder ohne _linear")
    ap.add_argument("--margins", type=float, nargs="+", default=[0.2, 0.1, 0.05])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 1e-4])
    return ap.parse_args()


def waehle(val_identitaet, val_beste, min_delta):
    """Identitaet, solange keine trainierte Epoche sie um min_delta schlaegt."""
    return "adapter" if val_beste > val_identitaet + min_delta else "identitaet"


def _test_recall(adapter, emb, db_rows, q_rows, query_meta, db_meta, device, k_max, block=2048):
    """R@k auf test (Alle Queries), exakte Suche ueber das Skalarprodukt."""
    import torch

    from src.evaluation import evaluate_retrieval

    adapter.eval()
    with torch.no_grad():
        db = torch.cat([adapter(torch.from_numpy(np.asarray(emb[db_rows[i:i + block]], dtype=np.float32))
                                .to(device))
                        for i in range(0, len(db_rows), block)])
        treffer = []
        for i in range(0, len(q_rows), block):
            q = adapter(torch.from_numpy(np.asarray(emb[q_rows[i:i + block]], dtype=np.float32))
                        .to(device))
            treffer.append(torch.topk(q @ db.T, k_max, dim=1).indices.cpu().numpy())
    adapter.train()
    del db
    eintrag = evaluate_retrieval(np.concatenate(treffer), query_meta, db_meta, CFG,
                                 verbose=False)
    schwelle = str(int(float(CFG["vpr"]["uncertain_radius_m"])))
    return eintrag["schwellen"][schwelle]["recall"]


def main():
    args = _args()
    import torch

    from src.adapter_training import (
        TripletDataset,
        make_loader,
        split_fit_val,
        train_adapter,
        val_halves,
        val_recall_at_1,
    )
    from src.device import pick_device
    from src.models.adapter import LinearAdapter
    from src.pairs import train_positive_pairs
    from src.run_guard import embedding_fingerprint, require_fingerprint

    vpr, training = CFG["vpr"], CFG["vpr"]["adapter_training"]
    seed, device, method = int(vpr["split_seed"]), pick_device(), args.method
    emb_dir = PATHS.embedding_dir(method)
    npy = emb_dir / f"{method}_embeddings.npy"
    meta = pd.read_parquet(emb_dir / f"{method}_metadata.parquet")
    require_fingerprint(npy, embedding_fingerprint(CFG, method, "none", meta), what="Basis-Embeddings")
    emb = np.load(npy, mmap_mode="r")

    # Wie 05, Zelle "fit / val".
    fit_mask, val_mask = split_fit_val(meta, float(vpr["val_fraction"]), seed)
    fit_meta = meta[fit_mask].reset_index(drop=True)
    val_meta = meta[val_mask].reset_index(drop=True)
    fit_emb = np.ascontiguousarray(emb[np.flatnonzero(fit_mask)], dtype=np.float32)
    val_emb = np.ascontiguousarray(emb[np.flatnonzero(val_mask)], dtype=np.float32)
    db_mask, q_mask = val_halves(val_meta)
    val_args = (val_emb[db_mask], val_emb[q_mask], val_meta[db_mask].reset_index(drop=True),
                val_meta[q_mask].reset_index(drop=True), CFG, float(vpr["val_radius_m"]), device)

    paare = train_positive_pairs(meta, float(vpr["positive_radius_m"]), float(vpr["max_heading_diff_deg"]))
    fit_index = pd.Series(np.arange(len(fit_meta)), index=fit_meta["image_id"].to_numpy())
    a = fit_index.reindex(paare["anchor_image_id"].to_numpy()).to_numpy()
    p = fit_index.reindex(paare["positive_image_id"].to_numpy()).to_numpy()
    ok = ~np.isnan(a) & ~np.isnan(p)
    positive_pairs = list(zip(a[ok].astype(int), p[ok].astype(int), paare["distance_m"].to_numpy()[ok]))
    dataset = TripletDataset(
        fit_emb, positive_pairs, fit_meta,
        positive_radius_m=float(vpr["positive_radius_m"]),
        uncertain_radius_m=float(vpr["uncertain_radius_m"]),
        hard_negative_min_m=float(vpr["hard_negative_min_m"]),
        hard_negative_max_m=float(vpr["hard_negative_max_m"]),
        hard_negative_probability=float(training["hard_negative_probability"]),
    )

    # test wie in 06/07: query gegen database, Zeilen in der Reihenfolge der Metadaten.
    q_rows = np.flatnonzero((meta["split"] == "query").to_numpy())
    db_rows = np.flatnonzero((meta["split"] == "database").to_numpy())
    query_meta = meta.iloc[q_rows].reset_index(drop=True)
    db_meta = meta.iloc[db_rows].reset_index(drop=True)
    k_max = max(int(k) for k in CFG["retrieval"]["k_values"])
    min_delta = float(training.get("min_delta", 0.0))

    identitaet = LinearAdapter(embedding_dim=emb.shape[1]).to(device)
    with torch.no_grad():
        val_id, n_val = val_recall_at_1(identitaet, *val_args)
    test_id = _test_recall(identitaet, emb, db_rows, q_rows, query_meta, db_meta, device, k_max)
    soll = PATHS.evaluation / f"{method}.json"
    kontrolle = None
    if soll.exists():
        schwelle = str(int(float(vpr["uncertain_radius_m"])))
        kontrolle = json.loads(soll.read_text(encoding="utf-8"))["auswertungen"]["Alle Queries"][
            "schwellen"][schwelle]["recall"]["1"]
    print(f"{method}: {len(positive_pairs):,} Paare, {n_val:,} val-Queries, Geraet {device}")
    print(f"Identitaet: val {val_id:.3f}   test R@1 {test_id['1']:.3f}"
          + (f"   (07: {kontrolle:.3f})" if kontrolle is not None else ""))
    if kontrolle is not None and abs(test_id["1"] - kontrolle) > 0.002:
        print("  Achtung: test weicht von 07 ab -- andere Zeilen oder andere Suche, nicht deuten.")

    zeilen = []
    for margin in args.margins:
        for lr in args.lrs:
            # Je Kombination derselbe Zufall wie in 05: Negative (numpy) und
            # Mischen (torch) ziehen dann in derselben Reihenfolge.
            np.random.seed(seed)
            torch.manual_seed(seed)
            loader = make_loader(dataset, int(training["batch_size"]), training.get("max_pairs_per_epoch"))
            verlauf = []

            def val_fn(model, verlauf=verlauf):
                recall, n = val_recall_at_1(model, *val_args)
                verlauf.append(float(recall))
                return recall, n

            t0 = time.perf_counter()
            adapter = LinearAdapter(embedding_dim=emb.shape[1]).to(device)
            state, epoche, val_best, _ = train_adapter(
                adapter, loader, val_fn, epochs=int(training["epochs"]), learning_rate=lr,
                margin=margin, device=device, patience=training.get("patience"),
                min_delta=min_delta, log=lambda zeile: print("   " + zeile),
            )
            adapter.load_state_dict(state)
            test_best = _test_recall(adapter, emb, db_rows, q_rows, query_meta, db_meta, device, k_max)
            wahl = waehle(val_id, val_best, min_delta)
            z = {
                "margin": margin, "learning_rate": lr,
                "val_verlauf": verlauf, "beste_epoche": int(epoche), "val_beste_epoche": float(val_best),
                "test_beste_epoche": test_best, "wahl": wahl,
                "test_wahl_r1": test_best["1"] if wahl == "adapter" else test_id["1"],
                "minuten": round((time.perf_counter() - t0) / 60, 1),
            }
            zeilen.append(z)
            print(f"margin {margin:<5g} lr {lr:<7g} beste Epoche {epoche:>2}  val {val_best:.3f} "
                  f"({val_best - val_id:+.3f})  test {test_best['1']:.3f} ({test_best['1'] - test_id['1']:+.3f})"
                  f"  Wahl: {wahl:<10} {z['minuten']:.0f} min")
            del adapter, state

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"adapter_sweep_{method}.json"
    out.write_text(json.dumps({
        "datum": pd.Timestamp.now().strftime("%Y-%m-%d"), "method": method,
        "n_positive_pairs": len(positive_pairs), "n_val_queries_loesbar": int(n_val),
        "val_identitaet": float(val_id), "test_identitaet": test_id, "test_07": kontrolle,
        "min_delta": min_delta, "epochs": int(training["epochs"]), "patience": training.get("patience"),
        "kombinationen": zeilen,
    }, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
