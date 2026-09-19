"""
Adaptertraining -- alles, was 05 braucht, als Funktionen: fit/val-Split
auf Sequenz-Ebene, das Triplet-Dataset mit Hard Negatives, die val-Metrik
ueber src/evaluation, die Trainingsschleife mit Early Stopping, und das
blockweise Anwenden auf die Embeddings.

Bandaufteilung um jeden Anchor (Radien aus config.yaml -> vpr):
    <= positive_radius_m                        Positive
    positive_radius_m .. uncertain_radius_m     ausgeschlossen
    hard_negative_min_m .. hard_negative_max_m  Hard Negative
    > uncertain_radius_m                        Easy Negative

database und query werden hier nie angefasst: Anchor, Positive und
Negative kommen aus dem fit-Teil von train, die val-Metrik aus dem
val-Teil.

ZUM SEED: split_fit_val bekommt ihn als Argument, das Negative-Sampling in
TripletDataset.sample_negative zieht dagegen aus dem GLOBALEN numpy-RNG,
und make_loader mischt ueber den globalen torch-RNG. Reproduzierbar ist das
Training deshalb nur, wenn der Aufrufer vorher beide setzt -- 05 tut das in
seiner ersten Zelle:

    np.random.seed(SEED)
    torch.manual_seed(SEED)

Das bleibt bewusst so: auf einen eigenen Generator umzustellen wuerde die
Ziehungsreihenfolge aendern, und damit waeren die Adapter-Zahlen unter
results/ nicht mehr die, die dieser Code erzeugt. Wer das Modul ausserhalb
von 05 benutzt, setzt die beiden Zeilen selbst.
"""

import random
import time

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from .evaluation import evaluate_retrieval
from .geo import to_metric_xy


def split_fit_val(metadata, val_fraction, seed):
    """Masken fuer fit und val -- Sequenzen als Einheit, wie beim Hauptsplit."""
    train = (metadata["split"] == "train").to_numpy()
    if not train.any():
        raise RuntimeError("Keine train-Embeddings -- 04 muss mit allen Splits gelaufen sein.")
    seq = metadata["sequence_id"].astype(str)
    train_seqs = sorted(seq[train].unique())
    random.Random(seed).shuffle(train_seqs)
    n_val = max(1, int(val_fraction * len(train_seqs)))
    val_seqs = set(train_seqs[:n_val])
    val = train & seq.isin(val_seqs).to_numpy()
    return train & ~val, val


def val_halves(val_metadata):
    """Die val-Sequenzen in Mini-Database und Mini-Query teilen (Masken)."""
    seqs = sorted(val_metadata["sequence_id"].astype(str).unique())
    half = max(1, len(seqs) // 2)
    db_seqs, q_seqs = set(seqs[:half]), set(seqs[half:]) or set(seqs[:half])
    seq = val_metadata["sequence_id"].astype(str)
    return seq.isin(db_seqs).to_numpy(), seq.isin(q_seqs).to_numpy()


class TripletDataset(Dataset):
    """Anchor, Positive, Negative -- alle aus dem fit-Teil von train."""

    def __init__(self, fit_embeddings, positive_pairs, fit_metadata, positive_radius_m,
                 uncertain_radius_m, hard_negative_min_m, hard_negative_max_m,
                 hard_negative_probability=0.75):
        if not (positive_radius_m <= uncertain_radius_m <= hard_negative_min_m):
            raise ValueError("Es muss gelten: positive <= uncertain <= hard_negative_min.")
        self.fit_embeddings = fit_embeddings
        self.positive_pairs = positive_pairs
        self.hard_negative_probability = hard_negative_probability

        self.positive_by_anchor = {}
        for a, p, _ in positive_pairs:
            self.positive_by_anchor.setdefault(a, set()).add(p)

        # Nachbarschaften einmal vorberechnen statt bei jedem __getitem__.
        xy, _ = to_metric_xy(fit_metadata["lat"].to_numpy(), fit_metadata["lon"].to_numpy())
        self.n_fit = len(xy)
        tree = cKDTree(xy)
        anchors = sorted(self.positive_by_anchor)
        pts = xy[anchors]
        near_uncertain = tree.query_ball_point(pts, r=uncertain_radius_m)
        near_hard_max = tree.query_ball_point(pts, r=hard_negative_max_m)
        near_hard_min = tree.query_ball_point(pts, r=hard_negative_min_m)
        self.hard_by_anchor, self.blocked_by_anchor = {}, {}
        for slot, ai in enumerate(anchors):
            blocked = set(near_uncertain[slot]) | self.positive_by_anchor[ai] | {ai}
            hard = set(near_hard_max[slot]) - set(near_hard_min[slot])
            self.hard_by_anchor[ai] = np.fromiter(hard, dtype=np.int64, count=len(hard))
            self.blocked_by_anchor[ai] = blocked
        self.mean_hard = float(np.mean([len(v) for v in self.hard_by_anchor.values()]))

    def __len__(self):
        return len(self.positive_pairs)

    def sample_negative(self, anchor_index):
        hard = self.hard_by_anchor.get(anchor_index)
        blocked = self.blocked_by_anchor.get(anchor_index, frozenset())
        if hard is not None and len(hard) and np.random.random() < self.hard_negative_probability:
            return int(hard[np.random.randint(len(hard))])
        # Easy Negative per Rejection Sampling: gesperrt sind nur ein paar
        # Dutzend von n_fit Kandidaten, der erste Wurf sitzt fast immer.
        for _ in range(32):
            candidate = int(np.random.randint(self.n_fit))
            if candidate not in blocked:
                return candidate
        if hard is not None and len(hard):
            return int(hard[np.random.randint(len(hard))])
        raise RuntimeError(f"Keine negativen Kandidaten fuer Anchor {anchor_index}.")

    def __getitem__(self, index):
        a, p, _ = self.positive_pairs[index]
        n = self.sample_negative(a)
        return tuple(torch.from_numpy(self.fit_embeddings[i]).float() for i in (a, p, n))


def make_loader(dataset, batch_size, max_pairs_per_epoch=None):
    if max_pairs_per_epoch and len(dataset) > max_pairs_per_epoch:
        sampler = RandomSampler(dataset, replacement=False, num_samples=max_pairs_per_epoch)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def cosine_triplet_loss(margin):
    return nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1 - F.cosine_similarity(x, y, dim=-1), margin=margin,
    )


@torch.inference_mode()
def val_recall_at_1(adapter, val_db_emb, val_q_emb, val_db_meta, val_q_meta, cfg, radius_m, device):
    """Recall@1 des Mini-Retrievals auf den val-Sequenzen -- dieselbe Rechnung wie 07."""
    adapter.eval()
    db = adapter(torch.from_numpy(np.asarray(val_db_emb)).float().to(device))
    qr = adapter(torch.from_numpy(np.asarray(val_q_emb)).float().to(device))
    beste = torch.argmax(qr @ db.T, dim=1).cpu().numpy()[:, None]
    adapter.train()
    val_cfg = {**cfg, "retrieval": {**cfg["retrieval"], "thresholds": [radius_m], "k_values": [1]}}
    eintrag = evaluate_retrieval(beste, val_q_meta, val_db_meta, val_cfg,
                                 verbose=False)["schwellen"][str(radius_m)]
    recall = eintrag["recall"]["1"]
    return (recall if recall is not None else float("nan")), eintrag["loesbar"]


def train_adapter(adapter, loader, val_fn, epochs, learning_rate, margin, device,
                  patience=None, min_delta=0.0, log=print):
    """
    Trainiert mit Early Stopping auf val-Recall@1 und gibt das beste Modell
    zurueck: (state_dict, beste_epoche, bester_recall, n_val).
    """
    from tqdm.auto import tqdm

    loss_fn = cosine_triplet_loss(margin)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate)
    adapter.train()
    t0 = time.perf_counter()
    best = {"state": None, "epoch": -1, "recall": -1.0, "n_val": 0}
    stale = 0
    for epoch in range(epochs):
        total = 0.0
        for anchor, positive, negative in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            anchor, positive, negative = (x.to(device) for x in (anchor, positive, negative))
            optimizer.zero_grad()
            loss = loss_fn(adapter(anchor), adapter(positive), adapter(negative))
            loss.backward()
            optimizer.step()
            total += loss.item()
        recall, n_val = val_fn(adapter)
        marker = ""
        if n_val > 0 and recall > best["recall"] + min_delta:
            best = {"state": {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()},
                    "epoch": epoch + 1, "recall": recall, "n_val": n_val}
            stale = 0
            marker = "  <- bestes Modell"
        else:
            # Zuwachs unter min_delta zaehlt nicht, sonst haelt Rauschen die
            # Schleife beliebig lange am Leben.
            stale += 1
            if patience:
                marker = f"  ({stale}/{patience} ohne Verbesserung)"
        verstrichen = time.perf_counter() - t0
        log(f"Epoch {epoch + 1:2d}: loss = {total / max(len(loader), 1):.4f}   "
            f"val Recall@1 = {recall:.4f}   {verstrichen / 60:5.1f} min{marker}")
        if patience and stale >= patience:
            log(f"Early Stopping nach Epoche {epoch + 1}.")
            break
    if best["state"] is None:
        raise RuntimeError("val-Recall war in keiner Epoche auswertbar -- val-Split pruefen.")
    return best["state"], best["epoch"], best["recall"], best["n_val"]


@torch.inference_mode()
def apply_adapter(adapter, embeddings, out_path, device, block=8192):
    """
    Adapter blockweise auf alle Embeddings anwenden und als memmap schreiben --
    bei MegaLoc waeren 332.867 x 8448 float32 nochmal 11 GB im Speicher.
    Gibt den mittleren Cosinus zur Baseline zurueck.
    """
    from tqdm.auto import tqdm

    adapter.eval()
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=embeddings.shape)
    kosinus = 0.0
    for start in tqdm(range(0, len(embeddings), block), desc="Adapter anwenden", leave=False):
        roh = np.ascontiguousarray(embeddings[start:start + block])
        ergebnis = adapter(torch.from_numpy(roh).float().to(device)).cpu().numpy()
        ergebnis /= np.linalg.norm(ergebnis, axis=1, keepdims=True)
        assert np.isfinite(ergebnis).all()
        out[start:start + block] = ergebnis
        kosinus += float((ergebnis * roh).sum())
    out.flush()
    return kosinus / len(embeddings)
