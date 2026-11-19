"""Segmentation scores for small puncta.

- instance F1: predicted and annotated objects matched one-to-one (Hungarian) at an IoU threshold, from 0.1 to 0.9
- detection F1: object centroids matched one-to-one within 0.5 um, in physical units on the anisotropic grid
- object confusion: from overlaps of at least 10% of the smaller object, every annotated object is correct, missed,
  split or fused, and every predicted object without overlap is spurious
- Dice of the foreground, voxel by voxel
- counts are pooled over the volumes of a set before F1 is computed; intervals resample units (crops, or 64 x 64
  tiles of a frame, an object belonging to the tile of its centroid)
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

IOUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TILE = 64


def instance_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """IoU of every annotated (rows) and predicted (columns) object."""
    a, b = gt.ravel().astype(np.int64), pred.ravel().astype(np.int64)
    fg = (a > 0) & (b > 0)
    ng, npred = int(a.max()), int(b.max())
    iou = np.zeros((ng + 1, npred + 1))
    if fg.any():
        k, c = np.unique(a[fg] * (npred + 1) + b[fg], return_counts=True)
        ga, pb = k // (npred + 1), k % (npred + 1)
        gs, ps = np.bincount(a, minlength=ng + 1), np.bincount(b, minlength=npred + 1)
        iou[ga, pb] = c / (gs[ga] + ps[pb] - c)
    g_ids, p_ids = np.unique(a[a > 0]), np.unique(b[b > 0])
    return iou[np.ix_(g_ids, p_ids)] if len(g_ids) and len(p_ids) else np.zeros((len(g_ids), len(p_ids)))


def centroids(labels: np.ndarray, spacing) -> tuple[np.ndarray, np.ndarray]:
    ids = np.unique(labels[labels > 0])
    c = np.array(ndi.center_of_mass(labels > 0, labels, ids)).reshape(-1, 3) * np.asarray(spacing, float)
    return ids, c


def match_counts(pred: np.ndarray, gt: np.ndarray, spacing, r_um: float = 0.5, min_frac: float = 0.1) -> dict:
    """Counts of one volume: matches at every IoU threshold, detections and the object confusion."""
    m = instance_iou(pred, gt)
    out = {"n_gt": m.shape[0], "n_pred": m.shape[1]}
    r, c = linear_sum_assignment(-m) if m.size else ([], [])
    for t in IOUS:
        out[f"tp@{t}"] = int(sum(m[i, j] >= t for i, j in zip(r, c)))
    _, gc = centroids(gt, spacing)
    _, pc = centroids(pred, spacing)
    out["det_tp"] = 0
    if len(gc) and len(pc):
        d = np.linalg.norm(gc[:, None] - pc[None], axis=-1)
        r, c = linear_sum_assignment(d)
        out["det_tp"] = int((d[r, c] <= r_um).sum())
    out.update(confusion(pred, gt, min_frac))
    return out


def confusion(pred: np.ndarray, gt: np.ndarray, min_frac: float = 0.1) -> dict:
    g_ids, p_ids = np.unique(gt[gt > 0]), np.unique(pred[pred > 0])
    a, b = gt.ravel().astype(np.int64), pred.ravel().astype(np.int64)
    fg = (a > 0) & (b > 0)
    nb = int(b.max()) + 1
    k, n = np.unique(a[fg] * nb + b[fg], return_counts=True)
    ga, pb = k // nb, k % nb
    gs, ps = np.bincount(a), np.bincount(b)
    keep = n >= min_frac * np.minimum(gs[ga], ps[pb])
    ga, pb = ga[keep], pb[keep]
    g_deg, p_deg = dict.fromkeys(g_ids.tolist(), 0), dict.fromkeys(p_ids.tolist(), 0)
    for g, p in zip(ga, pb):
        g_deg[g] += 1
        p_deg[p] += 1
    fused_p = {p for p, d in p_deg.items() if d >= 2}
    return {"correct": sum(1 for g, p in zip(ga, pb) if g_deg[g] == 1 and p_deg[p] == 1),
            "missed": sum(d == 0 for d in g_deg.values()), "split": sum(d >= 2 for d in g_deg.values()),
            "fused": len({g for g, p in zip(ga, pb) if p in fused_p}),
            "spurious": sum(d == 0 for d in p_deg.values())}


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred > 0, gt > 0
    return float(2 * np.logical_and(p, g).sum() / max(1, p.sum() + g.sum()))


def unit_counts(pred: np.ndarray, gt: np.ndarray, spacing, unit: str, tile: int | None = TILE) -> dict:
    """Matches at IoU 0.5 and 0.7 and detections per unit: the whole volume (`tile=None`) or in-plane tiles."""
    g_ids, gc = centroids(gt, spacing)
    p_ids, pc = centroids(pred, spacing)
    if tile is None:
        gu, pu = np.full(len(g_ids), unit, object), np.full(len(p_ids), unit, object)
    else:
        nx = int(np.ceil(gt.shape[2] / tile))
        where = lambda c: np.array([f"{unit}:{int(y / spacing[1]) // tile * nx + int(x / spacing[2]) // tile}"
                                    for _, y, x in c], object)
        gu, pu = where(gc), where(pc)
    out = {u: {"n_gt": int((gu == u).sum()), "n_pred": int((pu == u).sum()), "tp@0.5": 0, "tp@0.7": 0, "det_tp": 0}
           for u in set(gu) | set(pu)}
    m = instance_iou(pred, gt)
    if m.size:
        r, c = linear_sum_assignment(-m)
        for t in (0.5, 0.7):
            for i in r[m[r, c] >= t]:
                out[gu[i]][f"tp@{t}"] += 1
    if len(gc) and len(pc):
        d = np.linalg.norm(gc[:, None] - pc[None], axis=-1)
        r, c = linear_sum_assignment(d)
        for i in r[d[r, c] <= 0.5]:
            out[gu[i]]["det_tp"] += 1
    return out


def summarize(rows: list[dict], dices: list[float]) -> dict:
    s = {k: sum(r[k] for r in rows) for k in rows[0]}
    n = max(1, s["n_gt"] + s["n_pred"])
    out = {"n_gt": s["n_gt"], "n_pred": s["n_pred"], "f1": {t: 2 * s[f"tp@{t}"] / n for t in IOUS},
           "precision@0.5": s["tp@0.5"] / max(1, s["n_pred"]), "recall@0.5": s["tp@0.5"] / max(1, s["n_gt"]),
           "det_f1": 2 * s["det_tp"] / n, "dice": float(np.mean(dices))}
    for k in ("correct", "missed", "split", "fused"):
        out[k] = s[k] / max(1, s["n_gt"])
    out["spurious"] = s["spurious"] / max(1, s["n_pred"])
    return out


def bootstrap_ci(units: dict, metric: str = "tp@0.5", n: int = 2000, seed: int = 0, alpha: float = 0.05) -> list[float]:
    """Percentile interval of F1 (`tp@0.5`, `tp@0.7`) or detection F1 (`det_tp`) over resampled units."""
    keys = sorted(units)
    tp = np.array([units[k][metric] for k in keys], float)
    den = np.array([units[k]["n_gt"] + units[k]["n_pred"] for k in keys], float)
    idx = np.random.default_rng(seed).integers(0, len(keys), (n, len(keys)))
    f1 = 2 * tp[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return [float(np.percentile(f1, 100 * alpha / 2)), float(np.percentile(f1, 100 * (1 - alpha / 2)))]
