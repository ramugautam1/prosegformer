"""Scoring a checkpoint on annotated data.

- every volume is read from the output of its own annotation convention: frames from the frame output, crops from the
  crop output, each at the threshold the checkpoint chose on validation
- objects are 26-connected components of at least 4 voxels, the smallest annotated size
- frames and crops are reported separately; intervals are 95% bootstrap intervals over crops or 64 x 64 tiles
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .data import Volume, load_split, spacing
from .inference import OUTPUTS, components, device_of, load_checkpoint, predict
from .metrics import bootstrap_ci, dice, match_counts, summarize, unit_counts
from .model import size_multiple

log = logging.getLogger("prosegformer")
MIN_SIZE = 4


def output_for(kind: str, outputs: int) -> int:
    return OUTPUTS["crop"] if kind == "crops" and outputs > 1 else OUTPUTS["frame"]


def predict_set(model, vols: list[Volume], device, outputs: int, mirror: bool = True, multiple: int = 16) -> dict:
    return {v.name: predict(model, v.image, device, output_for(v.kind, outputs), mirror, multiple=multiple)
            for v in vols}


def score(probs: dict, vols: list[Volume], thresholds: dict, sp, intervals: bool = False) -> dict:
    """{kind: summary} with thresholds {"frame": t, "crop": t}."""
    out = {}
    for kind in ("frames", "crops"):
        part = [v for v in vols if v.kind == kind]
        if not part:
            continue
        t = thresholds["crop" if kind == "crops" and thresholds.get("crop") is not None else "frame"]
        rows, dices, units = [], [], {}
        for v in part:
            binary = probs[v.name] > t
            pred = components(binary, 26, MIN_SIZE)
            rows.append(match_counts(pred, v.labels, sp))
            dices.append(dice(binary, v.labels))
            if intervals:
                unit = v.name.split("_t")[0]
                for u, c in unit_counts(pred, v.labels, sp, unit, None if kind == "crops" else 64).items():
                    acc = units.setdefault(u, dict.fromkeys(c, 0))
                    for k, x in c.items():
                        acc[k] += x
        s = summarize(rows, dices)
        s["threshold"] = t
        if intervals:
            s["ci"] = {"f1@0.5": bootstrap_ci(units, "tp@0.5"), "f1@0.7": bootstrap_ci(units, "tp@0.7"),
                       "det_f1": bootstrap_ci(units, "det_tp"), "units": len(units)}
        out[kind] = s
    return out


def selection_score(result: dict) -> float:
    """Mean instance F1 at IoU 0.5 over frames and crops."""
    return float(np.mean([result[k]["f1"][0.5] for k in ("frames", "crops") if k in result]))


def format_summary(result: dict) -> str:
    lines = []
    for kind, s in result.items():
        ci = s.get("ci", {})
        f = lambda key, v: f"{v:.3f}" + (f" [{ci[key][0]:.3f}, {ci[key][1]:.3f}]" if key in ci else "")
        lines.append(f"{kind:6s} n={s['n_gt']:5d}  F1@0.5 {f('f1@0.5', s['f1'][0.5])}"
                     f"  F1@0.7 {f('f1@0.7', s['f1'][0.7])}"
                     f"  det F1 {f('det_f1', s['det_f1'])}  Dice {s['dice']:.3f}  (threshold {s['threshold']:.2f})")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score a ProSegFormer checkpoint on annotated volumes.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="dataset folder, e.g. data/bazooka")
    ap.add_argument("--split", default="test", help="train, val, test or all")
    ap.add_argument("--fold", type=int, help="fold of a dataset with folds (Gish, E-cadherin)")
    ap.add_argument("--threshold", type=float, help="one threshold for every volume (default: the checkpoint's)")
    ap.add_argument("--no-mirror", action="store_true", help="no mirror averaging")
    ap.add_argument("--device")
    ap.add_argument("--json", type=Path, help="write the scores here")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dev = device_of(a.device)
    model, info = load_checkpoint(a.checkpoint, dev)
    cfg = info["config"]
    thr = {"frame": a.threshold, "crop": a.threshold} if a.threshold is not None else dict(info["thresholds"])
    vols = load_split(a.data, a.split, a.fold)
    if any(v.kind == "crops" for v in vols) and (cfg["outputs"] < 2 or thr.get("crop") is None):
        thr["crop"] = thr["frame"]
    probs = predict_set(model, vols, dev, cfg["outputs"], not a.no_mirror, size_multiple(cfg))
    res = score(probs, vols, thr, spacing(a.data), intervals=True)
    log.info(format_summary(res))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"checkpoint": str(a.checkpoint), "data": str(a.data), "split": a.split,
                                      "fold": a.fold, "mirror": not a.no_mirror, "scores": res}, indent=1,
                                     default=float))


if __name__ == "__main__":
    main()
