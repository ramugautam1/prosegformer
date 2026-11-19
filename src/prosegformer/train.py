"""Training ProSegFormer.

- every step draws `frame_batch` frame patches and `crop_batch` whole crops; with two outputs, output 0 learns the
  frames and output 1 the crops, each only on its own kind of volume
- loss: binary cross-entropy plus soft Dice at full resolution, and the same on the auxiliary outputs at 1/2 and 1/4
  against max-pooled targets; the frame and crop losses are averaged
- AdamW with linear warm-up and cosine decay, mixed precision, gradient clipping, and an exponential moving average
  of the weights, which is what is validated and saved
- every `eval_every` steps the averaged model is scored on validation (threshold 0.5, no mirroring); the best
  checkpoint by mean frame and crop instance F1 at IoU 0.5 is kept
- the threshold of each output is then chosen on validation, with the inference used at test time
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import __version__
from .config import load_config
from .data import CropSamples, FramePatches, load_split, spacing
from .evaluate import format_summary, predict_set, score, selection_score
from .inference import device_of
from .model import DEFAULT, build_model, size_multiple

log = logging.getLogger("prosegformer")

DEFAULTS = {
    "data": "data/bazooka",
    "model": dict(DEFAULT),
    "init": None,
    "init_heads": False,
    "optim": {"steps": 10000, "lr": 1.0e-3, "weight_decay": 0.05, "warmup": 500, "ema": 0.999, "clip": 1.0,
              "frame_batch": 8, "crop_batch": 16, "crop_weight": 1.0, "patch": 64, "fg_prob": 0.7},
    "loss": {"aux_weights": [0.5, 0.25]},
    "eval": {"every": 1000, "thresholds": [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
             "mirror": True},
    "seed": 0,
    "workers": 6,
}


def seg_loss(logit, y, aux=(), weights=(0.5, 0.25)):
    def one(lg, t):
        p = torch.sigmoid(lg)
        soft_dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
        return F.binary_cross_entropy_with_logits(lg, t) + soft_dice
    loss = one(logit.float(), y)
    for w, a in zip(weights, aux):
        loss = loss + w * one(a.float(), F.adaptive_max_pool3d(y, a.shape[2:]))
    return loss


def init_weights(model, path: Path, heads: bool = False) -> int:
    """Load every tensor of a checkpoint (or plain state dict) that fits; the output layers only with `heads`
    (fine-tuning a trained model rather than starting from a pretrained encoder)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    own = model.state_dict()
    skip = () if heads else ("head.", "aux.")
    state = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape and not k.startswith(skip)}
    model.load_state_dict(state, strict=False)
    return len(state)


def choose_thresholds(probs, vols, sp, grid) -> dict:
    """Best threshold per output on validation (first maximum of instance F1 at IoU 0.5)."""
    out = {}
    for kind, key in (("frames", "frame"), ("crops", "crop")):
        part = [v for v in vols if v.kind == kind]
        if part:
            res = {t: score(probs, part, {"frame": t, "crop": t}, sp)[kind]["f1"][0.5] for t in grid}
            out[key] = max(grid, key=lambda t: res[t])
    return out


def _grad_scaler(enabled: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _batches(ds, batch, workers):
    return iter(DataLoader(ds, batch_size=batch, num_workers=workers, persistent_workers=workers > 0))


def train(cfg: dict, out: Path, fold: int | None = None, device: str | None = None) -> dict:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev = device_of(device)
    amp = dev.type == "cuda"
    out.mkdir(parents=True, exist_ok=True)
    o, mcfg = cfg["optim"], cfg["model"]
    sp = spacing(cfg["data"])
    tr = load_split(cfg["data"], "train", fold)
    va = load_split(cfg["data"], "val", fold)
    frames = [v for v in tr if v.kind == "frames"]
    crops = [v for v in tr if v.kind == "crops"]
    crop_out = mcfg["outputs"] - 1
    model = build_model(mcfg).to(dev)
    if cfg.get("init"):
        n = init_weights(model, Path(cfg["init"]), cfg.get("init_heads", False))
        log.info("init from %s: %d tensors", cfg["init"], n)
    ema = build_model(mcfg).to(dev).eval()
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("%.2f M parameters; %d frames, %d crop frames for training", n_params, len(frames), len(crops))
    w = cfg["workers"]
    frame_iter = _batches(FramePatches(frames, o["patch"], o["fg_prob"], seed=cfg["seed"]), o["frame_batch"], w // 2) \
        if frames else None
    crop_iter = _batches(CropSamples(crops, seed=cfg["seed"]), o["crop_batch"], w // 2) if crops else None
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": o["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0}], lr=o["lr"])
    steps, warm = o["steps"], o["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = _grad_scaler(amp)
    aw = tuple(cfg["loss"]["aux_weights"])
    multiple = size_multiple(mcfg)
    best, history, t0 = (-1.0, 0), [], time.time()

    def step_loss(batch_iter, channel):
        x, y = (t.to(dev, non_blocking=True) for t in next(batch_iter))
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            pred, aux = model(x, aux=True)
        return seg_loss(pred[:, channel:channel + 1], y, [a[:, channel:channel + 1] for a in aux], aw)

    for step in range(1, steps + 1):
        model.train()
        losses = []
        if frame_iter is not None:
            losses.append((step_loss(frame_iter, 0), 1.0))
        if crop_iter is not None:
            losses.append((step_loss(crop_iter, crop_out), o["crop_weight"]))
        loss = sum(l * k for l, k in losses) / sum(k for _, k in losses)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), o["clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        with torch.no_grad():
            d = min(o["ema"], (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        if step % cfg["eval"]["every"] == 0 or step == steps:
            probs = predict_set(ema, va, dev, mcfg["outputs"], mirror=False, multiple=multiple)
            r = score(probs, va, {"frame": 0.5, "crop": 0.5}, sp)
            val = selection_score(r)
            row = {"step": step, "loss": round(float(loss), 4), "val": round(val, 4),
                   **{f"{k}_f1@0.5": round(s["f1"][0.5], 4) for k, s in r.items()},
                   "min": round((time.time() - t0) / 60, 1)}
            history.append(row)
            log.info(json.dumps(row))
            with open(out / "log.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            if val > best[0]:
                best = (val, step)
                torch.save(ema.state_dict(), out / "best_weights.pt")
    ema.load_state_dict(torch.load(out / "best_weights.pt", map_location=dev))
    mirror = cfg["eval"]["mirror"]
    probs = predict_set(ema, va, dev, mcfg["outputs"], mirror=mirror, multiple=multiple)
    thresholds = choose_thresholds(probs, va, sp, cfg["eval"]["thresholds"])
    rv = score(probs, va, thresholds, sp)
    log.info("best step %d; validation at the chosen thresholds:\n%s", best[1], format_summary(rv))
    checkpoint = {"version": __version__, "model": ema.state_dict(), "config": mcfg, "thresholds": thresholds,
                  "inference": {"mirror": mirror, "tile": 64, "stride": 32}, "best_step": best[1],
                  "train": {"data": str(cfg["data"]), "fold": fold, "seed": cfg["seed"]}, "val": rv}
    torch.save(checkpoint, out / "model.pt")
    (out / "best_weights.pt").unlink()
    result = {"config": cfg, "fold": fold, "params_M": n_params, "best_step": best[1], "thresholds": thresholds,
              "val": rv, "history": history}
    (out / "result.json").write_text(json.dumps(result, indent=1, default=float))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train ProSegFormer.")
    ap.add_argument("config", type=Path, nargs="?", help="YAML configuration (see configs/)")
    ap.add_argument("--out", type=Path, required=True, help="run folder; model.pt is the checkpoint")
    ap.add_argument("--fold", type=int, help="fold of a dataset with folds")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override settings, e.g. optim.steps=500")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(DEFAULTS, a.config, a.set)
    if a.seed is not None:
        cfg["seed"] = a.seed
    train(cfg, a.out, a.fold, a.device)


if __name__ == "__main__":
    main()
