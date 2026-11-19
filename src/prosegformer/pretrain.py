"""Masked-image pretraining on unannotated movies.

- movies: T x Z x Y x X TIFF files in one folder; every frame is normalized with its own percentiles, which are
  computed once and cached next to the run (`stats.json`)
- patches: 13 x 64 x 64, centered near a bright voxel with probability 0.7, flipped and rotated in-plane
- blocks of 2 x 4 x 4 voxels are masked with probability 0.5; masked input voxels are set to the background level
  and their stem features replaced by a learned token
- the network, with a single output, reconstructs the normalized intensities; the loss is an L1 over masked voxels,
  each weighted by 1 + 4 * clip(intensity, 0, 2) so that the sparse puncta count
- the saved weights initialize `prosegformer-train` (`init:` in the configuration)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import __version__
from .config import load_config
from .inference import device_of
from .model import DEFAULT, build_model
from .train import _grad_scaler

log = logging.getLogger("prosegformer")

DEFAULTS = {
    "model": dict(DEFAULT),
    "optim": {"steps": 30000, "batch": 8, "lr": 1.0e-3, "weight_decay": 0.05, "warmup": 1000, "clip": 1.0},
    "mask": {"block": [2, 4, 4], "ratio": 0.5, "bright_weight": 4.0},
    "patch": 64,
    "bright_prob": 0.7,
    "seed": 0,
    "workers": 6,
}


def frame_stats(movies: list[Path], cache: Path) -> dict:
    stats = json.loads(cache.read_text()) if cache.exists() else {}
    for m in movies:
        if m.stem not in stats:
            a = tifffile.memmap(m)
            stats[m.stem] = [[float(v) for v in np.percentile(np.asarray(a[t], np.float32), [50, 99.9])]
                             for t in range(a.shape[0])]
            log.info("%s: %d frames", m.name, a.shape[0])
            cache.write_text(json.dumps(stats))
    return stats


class MoviePatches(Dataset):
    def __init__(self, movies: list[Path], stats: dict, patch: int = 64, bright_prob: float = 0.7, seed: int = 0,
                 n: int = 10 ** 8):
        self.movies, self.P, self.bright_prob, self.seed, self.n = movies, patch, bright_prob, seed, n
        self.stats = [np.array(stats[m.stem]) for m in movies]
        self.index = [(i, t) for i, m in enumerate(movies) for t in range(len(stats[m.stem]))]
        self.maps = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.maps is None:
            self.maps = [tifffile.memmap(m) for m in self.movies]
        r = np.random.default_rng((self.seed, 2, i))
        mi, t = self.index[r.integers(len(self.index))]
        lo, hi = self.stats[mi][t]
        vol = self.maps[mi][t]
        _, Y, X = vol.shape
        P = self.P
        y0, x0 = None, None
        if r.random() < self.bright_prob:
            wy, wx = r.integers(Y - 128 + 1), r.integers(X - 128 + 1)
            sub = (np.asarray(vol[:, wy:wy + 128, wx:wx + 128], np.float32) - lo) / max(hi - lo, 1.0)
            idx = np.argwhere(sub > 1.0)
            if len(idx):
                _, cy, cx = idx[r.integers(len(idx))]
                cy, cx = cy + wy, cx + wx
                y0 = int(np.clip(cy - r.integers(P), 0, Y - P))
                x0 = int(np.clip(cx - r.integers(P), 0, X - P))
        if y0 is None:
            y0, x0 = r.integers(Y - P + 1), r.integers(X - P + 1)
        x = np.clip((np.asarray(vol[:, y0:y0 + P, x0:x0 + P], np.float32) - lo) / max(hi - lo, 1.0), -0.5, 4.0)
        if r.random() < 0.5:
            x = x[:, ::-1]
        if r.random() < 0.5:
            x = x[:, :, ::-1]
        x = np.rot90(x, r.integers(4), axes=(1, 2))
        return torch.from_numpy(np.ascontiguousarray(x))[None]


def block_mask(B: int, shape, block, ratio: float, dev) -> torch.Tensor:
    Z, H, W = shape
    bz, by, bx = block
    g = (torch.rand(B, 1, math.ceil(Z / bz), H // by, W // bx, device=dev) < ratio).float()
    return F.interpolate(g, scale_factor=(bz, by, bx), mode="nearest")[:, :, :Z]


def pretrain(cfg: dict, movies: list[Path], out: Path, device: str | None = None) -> Path:
    torch.manual_seed(cfg["seed"])
    dev = device_of(device)
    amp = dev.type == "cuda"
    out.mkdir(parents=True, exist_ok=True)
    o, mk = cfg["optim"], cfg["mask"]
    mcfg = {**cfg["model"], "outputs": 1}
    model = build_model(mcfg, deep_supervision=False).to(dev)
    stats = frame_stats(movies, out / "stats.json")
    ds = MoviePatches(movies, stats, cfg["patch"], cfg["bright_prob"], cfg["seed"])
    w = cfg["workers"]
    batches = iter(DataLoader(ds, batch_size=o["batch"], num_workers=w, persistent_workers=w > 0,
                              prefetch_factor=4 if w else None))
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or n.endswith(".bias") or n == "mask_token" else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": o["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0}], lr=o["lr"], betas=(0.9, 0.95))
    steps, warm = o["steps"], o["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = _grad_scaler(amp)
    path, t0, running = out / "pretrained.pt", time.time(), []
    for step in range(1, steps + 1):
        model.train()
        x = next(batches).to(dev, non_blocking=True)
        m = block_mask(x.shape[0], x.shape[2:], mk["block"], mk["ratio"], dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            y = model(x * (1 - m), mask=m)
        weight = m * (1 + mk["bright_weight"] * x.clamp(0, 2))
        loss = ((y.float() - x).abs() * weight).sum() / weight.sum()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), o["clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        running.append(float(loss))
        if step % 500 == 0 or step == steps:
            row = {"step": step, "loss": round(float(np.mean(running)), 5), "min": round((time.time() - t0) / 60, 1)}
            running = []
            log.info(json.dumps(row))
            with open(out / "log.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
        if step % 5000 == 0 or step == steps:
            torch.save({"version": __version__, "model": model.state_dict(), "config": mcfg, "kind": "pretrained",
                        "movies": [m.name for m in movies], "step": step}, path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Masked-image pretraining of ProSegFormer on unannotated movies.")
    ap.add_argument("config", type=Path, nargs="?", help="YAML configuration (see configs/)")
    ap.add_argument("--movies", type=Path, required=True, help="folder of T x Z x Y x X TIFF movies")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(DEFAULTS, a.config, a.set)
    movies = sorted(a.movies.glob("*.tif"))
    if not movies:
        raise SystemExit(f"no TIFF movies in {a.movies}")
    log.info("pretrained weights: %s", pretrain(cfg, movies, a.out, a.device))


if __name__ == "__main__":
    main()
