"""Annotated volumes and training samples.

- a dataset folder holds `frames/` (single time points of whole fields, binary masks drawn as a whole) and, for
  Bazooka, `crops/` (small tracking crops over time, instance masks drawn one punctum at a time); see data/README.md
- every volume is normalized on its own as (x - p50) / (p99.9 - p50), clipped to [-0.5, 4]
- a crop movie is split into its time points, named `<crop>_t<frame>`
- training draws frame patches (13 x 64 x 64, centered on foreground with probability 0.7) and whole crops at their
  native size; both are flipped, rotated in-plane and jittered in intensity
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

KINDS = ("frames", "crops")


@dataclass
class Volume:
    name: str
    kind: str  # "frames" or "crops"
    image: np.ndarray  # normalized, float32, Z x Y x X
    labels: np.ndarray  # instance labels, Z x Y x X

    @property
    def foreground(self) -> np.ndarray:
        return (self.labels > 0).astype(np.float32)


def robust_norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 50), np.percentile(x, 99.9)
    return np.clip((x - lo) / max(hi - lo, 1.0), -0.5, 4.0)


def meta(root: Path) -> dict:
    return json.loads((Path(root) / "meta.json").read_text())


def spacing(root: Path) -> tuple[float, float, float]:
    return tuple(meta(root)["spacing_um"])


def split_names(root: Path, split: str, fold: int | None = None) -> dict[str, list[str]]:
    """{kind: names} of one split. Datasets with folds take `fold`; `all` is every annotated volume."""
    s = json.loads((Path(root) / "splits.json").read_text())
    if split == "all":
        return {k: sorted(p.stem for p in (Path(root) / k / "images").glob("*.tif")) for k in KINDS
                if (Path(root) / k).is_dir()}
    if "folds" in s:
        if fold is None:
            raise ValueError(f"{root} is split into folds: pass a fold (0-{len(s['folds']) - 1})")
        s = s["folds"][fold]
    part = s[split]
    return part if isinstance(part, dict) else {"frames": part}


def load_split(root: Path, split: str, fold: int | None = None) -> list[Volume]:
    root = Path(root)
    out = []
    for kind, names in split_names(root, split, fold).items():
        for n in names:
            img = tifffile.imread(root / kind / "images" / f"{n}.tif")
            lab = tifffile.imread(root / kind / "labels" / f"{n}.tif")
            if kind == "crops":
                out += [Volume(f"{n}_t{t}", kind, robust_norm(img[t]), lab[t]) for t in range(len(img))]
            else:
                out.append(Volume(n, kind, robust_norm(img), lab))
    return out


def augment(r: np.random.Generator, *arrays: np.ndarray, intensity: bool = True) -> list[np.ndarray]:
    if r.random() < 0.5:
        arrays = [a[:, ::-1] for a in arrays]
    if r.random() < 0.5:
        arrays = [a[:, :, ::-1] for a in arrays]
    k = r.integers(4)
    arrays = [np.rot90(a, k, axes=(1, 2)) for a in arrays]
    if intensity:
        img = arrays[0] * r.uniform(0.8, 1.25) + r.normal(0, 0.03)
        arrays = [img + r.normal(0, r.uniform(0, 0.05), img.shape)] + list(arrays[1:])
    return [np.ascontiguousarray(a, dtype=np.float32) for a in arrays]


class FramePatches(Dataset):
    """Random patches of whole-field volumes; sample i is a pure function of (seed, i)."""

    def __init__(self, vols: list[Volume], patch: int = 64, fg_prob: float = 0.7, seed: int = 0, n: int = 10 ** 7):
        self.vols, self.P, self.fg_prob, self.seed, self.n = vols, patch, fg_prob, seed, n
        self.fg_idx = [np.argwhere(v.labels > 0) for v in vols]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        r = np.random.default_rng((self.seed, i))
        k = r.integers(len(self.vols))
        img, lab = self.vols[k].image, self.vols[k].foreground
        _, Y, X = img.shape
        P = self.P
        if r.random() < self.fg_prob and len(self.fg_idx[k]):
            _, cy, cx = self.fg_idx[k][r.integers(len(self.fg_idx[k]))]
            y0 = int(np.clip(cy - r.integers(P), 0, Y - P))
            x0 = int(np.clip(cx - r.integers(P), 0, X - P))
        else:
            y0, x0 = r.integers(Y - P + 1), r.integers(X - P + 1)
        a, b = augment(r, img[:, y0:y0 + P, x0:x0 + P], lab[:, y0:y0 + P, x0:x0 + P])
        return torch.from_numpy(a)[None], torch.from_numpy(b)[None]


class CropSamples(Dataset):
    """Whole crops at their native size."""

    def __init__(self, vols: list[Volume], seed: int = 0, n: int = 10 ** 7):
        self.vols, self.seed, self.n = vols, seed, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        r = np.random.default_rng((self.seed, 1, i))
        v = self.vols[r.integers(len(self.vols))]
        a, b = augment(r, v.image, v.foreground)
        return torch.from_numpy(a)[None], torch.from_numpy(b)[None]
