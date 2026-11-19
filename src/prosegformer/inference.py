"""Prediction with a trained ProSegFormer.

- fields larger than a tile are predicted in overlapping 64 x 64 tiles (stride 32) that keep every slice, blended
  with Gaussian weights; smaller volumes, such as tracking crops, are predicted whole
- mirror averaging (default): the probabilities of the four in-plane flips are averaged
- the model has one output per annotation convention: `frame` for whole fields drawn as binary masks (use it for new
  data), `crop` for the instance masks of the tracking crops
- objects are connected components of the thresholded probability (26-connected, at least 4 voxels by default)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

from .data import robust_norm
from .model import build_model, size_multiple

OUTPUTS = {"frame": 0, "crop": 1}


def device_of(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path, device: torch.device | None = None, deep_supervision: bool = False):
    """-> (model in eval mode, checkpoint dict without the weights)."""
    device = device or device_of()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["config"], deep_supervision=deep_supervision)
    state = {k: v for k, v in ck["model"].items() if deep_supervision or not k.startswith("aux.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or set(missing) - {"mask_token"}:
        raise ValueError(f"{path} does not match its configuration: missing {missing}, unexpected {unexpected}")
    info = {k: v for k, v in ck.items() if k != "model"}
    return model.to(device).eval(), info


def _flips(mirror: bool) -> list[tuple[int, ...]]:
    return [(), (3,), (4,), (3, 4)] if mirror else [()]


@torch.no_grad()
def _forward(model, batch: torch.Tensor, output: int, mirror: bool) -> np.ndarray:
    amp = batch.device.type == "cuda"
    p = 0
    for f in _flips(mirror):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            o = model(batch.flip(f) if f else batch)
        o = torch.sigmoid(o[:, output:output + 1].float())
        p = p + (o.flip(f) if f else o)
    return (p / len(_flips(mirror)))[:, 0].cpu().numpy()


def _whole(model, image: np.ndarray, device, output: int, mirror: bool, multiple: int) -> np.ndarray:
    _, Y, X = image.shape
    py, px = -Y % multiple, -X % multiple
    x = np.pad(image, ((0, 0), (0, py), (0, px)), mode="reflect") if (py or px) else image
    p = _forward(model, torch.from_numpy(np.ascontiguousarray(x))[None, None].to(device), output, mirror)[0]
    return p[:, :Y, :X]


@torch.no_grad()
def predict(model, image: np.ndarray, device=None, output: int = 0, mirror: bool = True, tile: int = 64,
            stride: int = 32, batch: int = 8, multiple: int = 16) -> np.ndarray:
    """Foreground probability of a normalized Z x Y x X volume."""
    model.eval()
    device = device or next(model.parameters()).device
    Z, Y, X = image.shape
    if Y < tile or X < tile:
        return _whole(model, image, device, output, mirror, multiple)
    ys = sorted(set(list(range(0, Y - tile + 1, stride)) + [Y - tile]))
    xs = sorted(set(list(range(0, X - tile + 1, stride)) + [X - tile]))
    g = np.exp(-((np.arange(tile) - (tile - 1) / 2) ** 2) / (2 * (0.3 * tile) ** 2)).astype(np.float32)
    w = g[:, None] * g[None]
    acc, ws = np.zeros((Z, Y, X), np.float32), np.zeros((Y, X), np.float32)
    tiles = [(y0, x0) for y0 in ys for x0 in xs]
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        b = np.stack([image[:, y0:y0 + tile, x0:x0 + tile] for y0, x0 in chunk])[:, None]
        p = _forward(model, torch.from_numpy(np.ascontiguousarray(b)).to(device), output, mirror)
        for k, (y0, x0) in enumerate(chunk):
            acc[:, y0:y0 + tile, x0:x0 + tile] += p[k] * w
            ws[y0:y0 + tile, x0:x0 + tile] += w
    return acc / np.maximum(ws, 1e-6)


def components(binary: np.ndarray, connectivity: int = 26, min_size: int = 4) -> np.ndarray:
    """Connected components of at least `min_size` voxels, labelled 1..n."""
    struct = ndi.generate_binary_structure(3, 3 if connectivity == 26 else 1)
    lab, n = ndi.label(binary > 0, struct)
    if min_size > 1 and n:
        sizes = np.bincount(lab.ravel())
        lab[np.isin(lab, np.nonzero(sizes < min_size)[0])] = 0
        lab, _ = ndi.label(lab > 0, struct)
    return lab


class Segmenter:
    """A trained model with the thresholds chosen on its validation data.

        seg = Segmenter("checkpoints/prosegformer.pt")
        prob, labels = seg(stack)            # raw Z x Y x X stack
    """

    def __init__(self, checkpoint: Path, device: str | None = None, output: str = "frame", mirror: bool = True):
        self.device = device_of(device)
        self.model, self.info = load_checkpoint(checkpoint, self.device)
        if output not in OUTPUTS or OUTPUTS[output] >= self.info["config"]["outputs"]:
            raise ValueError(f"output {output!r} is not available in {checkpoint}")
        self.output, self.mirror = output, mirror
        self.threshold = (self.info.get("thresholds") or {}).get(output, 0.5)
        self.multiple = size_multiple(self.info["config"])

    def probability(self, stack: np.ndarray) -> np.ndarray:
        return predict(self.model, robust_norm(stack), self.device, OUTPUTS[self.output], self.mirror,
                       multiple=self.multiple)

    def __call__(self, stack: np.ndarray, threshold: float | None = None, connectivity: int = 26, min_size: int = 4):
        p = self.probability(stack)
        t = self.threshold if threshold is None else threshold
        return p, components(p > t, connectivity, min_size)
