"""Segment puncta in a 3D stack or movie.

- input: a stack or movie (see io.py); every time point is normalized and segmented on its own
- output folder:
  - `probability.tif`: foreground probability, T x Z x Y x X, 8-bit (0-255)
  - `mask.tif`: thresholded foreground, T x Z x Y x X, 0/1
  - `labels.tif`: puncta labelled 1..n in every time point, T x Z x Y x X
  - `objects.csv`: one row per punctum: time point, label, centroid and volume
  - `summary.json`: settings and counts
- `labels.tif` with `--connectivity 6 --min-size 6` gives the objects that ProTrackFormer's segmentation models
  were trained on; `mask.tif` can also be passed to `protrackformer-track` directly
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from . import __version__
from .inference import Segmenter
from .io import read_stack, voxel_size, write_hyperstack

log = logging.getLogger("prosegformer")


def objects_table(labels: np.ndarray, image: np.ndarray, voxel_um) -> list[dict]:
    rows = []
    vox = float(np.prod(voxel_um))
    for t in range(len(labels)):
        ids = np.unique(labels[t][labels[t] > 0])
        if not len(ids):
            continue
        com = np.array(ndi.center_of_mass(labels[t] > 0, labels[t], ids)).reshape(-1, 3) * np.asarray(voxel_um)
        size = ndi.sum_labels(np.ones_like(labels[t]), labels[t], ids)
        mean = ndi.mean(image[t].astype(np.float64), labels[t], ids)
        for i, c, n, m in zip(ids, com, size, mean):
            rows.append({"t": t, "label": int(i), "z_um": round(c[0], 3), "y_um": round(c[1], 3),
                         "x_um": round(c[2], 3),
                         "voxels": int(n), "volume_um3": round(n * vox, 4), "mean_intensity": round(float(m), 2)})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Segment protein puncta in a 3D stack or movie.")
    ap.add_argument("--image", type=Path, required=True, help="TIFF stack or movie, NIfTI, or a folder of t###.tif")
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/prosegformer.pt"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--channel", type=int, help="channel of a multi-channel file")
    ap.add_argument("--frames", help="first:last time point, inclusive, 0-based")
    ap.add_argument("--output", choices=["frame", "crop"], default="frame",
                    help="annotation convention: frame (whole fields, default) or crop (the tracking crops)")
    ap.add_argument("--threshold", type=float, help="probability threshold (default: the checkpoint's)")
    ap.add_argument("--connectivity", type=int, choices=[6, 26], default=26)
    ap.add_argument("--min-size", type=int, default=4, help="smallest object, in voxels")
    ap.add_argument("--voxel-size", type=float, nargs=3, metavar=("Z", "Y", "X"), help="um (default: from metadata)")
    ap.add_argument("--no-mirror", action="store_true", help="no mirror averaging (four times faster)")
    ap.add_argument("--device")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.time()
    frames = tuple(int(x) for x in a.frames.split(":")) if a.frames else None
    movie = read_stack(a.image, a.channel, frames)
    voxel_um = tuple(a.voxel_size) if a.voxel_size else voxel_size(a.image) or (0.5, 0.124, 0.124)
    seg = Segmenter(a.checkpoint, a.device, a.output, not a.no_mirror)
    thr = seg.threshold if a.threshold is None else a.threshold
    prob = np.zeros(movie.shape, np.float32)
    labels = np.zeros(movie.shape, np.uint32)
    for t in range(len(movie)):
        prob[t], labels[t] = seg(movie[t], thr, a.connectivity, a.min_size)
        log.info("time point %d/%d: %d puncta", t + 1, len(movie), int(labels[t].max()))
    a.out.mkdir(parents=True, exist_ok=True)
    write_hyperstack(a.out / "probability.tif", np.round(prob * 255).astype(np.uint8), voxel_um)
    write_hyperstack(a.out / "mask.tif", (prob > thr).astype(np.uint8), voxel_um)
    dtype = np.uint16 if labels.max() < 2 ** 16 else np.uint32
    write_hyperstack(a.out / "labels.tif", labels.astype(dtype), voxel_um)
    rows = objects_table(labels, movie, voxel_um)
    header = ["t", "label", "z_um", "y_um", "x_um", "voxels", "volume_um3", "mean_intensity"]
    with open(a.out / "objects.csv", "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in header) + "\n")
    summary = {"version": __version__, "image": str(a.image), "checkpoint": str(a.checkpoint),
               "shape": list(movie.shape),
               "voxel_um": list(voxel_um), "output": a.output, "threshold": thr, "mirror": not a.no_mirror,
               "connectivity": a.connectivity, "min_size": a.min_size,
               "puncta_per_time_point": [int(labels[t].max()) for t in range(len(labels))],
               "seconds": round(time.time() - t0, 1)}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    log.info("%d puncta in %d time points -> %s", len(rows), len(movie), a.out)


if __name__ == "__main__":
    main()
