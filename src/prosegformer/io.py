"""Reading stacks and movies, and writing results.

- input: a 3D TIFF (Z, Y, X), a 4D TIFF (T, Z, Y, X), a TIFF with a channel axis (pass the channel), a folder of
  per-frame `t###.tif` files, or a NIfTI file (x, y, z[, t])
- output volumes are ImageJ hyperstacks (T, Z, Y, X) with the voxel size in their metadata, so they open in Fiji and
  napari with the right scale
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile


def read_stack(path: Path, channel: int | None = None, frames: tuple[int, int] | None = None) -> np.ndarray:
    """Return a T x Z x Y x X array (T = 1 for a single stack). `frames` = (first, last), inclusive."""
    path = Path(path)
    if path.is_dir():
        files = sorted(f for f in path.iterdir() if re.fullmatch(r"t\d+\.tif", f.name))
        if not files:
            raise FileNotFoundError(f"no t###.tif files in {path}")
        if frames:
            files = files[frames[0]:frames[1] + 1]
        return np.stack([tifffile.imread(f) for f in files])
    if path.name.endswith((".nii", ".nii.gz")):
        import nibabel as nib

        arr = np.asarray(nib.load(str(path)).dataobj)
        arr = arr.transpose(3, 2, 1, 0) if arr.ndim == 4 else arr.transpose(2, 1, 0)[None]
    else:
        with tifffile.TiffFile(path) as tf:
            arr, axes = tf.series[0].asarray(), tf.series[0].axes.upper()
        if "C" in axes:
            if channel is None:
                raise ValueError(f"{path.name} has a channel axis ({axes}); pass --channel")
            arr = np.take(arr, channel, axis=axes.index("C"))
            axes = axes.replace("C", "")
        if arr.ndim == 3:
            arr = arr[None]
        elif arr.ndim != 4:
            raise ValueError(f"cannot read {path.name} with axes {axes} as T, Z, Y, X")
    if frames:
        arr = arr[frames[0]:frames[1] + 1]
    return np.ascontiguousarray(arr)


def voxel_size(path: Path) -> tuple[float, float, float] | None:
    """(z, y, x) in um from ImageJ TIFF metadata, if present."""
    try:
        with tifffile.TiffFile(path) as tf:
            meta = tf.imagej_metadata or {}
            res = tf.pages[0].tags.get("XResolution"), tf.pages[0].tags.get("YResolution")
            if "spacing" not in meta or None in res:
                return None
            x = res[0].value[1] / res[0].value[0]
            y = res[1].value[1] / res[1].value[0]
            return float(meta["spacing"]), float(y), float(x)
    except (OSError, ValueError, ZeroDivisionError, tifffile.TiffFileError):
        return None


def write_hyperstack(path: Path, arr: np.ndarray, voxel_um=(0.5, 0.124, 0.124)) -> None:
    """Write a T x Z x Y x X array as an ImageJ hyperstack."""
    z, y, x = voxel_um
    tifffile.imwrite(path, arr, imagej=True, resolution=(1 / x, 1 / y), compression="zlib",
                     metadata={"axes": "TZYX", "spacing": z, "unit": "um"})
