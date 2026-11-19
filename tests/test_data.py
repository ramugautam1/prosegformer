import json

import numpy as np
import pytest

from prosegformer.data import CropSamples, FramePatches, load_split, split_names
from tests.conftest import DATA


def test_bazooka_splits():
    expect = {"train": (6, 16), "val": (1, 4), "test": (1, 8)}
    seen = set()
    for split, (n_frames, n_crops) in expect.items():
        names = split_names(DATA / "bazooka", split)
        assert (len(names["frames"]), len(names["crops"])) == (n_frames, n_crops)
        assert not seen & set(names["crops"])
        seen |= set(names["crops"])
    assert len(seen) == 28


def test_bazooka_volumes():
    vols = load_split(DATA / "bazooka", "test")
    frames = [v for v in vols if v.kind == "frames"]
    crops = [v for v in vols if v.kind == "crops"]
    assert [v.image.shape for v in frames] == [(13, 320, 512)]
    assert len(crops) == 8 * 40 and all(v.image.shape == (13, 32, 32) for v in crops)
    assert crops[0].name == "Crop04_t0"
    for v in frames + crops[:5]:
        assert -0.5 <= v.image.min() and v.image.max() <= 4.0
        assert v.labels.dtype.kind in "iu"


@pytest.mark.parametrize("protein,n", [("gish", 7), ("ecad", 5)])
def test_folds_cover_every_stack(protein, n):
    folds = json.loads((DATA / protein / "splits.json").read_text())["folds"]
    tests = [t for f in folds for t in f["test"]]
    assert sorted(tests) == sorted(split_names(DATA / protein, "all")["frames"]) and len(tests) == n
    for f in folds:
        assert not set(f["train"]) & set(f["test"]) and not set(f["val"]) & set(f["test"])


def test_samples_are_reproducible(tiny_dataset):
    vols = load_split(tiny_dataset, "train")
    fp = FramePatches([v for v in vols if v.kind == "frames"], seed=3)
    cs = CropSamples([v for v in vols if v.kind == "crops"], seed=3)
    for ds, shape in ((fp, (1, 13, 64, 64)), (cs, (1, 13, 32, 32))):
        a, b = ds[7], ds[7]
        assert a[0].shape == shape and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
