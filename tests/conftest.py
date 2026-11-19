import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CHECKPOINTS = ROOT / "checkpoints"
TINY_MODEL = ["model.dims=[8,16,32]", "model.heads=[1,2,4]", "model.depths=[1,1,1]"]


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory) -> Path:
    """Two 13 x 64 x 64 frames and two 3-frame crops cut from the Bazooka benchmark."""
    out = tmp_path_factory.mktemp("tiny")
    src = DATA / "bazooka"
    for kind in ("frames", "crops"):
        for part in ("images", "labels"):
            (out / kind / part).mkdir(parents=True)
    for i, (name, y, x) in enumerate([("frame0", 96, 160), ("frame6", 64, 256)]):
        for part in ("images", "labels"):
            a = tifffile.imread(src / "frames" / part / f"{name}.tif")[:, y:y + 64, x:x + 64]
            tifffile.imwrite(out / "frames" / part / f"f{i}.tif", a)
    for name in ("Crop01", "Crop02"):
        for part in ("images", "labels"):
            movie = tifffile.imread(src / "crops" / part / f"{name}.tif")
            tifffile.imwrite(out / "crops" / part / f"{name}.tif", movie[:3])
    splits = {"train": {"frames": ["f0"], "crops": ["Crop01"]}, "val": {"frames": ["f1"], "crops": ["Crop02"]},
              "test": {"frames": ["f1"], "crops": ["Crop02"]}}
    (out / "splits.json").write_text(json.dumps(splits))
    (out / "meta.json").write_text(json.dumps({"spacing_um": [0.5, 0.124, 0.124]}))
    return out


@pytest.fixture
def blobs() -> np.ndarray:
    """Labels of three separate cubes in a 13 x 32 x 32 volume."""
    lab = np.zeros((13, 32, 32), np.int32)
    lab[4:7, 4:9, 4:9] = 1
    lab[5:8, 15:20, 15:20] = 2
    lab[2:4, 25:30, 3:8] = 3
    return lab
