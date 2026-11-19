import json

import numpy as np
import pytest
import tifffile

from prosegformer import evaluate, segment, train
from tests.conftest import TINY_MODEL

FAST = ["optim.steps=3", "optim.frame_batch=2", "optim.crop_batch=2", "optim.warmup=1", "eval.every=3",
        "eval.thresholds=[0.3,0.5]", "eval.mirror=false", "workers=0"]


@pytest.mark.slow
def test_train_evaluate_segment(tiny_dataset, tmp_path):
    run = tmp_path / "run"
    train.main(["--out", str(run), "--device", "cpu", "--set", f"data={tiny_dataset}", *TINY_MODEL, *FAST])
    result = json.loads((run / "result.json").read_text())
    assert set(result["thresholds"]) == {"frame", "crop"} and (run / "model.pt").exists()

    evaluate.main(["--checkpoint", str(run / "model.pt"), "--data", str(tiny_dataset), "--split", "test",
                   "--device", "cpu", "--no-mirror", "--json", str(tmp_path / "scores.json")])
    scores = json.loads((tmp_path / "scores.json").read_text())["scores"]
    assert set(scores) == {"frames", "crops"}

    movie = tifffile.imread(tiny_dataset / "crops/images/Crop02.tif")
    tifffile.imwrite(tmp_path / "movie.tif", movie)
    out = tmp_path / "seg"
    segment.main(["--image", str(tmp_path / "movie.tif"), "--checkpoint", str(run / "model.pt"), "--out", str(out),
                  "--device", "cpu", "--no-mirror", "--threshold", "0.5"])
    for f in ("probability.tif", "mask.tif", "labels.tif", "objects.csv", "summary.json"):
        assert (out / f).exists()
    labels = tifffile.imread(out / "labels.tif")
    assert labels.shape == movie.shape
    summary = json.loads((out / "summary.json").read_text())
    assert summary["puncta_per_time_point"] == [int(labels[t].max()) for t in range(len(labels))]
    assert len(open(out / "objects.csv").readlines()) == 1 + int(np.sum(summary["puncta_per_time_point"]))


@pytest.mark.slow
def test_pretrain_then_train(tiny_dataset, tmp_path):
    from prosegformer import pretrain
    from tests.conftest import DATA

    movies = tmp_path / "movies"
    movies.mkdir()
    frame = tifffile.imread(DATA / "bazooka/frames/images/frame0.tif")
    tifffile.imwrite(movies / "m1.tif", np.stack([frame[:, :128, :128], frame[:, 128:256, :128]]))
    pretrain.main(["--movies", str(movies), "--out", str(tmp_path / "ssl"), "--device", "cpu", "--set",
                   *TINY_MODEL, "optim.steps=2", "optim.batch=2", "optim.warmup=1", "workers=0"])
    weights = tmp_path / "ssl" / "pretrained.pt"
    assert weights.exists() and (tmp_path / "ssl" / "stats.json").exists()
    run = tmp_path / "run"
    train.main(["--out", str(run), "--device", "cpu", "--set", f"data={tiny_dataset}", f"init={weights}",
                *TINY_MODEL, *FAST])
    assert (run / "model.pt").exists()
