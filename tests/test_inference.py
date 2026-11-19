import numpy as np
import pytest
import torch

from prosegformer.inference import Segmenter, load_checkpoint, predict
from prosegformer.model import build_model
from tests.conftest import CHECKPOINTS


def test_tiled_and_whole_predictions_have_the_input_shape():
    torch.manual_seed(0)
    m = build_model({"dims": [8, 16, 32], "heads": [1, 2, 4], "depths": [1, 1, 1]}).eval()
    for shape in [(13, 96, 80), (13, 32, 32), (13, 30, 40)]:
        p = predict(m, np.random.rand(*shape).astype(np.float32), torch.device("cpu"))
        assert p.shape == shape and 0 <= p.min() and p.max() <= 1


def test_mirror_averaging_is_flip_equivariant():
    torch.manual_seed(0)
    m = build_model({"dims": [8, 16, 32], "heads": [1, 2, 4], "depths": [1, 1, 1]}).eval()
    x = np.random.rand(13, 32, 32).astype(np.float32)
    a = predict(m, x, torch.device("cpu"))
    b = predict(m, np.ascontiguousarray(x[:, ::-1]), torch.device("cpu"))[:, ::-1]
    assert np.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("name,outputs", [("prosegformer.pt", 2), ("prosegformer_lite.pt", 2),
                                          ("prosegformer_pretrained.pt", 1), ("prosegformer_lite_pretrained.pt", 1)])
def test_released_checkpoints_load(name, outputs):
    model, info = load_checkpoint(CHECKPOINTS / name, torch.device("cpu"))
    assert info["config"]["outputs"] == outputs
    if outputs == 2:
        assert set(info["thresholds"]) == {"frame", "crop"}


@pytest.mark.slow
def test_released_model_on_a_test_crop():
    import tifffile

    from prosegformer.metrics import match_counts, summarize
    from tests.conftest import DATA

    seg = Segmenter(CHECKPOINTS / "prosegformer.pt", "cpu", output="crop")
    img = tifffile.imread(DATA / "bazooka/crops/images/Crop04.tif")[:4]
    lab = tifffile.imread(DATA / "bazooka/crops/labels/Crop04.tif")[:4]
    rows = [match_counts(seg(img[t])[1], lab[t], (0.5, 0.124, 0.124)) for t in range(4)]
    assert summarize(rows, [1.0])["f1"][0.5] > 0.8
