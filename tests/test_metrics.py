import numpy as np

from prosegformer.inference import components
from prosegformer.metrics import bootstrap_ci, dice, match_counts, summarize, unit_counts

SP = (0.5, 0.124, 0.124)


def test_perfect_prediction(blobs):
    s = summarize([match_counts(blobs, blobs, SP)], [dice(blobs, blobs)])
    assert s["f1"][0.5] == s["f1"][0.9] == s["det_f1"] == s["dice"] == s["correct"] == 1.0
    assert s["missed"] == s["split"] == s["fused"] == s["spurious"] == 0


def test_missed_spurious_and_fused(blobs):
    pred = blobs.copy()
    pred[blobs == 3] = 0  # missed
    pred[10:12, 25:30, 25:30] = 4  # spurious
    c = match_counts(pred, blobs, SP)
    assert (c["missed"], c["spurious"], c["tp@0.5"]) == (1, 1, 2)
    fused = (blobs > 0).astype(int)
    fused[4:8, 4:20, 4:20] = 1
    c = match_counts(components(fused), blobs, SP)
    assert c["fused"] == 2


def test_components_connectivity():
    b = np.zeros((3, 4, 4), bool)
    b[0, 0, 0] = b[1, 1, 1] = True  # corner contact only
    assert components(b, 26, 1).max() == 1
    assert components(b, 6, 1).max() == 2
    assert components(b, 26, 4).max() == 0


def test_units_and_interval(blobs):
    u = unit_counts(blobs, blobs, SP, "v", tile=16)
    assert sum(x["n_gt"] for x in u.values()) == 3
    lo, hi = bootstrap_ci(u)
    assert lo == hi == 1.0
