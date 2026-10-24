import numpy as np

from reviewscope.calibration import expected_calibration_error, reliability_table


def test_perfectly_calibrated_has_near_zero_ece():
    rng = np.random.default_rng(0)
    probs = rng.uniform(size=200_000)
    labels = (rng.uniform(size=probs.size) < probs).astype(int)
    assert expected_calibration_error(probs, labels) < 0.01


def test_overconfident_model_is_caught():
    # Says 0.95 every time but is right 70% of the time.
    probs = np.full(1000, 0.95)
    labels = np.r_[np.ones(700), np.zeros(300)].astype(int)
    assert abs(expected_calibration_error(probs, labels) - 0.25) < 1e-9


def test_reliability_table_drops_empty_bins():
    t = reliability_table(np.array([0.05, 0.06, 0.92]), np.array([0, 0, 1]))
    assert t.bin.tolist() == [0, 9]
    assert t.n.tolist() == [2, 1]
