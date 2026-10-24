import numpy as np

from reviewscope.calibration import expected_calibration_error, macro_f1, reliability_table, roc_auc


def test_roc_auc_known_values():
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
    assert roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == 0.5


def test_macro_f1_punishes_always_majority():
    labels = np.r_[np.ones(80), np.zeros(20)].astype(int)
    always_yes = np.ones(100, dtype=int)
    # Accuracy would be 0.8; F1 is 0.889 for the positive class and 0 for the negative.
    assert abs(macro_f1(always_yes, labels) - (2 * 80 / 180) / 2) < 1e-9


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
