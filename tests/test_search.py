import numpy as np
import pandas as pd
import pytest

faiss = pytest.importorskip("faiss")

from reviewscope.search import ReviewIndex, build_index, recall_at_k  # noqa: E402


def _corpus(n=3000, d=64, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    ids = rng.integers(-(2**62), 2**62, size=n, dtype=np.int64)  # xxhash64 ids can be negative
    meta = pd.DataFrame({
        "review_id": ids,
        "app_id": np.arange(n) % 5,
        "app_name": [f"game {i % 5}" for i in range(n)],
        "text": [f"review {i}" for i in range(n)],
        "label": np.arange(n) % 2,
        "helpful": 0,
    })
    return v, ids, meta


@pytest.mark.parametrize("flat_max,kind", [(10_000, "flat"), (100, "ivfpq")])
def test_filter_returns_only_requested_game(flat_max, kind):
    v, ids, meta = _corpus()
    idx = ReviewIndex(build_index(v, ids, flat_max=flat_max, nprobe=64), meta, {"kind": kind, "nprobe": 64})
    hits = idx.search(v[0], k=10, app_id=3)
    assert len(hits) == 10
    assert {h.app_id for h in hits} == {3}
    hits = idx.search(v[0], k=10, app_id=3, label=1)
    assert {(h.app_id, h.label) for h in hits} == {(3, 1)}


def test_flat_index_is_exact():
    v, ids, _ = _corpus()
    index = build_index(v, ids, flat_max=10_000)
    assert recall_at_k(index, v, v[:50], ids) == 1.0


def test_unknown_game_returns_nothing():
    v, ids, meta = _corpus()
    idx = ReviewIndex(build_index(v, ids), meta, {"kind": "flat"})
    assert idx.search(v[0], app_id=999) == []
