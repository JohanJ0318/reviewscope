"""FAISS index over review embeddings, with an optional per-game filter.

Small corpora get an exact IndexFlatIP. Past --flat-max vectors the build
switches to IVF-PQ: vectors are clustered into nlist cells and stored as
product-quantised codes (48 bytes instead of 1,536 for a 384-d float vector),
and a query only scans nprobe cells. At 6.4M reviews that is ~0.3 GB of codes
against ~9.8 GB of raw floats, which is the reason to use a vector index here
at all. The cost is recall, so the build measures recall@k against exact search.

The game filter is an IDSelector applied inside the search, not a post-filter,
so asking for 10 results from one game returns 10 from that game.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Hit:
    review_id: int
    score: float
    app_id: int
    app_name: str
    text: str
    label: int
    helpful: int


def build_index(vectors: np.ndarray, ids: np.ndarray, flat_max: int = 100_000, nprobe: int = 16, seed: int = 7):
    import faiss

    n, d = vectors.shape
    if n <= flat_max:
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(d))
        index.add_with_ids(vectors, ids)
        return index
    nlist = int(4 * np.sqrt(n))
    m = 48 if d % 48 == 0 else 32 if d % 32 == 0 else 16
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(seed)
    train = vectors[rng.choice(n, size=min(n, max(100_000, 40 * nlist)), replace=False)]
    index.train(train)
    index.add_with_ids(vectors, ids)
    index.nprobe = nprobe
    return index


def recall_at_k(index, vectors: np.ndarray, queries: np.ndarray, ids: np.ndarray, k: int = 10) -> float:
    """Share of returned results that belong in the exact top-k (brute force on the raw vectors).

    Scored by similarity rather than by id, so duplicate reviews (identical vectors)
    tied at the k-th place count as correct whichever of them the index returns.
    """
    sims = queries @ vectors.T
    kth = -np.partition(-sims, k - 1, axis=1)[:, k - 1]
    row_of = {int(i): r for r, i in enumerate(ids)}
    _, got = index.search(queries, k)
    hits = 0
    for qi, returned in enumerate(got):
        rows = [row_of[int(i)] for i in returned if i != -1]
        hits += int((sims[qi, rows] >= kth[qi] - 1e-5).sum())
    return hits / (len(queries) * k)


class ReviewIndex:
    def __init__(self, index, meta: pd.DataFrame, info: dict):
        self.index = index
        self.meta = meta.reset_index(drop=True)
        # Positions, not a pandas index: lookups by 64-bit hash ids stay O(1) and warning-free.
        self.row_of = {int(r): i for i, r in enumerate(self.meta.review_id)}
        self.by_app = meta.groupby("app_id")["review_id"].apply(lambda s: s.to_numpy(dtype=np.int64)).to_dict()
        self.info = info

    @classmethod
    def load(cls, artifacts: str | Path) -> "ReviewIndex":
        import faiss

        d = Path(artifacts) / "index"
        index = faiss.read_index(str(d / "reviews.faiss"))
        if hasattr(index, "nprobe"):
            index.nprobe = json.loads((d / "info.json").read_text()).get("nprobe", 16)
        return cls(index, pd.read_parquet(d / "meta.parquet"), json.loads((d / "info.json").read_text()))

    def search(self, query_vec: np.ndarray, k: int = 10, app_id: int | None = None,
               label: int | None = None) -> list[Hit]:
        import faiss

        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        params = None
        if app_id is not None or label is not None:
            if app_id is not None:
                allowed = self.by_app.get(app_id)
                if allowed is None:
                    return []
                if label is not None:
                    labels = self.meta.label.to_numpy()[[self.row_of[int(r)] for r in allowed]]
                    allowed = allowed[labels == label]
            else:
                allowed = self.meta.review_id.to_numpy(dtype=np.int64)[self.meta.label.to_numpy() == label]
            if len(allowed) == 0:
                return []
            sel = faiss.IDSelectorBatch(allowed)
            params = faiss.SearchParametersIVF(sel=sel, nprobe=self.info.get("nprobe", 16)) \
                if self.info["kind"] == "ivfpq" else faiss.SearchParameters(sel=sel)
            k = min(k, len(allowed))
        scores, ids = self.index.search(q, k, params=params)
        hits = []
        for score, rid in zip(scores[0], ids[0]):
            if rid == -1:
                continue
            row = self.meta.iloc[self.row_of[int(rid)]]
            hits.append(Hit(int(rid), float(score), int(row.app_id), str(row.app_name), str(row.text),
                            int(row.label), int(row.helpful)))
        return hits
