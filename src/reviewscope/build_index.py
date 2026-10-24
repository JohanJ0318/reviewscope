"""Embed reviews with the exported ONNX embedder and build the FAISS index.

    python -m reviewscope.build_index --limit 1000000

Writes artifacts/index/{reviews.faiss, meta.parquet, info.json}. info.json
records the index kind, size on disk, build time and recall@10 against exact
search on held-out queries.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from reviewscope.data import load_all
from reviewscope.runtime import Embedder
from reviewscope.search import build_index, recall_at_k


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--limit", type=int, default=None, help="index at most this many reviews")
    p.add_argument("--flat-max", type=int, default=100_000, help="above this, use IVF-PQ instead of exact search")
    p.add_argument("--nprobe", type=int, default=16)
    p.add_argument("--queries", type=int, default=200)
    args = p.parse_args()

    import faiss

    out = Path(args.artifacts) / "index"
    out.mkdir(parents=True, exist_ok=True)
    df = load_all(args.processed, args.limit)
    print(f"embedding {len(df):,} reviews")
    emb = Embedder.from_artifacts(args.artifacts)
    t0 = time.perf_counter()
    vectors = emb.embed(df.text.tolist(), batch_size=128)
    embed_s = time.perf_counter() - t0
    ids = df.review_id.to_numpy(dtype=np.int64)

    t0 = time.perf_counter()
    index = build_index(vectors, ids, args.flat_max, args.nprobe)
    build_s = time.perf_counter() - t0
    kind = "flat" if len(df) <= args.flat_max else "ivfpq"

    # Held-out queries: real review texts, perturbed so they are not exact duplicates.
    rng = np.random.default_rng(0)
    q = vectors[rng.choice(len(vectors), size=min(args.queries, len(vectors)), replace=False)]
    q = q + rng.normal(0, 0.02, q.shape).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    recall = recall_at_k(index, vectors, q, ids, k=10)

    faiss.write_index(index, str(out / "reviews.faiss"))
    df.drop(columns=["split"]).to_parquet(out / "meta.parquet", index=False)
    info = {
        "kind": kind,
        "vectors": len(df),
        "dim": int(vectors.shape[1]),
        "nprobe": args.nprobe,
        "recall_at_10": recall,
        "index_mb": round((out / "reviews.faiss").stat().st_size / 1e6, 2),
        "raw_float32_mb": round(vectors.nbytes / 1e6, 2),
        "embed_seconds": round(embed_s, 1),
        "build_seconds": round(build_s, 1),
    }
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
