"""HTTP API: classify review text and search reviews. No torch in this process.

    uvicorn reviewscope.serve:app --port 8000

GET  /healthz
POST /classify   {"texts": ["..."]}                 -> P(recommended) per text
GET  /search?q=...&app_id=...&k=10                  -> nearest reviews
"""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from reviewscope.runtime import Classifier, Embedder
from reviewscope.search import ReviewIndex

ARTIFACTS = os.environ.get("REVIEWSCOPE_ARTIFACTS", "artifacts")
INT8 = os.environ.get("REVIEWSCOPE_INT8", "0") == "1"

app = FastAPI(title="reviewscope")


@lru_cache(maxsize=1)
def classifier() -> Classifier:
    return Classifier.from_artifacts(ARTIFACTS, int8=INT8)


@lru_cache(maxsize=1)
def embedder() -> Embedder:
    return Embedder.from_artifacts(ARTIFACTS)


@lru_cache(maxsize=1)
def index() -> ReviewIndex:
    return ReviewIndex.load(ARTIFACTS)


class ClassifyIn(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)


@app.get("/healthz")
def healthz() -> dict:
    # Loading here makes readiness mean "models are in memory", not "process is up".
    classifier(), embedder(), index()
    return {"ok": True, "index": index().info["kind"], "vectors": index().info["vectors"]}


@app.post("/classify")
def classify(body: ClassifyIn) -> dict:
    probs = classifier().prob_recommended(body.texts)
    return {"prob_recommended": [round(float(p), 4) for p in probs]}


@app.get("/search")
def search(q: str = Query(min_length=2), app_id: int | None = None, k: int = Query(10, ge=1, le=50)) -> dict:
    hits = index().search(embedder().embed([q])[0], k=k, app_id=app_id)
    if app_id is not None and not hits:
        raise HTTPException(404, f"no reviews indexed for app {app_id}")
    return {"hits": [h.__dict__ for h in hits]}
