"""Answer a question about one game from its reviews, with Claude on Amazon Bedrock.

    export BEDROCK_MODEL_ID=<a Claude model or inference profile id in your region>
    python -m reviewscope.rag_bedrock --app-id 10010 "What do players say about performance?"

Retrieval is balanced: half the snippets come from positive reviews and half
from negative ones, so the answer is not just the majority sentiment repeated.
Every claim must cite a snippet as [n]; citations that point at no snippet are
reported so they can be counted as hallucinations.

List the ids available to your account with:
    aws bedrock list-inference-profiles --query "inferenceProfileSummaries[].inferenceProfileId"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

from reviewscope.runtime import Embedder
from reviewscope.search import Hit, ReviewIndex

SYSTEM = (
    "You summarise what players say about a video game, using only the numbered review "
    "snippets provided. Cite every claim with the snippet number in square brackets, e.g. [3]. "
    "If the snippets do not answer the question, say so. Note where positive and negative "
    "reviews disagree. Keep it under 200 words."
)


def retrieve(index: ReviewIndex, emb: Embedder, question: str, app_id: int, k: int = 12) -> list[Hit]:
    q = emb.embed([question])[0]
    pos = index.search(q, k // 2, app_id=app_id, label=1)
    neg = index.search(q, k - len(pos), app_id=app_id, label=0)
    return sorted(pos + neg, key=lambda h: -h.score)


def build_prompt(question: str, hits: list[Hit]) -> str:
    lines = [f"Game: {hits[0].app_name}" if hits else "Game: unknown", "", "Review snippets:"]
    for i, h in enumerate(hits, 1):
        verdict = "recommended" if h.label else "not recommended"
        lines.append(f"[{i}] ({verdict}) {h.text[:600]}")
    lines += ["", f"Question: {question}"]
    return "\n".join(lines)


def check_citations(answer: str, n_snippets: int) -> dict:
    cited = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    return {"cited": sorted(cited), "invalid": sorted(c for c in cited if not 1 <= c <= n_snippets)}


def ask_bedrock(prompt: str, model_id: str, region: str | None, max_tokens: int = 600) -> dict:
    import boto3

    client = boto3.client("bedrock-runtime", region_name=region)
    t0 = time.perf_counter()
    resp = client.converse(
        modelId=model_id,
        system=[{"text": SYSTEM}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.2},
    )
    text = "".join(block.get("text", "") for block in resp["output"]["message"]["content"])
    return {"answer": text, "usage": resp.get("usage", {}), "latency_s": round(time.perf_counter() - t0, 2),
            "stop_reason": resp.get("stopReason")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question")
    p.add_argument("--app-id", type=int, required=True)
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--model-id", default=os.environ.get("BEDROCK_MODEL_ID"))
    p.add_argument("--region", default=os.environ.get("AWS_REGION"))
    p.add_argument("--dry-run", action="store_true", help="print the prompt; do not call Bedrock")
    args = p.parse_args()

    index = ReviewIndex.load(args.artifacts)
    hits = retrieve(index, Embedder.from_artifacts(args.artifacts), args.question, args.app_id, args.k)
    if not hits:
        raise SystemExit(f"no reviews indexed for app {args.app_id}")
    prompt = build_prompt(args.question, hits)
    if args.dry_run:
        print(prompt)
        return
    if not args.model_id:
        raise SystemExit("set BEDROCK_MODEL_ID or pass --model-id (see --help)")
    result = ask_bedrock(prompt, args.model_id, args.region)
    result["citations"] = check_citations(result["answer"], len(hits))
    print(result["answer"])
    print()
    print(json.dumps({k: v for k, v in result.items() if k != "answer"}, indent=2))


if __name__ == "__main__":
    main()
