"""Synthetic reviews in the Kaggle Steam Reviews schema, for CI and smoke runs.

Columns match andrewmvd/steam-reviews: app_id, app_name, review_text,
review_score (1 recommended / -1 not), review_votes (1 if voted helpful).
Each review mentions one or two aspects with a polarity, and 8% of labels are
flipped, so the classifier cannot be perfect and calibration has something to show.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

ASPECTS = {
    "performance": (["runs at a smooth 144 fps", "barely uses my GPU", "loads in seconds"],
                    ["stutters constantly", "drops to 20 fps in cities", "crashes on alt-tab"]),
    "price": (["worth every penny", "a steal on sale", "fair price for the content"],
              ["way overpriced", "full price for an unfinished game", "microtransactions everywhere"]),
    "story": (["the writing is genuinely moving", "great characters", "the ending got me"],
              ["the plot makes no sense", "dialogue is painful", "story is an afterthought"]),
    "multiplayer": (["matchmaking is quick", "servers are rock solid", "great community"],
                    ["full of cheaters", "servers are always down", "matchmaking takes forever"]),
    "bugs": (["surprisingly polished", "haven't hit a single bug", "patches land fast"],
             ["save files got corrupted", "quests break all the time", "bugs everywhere"]),
    "gameplay": (["combat feels amazing", "the core loop is addictive", "controls are tight"],
                 ["combat is clunky", "gets repetitive after an hour", "controls feel floaty"]),
}
OPENERS = ["", "After 40 hours: ", "Honestly, ", "TL;DR ", "Played with friends. ", "Refunded? No. "]
CLOSERS = ["", " Recommended.", " Would not buy again.", " 7/10.", " Wait for a sale.", " Masterpiece."]
NAMES = ["Iron", "Neon", "Hollow", "Star", "Rogue", "Ember", "Frost", "Echo", "Void", "Crimson"]
NOUNS = ["Frontier", "Protocol", "Kingdoms", "Drift", "Legacy", "Arena", "Odyssey", "Signal"]


def make_rows(n_apps: int, n_reviews: int, seed: int = 7):
    rng = random.Random(seed)
    apps = []
    for i in range(n_apps):
        name = f"{rng.choice(NAMES)} {rng.choice(NOUNS)} {i}"
        # Each game has its own baseline quality and one weak aspect.
        apps.append((10_000 + i * 10, name, rng.uniform(0.3, 0.9), rng.choice(list(ASPECTS))))
    for _ in range(n_reviews):
        app_id, name, quality, weak = rng.choice(apps)
        positive = rng.random() < quality
        aspects = rng.sample(list(ASPECTS), k=rng.choice([1, 2]))
        parts = []
        for a in aspects:
            good, bad = ASPECTS[a]
            pol = positive if a != weak else (positive and rng.random() < 0.3)
            parts.append(rng.choice(good if pol else bad))
        text = rng.choice(OPENERS) + ", ".join(parts).capitalize() + "." + rng.choice(CLOSERS)
        if rng.random() < 0.08:
            positive = not positive
        yield {
            "app_id": app_id,
            "app_name": name,
            "review_text": text,
            "review_score": 1 if positive else -1,
            "review_votes": int(rng.random() < 0.15),
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/raw/sample.csv")
    p.add_argument("--apps", type=int, default=60)
    p.add_argument("--reviews", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["app_id", "app_name", "review_text", "review_score", "review_votes"])
        w.writeheader()
        w.writerows(make_rows(args.apps, args.reviews, args.seed))
    print(f"wrote {args.reviews} reviews for {args.apps} apps to {out}")


if __name__ == "__main__":
    main()
