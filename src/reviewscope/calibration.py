"""Calibration report: ECE, a reliability diagram, and per-game predicted vs actual.

A classifier can be 90% accurate and still say "0.95" when it is right 70% of
the time. The reliability diagram shows that at a glance; ECE is the number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]


def reliability_table(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width bins on predicted probability; empty bins are dropped."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = which == b
        if mask.any():
            rows.append({
                "bin": b,
                "lo": edges[b],
                "hi": edges[b + 1],
                "n": int(mask.sum()),
                "mean_pred": float(probs[mask].mean()),
                "actual": float(labels[mask].mean()),
            })
    return pd.DataFrame(rows)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    t = reliability_table(probs, labels, n_bins)
    if t.empty:
        return 0.0
    return float((t.n * (t.mean_pred - t.actual).abs()).sum() / t.n.sum())


def roc_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Probability a random positive outranks a random negative (Mann-Whitney U, ties averaged)."""
    ranks = pd.Series(probs).rank().to_numpy()
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Unweighted mean of the F1 for each class, so the minority class counts equally."""
    scores = []
    for cls in (0, 1):
        tp = int(((preds == cls) & (labels == cls)).sum())
        fp = int(((preds == cls) & (labels != cls)).sum())
        fn = int(((preds != cls) & (labels == cls)).sum())
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(scores))


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot(preds: dict[str, pd.DataFrame], out_png: Path, min_reviews: int = 30, min_bin: int = 20) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    summary = {}
    for color, (name, df) in zip(SERIES, preds.items()):
        p, y = df.prob.to_numpy(), df.label.to_numpy()
        t = reliability_table(p, y)
        ece = expected_calibration_error(p, y)
        summary[name] = {"ece": ece, "accuracy": float(((p > 0.5) == y).mean()), "roc_auc": roc_auc(p, y),
                         "macro_f1": macro_f1(p > 0.5, y), "n": len(df)}
        # Sparse bins swing wildly; they still count in ECE, weighted by size, but are not drawn.
        shown = t[t.n >= min_bin]
        ax1.plot(shown.mean_pred, shown.actual, color=color, linewidth=2, marker="o", markersize=5,
                 label=f"{name}  ECE {ece:.3f}")

        games = df.groupby("app_id").agg(n=("label", "size"), pred=("prob", "mean"), actual=("label", "mean"))
        games = games[games.n >= min_reviews]
        if len(games):
            ax2.scatter(games.pred, games.actual, s=np.clip(games.n / 4, 12, 120), color=color,
                        alpha=0.75, edgecolors="white", linewidths=1, label=name)
            summary[name]["games"] = len(games)
            summary[name]["game_mae"] = float((games.pred - games.actual).abs().mean())

    for ax in (ax1, ax2):
        ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle="--", label="perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        _style(ax)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    ax1.set_title(f"Reliability per probability bin (bins with >= {min_bin} predictions)", fontsize=10, color=INK, loc="left")
    ax1.set_xlabel("mean predicted P(recommended)", fontsize=9, color=INK_2)
    ax1.set_ylabel("share actually recommended", fontsize=9, color=INK_2)
    ax2.set_title(f"Per game (test set, >= {min_reviews} reviews; size = reviews)", fontsize=10, color=INK, loc="left")
    ax2.set_xlabel("mean predicted P(recommended)", fontsize=9, color=INK_2)
    ax2.set_ylabel("actual positive rate", fontsize=9, color=INK_2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--modes", nargs="+", default=["lora", "head"])
    args = p.parse_args()
    art = Path(args.artifacts)
    preds = {m: pd.read_parquet(art / f"preds_test_{m}.parquet") for m in args.modes if (art / f"preds_test_{m}.parquet").exists()}
    if not preds:
        raise SystemExit(f"no preds_test_*.parquet in {art}; run train_lora first")
    summary = plot(preds, art / "plots" / "calibration.png")
    (art / "calibration.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
