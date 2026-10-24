"""Fine-tune a small encoder to predict "recommended" from review text, with LoRA.

The label is free: every Steam review carries the author's thumbs up/down, so
there are millions of labels and a held-out test set of unseen games is cheap.
That is what makes LoRA worth measuring here rather than just running.

--mode lora  trains low-rank adapters on attention plus the classifier head
--mode head  freezes the encoder and trains only the head (the baseline LoRA must beat)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from reviewscope.data import load_split


def encode(tok, texts: list[str], max_len: int):
    return tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")


def batches(df: pd.DataFrame, batch_size: int, shuffle: bool, seed: int):
    idx = np.arange(len(df))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield df.iloc[idx[start:start + batch_size]]


@torch.no_grad()
def predict(model, tok, df: pd.DataFrame, max_len: int, batch_size: int = 64) -> np.ndarray:
    model.eval()
    probs = []
    for chunk in batches(df, batch_size, shuffle=False, seed=0):
        enc = encode(tok, chunk["text"].tolist(), max_len)
        logits = model(**enc).logits
        probs.append(torch.softmax(logits, dim=-1)[:, 1].numpy())
    return np.concatenate(probs) if probs else np.array([])


def build_model(base: str, mode: str, r: int, alpha: int, targets: list[str] | None):
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2)
    if mode == "head":
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(("classifier", "pre_classifier", "score"))
        return model
    from peft import LoraConfig, TaskType, get_peft_model

    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=r, lora_alpha=alpha, lora_dropout=0.1, target_modules=targets)
    return get_peft_model(model, cfg)


def count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--base", default="distilbert-base-uncased")
    p.add_argument("--mode", choices=["lora", "head"], default="lora")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--targets", nargs="*", default=None, help="module names to adapt; PEFT picks defaults per architecture")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=None, help="default 5e-4 for lora, 1e-3 for head")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--max-train", type=int, default=200_000)
    p.add_argument("--max-eval", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = load_split(args.processed, "train", args.max_train, args.seed)
    val = load_split(args.processed, "val", args.max_eval, args.seed)
    test = load_split(args.processed, "test", args.max_eval, args.seed)
    print(f"train {len(train)}  val {len(val)}  test {len(test)}  positive rate {train.label.mean():.3f}")

    tok = AutoTokenizer.from_pretrained(args.base)
    model = build_model(args.base, args.mode, args.rank, args.alpha, args.targets)
    trainable, total = count_params(model)
    print(f"{args.mode}: training {trainable:,} of {total:,} parameters ({100 * trainable / total:.2f}%)")

    lr = args.lr or (5e-4 if args.mode == "lora" else 1e-3)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    history = []
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, chunk in enumerate(batches(train, args.batch_size, shuffle=True, seed=args.seed + epoch), 1):
            enc = encode(tok, chunk["text"].tolist(), args.max_len)
            loss = model(**enc, labels=torch.tensor(chunk["label"].values, dtype=torch.long)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{steps // args.epochs} loss {running / 50:.4f}")
                running = 0.0
        val_acc = float(((predict(model, tok, val, args.max_len) > 0.5) == val.label.values).mean())
        history.append({"epoch": epoch, "val_accuracy": val_acc})
        print(f"epoch {epoch} val accuracy {val_acc:.4f}")
    train_seconds = time.perf_counter() - t0

    test_probs = predict(model, tok, test, args.max_len)
    test_acc = float(((test_probs > 0.5) == test.label.values).mean())
    majority = float(max(test.label.mean(), 1 - test.label.mean()))
    print(f"test accuracy {test_acc:.4f}  (majority class {majority:.4f})")

    preds = test[["review_id", "app_id", "app_name", "label"]].assign(prob=test_probs)
    preds.to_parquet(out / f"preds_test_{args.mode}.parquet", index=False)

    metrics = {
        "mode": args.mode,
        "base": args.base,
        "trainable_params": trainable,
        "total_params": total,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_accuracy": test_acc,
        "majority_baseline": majority,
        "train_seconds": round(train_seconds, 1),
        "history": history,
        "lora": {"rank": args.rank, "alpha": args.alpha} if args.mode == "lora" else None,
    }
    (out / f"metrics_{args.mode}.json").write_text(json.dumps(metrics, indent=2))

    # Merge the adapters into the base weights so export and serving see a plain model.
    final = model.merge_and_unload() if args.mode == "lora" else model
    final.save_pretrained(out / f"classifier_{args.mode}")
    tok.save_pretrained(out / f"classifier_{args.mode}")
    print(f"saved {out / f'classifier_{args.mode}'}")


if __name__ == "__main__":
    main()
