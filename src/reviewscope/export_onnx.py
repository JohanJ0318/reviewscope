"""Export the classifier and the embedder to ONNX, check parity, and benchmark.

Why ONNX here: the serving image (Dockerfile, k8s/) runs on onnxruntime and
the Rust-backed `tokenizers` package only, so it never installs torch. This
script proves the exported graphs give the same answers as PyTorch, measures
what the swap costs or saves, and writes a fixed batch of token ids so the C++
runner in cpp/ can check its own outputs against the same expected logits.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from reviewscope.data import load_split

OPSET = 18


class MeanPooledEncoder(torch.nn.Module):
    """Sentence embedding = attention-masked mean of the last hidden state, L2-normalised."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, input_ids, attention_mask):
        hidden = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return torch.nn.functional.normalize(pooled, dim=-1)


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export(module: torch.nn.Module, sample: dict, path: Path, output: str) -> None:
    module.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    dims = {0: "batch", 1: "seq"}
    torch.onnx.export(
        module,
        (sample["input_ids"], sample["attention_mask"]),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=[output],
        dynamic_axes={"input_ids": dims, "attention_mask": dims, output: {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )


def ort_session(path: Path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def time_it(fn, repeats: int) -> dict:
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {"p50_ms": statistics.median(times), "p95_ms": times[int(0.95 * (len(times) - 1))]}


def write_cpp_fixture(enc: dict, logits: np.ndarray, out_dir: Path) -> None:
    """tokens.bin: int64 n, int64 seq, then n*seq ids, then n*seq mask. expected.bin: float32 n*2."""
    ids = enc["input_ids"].numpy().astype(np.int64)
    mask = enc["attention_mask"].numpy().astype(np.int64)
    with (out_dir / "tokens.bin").open("wb") as f:
        np.array(ids.shape, dtype=np.int64).tofile(f)
        ids.tofile(f)
        mask.tofile(f)
    logits.astype(np.float32).tofile(out_dir / "expected.bin")


def plot_bench(bench: dict, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from reviewscope.calibration import INK, INK_2, SERIES, _style

    runtimes = list(bench)
    batch_sizes = sorted({b for r in bench.values() for b in r}, key=int)
    fig, axes = plt.subplots(1, len(batch_sizes), figsize=(5 * len(batch_sizes), 2.8), dpi=150)
    axes = np.atleast_1d(axes)
    for ax, bs in zip(axes, batch_sizes):
        vals = [bench[r][bs]["p50_ms"] for r in runtimes]
        y = np.arange(len(runtimes))
        ax.barh(y, vals, height=0.6, color=SERIES[0])
        for yi, v in zip(y, vals):
            ax.text(v, yi, f"  {v:.1f} ms", va="center", fontsize=9, color=INK)
        ax.set_yticks(y, runtimes)
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.35)
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"p50 latency, batch {bs}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("milliseconds (lower is better)", fontsize=9, color=INK_2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--classifier", default=None, help="default: <artifacts>/classifier_lora")
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--eval-rows", type=int, default=2000)
    args = p.parse_args()

    from onnxruntime.quantization import QuantType, quantize_dynamic

    art = Path(args.artifacts)
    onnx_dir = art / "onnx"
    clf_dir = Path(args.classifier) if args.classifier else art / "classifier_lora"
    tok = AutoTokenizer.from_pretrained(clf_dir)
    clf = LogitsOnly(AutoModelForSequenceClassification.from_pretrained(clf_dir)).eval()

    test = load_split(args.processed, "test", args.eval_rows)
    texts = test.text.tolist()
    fixture = tok(texts[:64], truncation=True, max_length=args.max_len, padding="max_length", return_tensors="pt")

    export(clf, fixture, onnx_dir / "classifier.onnx", "logits")
    quantize_dynamic(str(onnx_dir / "classifier.onnx"), str(onnx_dir / "classifier.int8.onnx"), weight_type=QuantType.QInt8)
    tok.save_pretrained(onnx_dir / "classifier_tokenizer")

    emb_tok = AutoTokenizer.from_pretrained(args.embedder)
    emb = MeanPooledEncoder(AutoModel.from_pretrained(args.embedder)).eval()
    emb_sample = emb_tok(texts[:8], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
    export(emb, emb_sample, onnx_dir / "embedder.onnx", "embedding")
    emb_tok.save_pretrained(onnx_dir / "embedder_tokenizer")

    # Parity: same token ids in, compare outputs.
    sess = ort_session(onnx_dir / "classifier.onnx")
    sess8 = ort_session(onnx_dir / "classifier.int8.onnx")
    feed = {k: fixture[k].numpy() for k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        ref = clf(fixture["input_ids"], fixture["attention_mask"]).numpy()
    onnx_logits = sess.run(None, feed)[0]
    # The C++ runner checks itself against PyTorch, not against Python ORT (which it would match trivially).
    write_cpp_fixture(fixture, ref, onnx_dir)

    emb_sess = ort_session(onnx_dir / "embedder.onnx")
    with torch.no_grad():
        emb_ref = emb(emb_sample["input_ids"], emb_sample["attention_mask"]).numpy()
    emb_out = emb_sess.run(None, {k: emb_sample[k].numpy() for k in ("input_ids", "attention_mask")})[0]

    # Accuracy of each runtime on the same held-out rows; int8 is where it can slip.
    enc = tok(texts, truncation=True, max_length=args.max_len, padding=True, return_tensors="np")
    full = {k: enc[k].astype(np.int64) for k in ("input_ids", "attention_mask")}
    labels = test.label.to_numpy()
    acc = {}
    for name, s in (("onnx fp32", sess), ("onnx int8", sess8)):
        preds = np.concatenate([s.run(None, {k: v[i:i + 256] for k, v in full.items()})[0].argmax(1)
                                for i in range(0, len(labels), 256)])
        acc[name] = float((preds == labels).mean())
    with torch.no_grad():
        torch_preds = np.concatenate([
            clf(**{k: torch.from_numpy(v[i:i + 256]) for k, v in full.items()}).numpy().argmax(1)
            for i in range(0, len(labels), 256)])
    acc["torch"] = float((torch_preds == labels).mean())

    bench: dict[str, dict] = {"torch": {}, "onnx fp32": {}, "onnx int8": {}}
    torch.set_num_threads(4)
    for bs in (1, 32):
        t_in = {k: fixture[k][:bs] for k in ("input_ids", "attention_mask")}
        n_in = {k: v.numpy() for k, v in t_in.items()}
        with torch.no_grad():
            bench["torch"][bs] = time_it(lambda: clf(**t_in), args.repeats)
        bench["onnx fp32"][bs] = time_it(lambda: sess.run(None, n_in), args.repeats)
        bench["onnx int8"][bs] = time_it(lambda: sess8.run(None, n_in), args.repeats)

    sizes = {f.name: round(f.stat().st_size / 1e6, 1) for f in onnx_dir.glob("*.onnx")}
    report = {
        "classifier_max_abs_diff": float(np.abs(ref - onnx_logits).max()),
        "embedder_max_abs_diff": float(np.abs(emb_ref - emb_out).max()),
        "accuracy": acc,
        "eval_rows": len(labels),
        "latency": bench,
        "model_mb": sizes,
        "seq_len": args.max_len,
    }
    (art / "onnx_report.json").write_text(json.dumps(report, indent=2))
    plot_bench(bench, art / "plots" / "latency.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
