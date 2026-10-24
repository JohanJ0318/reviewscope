"""Torch-free inference: onnxruntime + the Rust `tokenizers` package.

Everything that runs after training (index build, search, RAG, the API) goes
through these two classes, so query embeddings at serve time are produced by
exactly the graph that embedded the corpus.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _tokenizer(path: Path, max_len: int):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    tok.enable_truncation(max_len)
    tok.enable_padding()
    return tok


def _session(path: Path, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class _OnnxModel:
    def __init__(self, model: Path, tokenizer: Path, max_len: int = 128, threads: int = 4):
        self.tok = _tokenizer(tokenizer, max_len)
        self.sess = _session(model, threads)

    def _run(self, texts: list[str], batch_size: int) -> np.ndarray:
        outs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch(texts[i:i + batch_size])
            feed = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            }
            outs.append(self.sess.run(None, feed)[0])
        return np.concatenate(outs) if outs else np.empty((0, 0), dtype=np.float32)


class Embedder(_OnnxModel):
    @classmethod
    def from_artifacts(cls, artifacts: str | Path, **kw) -> "Embedder":
        onnx = Path(artifacts) / "onnx"
        return cls(onnx / "embedder.onnx", onnx / "embedder_tokenizer", **kw)

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self._run(texts, batch_size).astype(np.float32)


class Classifier(_OnnxModel):
    @classmethod
    def from_artifacts(cls, artifacts: str | Path, int8: bool = False, **kw) -> "Classifier":
        onnx = Path(artifacts) / "onnx"
        name = "classifier.int8.onnx" if int8 else "classifier.onnx"
        return cls(onnx / name, onnx / "classifier_tokenizer", **kw)

    def prob_recommended(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        logits = self._run(texts, batch_size)
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e[:, 1] / e.sum(axis=1)
